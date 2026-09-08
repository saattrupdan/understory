import { tool } from "ai";
import { z } from "zod";
import type { KnowledgeBase } from "../okf/index.js";
import type { Concept, TreeNode } from "../okf/types.js";
import type { TraceRecorder } from "./trace.js";
import { recordHotDelete, recordHotWrite } from "./hot-memory.js";
import { resolveAgentLimits } from "./limits.js";
import { AgentRunContext } from "./run-context.js";

/** Bundle-relative concept path, e.g. "/tables/customers.md". */
const conceptPath = z
  .string()
  .describe('Bundle-relative path starting with "/", ending in .md');

const frontmatterSchema = z
  .object({
    type: z.string().min(1).describe("Concept kind, e.g. 'API Endpoint'. Required."),
    title: z.string().optional(),
    description: z.string().optional().describe("One-line summary"),
    resource: z.string().optional().describe("Canonical URI of the underlying asset"),
    tags: z.array(z.string()).optional(),
  })
  .passthrough()
  .describe("YAML frontmatter. Additional producer-defined keys are allowed.");

const logSummary = z
  .string()
  .describe(
    "One past-tense sentence for the update log, with bundle-relative links, e.g. 'Added [Billing API](/apis/billing-api.md).'"
  );

export interface ReadPage {
  path: string;
  frontmatter: Concept["frontmatter"];
  body: string;
  offset: number;
  total_chars: number;
  truncated: boolean;
  next_offset: number | null;
}

function readPage(concept: Concept, offset: number, maxChars: number): ReadPage {
  if (!Number.isSafeInteger(offset) || offset < 0 || offset > concept.body.length) {
    throw new Error(
      `Invalid offset ${offset} for ${concept.path}; expected an integer from 0 to ${concept.body.length}`
    );
  }
  const body = concept.body.slice(offset, offset + maxChars);
  const nextOffset = offset + body.length;
  const truncated = nextOffset < concept.body.length;
  return {
    path: concept.path,
    frontmatter: concept.frontmatter,
    body,
    offset,
    total_chars: concept.body.length,
    truncated,
    next_offset: truncated ? nextOffset : null,
  };
}

function boundText(text: string, maxChars: number): string {
  if (text.length <= maxChars) return text;
  const marker = `\n... [truncated; total_chars=${text.length}]`;
  if (marker.length >= maxChars) return text.slice(0, maxChars);
  const available = maxChars - marker.length;
  const lineEnd = text.lastIndexOf("\n", available);
  return text.slice(0, lineEnd > 0 ? lineEnd : available) + marker;
}

function isReadPage(value: unknown): value is ReadPage {
  return typeof value === "object" && value !== null && "body" in value && typeof value.body === "string";
}

export function buildReadTools(
  kb: KnowledgeBase,
  trace?: TraceRecorder,
  state: AgentRunContext = new AgentRunContext(resolveAgentLimits())
) {
  return {
    search_knowledge: tool({
      description:
        "Search the knowledge base by keywords, optionally filtered by concept type and/or tags. Returns ranked hits with paths and snippets. NOTE: matching is keyword-based, not semantic — a miss does NOT mean the knowledge is absent; it may be worded differently.",
      inputSchema: z.object({
        query: z.string().describe("Keywords to search for. May be empty when filtering by type/tags only."),
        type: z.string().optional().describe("Exact concept type filter"),
        tags: z.array(z.string()).optional().describe("Require ALL of these tags"),
      }),
      execute: async ({ query, type, tags }) => {
        const hits = await kb.search(query, { type, tags });
        trace?.record("search_knowledge", query, hits.map((h) => h.path));
        if (hits.length > 0) {
          return state.fits(hits)
            ? state.result(hits)
            : state.result({
                truncated: true,
                notice: "Search results were truncated by the per-run output budget; narrow the search or start a fresh request.",
                hits,
              });
        }
        // Keyword miss ≠ knowledge absent. Put the map in the tool result so
        // the model's next step is to read plausible concepts, not give up.
        // Paths and types only: this lands in the transcript at every missed
        // step, and descriptions would triple its cost for no navigation gain.
        const tree = boundText(formatTree(await kb.listTree(), 0, false), state.remaining);
        return state.result({
          hits: [],
          notice:
            "No keyword matches — but this search is literal, not semantic. Before concluding it is absent, retry with synonyms and review the layout before reading plausible concepts.",
          bundle_layout: tree,
        });
      },
    }),
    read_concept: tool({
      description:
        "Read one concept's frontmatter and a bounded body page. If truncated, use next_offset to page through the body before replacing it.",
      inputSchema: z.object({
        path: conceptPath,
        offset: z.number().int().min(0).default(0).describe("Body character offset"),
      }),
      execute: async ({ path, offset = 0 }) => {
        const c = await kb.readConcept(path);
        trace?.record("read_concept", c.path, [c.path]);
        const page = readPage(c, offset, state.maxDocumentChars);
        const result = state.result(page);
        if (isReadPage(result)) {
          state.recordBodyPage(c.path, offset, c.body, page.body, result.body);
        }
        return result;
      },
    }),
    read_concepts: tool({
      description:
        "Read several concepts in one call with frontmatter retained and a bounded aggregate body. " +
        "Each body may be paged with read_concept if its metadata says truncated.",
      inputSchema: z.object({
        paths: z.array(conceptPath).min(1).max(12).describe("Concept paths to read together"),
      }),
      execute: async ({ paths }) => {
        trace?.record("read_concepts", paths.slice(0, 3).join(", "), paths);
        const concepts: ReadPage[] = [];
        const missing: string[] = [];
        let returnedChars = 0;
        let totalChars = 0;
        const sourceConcepts = new Map<string, Concept>();
        const sourcePages = new Map<string, ReadPage>();
        for (const p of paths) {
          try {
            const c = await kb.readConcept(p);
            sourceConcepts.set(c.path, c);
            totalChars += c.body.length;
            const page = readPage(c, 0, state.maxDocumentChars);
            sourcePages.set(c.path, page);
            returnedChars += page.body.length;
            concepts.push(page);
          } catch {
            missing.push(p); // Reported rather than guessed at.
          }
        }
        const result = state.result({
          read: concepts,
          missing,
          returned_body_chars: returnedChars,
          total_body_chars: totalChars,
          truncated: returnedChars < totalChars,
          max_body_chars: state.maxDocumentChars,
        });
        if (typeof result === "object" && result !== null && "read" in result && Array.isArray(result.read)) {
          for (const page of result.read) {
            if (!isReadPage(page)) continue;
            const source = sourceConcepts.get(page.path);
            const sourcePage = sourcePages.get(page.path);
            if (source && sourcePage) {
              state.recordBodyPage(source.path, 0, source.body, sourcePage.body, page.body);
            }
          }
        }
        return result;
      },
    }),
    list_directory: tool({
      description:
        "List the bundle's compact directory tree with concept types (descriptions omitted). Use to understand structure and decide where new concepts belong.",
      inputSchema: z.object({}),
      execute: async () => {
        trace?.record("list_directory", "", []);
        return state.result(boundText(formatTree(await kb.listTree(), 0, false), state.remaining));
      },
    }),
    lint_knowledge: tool({
      description:
        "Graph health check: orphaned concepts (nothing links to them) and broken links. Use to find what needs wiring into the graph or fixing.",
      inputSchema: z.object({}),
      execute: async () => {
        trace?.record("lint_knowledge", "", []);
        const report = await kb.lint();
        return state.fits(report)
          ? state.result(report)
          : state.result({
              truncated: true,
              notice: "Lint output was truncated by the per-run output budget; start a fresh request for the full report.",
              report,
            });
      },
    }),
  };
}

export function buildWriteTools(
  kb: KnowledgeBase,
  filesChanged: Set<string>,
  trace?: TraceRecorder,
  state: AgentRunContext = new AgentRunContext(resolveAgentLimits())
) {
  return {
    write_concept: tool({
      description:
        "Create a new concept only; an existing path is rejected and must be changed with patch_concept. Frontmatter must include a non-empty 'type'. index.md and log.md maintenance is automatic — never write those.",
      inputSchema: z.object({
        path: conceptPath,
        frontmatter: frontmatterSchema,
        body: z.string().describe("Markdown body (no frontmatter block)"),
        log_summary: logSummary,
      }),
      execute: async ({ path, frontmatter, body, log_summary }) => {
        const c = await kb.createConcept(path, frontmatter, body, log_summary);
        filesChanged.add(c.path);
        recordHotWrite(c.path);
        trace?.record("write_concept", c.path, [c.path], true);
        return state.result({ written: c.path });
      },
    }),
    patch_concept: tool({
      description:
        "Targeted update of an existing concept: merge frontmatter keys (null deletes a key), replace one top-level '# Section' body section, or replace the whole body. replace_body requires that the complete unchanged body was read in this run; otherwise use replace_section. Prefer this over write_concept for all existing paths.",
      inputSchema: z.object({
        path: conceptPath,
        frontmatter: z
          .record(z.unknown())
          .optional()
          .describe("Frontmatter keys to merge; set a key to null to remove it"),
        replace_section: z
          .object({
            heading: z
              .string()
              .min(1)
              .describe("Top-level heading name, e.g. 'Schema'. Must be non-empty — to replace the whole body use replace_body instead."),
            content: z.string().describe("New content for that section"),
          })
          .optional(),
        replace_body: z
          .string()
          .optional()
          .describe("Replace the entire markdown body (frontmatter untouched). Use for restructuring; prefer replace_section for targeted edits."),
        log_summary: logSummary,
      }),
      execute: async ({ path, frontmatter, replace_section, replace_body, log_summary }) => {
        const expectedBodyHash =
          replace_body === undefined
            ? undefined
            : state.expectedBodyHash(path, (await kb.readConcept(path)).body);
        const c = await kb.patchConcept(
          path,
          {
            frontmatter,
            replaceSection: replace_section
              ? { heading: replace_section.heading, content: replace_section.content }
              : undefined,
            replaceBody: replace_body,
          },
          log_summary,
          expectedBodyHash
        );
        filesChanged.add(c.path);
        recordHotWrite(c.path);
        trace?.record("patch_concept", c.path, [c.path], true);
        return state.result({ patched: c.path });
      },
    }),
    delete_concept: tool({
      description:
        "Permanently delete a concept file. Prefer deprecation (tag 'deprecated' via patch_concept) unless content is wrong/harmful or deletion was explicitly requested.",
      inputSchema: z.object({
        path: conceptPath,
        log_summary: logSummary,
      }),
      execute: async ({ path, log_summary }) => {
        await kb.deleteConcept(path, log_summary);
        filesChanged.add(path);
        recordHotDelete(path);
        trace?.record("delete_concept", path, [path], true);
        return state.result({ deleted: path });
      },
    }),
  };
}

/** Compact indented listing for prompts and the list_directory tool. */
export function formatTree(node: TreeNode, depth = 0, descriptions = true): string {
  const lines: string[] = [];
  if (depth === 0) lines.push("/");
  for (const child of node.children ?? []) {
    const indent = "  ".repeat(depth + 1);
    if (child.kind === "directory") {
      lines.push(`${indent}${child.name}/`);
      lines.push(formatTree(child, depth + 1, descriptions));
    } else if (child.kind === "concept") {
      const meta = [child.type, descriptions ? child.description : undefined]
        .filter(Boolean)
        .join(" — ");
      lines.push(`${indent}${child.name}${meta ? `  [${meta}]` : ""}`);
    }
  }
  return lines.filter(Boolean).join("\n");
}
