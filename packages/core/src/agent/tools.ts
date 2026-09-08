import { tool } from "ai";
import { z } from "zod";
import { BundleError, type KnowledgeBase } from "../okf/index.js";
import type { Concept, TreeNode } from "../okf/types.js";
import type { TraceRecorder } from "./trace.js";
import { recordHotDelete, recordHotWrite } from "./hot-memory.js";
import { resolveAgentLimits } from "./limits.js";
import { AgentRunContext, fitText } from "./run-context.js";

const MAX_CONCEPT_PATH_CHARS = 512;
const MAX_QUERY_CHARS = 2_048;
const MAX_TYPE_CHARS = 256;
const MAX_TAG_CHARS = 128;
const MAX_TAGS = 32;
const MAX_LOG_SUMMARY_CHARS = 1_000;

/** Bundle-relative concept path, e.g. "/tables/customers.md". */
const conceptPath = z
  .string()
  .max(MAX_CONCEPT_PATH_CHARS)
  .regex(/^\/(?!\/).+\.md$/)
  .refine((value) =>
    value.split("/").slice(1).every((segment) => segment.length > 0 && segment !== "." && segment !== "..")
  )
  .describe('Canonical bundle-relative path starting with exactly one "/", ending in .md');

const frontmatterSchema = z
  .object({
    type: z
      .string()
      .min(1)
      .max(MAX_TYPE_CHARS)
      .describe("Concept kind, e.g. 'API Endpoint'. Required."),
    title: z.string().max(512).optional(),
    description: z.string().max(4_000).optional().describe("One-line summary"),
    resource: z.string().max(2_048).optional().describe("Canonical URI of the underlying asset"),
    tags: z.array(z.string().max(MAX_TAG_CHARS)).max(MAX_TAGS).optional(),
  })
  .passthrough()
  .describe("YAML frontmatter. Additional producer-defined keys are allowed.");

const logSummary = z
  .string()
  .max(MAX_LOG_SUMMARY_CHARS)
  .describe(
    "One past-tense sentence for the update log, with bundle-relative links, e.g. 'Added [Billing API](/apis/billing-api.md).'"
  );

export interface ReadPage {
  path: string;
  frontmatter: Concept["frontmatter"];
  frontmatter_truncated: boolean;
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
    frontmatter_truncated: false,
    body,
    offset,
    total_chars: concept.body.length,
    truncated,
    next_offset: truncated ? nextOffset : null,
  };
}

function boundedReadPage(
  page: ReadPage,
  bodyLimit: number,
  budget: number,
  state: AgentRunContext
): ReadPage | undefined {
  const emptyBodyPage = (
    frontmatter: Concept["frontmatter"],
    frontmatterTruncated: boolean
  ): ReadPage => ({
    path: page.path,
    frontmatter,
    frontmatter_truncated: frontmatterTruncated,
    body: "",
    offset: page.offset,
    total_chars: page.total_chars,
    truncated: page.offset < page.total_chars,
    next_offset: page.offset < page.total_chars ? page.offset : null,
  });

  const sourceBody = page.body.slice(0, bodyLimit);
  if (sourceBody.length === 0) {
    const frontmatter = fitFrontmatter(budget);
    if (!frontmatter) return undefined;
    return emptyBodyPage(frontmatter.value, frontmatter.truncated);
  }

  // Reserve half of the practical result budget for the body before fitting
  // frontmatter. At the defaults this is the complete 12k body page; with a
  // small budget it still prevents metadata from consuming the whole page.
  const reservedBody = Math.min(sourceBody.length, Math.max(1, Math.floor(budget / 2)));
  let low = 1;
  let high = sourceBody.length;
  let best: { body: string; frontmatter: Concept["frontmatter"]; truncated: boolean } | undefined;

  // First establish that the reserved quota fits. If metadata overhead leaves
  // less room, reduce the body only as far as necessary to retain valid metadata.
  const reserveFit = fitForBody(reservedBody);
  if (reserveFit) best = reserveFit;
  else {
    high = reservedBody - 1;
    while (low <= high) {
      const middle = Math.floor((low + high) / 2);
      const candidate = fitForBody(middle);
      if (candidate) {
        best = candidate;
        low = middle + 1;
      } else high = middle - 1;
    }
    if (!best) return undefined;
    low = best.body.length + 1;
    high = sourceBody.length;
  }

  // Once the body quota is secured, use remaining capacity for more body, not
  // for restoring frontmatter. This makes large metadata fail soft.
  while (low <= high) {
    const middle = Math.floor((low + high) / 2);
    const candidate = fitForBody(middle);
    if (candidate) {
      best = candidate;
      low = middle + 1;
    } else high = middle - 1;
  }
  return best ? makePage(best.body, best.frontmatter, best.truncated) : undefined;

  function fitForBody(bodyLength: number):
    | { body: string; frontmatter: Concept["frontmatter"]; truncated: boolean }
    | undefined {
    const body = sourceBody.slice(0, bodyLength);
    let fmLow = 0;
    let fmHigh = budget;
    let bestFrontmatter = {} as Concept["frontmatter"];
    while (fmLow <= fmHigh) {
      const middle = Math.floor((fmLow + fmHigh) / 2);
      const candidate = state.fit(page.frontmatter, middle);
      const frontmatter =
        candidate && typeof candidate === "object" && !Array.isArray(candidate)
          ? (candidate as Concept["frontmatter"])
          : ({} as Concept["frontmatter"]);
      if (JSON.stringify(makePage(body, frontmatter, true)).length <= budget) {
        bestFrontmatter = frontmatter;
        fmLow = middle + 1;
      } else fmHigh = middle - 1;
    }
    const truncated = JSON.stringify(bestFrontmatter) !== JSON.stringify(page.frontmatter);
    const result = makePage(body, bestFrontmatter, truncated);
    return JSON.stringify(result).length <= budget
      ? { body, frontmatter: bestFrontmatter, truncated }
      : undefined;
  }

  function makePage(
    body: string,
    frontmatter: Concept["frontmatter"],
    frontmatterTruncated: boolean
  ): ReadPage {
    const nextOffset = page.offset + body.length;
    const truncated = nextOffset < page.total_chars;
    return {
      path: page.path,
      frontmatter,
      frontmatter_truncated: frontmatterTruncated,
      body,
      offset: page.offset,
      total_chars: page.total_chars,
      truncated,
      next_offset: truncated ? nextOffset : null,
    };
  }

  function fitFrontmatter(
    resultBudget: number
  ): { value: Concept["frontmatter"]; truncated: boolean } | undefined {
    let fmLow = 0;
    let fmHigh = resultBudget;
    let bestFrontmatter = {} as Concept["frontmatter"];
    while (fmLow <= fmHigh) {
      const middle = Math.floor((fmLow + fmHigh) / 2);
      const candidate = state.fit(page.frontmatter, middle);
      const frontmatter =
        candidate && typeof candidate === "object" && !Array.isArray(candidate)
          ? (candidate as Concept["frontmatter"])
          : ({} as Concept["frontmatter"]);
      if (JSON.stringify(emptyBodyPage(frontmatter, true)).length <= resultBudget) {
        bestFrontmatter = frontmatter;
        fmLow = middle + 1;
      } else fmHigh = middle - 1;
    }
    const truncated = JSON.stringify(bestFrontmatter) !== JSON.stringify(page.frontmatter);
    const empty = emptyBodyPage(bestFrontmatter, truncated);
    return JSON.stringify(empty).length <= resultBudget
      ? { value: bestFrontmatter, truncated }
      : undefined;
  }
}

function boundText(text: string, maxChars: number): string {
  if (text.length <= maxChars) return text;
  const marker = `\n... [truncated; total_chars=${text.length}]`;
  const available = Math.max(0, maxChars - marker.length);
  const lineEnd = text.lastIndexOf("\n", available);
  const prefixLength = lineEnd > 0 ? lineEnd : available;
  return fitText(text.slice(0, prefixLength), maxChars, marker);
}

function isReadPage(value: unknown): value is ReadPage {
  return typeof value === "object" && value !== null && "body" in value && typeof value.body === "string";
}

function isExpectedReadError(error: unknown): boolean {
  return (
    (error instanceof BundleError && error.code !== "INVALID_FRONTMATTER") ||
    (error instanceof Error && error.message.startsWith("Invalid offset "))
  );
}

function readError(path: string, error: unknown): {
  path: string;
  read: false;
  error: { code: string };
} {
  const code = error instanceof BundleError ? error.code.toLowerCase() : "invalid_offset";
  return { path: boundedConceptPath(path), read: false, error: { code } };
}

function boundedConceptPath(value: string): string {
  return value.slice(0, MAX_CONCEPT_PATH_CHARS);
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
        query: z
          .string()
          .max(MAX_QUERY_CHARS)
          .describe("Keywords to search for. May be empty when filtering by type/tags only."),
        type: z.string().max(MAX_TYPE_CHARS).optional().describe("Exact concept type filter"),
        tags: z
          .array(z.string().max(MAX_TAG_CHARS))
          .max(MAX_TAGS)
          .optional()
          .describe("Require ALL of these tags"),
      }),
      execute: async ({ query, type, tags }) => {
        const hits = await kb.search(query, { type, tags });
        trace?.record(
          "search_knowledge",
          boundText(query, MAX_QUERY_CHARS),
          hits.map((h) => boundedConceptPath(h.path))
        );
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
        const tree = boundText(formatTree(await kb.listTree(), 0, false), state.payloadBudget);
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
        let c: Concept;
        try {
          c = await kb.readConcept(path);
        } catch (error) {
          if (!isExpectedReadError(error)) throw error;
          return state.result(readError(path, error));
        }
        trace?.record("read_concept", c.path, [c.path]);
        let page: ReadPage;
        try {
          page = readPage(c, offset, state.maxDocumentChars);
        } catch (error) {
          if (!isExpectedReadError(error)) throw error;
          return state.result(readError(c.path, error));
        }
        const bounded = boundedReadPage(page, page.body.length, state.payloadBudget, state);
        const result = bounded === undefined ? state.exhausted() : state.consume(bounded);
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
        trace?.record(
          "read_concepts",
          paths.slice(0, 3).map((path) => boundedConceptPath(path)).join(", "),
          paths.map((path) => boundedConceptPath(path))
        );
        const sourceConcepts = new Map<string, Concept>();
        const sourcePages = new Map<string, ReadPage>();
        const sourceByCanonicalPath = new Map<string, Concept>();
        const sourcePagesByCanonicalPath = new Map<string, ReadPage>();
        const missing: string[] = [];
        for (const requested of paths) {
          try {
            const c = await kb.readConcept(requested);
            const page = readPage(c, 0, state.maxDocumentChars);
            // Keep the request key for allocation, but retain the canonical
            // result path for callers and body-coverage bookkeeping.
            sourceConcepts.set(requested, c);
            sourcePages.set(requested, page);
            sourceByCanonicalPath.set(c.path, c);
            sourcePagesByCanonicalPath.set(c.path, page);
          } catch (error) {
            if (!isExpectedReadError(error)) throw error;
            missing.push(boundedConceptPath(requested)); // Reported rather than guessed at.
          }
        }
        const totalChars = [...sourceConcepts.values()].reduce(
          (total, concept) => total + concept.body.length,
          0
        );
        const base = {
          read: [] as ReadPage[],
          missing,
          omitted: [] as string[],
          returned_body_chars: 0,
          total_body_chars: totalChars,
          truncated: false,
          max_body_chars: state.maxDocumentChars,
          continuation:
            "Page included bodies with each page's next_offset; retry omitted paths in a fresh request.",
        };
        if (JSON.stringify(base).length > state.payloadBudget) return state.exhausted();

        for (const requested of paths) {
          const source = sourceConcepts.get(requested);
          const page = sourcePages.get(requested);
          if (!source || !page) continue;
          let low = page.body.length === 0 ? 0 : 1;
          let high = page.body.length;
          let best: ReadPage | undefined;
          while (low <= high) {
            const middle = Math.floor((low + high) / 2);
            const candidate = boundedReadPage(page, middle, state.payloadBudget, state);
            if (!candidate) {
              high = middle - 1;
              continue;
            }
            const result = {
              ...base,
              read: [...base.read, candidate],
              returned_body_chars: base.returned_body_chars + candidate.body.length,
            };
            if (JSON.stringify(result).length <= state.payloadBudget) {
              best = candidate;
              low = middle + 1;
            } else {
              high = middle - 1;
            }
          }
          if (best === undefined) {
            base.omitted.push(boundedConceptPath(requested));
            base.truncated = true;
            continue;
          }
          base.read.push(best);
          base.returned_body_chars += best.body.length;
          base.truncated ||= best.truncated;
        }
        base.truncated ||= base.omitted.length > 0 || base.missing.length > 0;
        const result = state.consume(base);
        if (
          typeof result === "object" &&
          result !== null &&
          "read" in result &&
          Array.isArray(result.read)
        ) {
          for (const page of result.read) {
            if (!isReadPage(page)) continue;
            const source = sourceByCanonicalPath.get(page.path);
            const sourcePage = sourcePagesByCanonicalPath.get(page.path);
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
        return state.result(boundText(formatTree(await kb.listTree(), 0, false), state.payloadBudget));
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
