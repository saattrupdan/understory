import type { Bundle } from "./bundle.js";
import { scanGraph } from "./graph.js";

const MIN_DUPLICATE_BLOCK_CHARS = 40;
const MAX_DUPLICATE_EXCERPT_CHARS = 160;

export interface LintFinding {
  path: string;
  type?: string;
  title?: string;
}

export interface BrokenLink {
  /** Concept containing the dangling link. */
  path: string;
  /** The missing bundle-relative target. */
  target: string;
}

export interface SectionOccurrence {
  /** One-based source line. */
  line: number;
  /** Markdown heading level (1–6). */
  level: number;
}

export interface DuplicateSectionTitle {
  path: string;
  /** First spelling of the repeated title. */
  title: string;
  occurrences: SectionOccurrence[];
}

export interface DuplicateContentBlock {
  path: string;
  /** Bounded preview of the first repeated block. */
  excerpt: string;
  /** One-based start line of each occurrence. */
  lines: number[];
}

export interface LintReport {
  conceptCount: number;
  /** Distinct inter-concept link edges (source → target, deduped per source). */
  linkCount: number;
  /** Concepts no other concept links to (index/log catalogs don't count). */
  orphans: LintFinding[];
  /** Outbound links pointing at nonexistent concepts. */
  brokenLinks: BrokenLink[];
  /** Repeated Markdown heading titles within one concept. */
  duplicateSectionTitles: DuplicateSectionTitle[];
  /** Repeated substantive Markdown blocks within one concept. */
  duplicateContentBlocks: DuplicateContentBlock[];
  healthy: boolean;
}

interface LocatedValue {
  value: string;
  line: number;
}

interface LocatedHeading extends LocatedValue {
  level: number;
  parent: string;
}

/** Normalize prose enough to catch copy/paste repeats without fuzzy matching. */
function duplicateKey(value: string): string {
  return value.trim().replace(/\s+/g, " ").toLowerCase();
}

function excerpt(value: string): string {
  const compact = value.trim().replace(/\s+/g, " ");
  return compact.length <= MAX_DUPLICATE_EXCERPT_CHARS
    ? compact
    : `${compact.slice(0, MAX_DUPLICATE_EXCERPT_CHARS - 1)}…`;
}

function repeatedEntries<T extends LocatedValue>(
  entries: T[],
  keyFor: (entry: T) => string = (entry) => duplicateKey(entry.value)
): Map<string, T[]> {
  const grouped = new Map<string, T[]>();
  for (const entry of entries) {
    const key = keyFor(entry);
    const matches = grouped.get(key) ?? [];
    matches.push(entry);
    grouped.set(key, matches);
  }
  return new Map([...grouped].filter(([, matches]) => matches.length > 1));
}

function duplicateBodyContent(body: string, path: string): {
  sectionTitles: DuplicateSectionTitle[];
  contentBlocks: DuplicateContentBlock[];
} {
  const headings: LocatedHeading[] = [];
  const blocks: LocatedValue[] = [];
  const lines = body.replace(/\r\n?/g, "\n").split("\n");
  const activeHeadings: string[] = [];
  let blockLines: string[] = [];
  let blockStart = 1;
  let fence: "`" | "~" | undefined;

  const flushBlock = () => {
    if (blockLines.length === 0) return;
    const value = blockLines.join("\n").trim();
    const isIndentedCode = blockLines.every((line) => /^( {4}|\t)/.test(line));
    if (!isIndentedCode && duplicateKey(value).length >= MIN_DUPLICATE_BLOCK_CHARS) {
      blocks.push({ value, line: blockStart });
    }
    blockLines = [];
  };

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    const lineNumber = index + 1;
    const fenceMatch = line.match(/^ {0,3}(`{3,}|~{3,})/);

    if (fence !== undefined) {
      if (fenceMatch?.[1][0] === fence) fence = undefined;
      continue;
    }
    if (fenceMatch) {
      flushBlock();
      fence = fenceMatch[1][0] as "`" | "~";
      continue;
    }

    const headingMatch = line.match(/^ {0,3}(#{1,6})(?:[ \t]+(.*?))?[ \t]*$/);
    if (headingMatch) {
      flushBlock();
      const level = headingMatch[1].length;
      const value = (headingMatch[2] ?? "").replace(/[ \t]+#+[ \t]*$/, "").trim();
      if (value !== "") {
        activeHeadings.length = level - 1;
        headings.push({
          value,
          line: lineNumber,
          level,
          parent: activeHeadings.map((title, parentLevel) => `${parentLevel + 1}:${title}`).join("/"),
        });
        activeHeadings[level - 1] = duplicateKey(value);
      }
      continue;
    }
    if (line.trim() === "") {
      flushBlock();
      continue;
    }

    if (blockLines.length === 0) blockStart = lineNumber;
    blockLines.push(line);
  }
  flushBlock();

  const repeatedHeadings = repeatedEntries(
    headings,
    (heading) => `${heading.parent}\n${heading.level}:${duplicateKey(heading.value)}`
  );
  const sectionTitles = [...repeatedHeadings.values()].map((matches) => ({
    path,
    title: matches[0].value,
    occurrences: matches.map(({ line, level }) => ({ line, level })),
  }));
  const contentBlocks = [...repeatedEntries(blocks).values()].map((matches) => ({
    path,
    excerpt: excerpt(matches[0].value),
    lines: matches.map(({ line }) => line),
  }));

  return { sectionTitles, contentBlocks };
}

/**
 * Deterministic knowledge health check: graph defects and repeated content.
 * Duplicate checks are intentionally exact after case/whitespace normalization;
 * semantic near-duplicates remain the consolidation agent's responsibility.
 */
export async function lintBundle(bundle: Bundle): Promise<LintReport> {
  const { nodes, edges, brokenLinks, inbound } = await scanGraph(bundle);

  const orphans: LintFinding[] = nodes
    .filter((n) => (inbound.get(n.path) ?? 0) === 0)
    .map((n) => ({ path: n.path, type: n.type, title: n.title }));
  const duplicateSectionTitles: DuplicateSectionTitle[] = [];
  const duplicateContentBlocks: DuplicateContentBlock[] = [];

  for (const node of nodes) {
    try {
      const concept = await bundle.readConcept(node.path);
      const duplicates = duplicateBodyContent(concept.body, node.path);
      duplicateSectionTitles.push(...duplicates.sectionTitles);
      duplicateContentBlocks.push(...duplicates.contentBlocks);
    } catch {
      // Match graph scanning's permissive behavior for unreadable concepts.
    }
  }

  return {
    conceptCount: nodes.length,
    linkCount: edges.length,
    orphans,
    brokenLinks,
    duplicateSectionTitles,
    duplicateContentBlocks,
    healthy:
      orphans.length === 0 &&
      brokenLinks.length === 0 &&
      duplicateSectionTitles.length === 0 &&
      duplicateContentBlocks.length === 0,
  };
}
