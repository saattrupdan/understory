/** Frontmatter of an OKF concept. `type` is the only required field (spec §5). */
export interface ConceptFrontmatter {
  type: string;
  title?: string;
  description?: string;
  resource?: string;
  tags?: string[];
  timestamp?: string;
  /** Producer-defined keys are permitted and preserved. */
  [key: string]: unknown;
}

export interface Concept {
  /** Bundle-relative path, always starting with "/" (e.g. "/tables/customers.md"). */
  path: string;
  frontmatter: ConceptFrontmatter;
  body: string;
  raw: string;
}

export interface TreeNode {
  name: string;
  path: string;
  kind: "directory" | "concept" | "reserved";
  /** Present on concepts. */
  type?: string;
  title?: string;
  description?: string;
  children?: TreeNode[];
}

export interface SearchHit {
  path: string;
  type: string;
  title?: string;
  description?: string;
  /** Snippet of body text around the first match, if the match was in the body. */
  snippet?: string;
  /** Ranking score. Its absolute value is not a measure of match quality. */
  score: number;
  /**
   * Corpus-aware literal evidence, used to decide whether a hit is confident
   * enough for deterministic recall. Common terms contribute close to zero.
   */
  confidence?: number;
  /** Number of original query token groups matched anywhere, including paths. */
  matchedGroups?: number;
  /** Number of original query groups matched by content text or variants. */
  contentGroups?: number;
  /** Distinctive, content-backed groups that contribute to confidence. */
  distinctiveGroups?: number;
  /** Distinctive complete compound/filename-like content groups. */
  exactCompoundGroups?: number;
  /** Whether this hit satisfies the deterministic recall evidence rule. */
  confidenceQualified?: boolean;
}

export type LogAction = "Creation" | "Update" | "Deletion";

export interface LogEntry {
  date: string; // YYYY-MM-DD
  action: LogAction;
  summary: string;
}

export interface ConformanceIssue {
  path: string;
  severity: "error" | "warning";
  message: string;
}

export interface ConformanceReport {
  conformant: boolean;
  conceptCount: number;
  directoryCount: number;
  issues: ConformanceIssue[];
}

export const RESERVED_FILENAMES = new Set(["index.md", "log.md"]);
