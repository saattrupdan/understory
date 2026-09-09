import type { Bundle } from "./bundle.js";
import type { SearchHit } from "./types.js";

export interface SearchOptions {
  type?: string;
  tags?: string[];
  limit?: number;
}

// Keep punctuation as a separator, but retain symbols: an emoji, currency sign,
// or other non-punctuation Unicode mark is still a meaningful search query.
const TOKEN_PATTERN = /[\p{L}\p{N}\p{M}\p{S}]+(?:[-._/][\p{L}\p{N}\p{M}\p{S}]+)*/gu;
const COMPOUND_SEPARATOR = /[/._-]+/u;
const MAX_QUERY_CHARS = 4096;
const MAX_QUERY_TERMS = 128;
const MIN_BROAD_TERM_LENGTH = 2;

/** Unicode-normalise and apply JavaScript's locale-independent lower-casing. */
function normalise(value: string): string {
  // NFKC makes canonically equivalent text and compatibility forms searchable
  // alike. Lower-casing can itself introduce combining marks, so compose once
  // more afterwards before both query and document tokenisation.
  return value.normalize("NFKC").toLowerCase().normalize("NFC");
}

/**
 * Keep the original query token as well as its useful filename-like pieces.
 * The full token preserves exact path/title matches; the pieces make queries
 * such as `branch/install` useful against prose that names those parts apart.
 * Unicode properties retain letters, numbers, marks, and symbols in every
 * script. Punctuation-only input therefore remains the only browse query.
 */
function queryTerms(query: string): string[] {
  const terms = new Set<string>();
  const tokens = normalise(query.slice(0, MAX_QUERY_CHARS)).match(TOKEN_PATTERN) ?? [];
  for (const token of tokens) {
    terms.add(token);
    for (const part of token.split(COMPOUND_SEPARATOR)) {
      if (part) terms.add(part);
    }
    if (terms.size >= MAX_QUERY_TERMS) break;
  }
  return [...terms].slice(0, MAX_QUERY_TERMS);
}

/** Tokens used for whole-token bonuses, including compound components. */
function fieldTokens(value: string): Set<string> {
  const tokens = new Set<string>();
  for (const token of value.match(TOKEN_PATTERN) ?? []) {
    tokens.add(token);
    for (const part of token.split(COMPOUND_SEPARATOR)) {
      if (part) tokens.add(part);
    }
  }
  return tokens;
}

function rarity(documentFrequency: number, documentCount: number): number {
  // Do not give ubiquitous question scaffolding a full-strength score. The
  // small floor preserves broad search (including a query whose term occurs in
  // every document), while distinctive terms remain several times stronger.
  return Math.min(4, Math.max(0.2, Math.log((documentCount + 1) / (documentFrequency + 1))));
}

function confidenceRarity(documentFrequency: number, documentCount: number): number {
  if (documentFrequency === 0 || documentCount === 0) return 0;
  // Unlike ranking weights, confidence must approach zero when a term is in
  // the whole corpus. Smoothing retains useful evidence in very small bundles.
  return Math.max(0, 2 * Math.log((documentCount + 1) / (documentFrequency + 0.5)));
}

function matches(field: string, tokens: Set<string>, term: string): boolean {
  // Exact tokens are the evidence used for confidence. Substrings remain a
  // lower-grade ranking signal for compatibility with ordinary prose queries,
  // but one-character Latin terms are not allowed to match every word in a
  // document ("I" is the common example in a question). A one-character CJK
  // query is different: it commonly appears inside a longer uninterrupted
  // token and must retain the broad-match behaviour.
  const canUseSubstring =
    term.length >= MIN_BROAD_TERM_LENGTH || !/^[a-z]$/u.test(term);
  return tokens.has(term) || (canUseSubstring && field.includes(term));
}

/**
 * Naive in-memory scan over all concepts — fine into the thousands of files.
 * Scores preserve broad substring matching, while rarity and whole-token
 * matches make compound/path-specific queries useful retrieval signals.
 */
export async function searchBundle(
  bundle: Bundle,
  query: string,
  options: SearchOptions = {}
): Promise<SearchHit[]> {
  const terms = queryTerms(query);
  const paths = await bundle.listConceptPaths();
  const documents: Array<{
    conceptPath: string;
    concept: Awaited<ReturnType<Bundle["readConcept"]>>;
    fields: { title: string; description: string; tags: string; body: string; path: string };
    tokens: {
      title: Set<string>;
      description: Set<string>;
      tags: Set<string>;
      body: Set<string>;
      path: Set<string>;
    };
  }> = [];

  for (const conceptPath of paths) {
    let concept;
    try {
      concept = await bundle.readConcept(conceptPath);
    } catch {
      continue; // Permissive: skip unreadable files.
    }
    const fm = concept.frontmatter;

    if (options.type && normalise(fm.type ?? "") !== normalise(options.type)) continue;
    if (options.tags?.length) {
      const conceptTags = (Array.isArray(fm.tags) ? fm.tags : []).map((t) =>
        normalise(String(t))
      );
      if (!options.tags.every((t) => conceptTags.includes(normalise(t)))) continue;
    }

    const fields = {
      title: normalise((fm.title ?? "").toString()),
      description: normalise((fm.description ?? "").toString()),
      tags: normalise((Array.isArray(fm.tags) ? fm.tags : []).join(" ")),
      body: normalise(concept.body),
      path: normalise(conceptPath),
    };
    documents.push({
      conceptPath,
      concept,
      fields,
      tokens: {
        title: fieldTokens(fields.title),
        description: fieldTokens(fields.description),
        tags: fieldTokens(fields.tags),
        body: fieldTokens(fields.body),
        path: fieldTokens(fields.path),
      },
    });
  }

  const documentFrequency = new Map<string, number>();
  for (const term of terms) {
    let count = 0;
    for (const document of documents) {
      const { fields, tokens } = document;
      if (
        matches(fields.title, tokens.title, term) ||
        matches(fields.path, tokens.path, term) ||
        matches(fields.description, tokens.description, term) ||
        matches(fields.tags, tokens.tags, term) ||
        matches(fields.body, tokens.body, term)
      ) {
        count += 1;
      }
    }
    documentFrequency.set(term, count);
  }

  const hits: SearchHit[] = [];
  for (const document of documents) {
    const { conceptPath, concept, fields, tokens } = document;
    let score = 0;
    let confidence = 0;
    let pathScore = 0;
    let firstBodyMatch = -1;
    for (const term of terms) {
      const frequency = documentFrequency.get(term) ?? 0;
      const weight = rarity(frequency, documents.length);
      const evidenceWeight = confidenceRarity(frequency, documents.length);
      const titleExact = tokens.title.has(term);
      const descriptionExact = tokens.description.has(term);
      const tagsExact = tokens.tags.has(term);
      const bodyExact = tokens.body.has(term);

      if (matches(fields.title, tokens.title, term)) score += 8 * weight;
      if (matches(fields.description, tokens.description, term)) score += 5 * weight;
      if (matches(fields.tags, tokens.tags, term)) score += 5 * weight;
      if (matches(fields.body, tokens.body, term)) {
        score += 2 * weight;
        const bodyIdx = fields.body.indexOf(term);
        if (bodyIdx !== -1 && firstBodyMatch === -1) firstBodyMatch = bodyIdx;
      }

      // Exact semantic tokens are the main ranking and confidence evidence.
      // A title hit is strongest, followed by metadata and then body content.
      if (titleExact) {
        score += 20 * weight;
        confidence += 20 * evidenceWeight;
      }
      if (descriptionExact) {
        score += 8 * weight;
        confidence += 8 * evidenceWeight;
      }
      if (tagsExact) {
        score += 8 * weight;
        confidence += 8 * evidenceWeight;
      }
      if (bodyExact) {
        score += 4 * weight;
        confidence += 4 * evidenceWeight;
      }

      // Paths are bounded ranking hints only. In particular, they contribute
      // no confidence: a filename or directory name can never independently
      // make deterministic recall trust a hit.
      if (matches(fields.path, tokens.path, term)) {
        pathScore += (tokens.path.has(term) ? 4 : 2) * weight;
      }
    }
    score += Math.min(14, pathScore);

    // Empty query with type/tag filters = browse mode: include everything that passed filters.
    // Non-ASCII words and symbols are retained above, so they cannot accidentally enter this branch.
    if (terms.length === 0) score = 1;
    if (score === 0) continue;

    hits.push({
      path: conceptPath,
      type: concept.frontmatter.type ?? "unknown",
      title: concept.frontmatter.title as string | undefined,
      description: concept.frontmatter.description as string | undefined,
      snippet:
        firstBodyMatch >= 0
          ? concept.body
              .slice(Math.max(0, firstBodyMatch - 60), firstBodyMatch + 120)
              .replace(/\s+/g, " ")
              .trim()
          : undefined,
      score,
      confidence,
    });
  }

  hits.sort((a, b) => b.score - a.score);
  return hits.slice(0, options.limit ?? 20);
}

/** Distinct `type` values in use across the bundle (fed to the agent's system prompt). */
export async function listTypes(bundle: Bundle): Promise<string[]> {
  const paths = await bundle.listConceptPaths();
  const types = new Set<string>();
  for (const p of paths) {
    try {
      const { frontmatter } = await bundle.readConcept(p);
      if (frontmatter.type) types.add(frontmatter.type);
    } catch {
      // skip
    }
  }
  return [...types].sort();
}
