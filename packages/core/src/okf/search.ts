import type { Bundle } from "./bundle.js";
import type { SearchHit } from "./types.js";

export interface SearchOptions {
  type?: string;
  tags?: string[];
  limit?: number;
}

const TOKEN_PATTERN = /[\p{L}\p{N}\p{M}]+(?:[-._/][\p{L}\p{N}\p{M}]+)*/gu;
const COMPOUND_SEPARATOR = /[/.\-_]+/u;

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
 * Unicode properties retain letters, numbers and combining marks in every
 * script. Even one-code-point tokens are meaningful in scripts such as Han.
 */
function queryTerms(query: string): string[] {
  const terms = new Set<string>();
  const tokens = normalise(query).match(TOKEN_PATTERN) ?? [];
  for (const token of tokens) {
    terms.add(token);
    for (const part of token.split(COMPOUND_SEPARATOR)) terms.add(part);
  }
  return [...terms];
}

/** Tokens used for whole-token bonuses, including compound components. */
function fieldTokens(value: string): Set<string> {
  const tokens = new Set<string>();
  for (const token of value.match(TOKEN_PATTERN) ?? []) {
    tokens.add(token);
    for (const part of token.split(COMPOUND_SEPARATOR)) tokens.add(part);
  }
  return tokens;
}

function rarity(documentFrequency: number, documentCount: number): number {
  // Keep ranking weights bounded: rare terms should win over generic prose,
  // but a single unusual token must not make every other term irrelevant.
  return Math.min(4, Math.max(1, Math.log((documentCount + 1) / (documentFrequency + 1)) + 1));
}

function confidenceRarity(documentFrequency: number, documentCount: number): number {
  if (documentFrequency === 0 || documentCount === 0) return 0;
  // Unlike the ranking weight, confidence must approach zero when a term is in
  // the whole corpus. Smoothing retains useful evidence in very small bundles.
  return 2 * Math.log((documentCount + 1) / (documentFrequency + 0.5));
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
    tokens: { title: Set<string>; path: Set<string> };
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
      tokens: { title: fieldTokens(fields.title), path: fieldTokens(fields.path) },
    });
  }

  const documentFrequency = new Map<string, number>();
  for (const term of terms) {
    let count = 0;
    for (const document of documents) {
      const { fields } = document;
      if (
        fields.title.includes(term) ||
        fields.path.includes(term) ||
        fields.description.includes(term) ||
        fields.tags.includes(term) ||
        fields.body.includes(term)
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
    let firstBodyMatch = -1;
    for (const term of terms) {
      const frequency = documentFrequency.get(term) ?? 0;
      const weight = rarity(frequency, documents.length);
      const evidenceWeight = confidenceRarity(frequency, documents.length);
      if (fields.title.includes(term)) {
        score += 10 * weight;
        confidence += 10 * evidenceWeight;
      }
      if (fields.path.includes(term)) {
        score += 6 * weight;
        // Paths are excellent ranking hints but weak semantic evidence on
        // their own: directory and extension components repeat everywhere.
        confidence += 2 * evidenceWeight;
      }
      if (fields.description.includes(term)) {
        score += 5 * weight;
        confidence += 5 * evidenceWeight;
      }
      if (fields.tags.includes(term)) {
        score += 5 * weight;
        confidence += 5 * evidenceWeight;
      }
      const bodyIdx = fields.body.indexOf(term);
      if (bodyIdx !== -1) {
        score += 2 * weight;
        confidence += 2 * evidenceWeight;
        if (firstBodyMatch === -1) firstBodyMatch = bodyIdx;
      }

      // A whole token in a title/path is much more selective than a substring
      // in a long body. Components of `foo-bar.md` count as whole tokens too.
      if (tokens.title.has(term)) {
        score += 24 * weight;
        confidence += 24 * evidenceWeight;
      }
      if (tokens.path.has(term)) {
        score += 20 * weight;
        confidence += 5 * evidenceWeight;
      }
    }
    // Empty query with type/tag filters = browse mode: include everything that passed filters.
    // Non-ASCII words are retained above, so they cannot accidentally enter this branch.
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
