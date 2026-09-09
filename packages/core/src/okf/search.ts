import type { Bundle } from "./bundle.js";
import type { SearchHit } from "./types.js";

export interface SearchOptions {
  type?: string;
  tags?: string[];
  limit?: number;
}

/**
 * Keep the original query token as well as its useful filename-like pieces.
 * The full token preserves exact path/title matches; the pieces make queries
 * such as `branch/install` useful against prose that names those parts apart.
 */
function queryTerms(query: string): string[] {
  const terms = new Set<string>();
  const tokens = query.toLowerCase().match(/[a-z0-9]+(?:[-._/][a-z0-9]+)*/g) ?? [];
  for (const token of tokens) {
    if (token.length > 1) terms.add(token);
    for (const part of token.split(/[\/.\-_]+/)) {
      if (part.length > 1) terms.add(part);
    }
  }
  return [...terms];
}

/** Tokens used for whole-token bonuses, including compound components. */
function fieldTokens(value: string): Set<string> {
  const tokens = new Set<string>();
  for (const token of value.toLowerCase().match(/[a-z0-9]+(?:[-._/][a-z0-9]+)*/g) ?? []) {
    tokens.add(token);
    for (const part of token.split(/[\/.\-_]+/)) {
      if (part.length > 1) tokens.add(part);
    }
  }
  return tokens;
}

function rarity(documentFrequency: number, documentCount: number): number {
  // Keep the weighting bounded: rare terms should win over generic prose, but
  // a single unusual token must not make every other matching term irrelevant.
  return Math.min(4, Math.max(1, Math.log((documentCount + 1) / (documentFrequency + 1)) + 1));
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

    if (options.type && fm.type?.toLowerCase() !== options.type.toLowerCase()) continue;
    if (options.tags?.length) {
      const conceptTags = (Array.isArray(fm.tags) ? fm.tags : []).map((t) =>
        String(t).toLowerCase()
      );
      if (!options.tags.every((t) => conceptTags.includes(t.toLowerCase()))) continue;
    }

    const fields = {
      title: (fm.title ?? "").toString().toLowerCase(),
      description: (fm.description ?? "").toString().toLowerCase(),
      tags: (Array.isArray(fm.tags) ? fm.tags : []).join(" ").toLowerCase(),
      body: concept.body.toLowerCase(),
      path: conceptPath.toLowerCase(),
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
    let firstBodyMatch = -1;
    for (const term of terms) {
      const weight = rarity(documentFrequency.get(term) ?? 0, documents.length);
      if (fields.title.includes(term)) score += 10 * weight;
      if (fields.path.includes(term)) score += 6 * weight;
      if (fields.description.includes(term)) score += 5 * weight;
      if (fields.tags.includes(term)) score += 5 * weight;
      const bodyIdx = fields.body.indexOf(term);
      if (bodyIdx !== -1) {
        score += 2 * weight;
        if (firstBodyMatch === -1) firstBodyMatch = bodyIdx;
      }

      // A whole token in a title/path is much more selective than a substring
      // in a long body. Components of `foo-bar.md` count as whole tokens too.
      if (tokens.title.has(term)) score += 24 * weight;
      if (tokens.path.has(term)) score += 20 * weight;
    }
    // Empty query with type/tag filters = browse mode: include everything that passed filters.
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
