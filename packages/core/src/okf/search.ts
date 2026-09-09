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
const MAX_QUERY_GROUPS = 64;
const MAX_GROUP_TERMS = 12;
const MAX_SCORING_GROUPS = 4;
const MIN_BROAD_TERM_LENGTH = 2;
const DISTINCTIVE_RARITY = 0.2;

// This is deliberately a small, explicit vocabulary rather than a stemmer. It
// covers the English forms that commonly occur in installation questions while
// leaving short words, names, and non-Latin scripts untouched.
const MORPHOLOGY = new Map([
  ["installed", "install"],
  ["installing", "install"],
  ["installation", "install"],
]);

// Question scaffolding should help ranking, but should not become confidence
// merely because a small corpus happens to contain it once.
const COMMON_CONFIDENCE_TERMS = new Set([
  "a",
  "an",
  "and",
  "are",
  "can",
  "do",
  "does",
  "for",
  "how",
  "i",
  "in",
  "is",
  "it",
  "of",
  "on",
  "or",
  "should",
  "the",
  "to",
  "use",
  "what",
  "where",
  "which",
  "with",
]);

type FieldName = "title" | "description" | "tags" | "body" | "path";
type ContentFieldName = Exclude<FieldName, "path">;

interface TokenGroup {
  /** The original complete token, including filename-like separators. */
  full: string;
  /** Morphological root used to identify duplicate query evidence. */
  root: string;
  /** The complete token plus bounded component/morphological variants. */
  terms: string[];
  /** Variants of the complete token only, used for exact compound matches. */
  fullVariants: string[];
}

interface FieldData {
  text: string;
  groups: TokenGroup[];
  terms: Set<string>;
}

interface Document {
  conceptPath: string;
  concept: Awaited<ReturnType<Bundle["readConcept"]>>;
  fields: Record<FieldName, FieldData>;
}

interface GroupEvidence {
  matched: boolean;
  contentTerms: Set<string>;
  exactPathTerms: Set<string>;
  exactFullContent: boolean;
  exactFullPath: boolean;
  bestContentRarity: number;
  bestContentFrequency: number;
  bestRankingRarity: number;
  bodyIndex: number;
}

/** Unicode-normalise and apply JavaScript's locale-independent lower-casing. */
function normalise(value: string): string {
  // NFKC makes canonically equivalent text and compatibility forms searchable
  // alike. Lower-casing can itself introduce combining marks, so compose once
  // more afterwards before both query and document tokenisation.
  return value.normalize("NFKC").toLowerCase().normalize("NFC");
}

function morphologicalVariants(token: string): string[] {
  const variant = MORPHOLOGY.get(token);
  return variant && variant !== token ? [variant] : [];
}

function tokenGroup(token: string): TokenGroup {
  const parts = token.split(COMPOUND_SEPARATOR).filter(Boolean);
  const roots = parts.map((part) => MORPHOLOGY.get(part) ?? part);
  const root = roots.join(token.match(COMPOUND_SEPARATOR)?.[0] ?? "");
  const terms = new Set<string>([token]);
  const fullVariants = new Set<string>([token, ...morphologicalVariants(token)]);

  for (const part of parts.slice(0, MAX_GROUP_TERMS)) {
    terms.add(part);
    for (const variant of morphologicalVariants(part)) terms.add(variant);
  }
  for (const variant of fullVariants) terms.add(variant);

  return {
    full: token,
    root,
    terms: [...terms].slice(0, MAX_GROUP_TERMS),
    fullVariants: [...fullVariants],
  };
}

/**
 * Tokenise into provenance-preserving groups. Components aid ranking, but a
 * compound such as `make_icons.py` remains one piece of user evidence.
 */
function tokenGroups(value: string): TokenGroup[] {
  const groups: TokenGroup[] = [];
  const seen = new Set<string>();
  for (const token of value.match(TOKEN_PATTERN) ?? []) {
    if (seen.has(token)) continue;
    seen.add(token);
    groups.push(tokenGroup(token));
  }
  return groups;
}

function fieldData(value: string): FieldData {
  const text = normalise(value);
  const groups = tokenGroups(text);
  const terms = new Set<string>();
  for (const group of groups) {
    for (const term of group.terms) terms.add(term);
  }
  return { text, groups, terms };
}

function queryGroups(query: string): TokenGroup[] {
  const groups = tokenGroups(normalise(query.slice(0, MAX_QUERY_CHARS))).slice(0, MAX_QUERY_GROUPS);
  const canonical = new Map<string, TokenGroup>();

  for (const group of groups) {
    const existing = canonical.get(group.root);
    if (!existing) {
      canonical.set(group.root, group);
      continue;
    }

    // Keep all bounded variants from a morphological family so that merging
    // `installed`, `installing`, and `installation` does not lose stemming or
    // exact compound matching. The family still contributes only once.
    existing.terms = [...new Set([...existing.terms, ...group.terms])].slice(
      0,
      MAX_GROUP_TERMS
    );
    existing.fullVariants = [...new Set([...existing.fullVariants, ...group.fullVariants])];
  }

  return [...canonical.values()];
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
  // the whole corpus. Smoothing retains useful evidence in small bundles.
  return Math.max(0, 2 * Math.log((documentCount + 1) / (documentFrequency + 0.5)));
}

function matches(field: FieldData, term: string): boolean {
  // Content matches are evidence for confidence, but corpus frequency still
  // decides whether they are distinctive. One-character Latin terms are not
  // allowed to match every word in a document ("I" is the common example in a
  // question). A one-character CJK query is different: it commonly appears
  // inside a longer uninterrupted token and must retain broad-match behaviour.
  const canUseSubstring =
    term.length >= MIN_BROAD_TERM_LENGTH || !/^[a-z]$/u.test(term);
  return field.terms.has(term) || (canUseSubstring && field.text.includes(term));
}

function exactFullMatch(field: FieldData, group: TokenGroup): boolean {
  return field.groups.some((candidate) =>
    candidate.fullVariants.some((variant) => group.fullVariants.includes(variant))
  );
}

function fieldHasExactTerm(field: FieldData, group: TokenGroup): Set<string> {
  return new Set(group.terms.filter((term) => field.terms.has(term)));
}

function groupIsFilenameLike(group: TokenGroup): boolean {
  return COMPOUND_SEPARATOR.test(group.full) || /\.[a-z0-9]{1,8}$/u.test(group.full);
}

function evaluateGroup(
  group: TokenGroup,
  document: Document,
  documentFrequency: Map<string, number>,
  documentCount: number
): GroupEvidence {
  const contentTerms = new Set<string>();
  const exactPathTerms = fieldHasExactTerm(document.fields.path, group);
  let exactFullContent = false;
  let exactFullPath = false;
  let matched = false;
  let bestContentRarity = 0;
  let bestContentFrequency = documentCount;
  let bestRankingRarity = 0;
  let bodyIndex = -1;

  for (const fieldName of ["title", "description", "tags", "body"] as const) {
    const field = document.fields[fieldName];
    const exact = fieldHasExactTerm(field, group);
    for (const term of exact) {
      contentTerms.add(term);
      const frequency = documentFrequency.get(term) ?? documentCount;
      const termConfidenceRarity = confidenceRarity(frequency, documentCount);
      if (termConfidenceRarity > bestContentRarity) {
        bestContentRarity = termConfidenceRarity;
        bestContentFrequency = frequency;
      }
      bestRankingRarity = Math.max(bestRankingRarity, rarity(frequency, documentCount));
      if (fieldName === "body" && bodyIndex === -1) {
        bodyIndex = group.terms
          .map((candidate) => field.text.indexOf(candidate))
          .find((index) => index >= 0) ?? -1;
      }
    }
    if (matches(field, group.full) || group.terms.some((term) => matches(field, term))) {
      matched = true;
      for (const term of group.terms) {
        if (!matches(field, term)) continue;
        // Broad substring matches are content evidence too (for example the
        // query "work" against "works"), but their corpus frequency still
        // controls whether they can contribute confidence.
        contentTerms.add(term);
        const frequency = documentFrequency.get(term) ?? documentCount;
        const termConfidenceRarity = confidenceRarity(frequency, documentCount);
        if (termConfidenceRarity > bestContentRarity) {
          bestContentRarity = termConfidenceRarity;
          bestContentFrequency = frequency;
        }
        bestRankingRarity = Math.max(bestRankingRarity, rarity(frequency, documentCount));
      }
    }
    if (exactFullMatch(field, group)) {
      exactFullContent = true;
      matched = true;
    }
  }

  if (exactPathTerms.size > 0 || matches(document.fields.path, group.full)) matched = true;
  exactFullPath = exactFullMatch(document.fields.path, group);

  return {
    matched,
    contentTerms,
    exactPathTerms,
    exactFullContent,
    exactFullPath,
    bestContentRarity,
    bestContentFrequency,
    bestRankingRarity,
    bodyIndex,
  };
}

function fieldWeight(fieldName: ContentFieldName): number {
  switch (fieldName) {
    case "title":
      return 24;
    case "description":
      return 8;
    case "tags":
      return 8;
    case "body":
      return 4;
  }
}

function exactFieldWeight(fieldName: ContentFieldName): number {
  switch (fieldName) {
    case "title":
      return 18;
    case "description":
      return 7;
    case "tags":
      return 7;
    case "body":
      return 3;
  }
}

/**
 * Naive in-memory scan over all concepts — fine into the thousands of files.
 * Query groups keep compound provenance: components can improve ranking, but
 * confidence is accumulated once per corroborated group.
 */
export async function searchBundle(
  bundle: Bundle,
  query: string,
  options: SearchOptions = {}
): Promise<SearchHit[]> {
  const groups = queryGroups(query);
  const paths = await bundle.listConceptPaths();
  const documents: Document[] = [];

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

    documents.push({
      conceptPath,
      concept,
      fields: {
        title: fieldData((fm.title ?? "").toString()),
        description: fieldData((fm.description ?? "").toString()),
        tags: fieldData(Array.isArray(fm.tags) ? fm.tags.join(" ") : ""),
        body: fieldData(concept.body),
        path: fieldData(conceptPath),
      },
    });
  }

  // Frequencies are calculated per term for useful IDF-like ranking. They are
  // never used as separate confidence evidence: evaluateGroup below folds them
  // back into their original query group.
  const documentFrequency = new Map<string, number>();
  for (const group of groups) {
    for (const term of group.terms) {
      let count = 0;
      for (const document of documents) {
        if (
          (["title", "description", "tags", "body", "path"] as const).some((fieldName) =>
            matches(document.fields[fieldName], term)
          )
        ) {
          count += 1;
        }
      }
      documentFrequency.set(term, count);
    }
  }

  const hits: SearchHit[] = [];
  for (const document of documents) {
    let score = 0;
    let pathScore = 0;
    let firstBodyMatch = -1;
    let matchedGroups = 0;
    let contentGroups = 0;
    let distinctiveGroups = 0;
    let exactCompoundGroups = 0;
    let confidence = 0;
    const groupScores: number[] = [];

    for (const group of groups) {
      const evidence = evaluateGroup(group, document, documentFrequency, documents.length);
      if (!evidence.matched) continue;
      matchedGroups += 1;
      if (evidence.bodyIndex >= 0 && firstBodyMatch === -1) firstBodyMatch = evidence.bodyIndex;

      const groupWeight = Math.max(0.2, evidence.bestRankingRarity);
      let groupScore = 0;
      for (const fieldName of ["title", "description", "tags", "body"] as const) {
        const field = document.fields[fieldName];
        const broad = matches(field, group.full) || group.terms.some((term) => matches(field, term));
        if (!broad) continue;
        const exact = fieldHasExactTerm(field, group).size > 0 || exactFullMatch(field, group);
        groupScore += fieldWeight(fieldName) * groupWeight;
        if (exact) groupScore += exactFieldWeight(fieldName) * groupWeight;
      }

      // A complete compound is much more useful for ranking than a document
      // that happens to contain just one component. Additional components add
      // only bounded coverage, not independent user evidence.
      const componentCount = Math.max(1, group.terms.length - 1);
      const matchedComponentCount = group.terms.filter((term) =>
        ["title", "description", "tags", "body"].some((fieldName) =>
          document.fields[fieldName as ContentFieldName].terms.has(term)
        )
      ).length;
      groupScore += Math.min(8, (matchedComponentCount / componentCount) * 8);
      if (evidence.exactFullContent) groupScore += 14 * groupWeight;
      if (groupIsFilenameLike(group) && matchedComponentCount >= componentCount) {
        // A compound query is an independent coordination signal: matching
        // all of its components is stronger than accumulating unrelated words.
        // Bound the bonus so a long filename cannot dominate by size alone.
        groupScore += Math.min(80, groupScore);
      }
      groupScores.push(groupScore);

      if (evidence.exactPathTerms.size > 0 || evidence.exactFullPath) {
        pathScore += (evidence.exactFullPath ? 5 : 2) * groupWeight;
      } else if (matches(document.fields.path, group.full)) {
        pathScore += 2 * groupWeight;
      }

      if (evidence.contentTerms.size > 0) contentGroups += 1;
      const confidenceTerms = [...evidence.contentTerms].filter(
        (term) => !COMMON_CONFIDENCE_TERMS.has(term)
      );
      const distinctive =
        confidenceTerms.length > 0 &&
        evidence.bestContentRarity >= DISTINCTIVE_RARITY &&
        (documents.length <= 3 || evidence.bestContentFrequency * 2 <= documents.length);
      if (distinctive) {
        distinctiveGroups += 1;
        // Ten points per corroborated group makes the public gate explicit:
        // two distinctive content groups qualify ordinary prose. A complete,
        // distinctive compound/filename-like group can qualify by itself.
        confidence += 10 + Math.min(8, evidence.bestContentRarity * 1.5);
        if (evidence.exactFullContent && groupIsFilenameLike(group)) {
          exactCompoundGroups += 1;
          confidence += 10;
        }
      }
    }

    // Long natural-language questions contain many generic words. Summing all
    // of them lets a distractor win by accumulation, even when it has none of
    // the query's strongest evidence. Query groups remain independent, but
    // only the strongest bounded set contributes to ranking.
    groupScores.sort((a, b) => b - a);
    for (const contribution of groupScores.slice(0, MAX_SCORING_GROUPS)) {
      score += contribution;
    }
    score += Math.min(14, pathScore);
    const confidenceQualified =
      distinctiveGroups >= 2 || exactCompoundGroups >= 1;

    // Empty query with type/tag filters = browse mode: include everything that
    // passed filters. Symbols and non-ASCII words remain real query groups.
    if (groups.length === 0) {
      score = 1;
      confidence = 0;
    }
    if (score === 0) continue;

    hits.push({
      path: document.conceptPath,
      type: document.concept.frontmatter.type ?? "unknown",
      title: document.concept.frontmatter.title as string | undefined,
      description: document.concept.frontmatter.description as string | undefined,
      snippet:
        firstBodyMatch >= 0
          ? document.concept.body
              .slice(Math.max(0, firstBodyMatch - 60), firstBodyMatch + 120)
              .replace(/\s+/g, " ")
              .trim()
          : undefined,
      score,
      confidence,
      matchedGroups,
      contentGroups,
      distinctiveGroups,
      exactCompoundGroups,
      confidenceQualified,
    });
  }

  hits.sort((a, b) => b.score - a.score || (b.confidence ?? 0) - (a.confidence ?? 0) || a.path.localeCompare(b.path));
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
