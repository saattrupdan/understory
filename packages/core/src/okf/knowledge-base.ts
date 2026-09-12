import path from "node:path";
import { sha256 } from "../util/hash.js";
import { throwIfAborted } from "../util/abort.js";
import { simpleGit, type SimpleGit } from "simple-git";
import { Bundle, replaceSection } from "./bundle.js";
import { pruneEmptyDirs, regenerateIndexChain } from "./indexer.js";
import { appendLog, readLog } from "./logger.js";
import { searchBundle, listTypes, type SearchOptions } from "./search.js";
import { validateBundle } from "./validate.js";
import { lintBundle, type LintReport } from "./lint.js";
import { buildGraph, type GraphData } from "./graph.js";
import type {
  Concept,
  ConceptFrontmatter,
  ConformanceReport,
  LogAction,
  LogEntry,
  SearchHit,
  TreeNode,
} from "./types.js";

export interface KnowledgeBaseOptions {
  /** Commit after each mutation. Requires the bundle to be inside a git repo. */
  gitAutocommit?: boolean;
}

/**
 * The one write-path into the bundle. Spec conformance (index.md, log.md,
 * frontmatter validation, timestamps) is enforced HERE, deterministically —
 * never delegated to the LLM. Mutations are serialized through a queue.
 */
export class KnowledgeBase {
  readonly bundle: Bundle;
  private readonly git: SimpleGit | null;
  /** Queues are shared by every instance targeting the same resolved bundle root. */
  private static readonly mutationQueues = new Map<string, Promise<void>>();

  constructor(bundleRoot: string, private readonly options: KnowledgeBaseOptions = {}) {
    this.bundle = new Bundle(bundleRoot);
    this.git = options.gitAutocommit ? simpleGit(this.bundle.root) : null;
  }

  // ── Reads (no queue) ────────────────────────────────────────────────

  readConcept(conceptPath: string): Promise<Concept> {
    return this.bundle.readConcept(conceptPath);
  }

  listTree(): Promise<TreeNode> {
    return this.bundle.listTree();
  }

  search(query: string, options?: SearchOptions): Promise<SearchHit[]> {
    return searchBundle(this.bundle, query, options);
  }

  listTypes(): Promise<string[]> {
    return listTypes(this.bundle);
  }

  readLog(): Promise<LogEntry[]> {
    return readLog(this.bundle);
  }

  validate(): Promise<ConformanceReport> {
    return validateBundle(this.bundle);
  }

  /** Graph health: orphaned concepts + broken links (deterministic, no LLM). */
  lint(): Promise<LintReport> {
    return lintBundle(this.bundle);
  }

  /** Inter-concept link graph (nodes + edges) for visualization. */
  graph(): Promise<GraphData> {
    return buildGraph(this.bundle);
  }

  // ── Mutations (serialized; auto index + log + optional commit) ──────
  // The queue covers every instance in this process for one resolved root.
  // `createConcept` additionally uses an exclusive filesystem create, so an
  // external creator cannot overwrite it. Replace preconditions are not a
  // cross-process CAS: external writers must provide their own coordination.

  createConcept(
    conceptPath: string,
    frontmatter: ConceptFrontmatter,
    body: string,
    logSummary: string,
    signal?: AbortSignal
  ): Promise<Concept> {
    return this.enqueue(async () => {
      throwIfAborted(signal);
      const canonical = this.bundle.toBundlePath(conceptPath);
      throwIfAborted(signal);
      if (await this.bundle.exists(canonical)) {
        throw new Error(`Concept already exists: ${canonical}; use patch_concept`);
      }
      let concept: Concept;
      try {
        // The existence check gives a useful error for the common case. `wx`
        // also closes the race with a creator outside this process.
        throwIfAborted(signal);
        concept = await this.bundle.writeConcept(canonical, frontmatter, body, { exclusive: true });
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code === "EEXIST") {
          throw new Error(`Concept already exists: ${canonical}; use patch_concept`);
        }
        throw error;
      }
      await this.afterMutation(concept.path, "Creation", logSummary);
      return concept;
    }, signal);
  }

  writeConcept(
    conceptPath: string,
    frontmatter: ConceptFrontmatter,
    body: string,
    logSummary: string,
    signal?: AbortSignal
  ): Promise<Concept> {
    return this.enqueue(async () => {
      throwIfAborted(signal);
      const existed = await this.bundle.exists(conceptPath);
      throwIfAborted(signal);
      const concept = await this.bundle.writeConcept(conceptPath, frontmatter, body);
      await this.afterMutation(concept.path, existed ? "Update" : "Creation", logSummary);
      return concept;
    }, signal);
  }

  patchConcept(
    conceptPath: string,
    changes: Parameters<Bundle["patchConcept"]>[1],
    logSummary: string,
    expectedBodyHash?: string,
    signal?: AbortSignal
  ): Promise<Concept> {
    return this.enqueue(async () => {
      throwIfAborted(signal);
      if (expectedBodyHash) {
        const current = await this.bundle.readConcept(conceptPath);
        throwIfAborted(signal);
        const actual = sha256(current.body);
        if (actual !== expectedBodyHash) {
          throw new Error(`Concept changed while it was being read: ${current.path}`);
        }
      }

      // Read and construct the patch while serialized, then check again at the
      // last possible point before the concept file is written.
      const existing = await this.bundle.readConcept(conceptPath);
      throwIfAborted(signal);
      const fm: ConceptFrontmatter = { ...existing.frontmatter };
      if (changes.frontmatter) {
        for (const [key, value] of Object.entries(changes.frontmatter)) {
          if (value === null) delete fm[key];
          else fm[key] = value;
        }
      }
      let body = changes.replaceBody ?? existing.body;
      if (changes.replaceSection) {
        body = replaceSection(body, changes.replaceSection.heading, changes.replaceSection.content);
      }
      throwIfAborted(signal);
      const concept = await this.bundle.writeConcept(existing.path, fm, body);
      await this.afterMutation(concept.path, "Update", logSummary);
      return concept;
    }, signal);
  }

  deleteConcept(conceptPath: string, logSummary: string, signal?: AbortSignal): Promise<void> {
    return this.enqueue(async () => {
      throwIfAborted(signal);
      const canonical = this.bundle.toBundlePath(conceptPath);
      throwIfAborted(signal);
      await this.bundle.deleteConcept(canonical);
      await this.afterMutation(canonical, "Deletion", logSummary);
    }, signal);
  }

  private enqueue<T>(fn: () => Promise<T>, signal?: AbortSignal): Promise<T> {
    const key = this.bundle.root;
    const run = () => {
      // This runs only after the preceding mutation has fully settled, so a
      // cancelled waiter never starts any filesystem work of its own.
      throwIfAborted(signal);
      return fn();
    };
    const previous = KnowledgeBase.mutationQueues.get(key) ?? Promise.resolve();
    const next = previous.then(run, run);
    const settled = next.then(
      () => undefined,
      () => undefined
    );
    const cleanup = settled.then(() => {
      if (KnowledgeBase.mutationQueues.get(key) === cleanup) {
        KnowledgeBase.mutationQueues.delete(key);
      }
    });
    KnowledgeBase.mutationQueues.set(key, cleanup);
    return next;
  }

  private async afterMutation(
    conceptPath: string,
    action: LogAction,
    logSummary: string
  ): Promise<void> {
    // Sweep husks first (dirs holding only their auto-generated index.md) so
    // the reindex below never resurrects a pruned directory. Whole-bundle:
    // cheap at this scale, and it also heals husks from before this feature.
    await pruneEmptyDirs(this.bundle);
    await regenerateIndexChain(this.bundle, path.posix.dirname(conceptPath));
    const linked = `[${conceptPath.split("/").pop()}](${conceptPath})`;
    await appendLog(this.bundle, action, logSummary || `${action} of ${linked}.`);
    if (this.git) {
      try {
        await this.git.add(".");
        await this.git.commit(`${action.toLowerCase()}: ${logSummary || conceptPath}`);
      } catch (err) {
        // Autocommit is best-effort; the KB write itself already succeeded.
        console.error(`[understory] git autocommit failed: ${(err as Error).message}`);
      }
    }
  }
}
