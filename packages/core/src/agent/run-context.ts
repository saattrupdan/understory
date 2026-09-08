import { createHash } from "node:crypto";
import type { AgentLimits } from "./limits.js";

const EXHAUSTION_NOTICE = "Tool output budget exhausted; start a fresh request or raise the setting.";
const EXHAUSTION_SERIALISED_LENGTH = JSON.stringify(EXHAUSTION_NOTICE).length;

/** A body page observed by the agent during this run. */
interface BodyRead {
  totalChars: number;
  hash: string;
  ranges: Array<[number, number]>;
}

/**
 * State shared by every tool invocation in one model run.
 *
 * The counter is deliberately updated in a synchronous section after each
 * asynchronous tool operation. JavaScript does not interleave that section,
 * which makes concurrent AI SDK tool calls safe without serialising the reads.
 */
export class AgentRunContext {
  private remainingChars: number;
  private readonly bodyReads = new Map<string, BodyRead>();

  constructor(private readonly limits: AgentLimits) {
    this.remainingChars = limits.maxToolResultChars;
  }

  get remaining(): number {
    return this.remainingChars;
  }

  fits(value: unknown): boolean {
    return serialisedLength(value) <= this.remainingChars;
  }

  /** Consume a value as the SDK will serialise it, truncating its structure first. */
  result<T>(value: T): T {
    const length = serialisedLength(value);
    if (length <= this.remainingChars) {
      if (
        this.remainingChars - length >= EXHAUSTION_SERIALISED_LENGTH ||
        this.remainingChars < EXHAUSTION_SERIALISED_LENGTH
      ) {
        this.remainingChars -= length;
        return value;
      }
      return this.exhausted() as T;
    }

    const fitted = fitValue(value, this.remainingChars);
    if (
      fitted !== undefined &&
      (this.remainingChars - serialisedLength(fitted) >= EXHAUSTION_SERIALISED_LENGTH ||
        this.remainingChars < EXHAUSTION_SERIALISED_LENGTH)
    ) {
      this.remainingChars -= serialisedLength(fitted);
      return fitted as T;
    }

    return this.exhausted() as T;
  }

  /** Reserve space for the compact tree embedded in the system prompt. */
  systemTree(tree: string): string {
    const treeBudget =
      this.remainingChars >= EXHAUSTION_SERIALISED_LENGTH
        ? this.remainingChars - EXHAUSTION_SERIALISED_LENGTH
        : this.remainingChars;
    const source =
      serialisedLength(tree) <= treeBudget
        ? tree
        : `${tree}\n... [system tree truncated; search and list_directory remain available]`;
    const fitted = fitString(source, treeBudget);
    if (fitted === undefined) return "";
    this.remainingChars -= serialisedLength(fitted);
    return fitted;
  }

  /** Return an explicit, bounded result when no useful payload remains. */
  exhausted(): string | undefined {
    const fitted = fitString(EXHAUSTION_NOTICE, this.remainingChars);
    if (fitted === undefined) return undefined;
    this.remainingChars -= serialisedLength(fitted);
    return fitted;
  }

  /** Record a page only when the complete page survived result bounding. */
  recordBodyPage(
    conceptPath: string,
    offset: number,
    fullBody: string,
    pageBody: string,
    returnedBody: string
  ): void {
    if (pageBody !== returnedBody) return;
    const existing = this.bodyReads.get(conceptPath);
    const hash = hashBody(fullBody);
    if (existing && existing.hash !== hash) {
      existing.ranges = [];
      return;
    }
    const read = existing ?? { totalChars: fullBody.length, hash, ranges: [] };
    read.ranges.push([offset, offset + pageBody.length]);
    read.ranges = mergeRanges(read.ranges);
    this.bodyReads.set(conceptPath, read);
  }

  /** Verify that replace_body is based on a complete, unchanged body read. */
  expectedBodyHash(conceptPath: string, body: string): string {
    const read = this.bodyReads.get(conceptPath);
    const hash = hashBody(body);
    if (
      !read ||
      read.totalChars !== body.length ||
      read.hash !== hash ||
      !coversBody(read.ranges, body.length)
    ) {
      throw new Error(
        `replace_body requires a complete, unchanged read of ${conceptPath}; page the full body with read_concept first or use replace_section`
      );
    }
    return hash;
  }

  get maxDocumentChars(): number {
    return this.limits.maxDocumentChars;
  }
}

export function hashBody(body: string): string {
  return createHash("sha256").update(body).digest("hex");
}

function serialisedLength(value: unknown): number {
  const encoded = JSON.stringify(value);
  return encoded === undefined ? 0 : encoded.length;
}

function fitString(value: string, budget: number): string | undefined {
  if (serialisedLength(value) <= budget) return value;
  if (budget < 2) return undefined;
  let low = 0;
  let high = value.length;
  let best = "";
  while (low <= high) {
    const middle = Math.floor((low + high) / 2);
    const candidate = value.slice(0, middle);
    if (serialisedLength(candidate) <= budget) {
      best = candidate;
      low = middle + 1;
    } else {
      high = middle - 1;
    }
  }
  return best;
}

function fitValue(value: unknown, budget: number): unknown {
  if (serialisedLength(value) <= budget) return value;
  if (budget < 2) return undefined;
  if (typeof value === "string") return fitString(value, budget);
  if (value === null || typeof value === "number" || typeof value === "boolean") {
    return serialisedLength(value) <= budget ? value : undefined;
  }
  if (Array.isArray(value)) return fitArray(value, budget);
  if (typeof value === "object") return fitObject(value as Record<string, unknown>, budget);
  return undefined;
}

function fitArray(value: unknown[], budget: number): unknown[] {
  const result: unknown[] = [];
  for (const item of value) {
    const candidate = [...result, item];
    if (serialisedLength(candidate) <= budget) {
      result.push(item);
      continue;
    }
    const available = budget - serialisedLength(result) - 1;
    const fitted = fitValue(item, Math.max(0, available));
    if (fitted === undefined) break;
    const withFitted = [...result, fitted];
    if (serialisedLength(withFitted) <= budget) result.push(fitted);
    break;
  }
  return result;
}

function fitObject(value: Record<string, unknown>, budget: number): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  for (const [key, item] of Object.entries(value)) {
    const withoutValue = serialisedLength({ ...result, [key]: null }) - 4;
    const available = budget - withoutValue;
    const fitted = fitValue(item, Math.max(0, available));
    if (fitted === undefined) continue;
    const candidate = { ...result, [key]: fitted };
    if (serialisedLength(candidate) <= budget) result[key] = fitted;
  }
  return result;
}

function mergeRanges(ranges: Array<[number, number]>): Array<[number, number]> {
  ranges.sort((a, b) => a[0] - b[0]);
  const merged: Array<[number, number]> = [];
  for (const range of ranges) {
    const previous = merged.at(-1);
    if (previous && range[0] <= previous[1]) previous[1] = Math.max(previous[1], range[1]);
    else merged.push([...range]);
  }
  return merged;
}

function coversBody(ranges: Array<[number, number]>, length: number): boolean {
  return ranges.length === 1 && ranges[0][0] === 0 && ranges[0][1] >= length;
}
