import { sha256 } from "../util/hash.js";
import {
  DEFAULT_AGENT_MAX_INPUT_CHARS,
  EXHAUSTION_NOTICE,
  EXHAUSTION_SERIALISED_LENGTH,
  MIN_AGENT_MAX_SYSTEM_CONTEXT_CHARS,
  MIN_AGENT_MAX_TOOL_RESULT_CHARS,
  TOOL_RESULT_CONTROL_OVERHEAD,
  inputLength,
  type AgentLimits,
} from "./limits.js";

const SYSTEM_TREE_MARKER = "\n... [system tree truncated; search and list_directory remain available]";
const SYSTEM_TYPES_MARKER = "\n... [system types truncated; search remains available]";
const TRUNCATION_MARKER = /\n\.\.\. \[truncated; total_chars=\d+\]$/;

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
export type AgentLimitsInput = Omit<AgentLimits, "maxInputChars"> &
  Partial<Pick<AgentLimits, "maxInputChars">>;

export class AgentRunContext {
  private readonly limits: Pick<
    AgentLimits,
    "maxSteps" | "maxDocumentChars" | "maxToolResultChars" | "maxSystemContextChars" | "maxInputChars"
  >;
  private remainingChars: number;
  private remainingSystemChars: number;
  private systemTreeWritten = false;
  private systemTypesWritten = false;
  private readonly bodyReads = new Map<string, BodyRead>();
  private writeInputChars = 0;

  constructor(limits: AgentLimitsInput) {
    this.limits = {
      ...limits,
      maxInputChars: limits.maxInputChars ?? DEFAULT_AGENT_MAX_INPUT_CHARS,
    };
    // Keep manually constructed contexts safe too; callers should not be able
    // to configure a budget in which even the control result cannot fit.
    this.remainingChars = Math.max(
      MIN_AGENT_MAX_TOOL_RESULT_CHARS,
      limits.maxToolResultChars
    );
    this.remainingSystemChars = Math.max(
      MIN_AGENT_MAX_SYSTEM_CONTEXT_CHARS,
      limits.maxSystemContextChars
    );
  }

  get remaining(): number {
    return this.remainingChars;
  }

  get remainingSystemContext(): number {
    return this.remainingSystemChars;
  }

  /** Budget available for useful payload before the next control notice. */
  get payloadBudget(): number {
    return Math.max(
      0,
      this.remainingChars - EXHAUSTION_SERIALISED_LENGTH - TOOL_RESULT_CONTROL_OVERHEAD
    );
  }

  fits(value: unknown): boolean {
    return serialisedLength(value) <= this.payloadBudget;
  }

  /**
   * Consume a complete, already structurally bounded tool result.
   *
   * The notice reserve is applied before consumption. A rejected payload is
   * replaced with a valid notice rather than returning undefined.
   */
  consume<T>(value: T): T {
    if (value === undefined) return this.exhausted() as T;
    const length = serialisedLength(value);
    if (length > this.payloadBudget) return this.exhausted() as T;
    this.remainingChars -= length;
    return value;
  }

  /** Consume a value, structurally fitting it against the reserved payload budget. */
  result<T>(value: T): T {
    const budget = this.payloadBudget;
    const fitted = fitValue(value, budget);
    if (fitted === undefined) return this.exhausted() as T;
    const length = serialisedLength(fitted);
    if (length > budget) return this.exhausted() as T;
    this.remainingChars -= length;
    return fitted as T;
  }

  /** Reserve space for dynamic system context, independently of tool results. */
  systemTree(tree: string): string {
    const result = this.consumeSystemText(
      tree,
      SYSTEM_TREE_MARKER,
      this.systemTypesWritten ? 0 : serialisedLength(SYSTEM_TYPES_MARKER)
    );
    this.systemTreeWritten = true;
    return result;
  }

  /** Bound the type map embedded in the system prompt. */
  systemTypes(types: string[]): string[] {
    if (types.length === 0) {
      this.systemTypesWritten = true;
      return [];
    }
    const value = types.join(", ");
    const bounded = this.consumeSystemText(
      value,
      SYSTEM_TYPES_MARKER,
      this.systemTreeWritten ? 0 : serialisedLength(SYSTEM_TREE_MARKER)
    );
    this.systemTypesWritten = true;
    return [bounded];
  }

  private consumeSystemText(value: string, marker: string, reserveAfter = 0): string {
    if (serialisedLength(value) <= this.remainingSystemChars - reserveAfter) {
      this.remainingSystemChars -= serialisedLength(value);
      return value;
    }
    const markerLength = serialisedLength(marker);
    if (markerLength + reserveAfter > this.remainingSystemChars) {
      return "";
    }
    let low = 0;
    let high = value.length;
    let best = marker;
    while (low <= high) {
      const middle = Math.floor((low + high) / 2);
      const candidate = value.slice(0, middle) + marker;
      if (serialisedLength(candidate) + reserveAfter <= this.remainingSystemChars) {
        best = candidate;
        low = middle + 1;
      } else {
        high = middle - 1;
      }
    }
    this.remainingSystemChars -= serialisedLength(best);
    return best;
  }

  /**
   * Return an explicit result when no useful payload remains.
   *
   * Once the notice itself has been emitted, repeated calls may exceed the
   * accounting budget by its fixed JSON framing; a tool result must never be
   * undefined merely because the run is exhausted.
   */
  exhausted(): string {
    if (this.remainingChars >= EXHAUSTION_SERIALISED_LENGTH) {
      this.remainingChars -= EXHAUSTION_SERIALISED_LENGTH;
    }
    return EXHAUSTION_NOTICE;
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
    const hash = sha256(fullBody);
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
    const hash = sha256(body);
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

  get maxInputChars(): number {
    return this.limits.maxInputChars;
  }

  /** Meter model-generated write arguments across all writes in this run. */
  assertWriteInput(value: unknown): void {
    const length = inputLength(value);
    if (length > this.maxInputChars || this.writeInputChars + length > this.maxInputChars) {
      throw new Error(`Write tool input exceeds AGENT_MAX_INPUT_CHARS (${this.maxInputChars} characters)`);
    }
    this.writeInputChars += length;
  }

  /** Fit a value without consuming the shared tool-result budget. */
  fit<T>(value: T, budget: number): T | undefined {
    return fitValue(value, budget) as T | undefined;
  }
}

export function serialisedLength(value: unknown): number {
  const encoded = JSON.stringify(value);
  return encoded === undefined ? 0 : encoded.length;
}

function fitString(value: string, budget: number): string | undefined {
  if (serialisedLength(value) <= budget) return value;
  const markerMatch = value.match(TRUNCATION_MARKER);
  if (markerMatch) return fitText(value.slice(0, -markerMatch[0].length), budget, markerMatch[0]);
  if (budget < 2) return undefined;
  return fitText(value, budget, "");
}

/** Fit text while reserving its visible truncation marker in JSON characters. */
export function fitText(value: string, budget: number, marker: string): string {
  if (!marker && serialisedLength(value) <= budget) return value;
  if (serialisedLength(marker) > budget) return marker;
  let low = 0;
  let high = value.length;
  let best = marker;
  while (low <= high) {
    const middle = Math.floor((low + high) / 2);
    const candidate = value.slice(0, middle) + marker;
    if (serialisedLength(candidate) <= budget) {
      best = candidate;
      low = middle + 1;
    } else {
      high = middle - 1;
    }
  }
  return best;
}

export function fitValue(value: unknown, budget: number): unknown {
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
