import type { StreamTextTransform, TextStreamPart } from "ai";

const KNOWN_TOOL_NAMES = [
  "read_concept",
  "read_concepts",
  "search_knowledge",
  "list_directory",
  "lint_knowledge",
  "write_concept",
  "patch_concept",
  "delete_concept",
] as const;

const TOOL_NAME_PATTERN = KNOWN_TOOL_NAMES.join("|");
const KNOWN_TOOL_CALL = new RegExp(`\\b(?:${TOOL_NAME_PATTERN})\\s*\\(`, "gi");
const TOOL_MARKER = /<\|tool_call_(?:start|end)\|>/gi;
const KNOWN_TOOL_SET = new Set<string>(KNOWN_TOOL_NAMES);

interface Range {
  start: number;
  end: number;
}

interface CallCandidate {
  start: number;
  end: number;
  complete: boolean;
}

function protectedRanges(answer: string): Range[] {
  const ranges: Range[] = [];
  // Documentation is allowed to quote protocol syntax. Do not hide arbitrary
  // quoted prose: only protect a quoted span when it contains a marker or a
  // complete-looking tool call.
  for (const pattern of [
    /```[\s\S]*?```/g,
    /`[^`\n]*`/g,
    /"(?:\\.|[^"\\])*"/g,
    /'[^'\n]*'/g,
  ]) {
    for (const match of answer.matchAll(pattern)) {
      const value = match[0];
      const start = match.index ?? 0;
      // A contraction apostrophe is not a quote delimiter. Without this
      // guard, the apostrophe in "Here's the call: [read_concept(path='x')]"
      // opens a range that hides the actual leaked call.
      if (
        value.startsWith("'") &&
        /[\p{L}\p{N}]/u.test(answer[start - 1] ?? "") &&
        /[\p{L}\p{N}]/u.test(answer[start + 1] ?? "")
      ) {
        continue;
      }
      if (
        value.includes("<|tool_call_") ||
        new RegExp(`(?:${TOOL_NAME_PATTERN})\\s*\\(`, "i").test(value)
      ) {
        ranges.push({ start, end: start + value.length });
      }
    }
  }
  return ranges;
}

function isProtected(index: number, ranges: Range[]): boolean {
  return ranges.some((range) => index >= range.start && index < range.end);
}

function matchingParen(answer: string, openParen: number): { end: number; complete: boolean } {
  let depth = 0;
  let quote: "'" | '"' | undefined;
  let escaped = false;
  for (let index = openParen; index < answer.length; index += 1) {
    const char = answer[index];
    if (quote) {
      if (escaped) escaped = false;
      else if (char === "\\") escaped = true;
      else if (char === quote) quote = undefined;
      continue;
    }
    if (char === "'" || char === '"') {
      quote = char;
      continue;
    }
    if (char === "(") depth += 1;
    else if (char === ")") {
      depth -= 1;
      if (depth === 0) return { end: index + 1, complete: true };
    }
  }
  return { end: answer.length, complete: false };
}

function callsIn(answer: string): CallCandidate[] {
  const ranges = protectedRanges(answer);
  const calls: CallCandidate[] = [];
  for (const match of answer.matchAll(KNOWN_TOOL_CALL)) {
    const start = match.index ?? 0;
    if (isProtected(start, ranges)) continue;
    const openParen = start + match[0].lastIndexOf("(");
    const result = matchingParen(answer, openParen);
    calls.push({ start, end: result.end, complete: result.complete });
  }
  return calls;
}

function isProtocolPreface(value: string): boolean {
  const prefix = value
    .replace(/^[\s\[({,;:]+/, "")
    .replace(/[\s:([{,;]+$/, "")
    .trim();
  if (!prefix) return true;
  if (/^sufficient$/i.test(prefix)) return true;
  const base =
    "(?:sure|okay|ok|alright|certainly|of course|here(?:['’]s| is)(?:\\s+(?:the\\s+)?(?:tool\\s+)?call(?:\\s+is)?)?|calling|call(?:ing)?|using|invoking|running|the tool call(?: is)?|i will (?:use|call|invoke|run)|i(?:['’]ll| will) (?:use|call|invoke|run)|let me (?:use|call|invoke|run)|i(?:['’]m| am) going to (?:use|call|invoke|run)|use|call|invoke|run)";
  return new RegExp(
    `^${base}(?:\\s+(?:the\\s+)?(?:${TOOL_NAME_PATTERN})(?:\\s+(?:tool|function))?|\\s+the\\s+(?:tool|function))?$`,
    "i"
  ).test(prefix);
}

function isProtocolSeparator(value: string): boolean {
  return /^[\s,;:[\]{}()]*$/.test(value);
}

function lineContaining(answer: string, index: number): string {
  const start = answer.lastIndexOf("\n", index - 1) + 1;
  const end = answer.indexOf("\n", index);
  return answer.slice(start, end < 0 ? answer.length : end);
}

function stripCallPunctuation(value: string): string {
  return value.trim().replace(/^(?:\[|\{)+/, "").replace(/(?:\]|\})+$/, "").trim();
}

function isTerminalProtocolSuffix(value: string): boolean {
  const suffix = value.trim().replace(/^[\s.,!?;:)}\]]+/, "");
  if (!suffix) return true;
  return /^(?:done|complete|completed|finished|success|successful|okay|ok|that's it|that is all)[.!]?$/i.test(
    suffix
  );
}

function isStandaloneCallLine(answer: string, call: CallCandidate): boolean {
  const line = lineContaining(answer, call.start);
  const before = line.slice(0, call.start - (answer.lastIndexOf("\n", call.start - 1) + 1));
  const after = line.slice(call.end - (answer.lastIndexOf("\n", call.end - 1) + 1));
  return (
    isProtocolSeparator(stripCallPunctuation(before)) &&
    isProtocolSeparator(stripCallPunctuation(after)) &&
    (call.complete || !after.trim())
  );
}

function isCallSequence(answer: string, calls: CallCandidate[]): boolean {
  if (calls.length === 0) return false;

  // A call on its own line is an unambiguous leaked protocol item, even when
  // the model first emitted an ordinary sentence. This catches:
  //   I will inspect the file.
  //   read_concept(path="x")
  if (calls.some((call) => isStandaloneCallLine(answer, call))) return true;

  const first = calls[0];
  const last = calls[calls.length - 1];
  for (let index = 1; index < calls.length; index += 1) {
    if (!isProtocolSeparator(answer.slice(calls[index - 1].end, calls[index].start))) {
      return false;
    }
  }

  const prefix = answer.slice(0, first.start);
  const prefixWithoutListPunctuation = prefix.replace(/[\s\[]+$/, "");
  const suffix = answer.slice(last.end);
  const atAnswerBoundary = !prefixWithoutListPunctuation.trim();
  const preface = isProtocolPreface(prefixWithoutListPunctuation);
  const linePrefix = lineContaining(answer, first.start).slice(
    0,
    first.start - (answer.lastIndexOf("\n", first.start - 1) + 1)
  );
  const atLineBoundary = isProtocolSeparator(stripCallPunctuation(linePrefix));

  if (!(atAnswerBoundary || preface || atLineBoundary)) return false;
  if (!isTerminalProtocolSuffix(suffix)) return false;

  // A truncated call is protocol only when it is at a structural boundary. A
  // name followed by an unfinished argument in ordinary prose is not enough.
  return calls.every((call) => call.complete || atAnswerBoundary || preface || atLineBoundary);
}

function isKnownToolCallObject(value: unknown): boolean {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const object = value as Record<string, unknown>;
  const name = [object.name, object.tool, object.toolName].find(
    (candidate): candidate is string => typeof candidate === "string"
  );
  if (name && KNOWN_TOOL_SET.has(name)) {
    return ["arguments", "input", "parameters", "params"].some((key) => key in object);
  }

  // OpenAI's function protocol nests the actual call under `function`. Keep
  // the argument-field requirement so ordinary JSON documentation mentioning a
  // tool name is still prose.
  const nested = object.function;
  if (!nested || typeof nested !== "object" || Array.isArray(nested)) return false;
  const nestedObject = nested as Record<string, unknown>;
  return (
    typeof nestedObject.name === "string" &&
    KNOWN_TOOL_SET.has(nestedObject.name) &&
    "arguments" in nestedObject
  );
}

function isProtocolJson(value: unknown): boolean {
  if (isKnownToolCallObject(value)) return true;
  return Array.isArray(value) && value.length > 0 && value.every(isKnownToolCallObject);
}

function normaliseSingleQuotedJson(value: string): string | undefined {
  let output = "";
  let quote: "'" | '"' | undefined;
  let escaped = false;
  for (const char of value) {
    if (quote) {
      if (escaped) {
        // JSON has no \\' escape. Treat it as the literal apostrophe while
        // preserving all normal JSON escapes.
        output += char === "'" ? "'" : `\\${char}`;
        escaped = false;
      } else if (char === "\\") {
        escaped = true;
      } else if (char === quote) {
        output += '"';
        quote = undefined;
      } else {
        output += char === '"' && quote === "'" ? '\\\"' : char;
      }
      continue;
    }
    if (char === "'" || char === '"') {
      quote = char;
      output += '"';
    } else {
      output += char;
    }
  }
  return quote || escaped ? undefined : output;
}

function parseJsonProtocol(answer: string): boolean {
  const value = answer.trim();
  if (!value.startsWith("{") && !value.startsWith("[")) return false;
  const withoutTerminalPunctuation = value.replace(/[\s.,!?;:]+$/, "");
  try {
    return isProtocolJson(JSON.parse(withoutTerminalPunctuation));
  } catch {
    const normalised = normaliseSingleQuotedJson(withoutTerminalPunctuation);
    if (normalised) {
      try {
        if (isProtocolJson(JSON.parse(normalised))) return true;
      } catch {
        // Continue with the truncated-envelope fallback below.
      }
    }
    // Providers sometimes stop halfway through a JSON protocol envelope. Keep
    // this anchored to a root object/array and require both the protocol name
    // and an argument field, rather than looking for a tool-name string.
    const name = new RegExp(
      `^[\\[{\\s]*(?:["'](?:name|tool|toolName)["']\\s*:\\s*["'])?(?:${TOOL_NAME_PATTERN})(?:["']|\\b)`,
      "i"
    );
    const argument = /["'](?:arguments|input|parameters|params)["']\s*:/i;
    return name.test(value) && argument.test(value);
  }
}

function markerEnvelope(answer: string): boolean {
  const ranges = protectedRanges(answer);
  const markers = [...answer.matchAll(TOOL_MARKER)].filter(
    (match) => !isProtected(match.index ?? 0, ranges)
  );
  if (markers.length === 0) return false;
  if (!answer.replace(TOOL_MARKER, "").trim()) return false;

  for (let index = 0; index < markers.length; index += 1) {
    const marker = markers[index];
    const start = marker.index ?? 0;
    const payloadStart = start + marker[0].length;
    const nextMarker = markers[index + 1];
    const payload = answer.slice(payloadStart, nextMarker?.index ?? answer.length);
    const payloadWithoutEnd = payload.replace(/<\|tool_call_end\|>/gi, "").trim();
    const before = answer.slice(0, start).trim();
    const end = payload.search(/<\|tool_call_end\|>/i);
    const afterEnd = end < 0 ? "" : payload.slice(end + "<|tool_call_end|>".length).trim();

    // A complete start/end envelope is protocol even when its payload uses a
    // historical or otherwise unknown tool name. A truncated envelope is
    // likewise protocol when its payload has call/object structure. Restrict
    // this broad rule to the answer boundary so marker documentation remains
    // valid prose.
    const wholeAnswer = !before && !afterEnd;
    const isStart = /start/i.test(marker[0]);
    const hasEnd = end >= 0;
    const hasProtocolShape =
      parseJsonProtocol(payloadWithoutEnd) ||
      /\b[A-Za-z_$][\w$.-]*\s*\(/.test(payloadWithoutEnd) ||
      /^[\[{]/.test(payloadWithoutEnd);
    if (wholeAnswer && isStart && payloadWithoutEnd && (hasEnd || hasProtocolShape)) return true;

    if (parseJsonProtocol(payloadWithoutEnd) || isCallSequence(payloadWithoutEnd, callsIn(payloadWithoutEnd))) {
      if (!before || isProtocolPreface(answer.slice(0, start))) return true;
    }
    // A payload-only marker pair at an answer boundary is malformed, while an
    // empty marker pair mentioned in prose remains documentation.
    if (
      hasEnd &&
      payloadWithoutEnd &&
      !afterEnd &&
      (!before || isProtocolPreface(answer.slice(0, start)))
    ) {
      return true;
    }
  }
  return false;
}

/** Return whether an answer is model-emitted tool protocol rather than prose. */
export function isMalformedAnswer(answer: string): boolean {
  if (!answer.trim()) return false;
  if (markerEnvelope(answer)) return true;
  if (parseJsonProtocol(answer)) return true;
  return isCallSequence(answer, callsIn(answer));
}

/** A stable, concise diagnostic for logs and failed traces. */
export const MALFORMED_ANSWER_MESSAGE =
  "Malformed model answer: textual tool-call protocol leakage";

export interface ProtocolLeakageGuard {
  /** AI SDK v5 `experimental_transform` hook. */
  transform: StreamTextTransform<any>;
  wasMalformed(): boolean;
}

export interface ProtocolLeakageGuardOptions {
  /** Optional dynamic detail, evaluated only when malformed text is found. */
  errorMessage?: () => string;
}

/**
 * Buffer text and terminal events for the complete stream step before
 * validating it. Text can be split over several blocks by AI SDK v5, so
 * validating at text-end would let a protocol sequence straddling two blocks
 * escape. A malformed step produces an AI SDK v5 `error` stream part before
 * releasing the provider terminal events, so the client can observe failure
 * before any completion marker.
 */
export function createProtocolLeakageGuard(
  options: ProtocolLeakageGuardOptions = {}
): ProtocolLeakageGuard {
  let malformed = false;
  let errorEmitted = false;
  let pendingText = "";
  let pendingTextParts: TextStreamPart<any>[] = [];
  const errorMessage = options.errorMessage ?? (() => MALFORMED_ANSWER_MESSAGE);

  const emitError = (controller: TransformStreamDefaultController<any>): void => {
    if (errorEmitted) return;
    errorEmitted = true;
    controller.enqueue({ type: "error", error: new Error(errorMessage()) });
  };

  const releaseText = (
    controller: TransformStreamDefaultController<any>
  ): boolean => {
    const parts = pendingTextParts;
    const text = pendingText;
    pendingTextParts = [];
    pendingText = "";
    if (malformed || isMalformedAnswer(text)) {
      malformed = true;
      emitError(controller);
      return false;
    }
    for (const part of parts) controller.enqueue(part);
    return true;
  };

  const transform: StreamTextTransform<any> = () =>
    new TransformStream<TextStreamPart<any>, any>({
      transform(part, controller) {
        if (
          part.type === "text-start" ||
          part.type === "text-delta" ||
          part.type === "text-end"
        ) {
          pendingTextParts.push(part);
          if (part.type === "text-delta") pendingText += part.text;
          return;
        }

        // Terminal events are deliberately held behind validation. In the
        // malformed case the client receives `error` first and no success
        // finish marker can claim that the answer completed successfully.
        if (part.type === "finish-step" || part.type === "finish") {
          // Preserve the provider's terminal shape so AI SDK can close its
          // stream, but only after the error part has been placed first.
          releaseText(controller);
          controller.enqueue(part);
          return;
        }

        controller.enqueue(part);
      },
      flush(controller) {
        releaseText(controller);
      },
    });

  return { transform, wasMalformed: () => malformed };
}
