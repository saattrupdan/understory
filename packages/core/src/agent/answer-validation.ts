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
const KNOWN_TOOL = new RegExp(`\\b(?:${KNOWN_TOOL_NAMES.join("|")})\\s*\\(`, "gi");
const KNOWN_TOOL_NAME = new RegExp(`\\b(?:${KNOWN_TOOL_NAMES.join("|")})\\b`, "i");
const TOOL_MARKER = /<\|tool_call_(?:start|end)\|>/gi;

interface Range {
  start: number;
  end: number;
}

interface CallCandidate {
  start: number;
  end: number;
  bracketed: boolean;
}

function protectedRanges(answer: string): Range[] {
  const ranges: Range[] = [];
  // Code and quoted examples are documentation. A single quote is only treated
  // as a quote when the quoted text itself contains protocol syntax; this avoids
  // treating the apostrophe in "Here's" as the start of a protected range.
  for (const pattern of [
    /```[\s\S]*?```/g,
    /`[^`\n]*`/g,
    /"(?:\\.|[^"\\])*"/g,
    /'\s*(?:\[|<\|tool_call_|read_concept|read_concepts|search_knowledge|list_directory|lint_knowledge|write_concept|patch_concept|delete_concept)[^'\n]*'/gi,
  ]) {
    for (const match of answer.matchAll(pattern)) {
      ranges.push({ start: match.index ?? 0, end: (match.index ?? 0) + match[0].length });
    }
  }
  return ranges;
}

function isProtected(index: number, ranges: Range[]): boolean {
  return ranges.some((range) => index >= range.start && index < range.end);
}

function matchingParen(answer: string, openParen: number): { end: number } {
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
      if (depth === 0) return { end: index + 1 };
    }
  }
  return { end: answer.length };
}

function callsIn(answer: string): CallCandidate[] {
  const ranges = protectedRanges(answer);
  const calls: CallCandidate[] = [];
  for (const match of answer.matchAll(KNOWN_TOOL)) {
    const start = match.index ?? 0;
    if (isProtected(start, ranges)) continue;
    const openParen = start + match[0].lastIndexOf("(");
    const result = matchingParen(answer, openParen);
    let end = result.end;
    let bracketed = false;
    let before = start - 1;
    while (before >= 0 && /[ \t]/.test(answer[before] ?? "")) before -= 1;
    if (answer[before] === "[") {
      bracketed = true;
    }
    const line = answer.slice(answer.lastIndexOf("\n", start - 1) + 1, start);
    if (line.lastIndexOf("[") > line.lastIndexOf("]")) bracketed = true;
    let after = end;
    while (after < answer.length && /[ \t]/.test(answer[after] ?? "")) after += 1;
    if (answer[after] === "]") end = after + 1;
    calls.push({ start, end, bracketed });
  }
  return calls;
}

function hasProtocolPrefix(prefix: string): boolean {
  const value = prefix.trim();
  // A colon is the protocol/documentation boundary: prose before it is an
  // introduction, while prose after a call is classified separately. The
  // recall path also deliberately emits the single protocol word SUFFICIENT.
  return !value || value.endsWith(":") || /^SUFFICIENT$/i.test(value);
}

function hasBriefProtocolSuffix(suffix: string): boolean {
  const value = suffix.trim().replace(/^[\s.,!?;:)}\]]+/, "");
  if (!value) return true;
  // Explanatory continuations are not protocol output, even when a call occurs
  // near the end of the sentence. Short terminal acknowledgements are common in
  // leaked model output and are deliberately covered here.
  if (/^(?:when|where|which|that|because|so that|to|for|is used|can be used|is a|is an|is the|returns|takes|expects)\b/i.test(value)) {
    return false;
  }
  return value.split(/\s+/).length <= 8;
}

function isCallSequence(answer: string, calls: CallCandidate[]): boolean {
  if (calls.length === 0) return false;
  const first = calls[0];
  const last = calls[calls.length - 1];
  const prefix = answer.slice(0, first.start);
  const protocolPrefix = prefix.replace(/\[\s*$/, "").trim();
  const suffix = answer.slice(last.end);
  const allBracketed = calls.every((call) => call.bracketed);
  const startsAtAnswerBoundary = !protocolPrefix;
  const linePrefix = answer.slice(answer.lastIndexOf("\n", first.start - 1) + 1, first.start);
  const startsAtLineBoundary =
    !linePrefix.replace(/\[\s*$/, "").trim() && hasProtocolPrefix(protocolPrefix);

  // Bare calls are protocol only at an answer/line boundary or after an
  // explicit protocol preface. This intentionally leaves documentation such as
  // "The guide ends with read_concept(path='x')" alone.
  const boundary =
    startsAtAnswerBoundary ||
    startsAtLineBoundary ||
    hasProtocolPrefix(protocolPrefix) ||
    (allBracketed && hasProtocolPrefix(protocolPrefix));
  if (!boundary) return false;

  // Calls in a sequence may be separated by JSON/list punctuation or whitespace.
  // Reject a candidate embedded in ordinary prose between two calls.
  for (let index = 1; index < calls.length; index += 1) {
    const between = answer.slice(calls[index - 1].end, calls[index].start).trim();
    if (between && !/^(?:[,;]|\]|\[|\{|\}|\n)+$/.test(between)) return false;
  }
  return (
    hasBriefProtocolSuffix(suffix) &&
    (startsAtAnswerBoundary || hasProtocolPrefix(protocolPrefix) ||
      (allBracketed && hasProtocolPrefix(protocolPrefix)))
  );
}

function jsonLikeToolPayload(answer: string): boolean {
  const value = answer.trim();
  if (!value.startsWith("{") && !value.startsWith("[")) return false;
  if (!KNOWN_TOOL_NAME.test(value)) return false;
  try {
    const parsed: unknown = JSON.parse(value.replace(/[\s.,!?]+$/, ""));
    const encoded = JSON.stringify(parsed);
    return KNOWN_TOOL_NAME.test(encoded);
  } catch {
    // A truncated JSON tool call is still protocol-shaped when it begins the
    // answer. Do not apply this to an object embedded in explanatory prose.
    return true;
  }
}

function markerEnvelope(answer: string): boolean {
  const ranges = protectedRanges(answer);
  const markers = [...answer.matchAll(TOOL_MARKER)].filter((match) => !isProtected(match.index ?? 0, ranges));
  if (markers.length === 0) return false;
  for (let index = 0; index < markers.length; index += 1) {
    const marker = markers[index];
    const start = marker.index ?? 0;
    const payloadStart = start + marker[0].length;
    const nextMarker = markers[index + 1];
    const payload = answer.slice(payloadStart, nextMarker?.index ?? answer.length);
    if (KNOWN_TOOL_NAME.test(payload) || jsonLikeToolPayload(payload)) return true;
    // A marker by itself is an actual protocol boundary only when it is the
    // complete answer (or a truncated answer beginning at the boundary).
    if (!answer.slice(0, start).trim() && (!nextMarker || !answer.slice(nextMarker.index! + nextMarker[0].length).trim())) {
      return true;
    }
  }
  return false;
}

/** Return whether an answer is model-emitted tool protocol rather than prose. */
export function isMalformedAnswer(answer: string): boolean {
  if (!answer.trim()) return false;
  if (markerEnvelope(answer)) return true;
  if (jsonLikeToolPayload(answer)) return true;
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

/**
 * Buffer text for the complete stream step before validating it. Text can be
 * split over several blocks by AI SDK v5, so validating at text-end would let a
 * protocol sequence straddling two blocks escape.
 */
export function createProtocolLeakageGuard(): ProtocolLeakageGuard {
  let malformed = false;
  let pendingText = "";
  let pendingTextParts: TextStreamPart<any>[] = [];

  const releaseText = (
    controller: TransformStreamDefaultController<TextStreamPart<any>>
  ): void => {
    const parts = pendingTextParts;
    const text = pendingText;
    pendingTextParts = [];
    pendingText = "";
    if (malformed || isMalformedAnswer(text)) {
      malformed = true;
      return;
    }
    for (const part of parts) controller.enqueue(part);
  };

  const transform: StreamTextTransform<any> = () =>
    new TransformStream<TextStreamPart<any>, TextStreamPart<any>>({
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

        // finish-step is the boundary for one model generation. finish is a
        // defensive fallback for providers that omit finish-step.
        if (part.type === "finish-step" || part.type === "finish") {
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
