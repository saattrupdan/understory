import type { StreamTextTransform, TextStreamPart } from "ai";

const KNOWN_TOOLS =
  "read_concept|read_concepts|search_knowledge|list_directory|lint_knowledge|" +
  "write_concept|patch_concept|delete_concept";
const TOOL_CALL = new RegExp(`(?:\\[\\s*)?(?:${KNOWN_TOOLS})\\s*\\(`, "gi");
const TOOL_MARKER = /<\|tool_call_(?:start|end)\|>/gi;

interface Range {
  start: number;
  end: number;
}

function protectedRanges(answer: string): Range[] {
  const ranges: Range[] = [];
  // Markdown code and quoted examples are documentation, not model protocol.
  for (const pattern of [/```[\s\S]*?```/g, /`[^`\n]*`/g, /"(?:\\.|[^"\\])*"/g]) {
    for (const match of answer.matchAll(pattern)) {
      ranges.push({ start: match.index ?? 0, end: (match.index ?? 0) + match[0].length });
    }
  }
  return ranges;
}

function isProtected(answer: string, index: number, ranges: Range[]): boolean {
  if (ranges.some((range) => index >= range.start && index < range.end)) return true;

  // A single-quoted Markdown example can contain single quotes in its tool
  // arguments, so a regular expression cannot safely find its closing quote.
  // Require the candidate to begin immediately after the opening quote (apart
  // from whitespace) and to have a closing quote after the call. This avoids
  // mistaking the apostrophe in a preface such as "Here's the call" for a quote.
  const lineStart = answer.lastIndexOf("\n", index - 1) + 1;
  let beforeIndex = index - 1;
  while (beforeIndex >= lineStart && /\s/.test(answer[beforeIndex] ?? "")) beforeIndex -= 1;
  if (answer[beforeIndex] === "'" && /[\])]\s*'/.test(answer.slice(index))) return true;
  return false;
}

function callEnd(answer: string, openParen: number): number {
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
    else if (char === ")" && --depth === 0) return index + 1;
  }
  // A cut-off call is still protocol-shaped. The caller will use the context
  // around the candidate to distinguish it from explanatory prose.
  return answer.length;
}

function isBareCallLeakage(answer: string, ranges: Range[]): boolean {
  for (const match of answer.matchAll(TOOL_CALL)) {
    const start = match.index ?? 0;
    if (isProtected(answer, start, ranges)) continue;

    const openParen = start + match[0].lastIndexOf("(");
    let end = callEnd(answer, openParen);
    const closingBracket = answer.slice(end).match(/^\s*\]/);
    if (closingBracket) end += closingBracket[0].length;

    const prefix = answer.slice(0, start).trim();
    const suffix = answer.slice(end).trim();
    const linePrefix = answer.slice(answer.lastIndexOf("\n", start - 1) + 1, start).trim();
    const prefixBeforeBracket = prefix.replace(/\[\s*$/, "").trim();
    const prefaced =
      prefix === "" ||
      prefix.endsWith(":") ||
      prefixBeforeBracket.endsWith(":") ||
      /^SUFFICIENT$/i.test(prefix) ||
      /^(?:sure|okay|ok|yes|no)[,!?]?\s*$/.test(prefixBeforeBracket) ||
      /^(?:here(?:'s| is)?|the\s+(?:requested\s+)?tool\s+call(?:\s+is)?|the\s+answer\s+is|result|output|calling|i(?:'ll| will)? use)\s*[:,-]?$/i.test(
        prefixBeforeBracket
      ) ||
      linePrefix === "";

    // Explanatory prose normally continues after the invocation. A call at
    // the end of a prefaced answer, or a truncated call, is leakage.
    if (prefaced && (suffix === "" || /^<\|tool_call_(?:start|end)\|>$/i.test(suffix))) {
      return true;
    }
  }
  return false;
}

/**
 * Return whether an answer is model-emitted tool protocol rather than prose.
 *
 * Detection is deliberately shape- and context-aware: markers must participate
 * in a marker/call sequence (or be the whole answer), and bare calls must look
 * like an emitted answer. Inline documentation, quoted examples, and backtick
 * examples remain ordinary prose.
 */
export function isMalformedAnswer(answer: string): boolean {
  const ranges = protectedRanges(answer);
  if (isBareCallLeakage(answer, ranges)) return true;

  for (const match of answer.matchAll(TOOL_MARKER)) {
    const start = match.index ?? 0;
    if (isProtected(answer, start, ranges)) continue;
    const before = answer.slice(0, start).trim();
    const after = answer.slice(start + match[0].length).trim();
    const markerIsWholeAnswer = before === "" && after === "";
    const markerTail = answer.slice(start + match[0].length);
    const nearbyCall = isBareCallLeakage(markerTail, protectedRanges(markerTail));
    if (markerIsWholeAnswer || nearbyCall) return true;
  }
  return false;
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
 * Gate text parts before AI SDK exposes them to a chat client.
 *
 * A finite prefix cannot tell ordinary prose from a protocol call that appears
 * later in the same answer. The smallest safe trade-off is therefore to buffer
 * one complete text part (normally the final synthesis) and release it only at
 * `text-end`; tool/reasoning parts continue through the SDK normally.
 */
export function createProtocolLeakageGuard(): ProtocolLeakageGuard {
  let malformed = false;
  let pendingStart: TextStreamPart<any> | undefined;
  let pendingText = "";

  const release = (
    controller: TransformStreamDefaultController<TextStreamPart<any>>,
    stopStream: () => void
  ): void => {
    if (isMalformedAnswer(pendingText)) {
      malformed = true;
      pendingStart = undefined;
      pendingText = "";
      stopStream();
      return;
    }
    if (pendingStart) controller.enqueue(pendingStart);
    if (pendingText) {
      controller.enqueue({
        type: "text-delta",
        id: pendingStart && "id" in pendingStart ? pendingStart.id : "guarded-text",
        text: pendingText,
      });
    }
    pendingStart = undefined;
    pendingText = "";
  };

  const transform: StreamTextTransform<any> = ({ stopStream }) =>
    new TransformStream<TextStreamPart<any>, TextStreamPart<any>>({
      transform(part, controller) {
        if (part.type === "text-start") {
          // A well-formed stream has one start/end pair. Flush defensively if a
          // provider starts a second one without ending the first.
          if (pendingStart || pendingText) release(controller, stopStream);
          pendingStart = part;
          pendingText = "";
        } else if (part.type === "text-delta") {
          pendingText += part.text;
        } else if (part.type === "text-end") {
          pendingText += "";
          release(controller, stopStream);
          if (!malformed) controller.enqueue(part);
        } else {
          controller.enqueue(part);
        }
      },
      flush(controller) {
        if (pendingStart || pendingText) release(controller, stopStream);
      },
    });

  return { transform, wasMalformed: () => malformed };
}
