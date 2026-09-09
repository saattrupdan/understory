const KNOWN_TOOLS =
  "(?:read_concept|read_concepts|search_knowledge|list_directory|lint_knowledge|" +
  "write_concept|patch_concept|delete_concept)";

const PROTOCOL_MARKER = /<\|tool_call_/i;
const MARKER_STYLE_CALL = new RegExp(
  `^\\s*(?:SUFFICIENT\\s+)?\\[\\s*${KNOWN_TOOLS}\\s*\\([\\s\\S]*\\]?\\s*$`,
  "i"
);
const COMPLETE_CALL = new RegExp(
  `^\\s*(?:SUFFICIENT\\s+)?${KNOWN_TOOLS}\\s*\\([\\s\\S]*\\)\\s*$`,
  "i"
);

/**
 * Return whether an answer is model-emitted tool protocol rather than prose.
 *
 * The bracketed form is deliberately anchored to the whole answer. This keeps
 * ordinary explanations that mention a tool name as valid answers while still
 * rejecting the marker-style syntax models sometimes copy from their training.
 */
export function isMalformedAnswer(answer: string): boolean {
  if (PROTOCOL_MARKER.test(answer)) return true;

  const candidate = answer.trim();
  return MARKER_STYLE_CALL.test(candidate) || COMPLETE_CALL.test(candidate);
}

/** A stable, concise diagnostic for logs and failed traces. */
export const MALFORMED_ANSWER_MESSAGE =
  "Malformed model answer: textual tool-call protocol leakage";
