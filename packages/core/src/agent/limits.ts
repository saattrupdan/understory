import { positiveIntegerEnv } from "../util/env.js";

export const DEFAULT_AGENT_MAX_STEPS = 8;
export const MIN_AGENT_MAX_STEPS = 2;
export const DEFAULT_AGENT_MAX_DOCUMENT_CHARS = 12_000;
export const DEFAULT_AGENT_MAX_TOOL_RESULT_CHARS = 24_000;
export const DEFAULT_AGENT_MAX_SYSTEM_CONTEXT_CHARS = 24_000;
/** JSON characters reserved for SDK tool-result framing, beyond the notice. */
export const TOOL_RESULT_CONTROL_OVERHEAD = 32;
/** Room for a visible `total_chars` truncation marker in bounded text. */
export const TOOL_RESULT_TRUNCATION_MARKER_RESERVE = 64;
export const EXHAUSTION_NOTICE =
  "Tool output budget exhausted; start a fresh request or raise the setting.";
export const EXHAUSTION_SERIALISED_LENGTH = JSON.stringify(EXHAUSTION_NOTICE).length;
/** Smallest useful tool budget: notice, framing, and visible marker room. */
export const MIN_AGENT_MAX_TOOL_RESULT_CHARS =
  EXHAUSTION_SERIALISED_LENGTH +
  TOOL_RESULT_CONTROL_OVERHEAD +
  TOOL_RESULT_TRUNCATION_MARKER_RESERVE;
/** Keeps both dynamic system-context truncation markers representable. */
export const MIN_AGENT_MAX_SYSTEM_CONTEXT_CHARS = 240;

export interface AgentLimits {
  maxSteps: number;
  maxDocumentChars: number;
  maxToolResultChars: number;
  maxSystemContextChars: number;
}

/** Resolve the context and agent-step bounds for one agent run. */
export function resolveAgentLimits(env: NodeJS.ProcessEnv = process.env): AgentLimits {
  return {
    maxSteps: Math.max(
      MIN_AGENT_MAX_STEPS,
      positiveIntegerEnv(env.AGENT_MAX_STEPS, DEFAULT_AGENT_MAX_STEPS)
    ),
    maxDocumentChars: positiveIntegerEnv(
      env.AGENT_MAX_DOCUMENT_CHARS,
      DEFAULT_AGENT_MAX_DOCUMENT_CHARS
    ),
    maxToolResultChars: Math.max(
      MIN_AGENT_MAX_TOOL_RESULT_CHARS,
      positiveIntegerEnv(env.AGENT_MAX_TOOL_RESULT_CHARS, DEFAULT_AGENT_MAX_TOOL_RESULT_CHARS)
    ),
    maxSystemContextChars: Math.max(
      MIN_AGENT_MAX_SYSTEM_CONTEXT_CHARS,
      positiveIntegerEnv(
        env.AGENT_MAX_SYSTEM_CONTEXT_CHARS,
        DEFAULT_AGENT_MAX_SYSTEM_CONTEXT_CHARS
      )
    ),
  };
}

export { positiveIntegerEnv } from "../util/env.js";
