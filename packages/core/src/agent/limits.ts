export const DEFAULT_AGENT_MAX_STEPS = 8;
export const MIN_AGENT_MAX_STEPS = 2;
export const DEFAULT_AGENT_MAX_DOCUMENT_CHARS = 12_000;
export const DEFAULT_AGENT_MAX_TOOL_RESULT_CHARS = 24_000;

export interface AgentLimits {
  maxSteps: number;
  maxDocumentChars: number;
  maxToolResultChars: number;
}

/** Parse a positive integer setting, falling back for missing or invalid input. */
export function positiveIntegerEnv(raw: string | undefined, fallback: number): number {
  const value = raw?.trim();
  if (!value || !/^[1-9]\d*$/.test(value)) return fallback;
  const parsed = Number(value);
  return Number.isSafeInteger(parsed) ? parsed : fallback;
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
    maxToolResultChars: positiveIntegerEnv(
      env.AGENT_MAX_TOOL_RESULT_CHARS,
      DEFAULT_AGENT_MAX_TOOL_RESULT_CHARS
    ),
  };
}
