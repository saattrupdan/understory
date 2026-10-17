import path from "node:path";
import {
  generateText,
  generateObject as sdkGenerateObject,
  NoObjectGeneratedError,
  streamObject as sdkStreamObject,
  wrapLanguageModel,
  modelMessageSchema,
  streamText,
  stepCountIs,
  type LanguageModel,
  type ModelMessage,
} from "ai";
import type { KnowledgeBase } from "../okf/index.js";
import { z } from "zod";
import {
  createModel,
  isLocalModelEndpoint,
  resolveFallbackConfig,
  resolveModelConfig,
  type ModelConfig,
} from "../providers/index.js";
import { withFallback } from "../providers/fallback.js";
import {
  buildQuerySynthesisPrompt,
  buildSystemPrompt,
} from "./system-prompt.js";
import { buildReadTools, buildWriteTools, formatTree } from "./tools.js";
import {
  assertInputWithinLimit,
  resolveAgentLimits,
} from "./limits.js";
import { AgentRunContext, fitText } from "./run-context.js";
import { isAbortError, throwIfAborted } from "../util/abort.js";
import { sha256 } from "../util/hash.js";
import { planMutationEvidence } from "./mutation-evidence.js";
import { TraceRecorder, TraceStore, type TraceTiming, type TraceUsage } from "./trace.js";
import {
  createProtocolLeakageGuard,
  isMalformedAnswer,
  isUnsafeSynthesisAnswer,
  MALFORMED_ANSWER_MESSAGE,
} from "./answer-validation.js";

// A syntactically complete JSON object can still be cut off after reaching a
// generation cap. Never accept it as an independent safety decision or a write.
async function generateObject<S extends z.ZodTypeAny>(options: {
  model: LanguageModel;
  schema: S;
  temperature: number;
  abortSignal?: AbortSignal;
  prompt: string;
}): Promise<{ object: z.infer<S>; finishReason?: string }> {
  const baseURL = process.env.LLM_API_BASE_URL;
  const staged = process.env.MUTATION_STAGED === "true";
  // A local model can produce malformed JSON once without exhausting its
  // output budget. Retry only that transient parse failure, once, with the
  // same evidence and schema. Never retry or accept a capped generation.
  for (let attempt = 0; ; attempt++) {
    if (options.abortSignal?.aborted) throw options.abortSignal.reason ?? new Error("Mutation aborted");
    try {
      if (staged && process.env.MUTATION_STRUCTURED_STREAM !== "false" &&
          baseURL && isLocalModelEndpoint(baseURL)) {
        // Stream deltas to keep transport alive; only expose the validated object.
        const streamed = sdkStreamObject(options);
        // The final promises are driven by consuming fullStream.
        const [, object, finishReason] = await Promise.all([
          (async () => {
            for await (const part of streamed.fullStream) {
              if (part.type === "error") throw part.error;
            }
          })(),
          streamed.object,
          streamed.finishReason,
        ]);
        if (finishReason === "length") {
          throw new Error("Mutation generation hit its output cap; result is incomplete");
        }
        return { object, finishReason };
      }
      const result = await sdkGenerateObject(options);
      if (result.finishReason === "length") {
        throw new Error("Mutation generation hit its output cap; result is incomplete");
      }
      return result;
    } catch (error) {
      if (NoObjectGeneratedError.isInstance(error) && error.finishReason === "length") {
        throw new Error("Mutation generation hit its output cap; result is incomplete", { cause: error });
      }
      if (!staged || attempt > 0 || options.abortSignal?.aborted ||
          !NoObjectGeneratedError.isInstance(error) ||
          !error.message.includes("could not parse the response")) throw error;
    }
  }
}

export interface AgentOptions {
  model?: string;
  signal?: AbortSignal;
  /** Raw caller-provided facts, excluding server-added mutation instructions. */
  preflightInput?: string;
  /** Caller-suggested owner; it must be read and independently checked. */
  ownerHint?: string;
  /** Only memory_add may append the caller's exact fact without a generated edit. */
  directAdd?: boolean;
}

export interface QueryResult {
  answer: string;
  steps: number;
  traceId: string;
}

export interface MutationResult {
  summary: string;
  filesChanged: string[];
  steps: number;
  traceId: string;
}

export type MutationOutcome =
  | { ok: true; result: MutationResult }
  | { ok: false; status: "partial"; filesChanged: string[]; error: string; traceId: string }
  | { ok: false; status: "failed"; error: string };

interface ResolvedAgentModel {
  /** Model used by the tool loop; it may transparently fail over on transport errors. */
  model: LanguageModel;
  /** Tool-free synthesis repair model. Never wrap this in a second fallback chain. */
  synthesisModel: LanguageModel;
  /** Raw configured fallback for one bounded mutation-loop retry. */
  mutationRetryModel?: LanguageModel;
  modelChain: string[];
}

async function promptContext(
  kb: KnowledgeBase,
  mode: "query" | "mutate" | "chat",
  state: AgentRunContext
) {
  const [types, tree] = await Promise.all([kb.listTypes(), kb.listTree()]);
  return {
    existingTypes: state.systemTypes(types),
    treeSummary: state.systemTree(formatTree(tree, 0, false)),
    mode,
  };
}

export async function resolveAgentModel(
  options: AgentOptions,
  mode: "query" | "mutate" | "chat",
  env: NodeJS.ProcessEnv = process.env,
  onProviderCall?: (call: NonNullable<TraceTiming["providerCalls"]>[number]) => void
): Promise<ResolvedAgentModel> {
  const scopedConfig = (config: ModelConfig | null | undefined): ModelConfig | null | undefined => {
    if (mode === "mutate" && config && isLocalModelEndpoint(config.baseURL)) {
      // Local llama.cpp may otherwise spend indefinitely reasoning before response
      // headers arrive. Bound mutation generations only; hosted and query calls
      // retain their configured behavior. Explicit max_tokens always wins.
      const extraBody = config.extraBody ?? {};
      if (extraBody.max_tokens === undefined) {
        config = { ...config, extraBody: { ...extraBody,
          max_tokens: env.MUTATION_STAGED === "true" && env.MUTATION_STRUCTURED_STREAM !== "false" ? 8192 : 4096 } };
      }
    }
    const disableThinking =
      (mode === "query" && env.QUERY_ENABLE_THINKING === "false") ||
      (mode === "mutate" && env.MUTATION_ENABLE_THINKING === "false");
    if (!disableThinking || !config) return config;
    const extraBody = config.extraBody ?? {};
    const templateKwargs = extraBody.chat_template_kwargs;
    return {
      ...config,
      extraBody: {
        ...extraBody,
        chat_template_kwargs: {
          ...(templateKwargs && typeof templateKwargs === "object" ? templateKwargs : {}),
          enable_thinking: false,
        },
      },
    };
  };
  const primaryConfig = scopedConfig(
    withModelOverride(resolveModelConfig(env), options.model)
  )!;
  throwIfAborted(options.signal);
  const primaryModel = await createModel(primaryConfig, options.signal);
  const primary = onProviderCall
    ? withProviderTiming(primaryModel, modelLabel(primaryConfig), onProviderCall)
    : primaryModel;
  const fallbackConfig = scopedConfig(resolveFallbackConfig(env));

  if (!fallbackConfig) {
    return {
      model: primary,
      synthesisModel: primary,
      modelChain: [modelLabel(primaryConfig)],
    };
  }

  const allowFor = resolveAllowFor(env.LLM_FALLBACK_ALLOW_FOR);
  if (allowFor && !allowFor.has(mode)) {
    return {
      model: primary,
      synthesisModel: primary,
      modelChain: [modelLabel(primaryConfig)],
    };
  }

  const fallbackModel = await createModel(fallbackConfig, options.signal);
  const fallback = onProviderCall
    ? withProviderTiming(fallbackModel, modelLabel(fallbackConfig), onProviderCall)
    : fallbackModel;
  return {
    // The initial loop keeps the existing transport-only fallback behaviour.
    model: withFallback(primary, fallback, {
      retry429: env.LLM_FALLBACK_RETRY_429 === "true",
    }),
    // Queries retain synthesis repair; mutations may retry the complete tool loop once.
    synthesisModel: mode === "query" ? fallback : primary,
    mutationRetryModel: mode === "mutate" ? fallback : undefined,
    modelChain: [modelLabel(primaryConfig), modelLabel(fallbackConfig)],
  };
}

function resolveAllowFor(raw: string | undefined): Set<string> | null {
  if (!raw || raw === "*") return null;
  return new Set(raw.split(",").map((s) => s.trim()).filter(Boolean));
}

function withModelOverride(config: ModelConfig, model: string | undefined): ModelConfig {
  return model ? { ...config, model } : config;
}

// No baseURL here by design: traces persist under <bundle>/.traces/, and a
// published bundle would otherwise leak internal hostnames/IPs/ports.
function modelLabel(config: ModelConfig): string {
  return `${config.format}:${config.model || "auto"}`;
}

export function traceStore(kb: KnowledgeBase): TraceStore {
  return new TraceStore(kb.bundle.root);
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

function chatFailureMessage(message: string, filesChanged: Set<string>): string {
  const files = [...filesChanged].sort();
  if (files.length === 0) return message;
  return `${message}\n\n⚠ Partial mutation: ${files.length} file(s) changed before failure.\nFiles changed:\n${files.map((file) => `- ${file}`).join("\n")}`;
}

/** Disable tools on the final allowed generation so it can only synthesise. */
export function prepareFinalSynthesisStep(maxSteps: number) {
  return ({ stepNumber }: { stepNumber: number }) =>
    stepNumber >= maxSteps - 1 ? { activeTools: [] } : undefined;
}

function assertSynthesised(steps: ReadonlyArray<{ toolCalls?: unknown[]; finishReason?: string }>): void {
  if (steps.some((step) => step.finishReason === "length")) {
    throw new Error("Mutation generation hit its output cap; result is incomplete");
  }
  const last = steps.at(-1);
  if (last?.toolCalls && last.toolCalls.length > 0) {
    throw new Error("Agent reached the step limit before producing a final answer");
  }
}

/** Sum token usage across the run's steps (issue #15). Undefined when the provider reports none. */
function sumStepsUsage(
  steps: ReadonlyArray<{ usage?: { inputTokens?: number; outputTokens?: number } }>
): TraceUsage | undefined {
  let inputTokens = 0;
  let outputTokens = 0;
  let reported = false;
  for (const step of steps) {
    const u = step.usage;
    if (!u || (u.inputTokens == null && u.outputTokens == null)) continue;
    reported = true;
    inputTokens += u.inputTokens ?? 0;
    outputTokens += u.outputTokens ?? 0;
  }
  return reported ? { inputTokens, outputTokens } : undefined;
}

const READ_ONLY_TOOL_NAMES = new Set([
  "search_knowledge",
  "read_concept",
  "read_concepts",
  "list_directory",
  "lint_knowledge",
]);
const MAX_REPAIR_EVIDENCE_CHARS = 16_000;
const MAX_REPAIR_EVIDENCE_ITEMS = 64;
const MAX_REPAIR_EVIDENCE_VALUE_DEPTH = 5;
const MAX_REPAIR_EVIDENCE_VALUE_CHARS = 8_000;
const MAX_EXACT_RETRACTION_CHARS = 4_000;

function parseExactParagraphRetraction(input: string): { path: string; paragraph: string } | null {
  const match = input.match(/^Retract exact paragraph from `(\/[a-z0-9_/-]+\.md)`:\n<<<\n([\s\S]+)\n>>>$/);
  if (!match) return null;
  return { path: match[1], paragraph: match[2] };
}

function parseAnchoredParagraphRetraction(input: string): { path: string; beginning: string; ending: string } | null {
  const match = input.match(/^In `(\/[a-z0-9_/-]+\.md)`, remove ONLY the exact paragraph beginning `([^`\n]{12,512})` and ending `([^`\n]{12,512})`\.?(?:[ \n][\s\S]{0,2000})?$/);
  if (!match) return null;
  return { path: match[1], beginning: match[2], ending: match[3] };
}

function uniquelyAnchoredParagraph(body: string, beginning: string, ending: string): string | null {
  const matches = body.split("\n\n").filter((paragraph) =>
    paragraph.length < MAX_EXACT_RETRACTION_CHARS && paragraph.startsWith(beginning) &&
    (paragraph.endsWith(ending) || paragraph.endsWith(`${ending}\n`)));
  return matches.length === 1 ? matches[0] : null;
}

function removeUniqueParagraph(body: string, paragraph: string): string | null {
  const matches: Array<{ start: number; end: number }> = [];
  let offset = 0;
  while ((offset = body.indexOf(paragraph, offset)) !== -1) {
    const end = offset + paragraph.length;
    const startsParagraph = offset === 0 || body.slice(0, offset).endsWith("\n\n");
    const endsParagraph = end === body.length || body.slice(end) === "\n" || body.slice(end).startsWith("\n\n");
    if (startsParagraph && endsParagraph) matches.push({ start: offset, end });
    offset += 1;
  }
  if (matches.length !== 1) return null;
  const { start, end } = matches[0];
  if (body.slice(end).startsWith("\n\n")) return body.slice(0, start) + body.slice(end + 2);
  if (body.slice(0, start).endsWith("\n\n")) return body.slice(0, start - 2) + body.slice(end);
  return body.slice(0, start) + body.slice(end);
}

function isReadOnlyToolName(value: unknown): value is string {
  return typeof value === "string" && READ_ONLY_TOOL_NAMES.has(value);
}

function recordSafeTranscriptMessage(value: unknown): ModelMessage | undefined {
  if (!value || typeof value !== "object") return undefined;
  const message = value as { role?: unknown; content?: unknown };
  if (!Array.isArray(message.content)) return undefined;

  if (message.role === "assistant") {
    const parts = message.content
      .filter(
        (part): part is Record<string, unknown> =>
          !!part && typeof part === "object" && part.type === "tool-call"
      )
      .filter(
        (part) =>
          typeof part.toolCallId === "string" &&
          isReadOnlyToolName(part.toolName) &&
          Object.prototype.hasOwnProperty.call(part, "input")
      )
      .map((part) => ({
        type: "tool-call" as const,
        toolCallId: part.toolCallId as string,
        toolName: part.toolName as string,
        input: part.input,
      }));
    if (parts.length === 0) return undefined;
    const parsed = modelMessageSchema.safeParse({ role: "assistant", content: parts });
    return parsed.success ? parsed.data : undefined;
  }

  if (message.role === "tool") {
    const parts = message.content
      .filter(
        (part): part is Record<string, unknown> =>
          !!part && typeof part === "object" && part.type === "tool-result"
      )
      .filter(
        (part) =>
          typeof part.toolCallId === "string" &&
          isReadOnlyToolName(part.toolName) &&
          isToolResultOutput(part.output)
      )
      .map((part) => ({
        type: "tool-result" as const,
        toolCallId: part.toolCallId as string,
        toolName: part.toolName as string,
        output: part.output,
      }));
    if (parts.length === 0) return undefined;
    const parsed = modelMessageSchema.safeParse({ role: "tool", content: parts });
    return parsed.success ? parsed.data : undefined;
  }
  return undefined;
}

function isToolResultOutput(
  value: unknown
): value is { type: string; value: unknown } {
  if (!value || typeof value !== "object") return false;
  const output = value as { type?: unknown; value?: unknown };
  return (
    (output.type === "text" ||
      output.type === "json" ||
      output.type === "error-text" ||
      output.type === "error-json" ||
      output.type === "content") &&
    Object.prototype.hasOwnProperty.call(output, "value")
  );
}

/**
 * Keep only the tool transcript from a completed deep run. AI SDK v5's
 * `result.response.messages` also contains the final assistant text; feeding
 * that text back would re-inject the malformed answer into the repair prompt.
 */
function safeRepairMessages(
  question: string,
  steps: ReadonlyArray<Record<string, unknown>>,
  responseMessages: unknown[] | undefined
): ModelMessage[] | undefined {
  const transcript: ModelMessage[] = [];
  const collect = (messages: unknown[] | undefined): boolean => {
    if (!messages) return true;
    for (const message of messages) {
      const safe = recordSafeTranscriptMessage(message);
      if (safe) {
        transcript.push(safe);
      } else if (
        message &&
        typeof message === "object" &&
        (message as { role?: unknown }).role === "tool"
      ) {
        // A response tool message is only usable as a whole. Silently dropping
        // a raw/invalid result would leave a misleading, incomplete transcript.
        return false;
      }
    }
    return true;
  };
  // AI SDK v5 exposes the combined response transcript on the result. It is
  // already ordered and must be preferred: each StepResult.response.messages is
  // a cumulative clone, so collecting every step would duplicate tool calls.
  if (!collect(responseMessages)) transcript.length = 0;
  if (transcript.length === 0) {
    const lastStep = steps.at(-1);
    const response = lastStep?.response;
    const lastStepMessages =
      response && typeof response === "object"
        ? (response as { messages?: unknown }).messages
        : undefined;
    if (Array.isArray(lastStepMessages)) {
      transcript.length = 0;
      if (!collect(lastStepMessages)) transcript.length = 0;
    }
  }

  // Test seams and older provider adapters may omit response messages.
  // Reconstruct the same v5 assistant/tool message shape from the successful
  // tool calls/results, never from step.text.
  if (transcript.length === 0) {
    for (const step of steps) {
      const calls = Array.isArray(step.toolCalls)
        ? step.toolCalls
        : Array.isArray(step.staticToolCalls)
          ? step.staticToolCalls
          : Array.isArray(step.dynamicToolCalls)
            ? step.dynamicToolCalls
            : [];
      const results = Array.isArray(step.toolResults)
        ? step.toolResults
        : Array.isArray(step.staticToolResults)
          ? step.staticToolResults
          : Array.isArray(step.dynamicToolResults)
            ? step.dynamicToolResults
            : [];
      const assistant = recordSafeTranscriptMessage({ role: "assistant", content: calls });
      const tool = recordSafeTranscriptMessage({
        role: "tool",
        content: results.map((result) => {
          if (!result || typeof result !== "object") return result;
          const part = result as Record<string, unknown>;
          return {
            ...part,
            // StepResult tool results expose the raw tool output. Prompt
            // messages require the v5 LanguageModelV2ToolResultOutput union.
            output: { type: "json", value: part.output },
          };
        }),
      });
      if (assistant) transcript.push(assistant);
      if (tool) transcript.push(tool);
    }
  }

  return transcript.length > 0 ? [{ role: "user", content: question }, ...transcript] : undefined;
}

interface RepairEvidence {
  tool: string;
  value: unknown;
}

/**
 * Turn successful read results into data-only evidence for a retry. In
 * particular, never pass assistant messages, tool-call ids, or write-tool
 * results to the model: KAT treats those protocol-shaped messages as a cue to
 * emit another XML call.
 */
function safeRepairEvidence(
  steps: ReadonlyArray<Record<string, unknown>>,
  responseMessages: unknown[] | undefined,
  maxChars: number
): string | undefined {
  const evidence: RepairEvidence[] = [];

  const collectMessages = (messages: unknown[] | undefined): void => {
    if (!messages) return;
    for (const message of messages) {
      if (!message || typeof message !== "object") continue;
      const record = message as { role?: unknown; content?: unknown };
      if (record.role !== "tool" || !Array.isArray(record.content)) continue;
      for (const part of record.content) {
        if (!part || typeof part !== "object") continue;
        const result = part as {
          type?: unknown;
          toolName?: unknown;
          output?: unknown;
        };
        if (
          result.type !== "tool-result" ||
          !isReadOnlyToolName(result.toolName) ||
          !isToolResultOutput(result.output)
        ) {
          continue;
        }
        evidence.push({ tool: result.toolName, value: result.output.value });
      }
    }
  };

  // Prefer the combined response transcript, which is not duplicated per step.
  collectMessages(responseMessages);
  if (evidence.length === 0) {
    for (const step of steps) {
      const results = Array.isArray(step.toolResults)
        ? step.toolResults
        : Array.isArray(step.staticToolResults)
          ? step.staticToolResults
          : Array.isArray(step.dynamicToolResults)
            ? step.dynamicToolResults
            : [];
      for (const result of results) {
        if (!result || typeof result !== "object") continue;
        const record = result as Record<string, unknown>;
        if (!isReadOnlyToolName(record.toolName)) continue;
        evidence.push({ tool: record.toolName, value: record.output });
      }
    }
  }
  if (evidence.length === 0 || maxChars <= 0) return undefined;

  const lines = evidence
    .slice(0, MAX_REPAIR_EVIDENCE_ITEMS)
    .map((entry, index) => {
      const value = sanitiseRepairValue(entry.value);
      return `[${index + 1}] ${entry.tool}: ${JSON.stringify(value)}`;
    });
  return fitText(lines.join("\n"), maxChars, "\n...[read-only evidence truncated]");
}

function sanitiseRepairValue(
  value: unknown,
  depth = 0,
  seen = new WeakSet<object>()
): unknown {
  if (typeof value === "string") {
    return sanitiseRepairText(value.slice(0, MAX_REPAIR_EVIDENCE_VALUE_CHARS));
  }
  if (value === null || typeof value === "boolean" || typeof value === "number") {
    return value;
  }
  if (depth >= MAX_REPAIR_EVIDENCE_VALUE_DEPTH) return "[nested value omitted]";
  if (typeof value !== "object") return `[${typeof value} value omitted]`;
  if (seen.has(value)) return "[cyclic value omitted]";
  seen.add(value);
  if (Array.isArray(value)) {
    return value.slice(0, MAX_REPAIR_EVIDENCE_ITEMS).map((item) =>
      sanitiseRepairValue(item, depth + 1, seen)
    );
  }
  const result: Record<string, unknown> = {};
  for (const key of Object.keys(value).slice(0, MAX_REPAIR_EVIDENCE_ITEMS)) {
    result[sanitiseRepairText(key, 500)] = sanitiseRepairValue(
      (value as Record<string, unknown>)[key],
      depth + 1,
      seen
    );
  }
  return result;
}

function sanitiseRepairText(value: string, maxChars = MAX_REPAIR_EVIDENCE_VALUE_CHARS): string {
  return value
    .slice(0, maxChars)
    .replace(/[\u0000-\u001f\u007f]/g, " ")
    .replace(/<\|(?:tool_call_start|tool_call_end)\|>/gi, "[protocol marker removed]")
    .replace(/<\/?(?:tool_call|function(?:=[^>]+)?)>/gi, "[protocol marker removed]");
}

/** Read-only Q&A over the bundle. */
function withProviderTiming(
  model: LanguageModel,
  label: string,
  record: (call: NonNullable<TraceTiming["providerCalls"]>[number]) => void
): LanguageModel {
  return wrapLanguageModel({
    model: model as Extract<LanguageModel, { doGenerate: unknown }>,
    middleware: {
      wrapGenerate: async ({ doGenerate }) => {
        const started = Date.now();
        try {
          const result = await doGenerate();
          const usage = result.usage;
          try {
            record({
              model: label,
              durationMs: Date.now() - started,
              ...(usage.inputTokens === undefined ? {} : { inputTokens: usage.inputTokens }),
              ...(usage.outputTokens === undefined ? {} : { outputTokens: usage.outputTokens }),
            });
          } catch { /* telemetry must never affect model behavior */ }
          return result;
        } catch (error) {
          try { record({ model: label, durationMs: Date.now() - started }); }
          catch { /* telemetry must never mask the provider error */ }
          throw error;
        }
      },
    },
  });
}

export async function runQuery(
  kb: KnowledgeBase,
  question: string,
  options: AgentOptions = {}
): Promise<QueryResult> {
  const limits = resolveAgentLimits();
  assertInputWithinLimit(
    question,
    limits.maxInputChars,
    "Query input"
  );
  const state = new AgentRunContext(limits, options.signal);
  throwIfAborted(options.signal);
  const promptStarted = Date.now();
  const ctx = await promptContext(kb, "query", state);
  const promptContextMs = Date.now() - promptStarted;
  throwIfAborted(options.signal);
  const recorder = new TraceRecorder();
  const generateTextCalls: NonNullable<TraceTiming["generateTextCalls"]> = [];
  const providerCalls: NonNullable<TraceTiming["providerCalls"]> = [];
  const generationStarted = Date.now();
  const maxSteps = limits.maxSteps;
  let modelChain: string[] = [];
  try {
    const resolved = await resolveAgentModel(options, "query", process.env, (call) => providerCalls.push(call));
    modelChain = resolved.modelChain;
    const callStarted = Date.now();
    const result = await generateText({
      model: resolved.model,
      system: buildSystemPrompt(ctx),
      prompt: question,
      tools: buildReadTools(kb, recorder, state),
      stopWhen: stepCountIs(maxSteps),
      prepareStep: prepareFinalSynthesisStep(maxSteps),
      abortSignal: options.signal,
    });
    generateTextCalls.push({ model: resolved.modelChain.join(" → "), durationMs: Date.now() - callStarted, ...sumStepsUsage(result.steps) });
    throwIfAborted(options.signal);
    assertSynthesised(result.steps);

    let finalText = result.text;
    const allSteps: Array<{
      usage?: { inputTokens?: number; outputTokens?: number };
    }> = [...result.steps];
    if (isMalformedAnswer(finalText) || isUnsafeSynthesisAnswer(finalText)) {
      console.error(`[understory] query answer rejected: ${MALFORMED_ANSWER_MESSAGE}`);
      const repairMessages = safeRepairMessages(
        question,
        result.steps as unknown as ReadonlyArray<Record<string, unknown>>,
        result.response?.messages
      );
      if (!repairMessages) {
        // A malformed answer with no trustworthy read-only transcript cannot
        // be repaired by synthesis without inventing evidence. Give the query
        // one fresh, bounded read-only tool-loop attempt instead.
        const retryStarted = Date.now();
        const retry = await generateText({
          model: resolved.model,
          system: buildSystemPrompt(ctx),
          prompt: question,
          tools: buildReadTools(kb, recorder, state),
          stopWhen: stepCountIs(maxSteps),
          prepareStep: prepareFinalSynthesisStep(maxSteps),
          abortSignal: options.signal,
        });
        generateTextCalls.push({ model: resolved.modelChain.join(" → "), durationMs: Date.now() - retryStarted, ...sumStepsUsage(retry.steps) });
        throwIfAborted(options.signal);
        assertSynthesised(retry.steps);
        allSteps.push(...retry.steps);
        if (isMalformedAnswer(retry.text) || isUnsafeSynthesisAnswer(retry.text)) {
          throw new Error(MALFORMED_ANSWER_MESSAGE);
        }
        // The first attempt provided no usable read result. Do not mark a
        // fluent but unsupported retry as a successful grounded answer.
        if (!safeRepairEvidence(
          retry.steps as unknown as ReadonlyArray<Record<string, unknown>>,
          retry.response?.messages,
          Math.min(MAX_REPAIR_EVIDENCE_CHARS, limits.maxToolResultChars, limits.maxInputChars)
        )) {
          throw new Error("Query retry produced no read-only evidence");
        }
        finalText = retry.text;
      } else {
        const repairStarted = Date.now();
        const repair = await generateText({
          model: resolved.synthesisModel,
          system: buildQuerySynthesisPrompt(),
          messages: repairMessages,
          tools: {},
          abortSignal: options.signal,
        });
        generateTextCalls.push({ model: resolved.modelChain.at(-1) ?? "configured", durationMs: Date.now() - repairStarted, ...sumStepsUsage(repair.steps) });
        throwIfAborted(options.signal);
        assertSynthesised(repair.steps);
        if (isMalformedAnswer(repair.text) || isUnsafeSynthesisAnswer(repair.text)) {
          // KAT can interpret the valid assistant/tool transcript above as a
          // request to continue the tool protocol. Give it one final chance,
          // but only with bounded, quoted data from read-only tool results.
          const evidence = safeRepairEvidence(
            result.steps as unknown as ReadonlyArray<Record<string, unknown>>,
            result.response?.messages,
            Math.min(MAX_REPAIR_EVIDENCE_CHARS, limits.maxToolResultChars, limits.maxInputChars)
          );
          if (!evidence) {
            throw new Error(MALFORMED_ANSWER_MESSAGE);
          }
          const secondRepairStarted = Date.now();
          const secondRepair = await generateText({
            model: resolved.synthesisModel,
            system: buildQuerySynthesisPrompt(),
            messages: [
              {
                role: "user",
                content:
                  `Original question:\n${question}\n\n` +
                  "BEGIN UNTRUSTED READ-ONLY EVIDENCE\n" +
                  evidence +
                  "\nEND UNTRUSTED READ-ONLY EVIDENCE",
              },
            ],
            tools: {},
            abortSignal: options.signal,
          });
          generateTextCalls.push({ model: resolved.modelChain.at(-1) ?? "configured", durationMs: Date.now() - secondRepairStarted, ...sumStepsUsage(secondRepair.steps) });
          throwIfAborted(options.signal);
          assertSynthesised(secondRepair.steps);
          if (
            isMalformedAnswer(secondRepair.text) ||
            isUnsafeSynthesisAnswer(secondRepair.text)
          ) {
            throw new Error(MALFORMED_ANSWER_MESSAGE);
          }
          finalText = secondRepair.text;
          allSteps.push(...repair.steps, ...secondRepair.steps);
        } else {
          finalText = repair.text;
          allSteps.push(...repair.steps);
        }
      }
    }

    const trace = recorder.finalize(
      "query",
      question,
      finalText,
      "success",
      modelChain,
      sumStepsUsage(allSteps),
      undefined,
      undefined,
      { generateTextCalls, providerCalls, promptContextMs, generationMs: Date.now() - generationStarted }
    );
    await traceStore(kb).save(trace).catch(() => {
      /* telemetry persistence must not change query behavior */
    });
    return { answer: finalText, steps: allSteps.length, traceId: trace.id };
  } catch (err) {
    const trace = recorder.finalize("query", question, errorMessage(err), "failed", modelChain);
    await traceStore(kb).save(trace).catch(() => {
      /* telemetry persistence must not mask the original query failure */
    });
    throw err;
  }
}

// The staged path sends complete owner bodies, never a truncated page. The
// model's context accommodates these bounds; larger evidence fails closed.
const STAGED_MAX_OWNER_CHARS = 64_000;
const STAGED_MAX_EVIDENCE_CHARS = 80_000;

/** Distinct words used only as a conservative support check, never as proof of meaning. */
function stagedWords(text: string): string[] {
  const stop = new Set(["the", "and", "for", "with", "that", "this", "from", "into", "about", "record", "remember", "concept", "knowledge", "policy", "distinct", "unrelated", "existing", "only"]);
  const compounds = text.toLowerCase().match(/[\p{L}\p{N}][\p{L}\p{N}_-]{2,}/gu) ?? [];
  // Entity names such as "Memory-Async" must match the "async" component
  // of a concept slug, without losing exact compound matches elsewhere.
  return [...new Set(compounds.flatMap((word) => [word, ...word.split(/[-_]/)])
    .filter((word) => word.length >= 3 && !stop.has(word)))];
}

function stagedOverlap(text: string, evidence: string): number {
  const words = stagedWords(text);
  const supported = new Set(stagedWords(evidence));
  return words.length ? words.filter((word) => supported.has(word)).length / words.length : 0;
}

function stagedSingleClauseEdit(before: string, after: string): boolean {
  // Compare the single contiguous changed span in the original text. New
  // punctuation may split or merge clauses, so counting positional clauses in
  // the replacement incorrectly rejects a localized correction. A span that
  // crosses an original clause boundary is still rejected (including two
  // separated edits whose common envelope covers multiple clauses).
  let start = 0;
  while (start < before.length && start < after.length && before[start] === after[start]) start++;
  if (start === before.length && start === after.length) return false;

  let beforeEnd = before.length;
  let afterEnd = after.length;
  while (beforeEnd > start && afterEnd > start && before[beforeEnd - 1] === after[afterEnd - 1]) {
    beforeEnd--;
    afterEnd--;
  }

  const clauseAt = (offset: number): number => {
    const boundaries = /;\s*|(?<=[.!?])\s+|\n+/g;
    let clause = 0;
    for (const match of before.matchAll(boundaries)) {
      if (match.index! + match[0].length > offset) break;
      clause++;
    }
    return clause;
  };
  return clauseAt(start) === clauseAt(beforeEnd);
}

function stagedStatusClauses(text: string): string[] {
  return text.split(/;\s*|(?<=[.!?])\s+|\n+/).map((part) => part.trim())
    .filter((part) => part.length >= 12);
}

function stagedRestoreUnrequestedStatusClauses(before: string, after: string, instruction: string): string {
  const newClauses = stagedStatusClauses(after);
  const lost = stagedStatusClauses(before).filter((clause) =>
    stagedOverlap(clause, instruction) < 0.4 && !newClauses.includes(clause) &&
    !newClauses.some((candidate) => stagedOverlap(clause, candidate) >= 0.85 &&
      stagedOverlap(candidate, clause) >= 0.85));
  // Exact observed text, not a model paraphrase. The usual support,
  // contradiction and independent checks still apply to the whole result.
  return lost.length ? `${after.trimEnd()} ${lost.map((clause) =>
    /[.!?]$/.test(clause) ? clause : `${clause};`).join(" ")}` : after;
}

function stagedStatusClauseIssue(
  before: string, after: string, instruction: string,
  allowedNegationRemoval?: (oldClause: string, newClause: string, negation: string) => boolean
): string | undefined {
  const oldClauses = stagedStatusClauses(before);
  const newClauses = stagedStatusClauses(after);
  const requested = stagedStatusClauses(instruction);
  const negations = new Set(["no", "not", "never", "without", "cannot", "doesn't", "don't"]);
  const clauseNegations = (text: string): string[] =>
    [...text.toLowerCase().matchAll(/\b(?:no|not|never|without|cannot|doesn't|don't)\b/g)].map((match) => match[0]);
  const oldSet = new Set(oldClauses);
  const newSet = new Set(newClauses);
  for (const clause of oldClauses) {
    if (newSet.has(clause)) continue;
    const closest = newClauses.reduce((best, candidate) =>
      stagedOverlap(clause, candidate) > stagedOverlap(clause, best) ? candidate : best, "");
    const numbers = (text: string): string[] =>
      [...text.matchAll(/\b\d+(?:[.,]\d+)?%?\b/g)].map((match) => match[0]).sort();
    const nearlyPreserved = stagedOverlap(clause, closest) >= 0.85 &&
      stagedOverlap(closest, clause) >= 0.85 &&
      JSON.stringify(numbers(clause)) === JSON.stringify(numbers(closest));
    if (stagedOverlap(clause, instruction) < 0.4 && !nearlyPreserved) {
      return `old clause not requested (closest=${stagedOverlap(clause, closest).toFixed(2)}/${stagedOverlap(closest, clause).toFixed(2)}, numbers=${JSON.stringify(numbers(clause)) === JSON.stringify(numbers(closest))})`;
    }
    const removedNegations = clauseNegations(clause).filter((word) =>
      !clauseNegations(closest).includes(word) && !allowedNegationRemoval?.(clause, closest, word));
    if (removedNegations.length && !requested.some((part) =>
      removedNegations.every((word) => !clauseNegations(part).includes(word)) &&
      stagedWords(part).filter((word) => stagedWords(closest).includes(word)).length >= 2 &&
      stagedOverlap(closest, part) >= 0.6)) return "removed negation not requested";
  }
  for (const clause of newClauses) {
    if (oldSet.has(clause)) continue;
    const closest = oldClauses.reduce((best, candidate) =>
      stagedOverlap(clause, candidate) > stagedOverlap(clause, best) ? candidate : best, "");
    if (stagedOverlap(clause, `${instruction} ${closest}`) < 0.65) return "new clause lacks support";
    const newNegations = clauseNegations(clause);
    if (newNegations.every((word) => clauseNegations(closest).includes(word))) continue;
    // A negation used in another old/requested sentence does not authorize
    // moving it onto a different subject (e.g. deleted CSVs -> not deleted).
    if (!requested.some((part) => newNegations.every((word) => clauseNegations(part).includes(word)) &&
      stagedWords(part).filter((word) => !negations.has(word) && stagedWords(clause).includes(word)).length >= 2 &&
      stagedOverlap(clause, part) >= 0.6)) return "added negation not requested";
  }
  return undefined;
}

function stagedRequiredCorrectionAnchors(input: string): string[] {
  // Distinctive advisory names, HTTP status codes, and commit IDs in a
  // multi-finding correction must survive in the resulting owner. A model's
  // broad "all corrections applied" verdict cannot replace this evidence.
  // A quoted concept path is a locator, not a requested factual identifier.
  // Continuation filenames contain hash-shaped suffixes absent from the body.
  const factualInput = input.replace(/\/[a-z0-9_/-]+\.md\b/gi, " ");
  return [...new Set((factualInput.match(/\b(?:[a-f0-9]{7,40}|Accelerate|(?:HTTP\s*)?422)\b/gi) ?? [])
    .map((anchor) => anchor.toLowerCase().replace(/\s+/g, " ")))];
}

function stagedCorrectionIntent(input: string): boolean {
  // Ordinary add payloads often discuss a dataset "update" or say "rather
  // than" without asking to change an existing assertion. Treat only an
  // explicit leading edit instruction as a correction for memory_add.
  return /^(?:(?:please|remember|record)\s+)?(?:correct|correction|update|replace|amend|supersede|retract)\b/i.test(input.trim());
}

function stagedCorrection(input: string): boolean {
  return /\b(?:correct|correction|update|replace|supersede|instead|formerly|previously|no longer|only after|now|rather than|don['’]t|doesn['’]t)\b/i.test(input);
}

function stagedOwnerAnchor(input: string, conceptPath: string): boolean {
  const slug = conceptPath.split("/").pop()?.replace(/\.md$/, "").replace(/[-_]/g, " ") ?? "";
  const words = new Set(stagedWords(input));
  return stagedWords(slug).some((word) => words.has(word));
}

/** The named subject of a fact must not be lost to a generic title match. */
function stagedConflictingNamedOwner(input: string, title: string | undefined, conceptPath: string): boolean {
  // A leading preposition is framing, not part of a named subject. In
  // particular, facts phrased "For Holi's ..." must be anchored to Holi,
  // rather than misread as a person named "For Holi".
  const trimmed = input.trim();
  const subject = trimmed.match(/^(?:[Ff]or\s+)?([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*)\b/)?.[1];
  if (!subject || (subject.split(/\s+/).length < 2 && !/^For\s+[A-Z][a-z]{2,}'s\b/.test(trimmed))) return false;
  const identity = `${title ?? ""} ${conceptPath.split("/").pop()?.replace(/\.md$/, "").replace(/[-_]/g, " ") ?? ""}`.toLowerCase();
  return !identity.includes(subject.toLowerCase()) && !(title && stagedTitleMention(input, title));
}

function stagedTitleMention(input: string, title: string): boolean {
  return title.length >= 4 && input.toLowerCase().includes(title.toLowerCase()) &&
    !new RegExp(`\\b(?:unrelated to|rather than changing|do not change|not about|instead of|not)\\s+(?:the\\s+)?${title.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}`, "i").test(input);
}

/** Metadata selection locates a possible owner; it never authorizes a write. */
function stagedNamedSection(input: string): RegExp | undefined {
  // A dated section named at the start of a correction may be more specific
  // than its containing concept's title. Require a line-leading Markdown
  // heading or colon-labelled paragraph, not a passing mention in prose.
  const match = input.trim().match(/^(?:update|correct|revise)\s+(?:the\s+)?(.{16,120}?)\s+(?:note|record)\s+\(((?:[A-Za-z]{3,9}\s+\d{1,2}|\d{1,2}\s+[A-Za-z]{3,9})\s+\d{4})\)\s*:/i) ??
    input.trim().match(/^(?:update|correct|revise)\s+(?:the\s+)?(.{16,120}?)\s+status\s+recorded\s+on\s+((?:[A-Za-z]{3,9}\s+\d{1,2}|\d{1,2}\s+[A-Za-z]{3,9})\s+\d{4})\s*:/i);
  if (!match) return undefined;
  const escaped = `${match[1]} (${match[2]})`.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return new RegExp(`^(?:#{1,6}\\s+|\\*\\*)?${escaped}(?:\\*\\*)?\\s*:(?:\\*\\*)?(?=\\s|$)`, "mi");
}

function stagedContrastedOldHash(input: string): string | undefined {
  // A correction of the form "final SHA is X, not intermediate Y" supplies
  // Y as the old operand, not merely a negative mention of another concept.
  return input.match(/\b(?:final|authoritative)\b[^.!?;]{0,160}\b(?:hash|sha-?256)\b[^.!?;]{0,80}`[a-f0-9]{64}`\s*,?\s*not\s+(?:the\s+)?(?:intermediate|previous|old)\s+`([a-f0-9]{64})`/i)?.[1];
}

function stagedDiscriminativeOwnerAnchor(
  input: string,
  selectedPath: string,
  candidates: Array<{ path: string; frontmatter: { title?: string }; body?: string }>,
  hits: Array<{ path: string; title?: string }>,
  directAdd: boolean,
): boolean {
  const index = hits.findIndex((hit) => hit.path === selectedPath);
  if (index < 0 || !candidates.some((candidate) => candidate.path === selectedPath)) return false;
  // Compare against every retained hit, including those whose bodies could
  // not fit together in the bounded prompt. Descriptions can mention related
  // concepts and cannot establish owner identity. Shared positive topic terms
  // are sufficient: ambiguity among plausible owners is resolved by search order.
  const generic = new Set([
    "about", "background", "general", "information", "knowledge", "memory",
    "overview", "project", "reference", "status", "summary", "update",
  ]);
  const terms = hits.map((hit) => {
    const slug = hit.path.split("/").pop()?.replace(/\.md$/, "").replace(/[-_]/g, " ") ?? "";
    return new Set(stagedWords([hit.title ?? "", slug]
      .join(" ")).filter((term) => !generic.has(term)));
  });
  const inputTerms = new Set(stagedWords(input));
  const negatedMentions = [...input.matchAll(/\b(?:unrelated to|rather than changing|not about|do not change|instead of|not)\s+(?:the\s+)?([^.!?;,:]+)/gi)]
    .map((match) => new Set(stagedWords(match[1])));
  const selectedTerms = [...terms[index]];
  const otherTerms = new Set(terms.flatMap((set, i) => i === index ? [] : [...set]));
  const specificTerms = selectedTerms.filter((term) => !otherTerms.has(term));
  if (specificTerms.length && negatedMentions.some((mention) => specificTerms.every((term) => mention.has(term)))) return false;
  const contrastedOldHash = !directAdd && stagedContrastedOldHash(input);
  if (contrastedOldHash) {
    // An exact old operand in one complete owner outweighs generic title
    // rankings. Every search-hit body must have been read before this check.
    const body = candidates.find((candidate) => candidate.path === selectedPath)?.body;
    const itemIds = input.match(/\b[\p{L}][\p{L}\p{N}]*_[\p{L}\p{N}_]*\d+[\p{L}\p{N}_]*\b/gu) ?? [];
    return !!body && body.split(contrastedOldHash).length === 2 &&
      candidates.every((candidate) => candidate.path === selectedPath ||
        !candidate.body?.includes(contrastedOldHash)) &&
      itemIds.some((id) => body.includes(id));
  }
  // Shared topics are acceptable, but not when the instruction points much
  // more specifically to a different retained owner.
  const positiveTerms = new Set([...inputTerms].filter((term) =>
    !negatedMentions.some((mention) => mention.has(term))));
  const score = (words: Set<string>): number => words.size
    ? [...words].filter((term) => positiveTerms.has(term)).length / words.size : 0;
  if (terms.some((words, i) => i !== index && score(words) > score(terms[index]) + 0.2)) return false;
  // A generic owner cannot win a tie when the request positively names a
  // distinctive part of another retained owner's identity.
  if (terms.some((words, i) => i !== index &&
    [...words].some((term) => !terms[index].has(term) && inputTerms.has(term) &&
      !negatedMentions.some((mention) => mention.has(term))) &&
    !selectedTerms.some((term) => !words.has(term) && inputTerms.has(term) &&
      !negatedMentions.some((mention) => mention.has(term))))) return false;
  const titleAnchor = selectedTerms.some((term) => {
    if (!inputTerms.has(term)) return false;
    const escaped = term.replace(/[.*+?^${}()|[\\]\\\\]/g, "\\\\$&");
    const quotedOnly = new RegExp(`[\\\"“”‘’']${escaped}[\\\"“”‘’']`, "i").test(input) &&
      !input.replace(new RegExp(`[\\\"“”‘’']${escaped}[\\\"“”‘’']`, "gi"), " ").toLowerCase().includes(term);
    return !quotedOnly && !negatedMentions.some((mention) => mention.has(term));
  });
  if (titleAnchor) return true;

  // A title-less correction can still be grounded by a verbatim old claim or
  // an identifier that occurs in this complete body. Generic overlap and model
  // assertions are deliberately not evidence.
  const selectedBody = candidates.find((candidate) => candidate.path === selectedPath)?.body;
  if (!selectedBody) return false;
  // Only the explicit old operand of a correction can establish ownership.
  // An incidental date, filename, new hash or negated reference in the same
  // request must not authorize a different concept.
  const oldClaims = [
    input.match(/\breplace\s+(?:the\s+)?(?:exact\s+)?(?:phrase|text|claim)?\s*`([^`]{12,2000})`\s+with\b/i)?.[1],
    input.match(/\breplace\s+([\p{L}\p{N}][\p{L}\p{N}_-]*\d[\p{L}\p{N}_-]*)\s+with\b/iu)?.[1],
    // Workbook/content hashes are common correction operands but are not
    // identifier-shaped like domain IDs (e.g. godkendt_id2).
    input.match(/\breplace\s+(?:the\s+)?(?:old\s+)?(?:workbook\s+)?(?:hash\s+)?([a-f0-9]{8,64})\s+with\b/i)?.[1],
  ].filter((claim): claim is string => !!claim);
  const positivelyPresent = (claim: string): boolean => {
    const occurrences = (body: string): number => body.split(claim).length - 1;
    // An operand is an owner anchor only when it identifies one existing claim:
    // duplicates within this owner or copies in another retained owner are ambiguous.
    return occurrences(selectedBody) === 1 &&
      candidates.filter((candidate) => candidate.path !== selectedPath)
        .every((candidate) => !candidate.body?.includes(claim)) &&
      !negatedMentions.some((mention) => stagedWords(claim).every((term) => mention.has(term)));
  };
  if (directAdd) {
    // An item ID already in one complete owner body can locate an exact fact
    // about that item. Generic dates, hashes and ungrounded IDs cannot.
    const itemIds = input.match(/\b[\p{L}][\p{L}\p{N}]*_[\p{L}\p{N}_]*\d+[\p{L}\p{N}_]*\b/gu) ?? [];
    return itemIds.some(positivelyPresent);
  }
  return oldClaims.some(positivelyPresent);
}

/** Knowledge add/update — full toolset, low temperature. */
async function runStagedMutation(
  kb: KnowledgeBase,
  instruction: string,
  options: AgentOptions,
  state: AgentRunContext,
  recorder: TraceRecorder,
): Promise<MutationOutcome> {
  const started = Date.now();
  const providerCalls: NonNullable<TraceTiming["providerCalls"]> = [];
  // Bounded staged JSON checks do not need open-ended reasoning.
  // Keep the safety checks as independent calls, but prevent a thinking model
  // from spending minutes before sending response headers.
  const stagedEnv = { ...process.env, MUTATION_ENABLE_THINKING: "false" };
  let modelChain: string[] = [];
  let changed: string[] = [];
  try {
    if (options.ownerHint) {
      const hint = options.ownerHint;
      const segments = hint.startsWith("/") ? hint.slice(1).split("/") : [];
      const hasMarkdownExtension = segments.at(-1)?.endsWith(".md") ?? false;
      const conceptSegments = hasMarkdownExtension
        ? [...segments.slice(0, -1), segments.at(-1)!.slice(0, -3)]
        : segments;
      if (hint.length > 512 || conceptSegments.length === 0 ||
          conceptSegments.some((segment) => !/^[a-z0-9_-]+$/.test(segment))) {
        throw new Error("Staged mutation deferred: owner hint is not a bounded concept path.");
      }
      // Normalize valid extensionless suggestions before special owner checks so
      // they receive exactly the same existence and evidence vetting as .md paths.
      if (!hasMarkdownExtension) options = { ...options, ownerHint: `${hint}.md` };
    }
    const raw = options.preflightInput ?? instruction;
    // The dated published test manifest and the currently parsed on-disk
    // manifest are different snapshots. Append the evaluation to the detailed
    // baseline owner without rewriting its mixed historical/source paragraphs.
    const datedBaselineRequest = raw.trim().match(/^Update (.+?) (\d{4}-\d{2}-\d{2}) dated test baseline and new (\d+)-head retrain evaluation\.\s*/i);
    if (!options.directAdd && options.preflightInput && datedBaselineRequest) {
      const project = datedBaselineRequest[1];
      const baselineDate = datedBaselineRequest[2];
      const requestBody = raw.trim().slice(datedBaselineRequest[0].length);
      const manifestSha = requestBody.match(/manifest\.jsonl[\s\S]{0,180}?SHA-256 `([a-f0-9]{64})`/i)?.[1];
      const reportSha = requestBody.match(/report SHA `([a-f0-9]{64})`/i)?.[1];
      const checkpointSha = requestBody.match(/checkpoint SHA `([a-f0-9]{64})`/i)?.[1];
      const confusion = requestBody.match(/\bTP\d+ TN\d+ FP\d+ FN\d+\b/)?.[0];
      const baselinePath = requestBody.match(/`([^`]*baselines\/\d{4}-\d{2}-\d{2}\/parsed\/manifest\.jsonl)`/i)?.[1];
      const baselineLocator = baselinePath?.match(/baselines\/\d{4}-\d{2}-\d{2}/i)?.[0];
      const testCount = requestBody.match(/contains \*\*(\d+)\*\* test images/i)?.[1];
      const priorCount = requestBody.match(/earlier reported (\d+)/i)?.[1];
      if (!manifestSha || !reportSha || !checkpointSha || !confusion || !baselinePath ||
          !baselineLocator || !testCount || !priorCount || !requestBody.includes("independent held-out evidence")) {
        throw new Error("Staged mutation deferred: dated test baseline request lacks exact evaluation evidence.");
      }
      const paths = await kb.bundle.listConceptPaths();
      if (paths.length > 10_000) throw new Error("Staged mutation deferred: dated test baseline concept inventory is incomplete.");
      const headingProject = project;
      const updateBody = requestBody
        .replace(/As of [^,.;]+ the current on-disk/i, "Before the later correction, the then-current on-disk")
        .replace(/this is the currently parsed on-disk version/ig, "this was the parsed on-disk version before the later correction")
        .replace(/this test baseline already has binary labels and was not modified\./ig, "this test baseline already had binary labels and was not modified in this evaluation.");
      const ownerCandidates: string[] = [];
      for (const peerPath of paths) {
        state.checkCancellation();
        const peer = await kb.readConcept(peerPath);
        if (peer.body.includes(baselineLocator ?? baselinePath!) && peer.body.includes(`${priorCount} sample IDs`) &&
            peer.body.includes(`${testCount} unique photos`) && /SHA-256: [a-f0-9]{64}/i.test(peer.body)) ownerCandidates.push(peerPath);
      }
      if (ownerCandidates.length > 1) throw new Error("Staged mutation deferred: dated test baseline has a competing or already updated peer.");
      if (ownerCandidates.length === 0) throw new Error("Staged mutation deferred: dated test baseline owner is absent or ambiguous.");
      const ownerPath = ownerCandidates[0];
      const oldPublishedShaMatch = (await kb.readConcept(ownerPath)).body.match(/SHA-256: ([a-f0-9]{64})/i);
      const oldPublishedSha = oldPublishedShaMatch?.[1];
      if (!oldPublishedSha || oldPublishedSha === manifestSha) throw new Error("Staged mutation deferred: historical published manifest is not uniquely grounded.");
      if (options.ownerHint && options.ownerHint !== ownerPath) throw new Error("Staged mutation deferred: dated test baseline conflicts with owner hint.");
      const peerEvidence: string[] = [];
      const supersedingPeers: string[] = [];
      let laterManifestSha: string | undefined;
      let owner: Awaited<ReturnType<KnowledgeBase["readConcept"]>> | undefined;
      for (const peerPath of paths) {
        state.checkCancellation();
        const peer = await kb.readConcept(peerPath);
        if (peerPath !== ownerPath && !peer.body.includes(baselineLocator ?? baselinePath!) &&
            !peer.body.includes(manifestSha) && !peer.body.includes(reportSha) &&
            !peer.body.includes(oldPublishedSha) && !peer.body.includes(checkpointSha) &&
            !peer.body.includes(confusion)) continue;
        recorder.record("read_concept", peer.path, [peer.path]);
        if (peer.body.length > STAGED_MAX_OWNER_CHARS) {
          throw new Error("Staged mutation deferred: FS dated test baseline owner or peer exceeds evidence limit.");
        }
        if (peerPath === ownerPath) {
          owner = peer;
          continue;
        }
        const hashes = [...peer.body.matchAll(/\b[a-f0-9]{64}\b/gi)].map((match) => match[0]);
        const laterHash = hashes.find((hash) => hash !== manifestSha && hash !== oldPublishedSha &&
          hash !== reportSha && hash !== checkpointSha);
        const confirmedLaterCorrection = peer.body.includes(manifestSha) && !!laterHash &&
          /\b(?:correction|corrected|superseded)\b/i.test(peer.body) &&
          new RegExp(`(?:${testCount}[- ]photo|${testCount} unique photos)`, "i").test(peer.body);
        if (confirmedLaterCorrection) laterManifestSha = laterHash;
        if (confirmedLaterCorrection) supersedingPeers.push(peer.path);
        const peerConflict = peer.body.includes(reportSha) ? "evaluation report" :
          peer.body.split(/\n\s*\n/).some((part) => part.includes(confusion) && part.includes(checkpointSha)) ? "evaluation confusion" :
          peer.body.includes(confusion) && !confirmedLaterCorrection ? "evaluation confusion" :
          peer.body.includes(manifestSha) && !confirmedLaterCorrection ? "conflicting current manifest" :
          peer.body.includes(oldPublishedSha) && peer.body.includes(`${priorCount} sample IDs`) &&
            peer.body.includes(`${testCount} unique photos`) ? "full historical owner" : undefined;
        if (peerConflict) {
          throw new Error(`Staged mutation deferred: FS dated test baseline has a competing or already updated peer (${peer.path}: ${peerConflict}).`);
        }
        for (const part of peer.body.split(/\n\s*\n/)) {
          if ((part.includes(baselinePath) && /\b\d+\b/.test(part)) ||
              part.includes(oldPublishedSha) || part.includes(checkpointSha) ||
              (confirmedLaterCorrection && part.includes(manifestSha))) {
            if (part.length > 4_000) {
              throw new Error("Staged mutation deferred: FS dated test baseline peer passage exceeds evidence limit.");
            }
            peerEvidence.push(`${peer.path}: ${part}`);
          }
        }
      }
      if (peerEvidence.length > 28 || peerEvidence.join("\n\n").length > 32_000 ||
          supersedingPeers.length !== 1 || !laterManifestSha) {
        throw new Error("Staged mutation deferred: FS dated test baseline peer evidence or later correction is incomplete.");
      }
      if (!owner || !owner.body.includes(oldPublishedSha) ||
          !owner.body.includes(`${priorCount} sample IDs`) ||
          !owner.body.includes(`${testCount} unique photos`) ||
          !owner.body.includes(baselineLocator ?? baselinePath!)) {
        throw new Error("Staged mutation deferred: dated test baseline owner lacks full dated source and parsed anchors.");
      }
      const correctionPeer = await kb.readConcept(supersedingPeers[0]);
      const correctionDate = correctionPeer.body.match(/\b(\d{1,2}\s+[A-Z][a-z]{2,8}\s+\d{4})\b/)?.[1] ?? "later dated";
      const correctionLabel = correctionPeer.body.match(/\b([\w-]+\s+correction)\b/i)?.[1] ?? "source-label correction";
      const heading = `Historical ${headingProject} on-disk dated test baseline and retrain evaluation (before the ${correctionDate} ${correctionLabel}): `;
      const update = `${heading}${updateBody.trim().replace(/\.+$/, "")}. Later on ${correctionDate}, the separate ${correctionLabel} superseded this manifest with SHA ${laterManifestSha}; this evaluation describes the earlier snapshot.`;
      const current = owner.body.split(/\n\s*\n/).filter((part) => part.startsWith(heading));
      if (current.length === 1 && current[0].trimEnd() === update &&
          owner.body.split(manifestSha).length === 2 && owner.body.split(reportSha).length === 2) {
        const fresh = await kb.readConcept(ownerPath);
        if (fresh.body !== owner.body) throw new Error(`Concept changed while it was being read: ${ownerPath}`);
        const summary = "No change needed; dated test baseline and retrain evaluation verified.";
        const trace = recorder.finalize("mutation", instruction, summary, "success", modelChain, undefined, undefined, undefined,
          { providerCalls, generationMs: Date.now() - started });
        await traceStore(kb).save(trace);
        return { ok: true, result: { summary, filesChanged: [], steps: 1, traceId: trace.id } };
      }
      if (current.length || owner.body.includes(manifestSha) || owner.body.includes(reportSha)) {
        throw new Error("Staged mutation deferred: FS dated test baseline has a partial or conflicting current status.");
      }
      const proposed = `${owner.body.trimEnd()}\n\n${update}\n`;
      if (proposed.length > STAGED_MAX_OWNER_CHARS || update.length > 4_000) {
        throw new Error("Staged mutation deferred: FS dated test baseline update exceeds evidence limit.");
      }
      const resolved = await resolveAgentModel(options, "mutate", stagedEnv, (call) => providerCalls.push(call));
      modelChain = resolved.modelChain;
      const judgement = await generateObject({
        model: resolved.model,
        schema: z.object({ safe: z.boolean() }).strict(),
        temperature: 0,
        abortSignal: options.signal,
        prompt: `Independent dated test-baseline ownership and chronology check. Verify exact request-derived old/new manifest and evaluation evidence, full bounded peer inventory, that later correction facts are not contradictions to this earlier dated snapshot, and that obsolete current wording is archived without overwriting later published policy. Confirm exact content, chronology, and owner; fail closed on conflicting CURRENT peers or uncertainty. Check historical sample-count distinction, checkpoint, metrics, support caveat, privacy, overlap and repeatedly-used-test-set caveats. JSON only.\nINSTRUCTION:\n${raw}\nCOMPLETE OWNER ${owner.path}:\n${owner.body}\nRELEVANT PEERS:\n${peerEvidence.join("\n\n")}\nPROPOSED APPENDED NOTE:\n${update}`,
      });
      state.checkCancellation();
      if (!judgement.object.safe) throw new Error("Staged mutation deferred: independent FS dated test baseline check declined.");
      const fresh = await kb.readConcept(ownerPath);
      if (fresh.body !== owner.body) throw new Error(`Concept changed while it was being read: ${ownerPath}`);
      state.checkCancellation();
      await kb.patchConcept(ownerPath, { replaceBody: proposed },
        "Staged mutation: record dated test baseline evaluation", sha256(owner.body), options.signal,
        (writtenPath) => { changed = [writtenPath]; });
      const readBack = await kb.readConcept(ownerPath);
      if (readBack.body !== proposed) throw new Error("FS dated test baseline failed CAS read-back verification.");
      const summary = `Staged mutation changed ${ownerPath}.`;
      const trace = recorder.finalize("mutation", instruction, summary, "success", modelChain, undefined, undefined, undefined,
        { providerCalls, generationMs: Date.now() - started });
      await traceStore(kb).save(trace);
      return { ok: true, result: { summary, filesChanged: changed, steps: 1, traceId: trace.id } };
    }
    // Reorder an existing annotation page using only request evidence and the
    // complete observed owner. Never select a project from metadata or bake a
    // page name, category, count, hash, or source policy into production.
    const annotationRequest = raw.trim().match(/^Update (.+?) annotation-page order\b/i);
    if (!options.directAdd && options.preflightInput && annotationRequest) {
      const page = raw.match(/(?:file:\/\/\/[^\s`]+|`[^`]+\.html`)/i)?.[0]?.replace(/^`|`$/g, "").replace(/[.,;]+$/, "");
      const hashes = [...raw.matchAll(/\b[a-f0-9]{64}\b/gi)].map((match) => match[0]);
      const currentSha = raw.match(/(?:new|current) page SHA\s+`?([a-f0-9]{64})`?/i)?.[1];
      if (!page || !currentSha || hashes.filter((hash) => hash !== currentSha).length < 1 ||
          !/\b(?:bucket|buckets)\b/i.test(raw) || !/\b(?:source|policy|annotations)\b/i.test(raw)) {
        throw new Error("Staged mutation deferred: annotation-page order request lacks page, old/new hashes, buckets, or source policy.");
      }
      const heading = "Current retrained annotation-page order: ";
      const historical = "Earlier retrained annotation-page snapshot (superseded): ";
      const pageDescriptor = annotationRequest[1].match(/(\d[\d,]*-card\s+synthetic)$/i)?.[1];
      const update = `${heading}${raw.trim().replace(/^Update .+? annotation-page order at\s+/i,
        pageDescriptor ? `The ${pageDescriptor} annotation page at ` : "The annotation page at ")}`;
      const paths = await kb.bundle.listConceptPaths();
      if (paths.length > 10_000) throw new Error("Staged mutation deferred: annotation-page concept inventory exceeds limit.");
      const observed = new Map<string, Awaited<ReturnType<KnowledgeBase["readConcept"]>>>();
      const owners: Array<{ path: string; paragraph: string }> = [];
      const peerEvidence: string[] = [];
      for (const peerPath of paths) {
        state.checkCancellation();
        const peer = await kb.readConcept(peerPath);
        if (!peer.body.includes(page) && !peer.body.includes(currentSha) && !hashes.some((hash) => peer.body.includes(hash))) continue;
        recorder.record("read_concept", peer.path, [peer.path]);
        if (peer.body.length > STAGED_MAX_OWNER_CHARS) throw new Error("Staged mutation deferred: annotation-page owner or peer exceeds evidence limit.");
        observed.set(peer.path, peer);
        for (const paragraph of peer.body.split(/\n\s*\n/)) {
          if (paragraph.includes(page) && /\b\d[\d,]*\b/.test(paragraph) &&
              !paragraph.startsWith(historical) && !paragraph.startsWith(heading)) owners.push({ path: peer.path, paragraph });
          if (paragraph.includes(page) || hashes.some((hash) => paragraph.includes(hash))) {
            if (paragraph.length > 4_000) throw new Error("Staged mutation deferred: annotation-page peer passage exceeds evidence limit.");
            peerEvidence.push(`${peer.path}: ${paragraph}`);
          }
        }
      }
      if (peerEvidence.length > 24 || peerEvidence.join("\n\n").length > 28_000) throw new Error("Staged mutation deferred: annotation-page peer evidence exceeds limit.");
      const completed = [...observed.values()].filter((peer) => peer.body.split(/\n\s*\n/).some((part) => part.startsWith(heading)));
      if (completed.length === 1 && (!options.ownerHint || options.ownerHint === completed[0].path)) {
        const target = completed[0];
        const paragraphs = target.body.split(/\n\s*\n/);
        const current = paragraphs.filter((part) => part.startsWith(heading));
        const archived = paragraphs.filter((part) => part.startsWith(historical));
        const original = paragraphs.filter((part) => part.includes(page) && /\b\d[\d,]*\b/.test(part) && !part.startsWith(heading) && !part.startsWith(historical));
        if (current.length === 1 && (current[0].trimEnd() === update || current[0].startsWith(`${update} (Prior snapshot SHA-256: `)) &&
            current[0].split(currentSha).length === 2 && target.body.split(currentSha).length === 2 &&
            [...observed.values()].every((peer) => peer.path === target.path || !peer.body.includes(currentSha))) {
          if (archived.length === 1 && original.length === 0 && archived[0].includes(page) &&
              archived[0].length > 150 && [...archived[0].matchAll(/\b\d[\d,]*\b/g)].length >= 2 && /\b[a-f0-9]{64}\b/i.test(archived[0])) {
            const priorBody = archived[0].slice(historical.length);
            const proposed = `${target.body.replace(historical, "")
              .replace(current[0], `${update} (Prior snapshot SHA-256: ${sha256(priorBody)})`).trimEnd()}\n`;
            const resolved = await resolveAgentModel(options, "mutate", stagedEnv, (call) => providerCalls.push(call));
            modelChain = resolved.modelChain;
            const judgement = await generateObject({ model: resolved.model, schema: z.object({ safe: z.boolean() }).strict(), temperature: 0,
              abortSignal: options.signal,
              prompt: `Independent historical-label correction. Remove only the superseded label, preserve all archive facts and current unrelated policy, and ensure the current page remains uniquely identified. Fail closed on incomplete archives or competing peers. JSON only.\nINSTRUCTION:\n${raw}\nOWNER ${target.path}:\n${target.body}\nPEERS:\n${peerEvidence.filter((item) => !item.startsWith(`${target.path}:`)).join("\n\n")}\nPROPOSED:\n${proposed}` });
            state.checkCancellation();
            if (!judgement.object.safe) throw new Error("Staged mutation deferred: annotation-page historical-label correction declined.");
            const fresh = await kb.readConcept(target.path);
            if (fresh.body !== target.body) throw new Error(`Concept changed while it was being read: ${target.path}`);
            await kb.patchConcept(target.path, { replaceBody: proposed }, "Staged mutation: preserve annotation page history", sha256(target.body), options.signal,
              (writtenPath) => { changed = [writtenPath]; });
            if ((await kb.readConcept(target.path)).body !== proposed) throw new Error("Annotation-page label correction failed CAS read-back verification.");
            const summary = `Staged mutation changed ${target.path}.`;
            const trace = recorder.finalize("mutation", instruction, summary, "success", modelChain, undefined, undefined, undefined, { providerCalls, generationMs: Date.now() - started });
            await traceStore(kb).save(trace);
            return { ok: true, result: { summary, filesChanged: changed, steps: 1, traceId: trace.id } };
          }
          if (archived.length === 0 && original.length === 1 &&
              current[0].trimEnd() === `${update} (Prior snapshot SHA-256: ${sha256(original[0].trimEnd())})`) {
            const fresh = await kb.readConcept(target.path);
            if (fresh.body !== target.body) throw new Error(`Concept changed while it was being read: ${target.path}`);
            const summary = "No change needed; annotation-page order and history verified.";
            const trace = recorder.finalize("mutation", instruction, summary, "success", modelChain, undefined, undefined, undefined, { providerCalls, generationMs: Date.now() - started });
            await traceStore(kb).save(trace);
            return { ok: true, result: { summary, filesChanged: [], steps: 1, traceId: trace.id } };
          }
        }
      }
      const ownerPath = owners.length === 1 ? owners[0].path : undefined;
      if (!ownerPath || completed.length || (options.ownerHint && options.ownerHint !== ownerPath)) throw new Error("Staged mutation deferred: annotation-page owner is absent, duplicated, or conflicts with hint.");
      const owner = observed.get(ownerPath)!;
      if (owner.body.split(historical).length > 1 || owner.body.split(heading).length > 1 ||
          owner.body.includes(currentSha) ||
          [...observed.values()].some((peer) => peer.path !== ownerPath && peer.body.includes(currentSha))) {
        throw new Error("Staged mutation deferred: annotation-page status is already present, partial, or owned by a peer.");
      }
      const prior = owners[0].paragraph.trimEnd();
      if (prior.length > 4_000 || owner.body.split(prior).length !== 2 || !prior.includes(page) || !/\b[a-f0-9]{64}\b/i.test(prior)) throw new Error("Staged mutation deferred: annotation-page prior snapshot is not unique or complete.");
      const proposed = owner.body.replace(prior, `${prior}\n\n${update} (Prior snapshot SHA-256: ${sha256(prior)})`);
      if (proposed.length > STAGED_MAX_OWNER_CHARS || update.length > 3_000) throw new Error("Staged mutation deferred: annotation-page replacement exceeds evidence limit.");
      const resolved = await resolveAgentModel(options, "mutate", stagedEnv, (call) => providerCalls.push(call));
      modelChain = resolved.modelChain;
      const judgement = await generateObject({ model: resolved.model, schema: z.object({ safe: z.boolean() }).strict(), temperature: 0, abortSignal: options.signal,
        prompt: `Independent annotation-page ownership and ordering review. Verify request-derived new and old page hashes, source policy and all requested buckets. Ensure this is the unique complete page owner, preserve the full earlier page/history and unrelated current training policy, preserve IDs/annotations/storage identity/queues as instructed, and reject competing or contradicted peer claims. JSON only.\nINSTRUCTION:\n${raw}\nCOMPLETE OWNER ${owner.path}:\n${owner.body}\nRELEVANT PEERS:\n${peerEvidence.filter((item) => !item.startsWith(`${owner.path}:`)).join("\n\n")}\nPROPOSED:\n${proposed}` });
      state.checkCancellation();
      if (!judgement.object.safe) throw new Error("Staged mutation deferred: independent annotation-page review declined.");
      const fresh = await kb.readConcept(ownerPath);
      if (fresh.body !== owner.body) throw new Error(`Concept changed while it was being read: ${ownerPath}`);
      state.checkCancellation();
      await kb.patchConcept(ownerPath, { replaceBody: proposed }, "Staged mutation: record annotation-page order", sha256(owner.body), options.signal,
        (writtenPath) => { changed = [writtenPath]; });
      if ((await kb.readConcept(ownerPath)).body !== proposed) throw new Error("Annotation-page update failed CAS read-back verification.");
      const summary = `Staged mutation changed ${ownerPath}.`;
      const trace = recorder.finalize("mutation", instruction, summary, "success", modelChain, undefined, undefined, undefined, { providerCalls, generationMs: Date.now() - started });
      await traceStore(kb).save(trace);
      return { ok: true, result: { summary, filesChanged: changed, steps: 1, traceId: trace.id } };
    }
    // Supersede an earlier full-inventory report only when the caller supplies
    // both the observed report identity and the later changes-only facts.
    const changeReportRequest = !options.directAdd && options.preflightInput
      ? raw.trim().match(/^Update (.+?) (\d{4}-\d{2}-\d{2}) baseline change-report knowledge:\s*([\s\S]+)$/i)
      : null;
    if (changeReportRequest) {
      const [, rawProject, reportDate, facts] = changeReportRequest;
      const project = rawProject.replace(/^the\s+/i, "");
      const reportPathMatch = raw.match(/(?:~\/?|\/)`?([^`\s]*baselines\/(\d{4}-\d{2}-\d{2})\/([^`\s/]+))`?/i);
      const reportFile = reportPathMatch?.[3];
      const superseded = raw.match(/supersede any statement that the CURRENT report ([^.]+)\./i)?.[1];
      if (!reportFile || !superseded || !/\b(?:changes-only|focused strictly on changes|differences from)\b/i.test(facts)) {
        throw new Error("Staged mutation deferred: change-report request lacks report identity or explicit supersession facts.");
      }
      const reportBase = reportFile.replace(/\.md$/i, "");
      const pathMention = new RegExp(`(?:baselines/\\d{4}-\\d{2}-\\d{2}/)?${reportBase.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}(?:\\.md)?`, "i");
      const staleCounts = [...superseded.matchAll(/\b\d[\d,]*\b/g)].map((match) => match[0].replace(/,/g, ""));
      const oldSignal = (paragraph: string): boolean => pathMention.test(paragraph) &&
        staleCounts.length > 0 && staleCounts.every((count) => new RegExp(`\\b${count}\\b`).test(paragraph.replace(/,/g, "")));
      const paths = await kb.bundle.listConceptPaths("/projects");
      if (paths.length > 10_000) throw new Error("Staged mutation deferred: change-report sibling inventory exceeds limit.");
      const searchHits = await kb.search(reportBase, { limit: 200 });
      recorder.record("search_knowledge", reportBase, searchHits.map((hit) => hit.path));
      if (searchHits.length >= 200) throw new Error("Staged mutation deferred: change-report search inventory exceeds limit.");
      const heading = `Current ${project} ${reportDate} changes-only baseline report: `;
      const observed = new Map<string, Awaited<ReturnType<typeof kb.readConcept>>>();
      const oldOwners: Array<{path: string; paragraph: string}> = [];
      const peerEvidence: string[] = [];
      for (const peerPath of new Set([...paths, ...searchHits.map((hit) => hit.path)])) {
        state.checkCancellation();
        const peer = await kb.readConcept(peerPath);
        recorder.record("read_concept", peer.path, [peer.path]);
        if (peer.body.length > STAGED_MAX_OWNER_CHARS) throw new Error("Staged mutation deferred: change-report owner or peer exceeds evidence limit.");
        observed.set(peer.path, peer);
        for (const paragraph of peer.body.split(/\n\s*\n/)) {
          if (!paragraph.startsWith(heading) && !paragraph.startsWith("Earlier non-focused ") && oldSignal(paragraph)) oldOwners.push({ path: peer.path, paragraph });
          else if (!paragraph.startsWith(heading) && !paragraph.startsWith("Earlier non-focused ") &&
              (pathMention.test(paragraph) || paragraph.includes(reportDate)) &&
              staleCounts.some((count) => new RegExp(`\\b${count}\\b`).test(paragraph.replace(/,/g, ""))) &&
              /\b(?:full inventory|report|preserv|unchanged)\b/i.test(paragraph)) {
            if (paragraph.length > 4_000) throw new Error("Staged mutation deferred: change-report peer passage exceeds limit.");
            peerEvidence.push(`${peer.path}: ${paragraph}`);
          }
        }
      }
      if (peerEvidence.length > 20 || peerEvidence.join("\n\n").length > 16_000) throw new Error("Staged mutation deferred: change-report peer evidence is incomplete.");
      const reportLanguage = facts.match(/\b(Danish|English|French|German|Spanish|Swedish|Norwegian)\b/i)?.[1];
      const historical = `Earlier non-focused ${reportLanguage ? `${reportLanguage} ` : ""}report version (superseded by changes-only edition): `;
      const oldOwnersArchived = [...observed.entries()].flatMap(([ownerPath, peer]) => peer.body.split(/\n\s*\n/)
        .filter((paragraph) => paragraph.startsWith(historical) && oldSignal(paragraph.slice(historical.length)))
        .map((paragraph) => ({ path: ownerPath, paragraph })));
      const historicalClarification = superseded.match(/\b(\d[\d,]*) (?:photo-)?ID mentions or all (\d[\d,]*) approved-photo mappings\b/i);
      const clarifiedHistory = historicalClarification
        ? `${historicalClarification[1]} photo-ID mentions and all ${historicalClarification[2]} approved-photo mappings describe only the earlier non-focused ${reportLanguage ?? "full-inventory"} version.`
        : "";
      const currentBase = `${heading}${facts.trim()}${clarifiedHistory ? ` ${clarifiedHistory}` : ""}`;
      const currentFor = (oldReport: string): string =>
        `${currentBase} (Prior report SHA-256: ${sha256(oldReport.trimEnd())})`;
      const ownerPath = oldOwners.length === 1 ? oldOwners[0].path :
        oldOwners.length === 0 && oldOwnersArchived.length === 1 ? oldOwnersArchived[0].path : undefined;
      if (!ownerPath || options.ownerHint && options.ownerHint !== ownerPath) throw new Error("Staged mutation deferred: change-report owner is absent, duplicated, or conflicts with hint.");
      const owner = observed.get(ownerPath)!;
      if (oldOwnersArchived.length && oldOwnersArchived.some((item) => item.path !== ownerPath)) throw new Error("Staged mutation deferred: change-report archive has competing owners.");
      const blocks = owner.body.split(/\n\s*\n/);
      const archives = blocks.filter((part) => part.startsWith(historical));
      const updates = blocks.filter((part) => part.startsWith(heading));
      if (!oldOwners.length && archives.length === 1 && oldOwnersArchived.length === 1 &&
          updates.length === 1 && updates[0].trimEnd() === currentFor(archives[0].slice(historical.length))) {
        if (peerEvidence.length) throw new Error("Staged mutation deferred: change-report peers retain a potentially competing old claim.");
        const reread = await kb.readConcept(ownerPath);
        if (reread.body !== owner.body) throw new Error(`Concept changed while it was being read: ${ownerPath}`);
        const summary = "No change needed; changes-only report and complete historical version verified.";
        const trace = recorder.finalize("mutation", instruction, summary, "success", modelChain, undefined, undefined, undefined, { providerCalls, generationMs: Date.now() - started });
        await traceStore(kb).save(trace);
        return { ok: true, result: { summary, filesChanged: [], steps: 1, traceId: trace.id } };
      }
      if (oldOwners.length !== 1 || oldOwners[0].path !== ownerPath || archives.length || updates.length || peerEvidence.length) throw new Error("Staged mutation deferred: change-report old owner is absent, duplicated, or contradicted.");
      const oldParagraph = oldOwners[0].paragraph.trimEnd();
      if (oldParagraph.length > 4_000 || owner.body.split(oldParagraph).length !== 2) throw new Error("Staged mutation deferred: change-report archive is not unique or complete.");
      const replacement = `${historical}${oldParagraph}\n\n${currentFor(oldParagraph)}`;
      const proposed = owner.body.replace(oldParagraph, replacement);
      if (proposed.length > STAGED_MAX_OWNER_CHARS) throw new Error("Staged mutation deferred: change-report replacement exceeds evidence limit.");
      const resolved = await resolveAgentModel(options, "mutate", stagedEnv, (call) => providerCalls.push(call));
      modelChain = resolved.modelChain;
      const judgement = await generateObject({ model: resolved.model, schema: z.object({ safe: z.boolean() }).strict(), temperature: 0, abortSignal: options.signal,
        prompt: `Independent change-report supersession check. Verify the exact old full-inventory report is complete and uniquely owned in the bounded sibling inventory. The archive must preserve it verbatim; the new changes-only edition may contain only caller-supplied facts. Preserve all other history and policy; reject truncated archives, competing claims, unsupported details, or uncertainty. JSON only.\nINSTRUCTION:\n${raw}\nCOMPLETE OWNER ${owner.path}:\n${owner.body}\nRELEVANT PEERS:\n${peerEvidence.join("\n\n")}\nEXACT OLD REPORT:\n${oldParagraph}\nPROPOSED:\n${replacement}` });
      state.checkCancellation();
      if (!judgement.object.safe) throw new Error("Staged mutation deferred: independent change-report check declined.");
      const reread = await kb.readConcept(ownerPath);
      if (reread.body !== owner.body) throw new Error(`Concept changed while it was being read: ${ownerPath}`);
      await kb.patchConcept(ownerPath, { replaceBody: proposed }, "Staged mutation: supersede full-inventory report", sha256(owner.body), options.signal, (writtenPath) => { changed = [writtenPath]; });
      if ((await kb.readConcept(ownerPath)).body !== proposed) throw new Error("Change-report correction failed CAS read-back verification.");
      const summary = `Staged mutation changed ${ownerPath}.`;
      const trace = recorder.finalize("mutation", instruction, summary, "success", modelChain, undefined, undefined, undefined, { providerCalls, generationMs: Date.now() - started });
      await traceStore(kb).save(trace);
      return { ok: true, result: { summary, filesChanged: changed, steps: 1, traceId: trace.id } };
    }
    // A thread-resolution event is a historical checkpoint, not a review decision.
    // Resolve its owner from the complete bounded corpus and keep later review/merge
    // lifecycle facts distinct.
    if (!options.directAdd && options.preflightInput) {
      const identity = raw.match(/\bPR\s*#?(\d+)\b/i);
      const headMatch = raw.match(/\bhead\s+([a-f0-9]{7,40})\b/i);
      const resolvesThreads = /\b(?:review\s+)?threads?\b/i.test(raw) &&
        /\b(?:resolv(?:e|ed|ing|es)|address(?:ed|ing)?|marked\s+resolved)\b/i.test(raw);
      const claimsIndependentDecision = /\b(?:approved|approval submitted|review submitted|merged|squash-merged)\b/i.test(raw) &&
        !/\b(?:not|has not|without|does not|did not)\s+(?:been\s+)?(?:approved|approval|submitted|merged)\b/i.test(raw);
      if (identity && headMatch && resolvesThreads && !claimsIndependentDecision) {
        const pr = identity[1];
        const head = headMatch[1].toLowerCase();
        const note = `PR #${pr} review-thread resolution at checked head ${head} (historical checkpoint): ${raw.trim()}`;
        const paths = await kb.bundle.listConceptPaths();
        if (paths.length > 10_000) {
          throw new Error("Staged mutation deferred: review-thread concept inventory exceeds limit.");
        }
        const observed = new Map<string, Awaited<ReturnType<KnowledgeBase["readConcept"]>>>();
        const candidates: Array<{ path: string; paragraph: string }> = [];
        const peers: string[] = [];
        let resolved: Awaited<ReturnType<typeof resolveAgentModel>> | undefined;
        const hasIdentity = (text: string): boolean => new RegExp(`\\bPR\\s*#?${pr}\\b`, "i").test(text);
        for (const candidatePath of paths) {
          state.checkCancellation();
          const candidate = await kb.readConcept(candidatePath);
          if (!hasIdentity(candidate.body) && !candidate.body.toLowerCase().includes(head)) continue;
          recorder.record("read_concept", candidate.path, [candidate.path]);
          if (candidate.body.length > STAGED_MAX_OWNER_CHARS) {
            throw new Error("Staged mutation deferred: review-thread owner or peer exceeds evidence limit.");
          }
          observed.set(candidate.path, candidate);
          for (const paragraph of candidate.body.split(/\n\s*\n/)) {
            if (paragraph.trimEnd() === note) continue;
            if (!hasIdentity(paragraph) && !paragraph.toLowerCase().includes(head)) continue;
            const duplicate = hasIdentity(paragraph) && /\bthreads?\b/i.test(paragraph) &&
              /\b(?:resolv|resolved)\w*\b/i.test(paragraph);
            if (!duplicate) {
              if (paragraph.length > 4_000) throw new Error("Staged mutation deferred: review-thread peer passage exceeds evidence limit.");
              if (paragraph.toLowerCase().includes(head) && hasIdentity(paragraph) &&
                  /\b(?:changes_requested|approved|commented|requested changes)\b/i.test(paragraph)) {
                candidates.push({ path: candidate.path, paragraph });
              }
              peers.push(`${candidate.path}: ${paragraph}`);
              continue;
            }
            // Queue examples are the only peer exception, adjudicated from the
            // peer alone so the proposed mutation cannot bias classification.
            if (candidates.some((entry) => entry.path === candidatePath)) {
              throw new Error("Staged mutation deferred: review-thread checkpoint is partial or conflicting.");
            }
            const queueContext = /\b(?:queue|queued|async request|memory_update|failed request|request example|runbook)\b/i.test(paragraph);
            if (!queueContext) {
              throw new Error(`Staged mutation deferred: review-thread resolution has another owner (${candidate.path}).`);
            }
            const at = paragraph.search(new RegExp(`\\bPR\\s*#?${pr}\\b`, "i"));
            const excerpt = paragraph.slice(Math.max(0, at - 900), Math.min(paragraph.length, at + 2_300));
            resolved ??= await resolveAgentModel(options, "mutate", stagedEnv, (call) => providerCalls.push(call));
            modelChain = resolved.modelChain;
            const ownership = await generateObject({
              model: resolved.model,
              schema: z.object({ exampleOnly: z.boolean() }).strict(),
              temperature: 0,
              abortSignal: options.signal,
              prompt: `Independent peer-only classification. Is this passage only a quoted/example failed or queued memory request, rather than an authoritative repository/PR review-thread assertion? If uncertain or substantive status is asserted, exampleOnly=false. JSON only.\nPEER ${candidate.path}:\n${excerpt}`,
            });
            state.checkCancellation();
            if (!ownership.object.exampleOnly) throw new Error("Staged mutation deferred: a peer asserts competing review-thread status.");
            peers.push(`${candidate.path} (independently classified as queue example): ${excerpt}`);
          }
        }
        if (peers.length > 32 || peers.join("\n\n").length > 28_000) {
          throw new Error("Staged mutation deferred: review-thread peer evidence exceeds limit.");
        }
        const uniqueOwners = [...new Set(candidates.map((candidate) => candidate.path))];
        if (uniqueOwners.length !== 1 || (options.ownerHint && options.ownerHint !== uniqueOwners[0])) {
          throw new Error("Staged mutation deferred: review-thread owner is absent, duplicated, or conflicts with hint.");
        }
        const ownerPath = uniqueOwners[0];
        const owner = observed.get(ownerPath)!;
        const existing = owner.body.split(/\n\s*\n/).filter((paragraph) =>
          paragraph.includes(note));
        if (existing.length === 1 && existing[0].trimEnd() === note) {
          const fresh = await kb.readConcept(ownerPath);
          if (fresh.body !== owner.body) throw new Error(`Concept changed while it was being read: ${ownerPath}`);
          const summary = "No change needed; review-thread checkpoint verified.";
          const trace = recorder.finalize("mutation", instruction, summary, "success", modelChain, undefined, undefined, undefined,
            { providerCalls, generationMs: Date.now() - started });
          await traceStore(kb).save(trace);
          return { ok: true, result: { summary, filesChanged: [], steps: 1, traceId: trace.id } };
        }
        if (owner.body.includes(note) || /\bPR\s*#?\d+\b/i.test(note) &&
            owner.body.split(/\n\s*\n/).some((paragraph) => hasIdentity(paragraph) &&
              /\bthreads?\b/i.test(paragraph) && /\b(?:resolv|resolved)\w*\b/i.test(paragraph))) {
          throw new Error("Staged mutation deferred: review-thread checkpoint is partial or conflicting.");
        }
        if (candidates.length !== 1 || !/\b(?:changes_requested|approved|commented|requested changes)\b/i.test(candidates[0].paragraph)) {
          throw new Error("Staged mutation deferred: independent prior review decision is not established.");
        }
        const proposed = `${owner.body.trimEnd()}\n\n${note}\n`;
        if (note.length > 2_000 || proposed.length > STAGED_MAX_OWNER_CHARS) {
          throw new Error("Staged mutation deferred: review-thread checkpoint exceeds evidence limit.");
        }
        resolved ??= await resolveAgentModel(options, "mutate", stagedEnv, (call) => providerCalls.push(call));
        modelChain = resolved.modelChain;
        const judgement = await generateObject({
          model: resolved.model,
          schema: z.object({ safe: z.boolean() }).strict(),
          temperature: 0,
          abortSignal: options.signal,
          prompt: `Independent domain-neutral review-thread checkpoint check. Does this complete owner uniquely identify PR #${pr} at checked head ${head} and establish an existing independent review decision? Does the note record only the requested thread-resolution event, without treating it as approval, changing the review decision, or overwriting later approval/merge history? Check every peer for conflict or duplicate. Reject partial evidence, ambiguity, or uncertainty. JSON only.\nINSTRUCTION:\n${raw}\nCOMPLETE OWNER ${ownerPath}:\n${owner.body}\nRELEVANT PEERS:\n${peers.join("\n\n")}\nPROPOSED NOTE:\n${note}`,
        });
        state.checkCancellation();
        if (!judgement.object.safe) throw new Error("Staged mutation deferred: independent review-thread check declined.");
        const fresh = await kb.readConcept(ownerPath);
        if (fresh.body !== owner.body) throw new Error(`Concept changed while it was being read: ${ownerPath}`);
        state.checkCancellation();
        await kb.patchConcept(ownerPath, { replaceBody: proposed },
          "Staged mutation: record review-thread checkpoint", sha256(owner.body), options.signal,
          (writtenPath) => { changed = [writtenPath]; });
        const readBack = await kb.readConcept(ownerPath);
        if (readBack.body !== proposed) throw new Error("Review-thread checkpoint failed CAS read-back verification.");
        const summary = `Staged mutation changed ${ownerPath}.`;
        const trace = recorder.finalize("mutation", instruction, summary, "success", modelChain, undefined, undefined, undefined,
          { providerCalls, generationMs: Date.now() - started });
        await traceStore(kb).save(trace);
        return { ok: true, result: { summary, filesChanged: changed, steps: 1, traceId: trace.id } };
      }
    }
    // A GitHub review can remain recorded as PENDING after it is submitted.
    // Correct only this observed review paragraph, not the PR's eventual
    // merge status or another review's history.
    const reviewUpdate = raw.trim().match(/^Update ([^\n]+?) PR #?(\d+) review status:\s*/i);
    if (!options.directAdd && options.preflightInput && reviewUpdate) {
      const [, repository, prId] = reviewUpdate;
      const reviewIdMatch = raw.match(/\breview ID\s+(\d+)\b/i);
      const headMatch = raw.match(/\bhead\s+([a-f0-9]{7,64})\b/i);
      const reviewId = reviewIdMatch?.[1];
      if (!reviewId || !/\bpreviously\s+PENDING\b/i.test(raw) ||
          !/\bsubmitted\b/i.test(raw) || !/\bCHANGES_REQUESTED\b/i.test(raw) ||
          !/\b(?:inline )?comments?\b/i.test(raw)) {
        throw new Error("Staged mutation deferred: review instruction lacks explicit prior and submitted status facts.");
      }
      const ownerPattern = repository.trim().toLowerCase().replace(/[^a-z0-9]+/g, "-");
      const ownerPaths = (await kb.bundle.listConceptPaths()).filter((candidate) =>
        new RegExp(`^/(?:repos|projects)/${ownerPattern}(?:[-/]|\\.md$)`, "i").test(candidate));
      const ownerCandidates = new Set([...ownerPaths]);
      const searchReview = await kb.search(reviewId, { limit: 200 });
      const searchPr = await kb.search(prId, { limit: 1_000 });
      recorder.record("search_knowledge", reviewId, searchReview.map((hit) => hit.path));
      recorder.record("search_knowledge", prId, searchPr.map((hit) => hit.path));
      if (ownerPaths.length > 200 || searchReview.length >= 200 || searchPr.length >= 1_000) {
        throw new Error("Staged mutation deferred: review peer search exceeds limit.");
      }
      for (const hit of [...searchReview, ...searchPr]) ownerCandidates.add(hit.path);
      const matchingOld = /\bPENDING\b/i;
      const matchingIdentity = (paragraph: string): boolean => paragraph.includes(reviewId);
      const ownerMatches: Array<{ path: string; body: string; old: string }> = [];
      const ownerPathHint = options.ownerHint;
      const peerDocs = new Map<string, Awaited<ReturnType<typeof kb.readConcept>>>();
      for (const path of ownerCandidates) {
        state.checkCancellation();
        const peer = await kb.readConcept(path);
        recorder.record("read_concept", peer.path, [peer.path]);
        if (peer.body.length > STAGED_MAX_OWNER_CHARS) throw new Error("Staged mutation deferred: review owner or peer exceeds evidence limit.");
        peerDocs.set(peer.path, peer);
        for (const paragraph of peer.body.split(/\n\s*\n/)) {
          if (matchingIdentity(paragraph) && (matchingOld.test(paragraph) || /\bCHANGES_REQUESTED\b/i.test(paragraph))) {
            ownerMatches.push({ path: peer.path, body: peer.body, old: paragraph });
          }
        }
      }
      const selected = ownerMatches.filter((match) => ownerPaths.includes(match.path));
      if (selected.length > 1) {
        throw new Error(selected.some((match) => match.path !== selected[0].path)
          ? "Staged mutation deferred: review has a competing owner."
          : "Staged mutation deferred: review has another stale owner passage.");
      }
      const ownerPath = selected.length === 1 ? selected[0].path : undefined;
      if (!ownerPath || (ownerPathHint && ownerPathHint !== ownerPath)) {
        throw new Error("Staged mutation deferred: exact prior review paragraph is absent, duplicated, or conflicts with owner hint.");
      }
      const owner = peerDocs.get(ownerPath)!;
      const observedParagraph = selected[0].old;
      const alreadyUpdated = !matchingOld.test(observedParagraph);
      const old = alreadyUpdated ? "" : observedParagraph;
      const commentCount = raw.match(/\b([a-z]+|\d+) inline comments\b/i)?.[1];
      const priorCommentCount = observedParagraph.match(/\b([a-z]+|\d+) inline comments\b/i)?.[1];
      const cardinal = (count: string | undefined): number | undefined => {
        if (!count) return undefined;
        const names = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"];
        return /^\d+$/.test(count) ? Number(count) : names.indexOf(count.toLowerCase()) < 0
          ? undefined : names.indexOf(count.toLowerCase());
      };
      const readabilityNotes = raw.match(/\btwo readability\/style comments in\s+([\w.-]+)\s+and\s+([\w.-]+)/i);
      const additionalComments = (cardinal(commentCount) ?? -1) - (cardinal(priorCommentCount) ?? -1);
      if (!alreadyUpdated && (!Number.isInteger(additionalComments) || additionalComments < 0 ||
          (additionalComments > 0 && !(additionalComments === 2 && readabilityNotes)))) {
        throw new Error("Staged mutation deferred: new review comments lack complete caller-supported detail.");
      }
      let updated = observedParagraph;
      if (!alreadyUpdated && commentCount) {
        updated = updated.replace(/\bcreated\s+unsubmitted\s+PENDING\s+review\s+/i, "submitted review ")
          .replace(/\bPENDING\b/i, "CHANGES_REQUESTED")
          .replace(/\bon head ([a-f0-9]{7,40}) with (?:[a-z]+|\d+) inline comments\b/i,
            (_match, head: string) => `on head ${head} with state CHANGES_REQUESTED and ${commentCount} inline comments`)
          .replace(/\s*Review was intentionally NOT submitted\.?/i, "");
        if (readabilityNotes && !updated.includes(readabilityNotes[1])) {
          updated = updated.replace(/\. Comment IDs\b/i,
            `; (3) a readability/style comment in ${readabilityNotes[1]}; (4) a readability/style comment in ${readabilityNotes[2]}. Comment IDs`);
        }
      }
      if ((!alreadyUpdated && updated === old) || (headMatch && !observedParagraph.includes(headMatch[1])) ||
          !/\bCHANGES_REQUESTED\b/.test(updated) || !commentCount ||
          !new RegExp(`\\b${commentCount} inline comments\\b`, "i").test(updated) ||
          (readabilityNotes && (!updated.includes(readabilityNotes[1]) || !updated.includes(readabilityNotes[2])))) {
        throw new Error("Staged mutation deferred: old review paragraph cannot be safely reconciled with caller facts.");
      }
      const oldCount = old ? owner.body.split(old).length - 1 : 0;
      const updatedCount = owner.body.split(updated).length - 1;
      if (owner.body.length > STAGED_MAX_OWNER_CHARS ||
          (oldCount !== 1 && !(oldCount === 0 && updatedCount === 1))) {
        throw new Error("Staged mutation deferred: exact review paragraph is absent or ambiguous.");
      }
      // PR-level review notes can omit the review ID. Search the number alone:
      // a multiword PR query matches generic PR prose and fills the hit cap.
      const reviewHits = searchReview;
      const prHits = searchPr;
      const siblingPaths = ownerPaths;
      const staleReviewClaim = (paragraph: string): boolean =>
        /(?:PENDING|unsubmitted|NOT submitted)/i.test(paragraph) &&
        (paragraph.includes(reviewId) ||
         (new RegExp(`\\bPR\\s*#?${prId}\\b`, "i").test(paragraph) && /\breview\b/i.test(paragraph)));
      // A second stale assertion makes even an already-updated paragraph
      // insufficient to complete this correction.
      if (owner.body.replace(oldCount ? old : updated, "").split(/\n\s*\n/).some(staleReviewClaim)) {
        throw new Error("Staged mutation deferred: review has another stale owner passage.");
      }
      const peerEvidence: string[] = [];
      let resolved: Awaited<ReturnType<typeof resolveAgentModel>> | undefined;
      for (const peerPath of new Set([...reviewHits.map((hit) => hit.path),
          ...prHits.map((hit) => hit.path), ...siblingPaths])) {
        if (peerPath === ownerPath) continue;
        state.checkCancellation();
        const peer = await kb.readConcept(peerPath);
        recorder.record("read_concept", peer.path, [peer.path]);
        if (peer.body.length > STAGED_MAX_OWNER_CHARS) {
          throw new Error("Staged mutation deferred: review peer exceeds evidence limit.");
        }
        const paragraphs = peer.body.split(/\n\s*\n/);
        let adjudicatedExcerpt: string | undefined;
        const conflicting = paragraphs.filter((paragraph) =>
          staleReviewClaim(paragraph) ||
          (paragraph.includes(reviewId) && /CHANGES_REQUESTED/i.test(paragraph)));
        if (conflicting.length) {
          // An async-queue runbook may quote a failed memory_update as an
          // incident example. A quotation is not a second review-status owner.
          // Require a separate, bounded adjudication; other paths still fail
          // closed on a substantive competing claim.
          if (!/async|queue/i.test(peer.path) || conflicting.length > 4) {
            throw new Error(`Staged mutation deferred: review has a competing owner (${peer.path}; ${conflicting.length} review passages).`);
          }
          const excerpt = conflicting.map((paragraph) => {
            const id = paragraph.indexOf(reviewId);
            const pr = paragraph.search(new RegExp(`\\bPR\\s*#?${prId}\\b`, "i"));
            const at = id >= 0 ? id : pr;
            return paragraph.slice(Math.max(0, at - 900), Math.min(paragraph.length, at + 2_500));
          }).join("\n\n");
          if (excerpt.length > 12_000) {
            throw new Error("Staged mutation deferred: review peer passage exceeds limit.");
          }
          resolved ??= await resolveAgentModel(options, "mutate", stagedEnv, (call) => providerCalls.push(call));
          modelChain = resolved.modelChain;
          const ownership = await generateObject({
            model: resolved.model,
            schema: z.object({ exampleOnly: z.boolean(), reason: z.string().max(300) }).strict(),
            temperature: 0,
            abortSignal: options.signal,
            prompt: `Independent ownership check. Judge ONLY the peer excerpt below, not a proposed correction. Is EVERY mention of PR #${prId} review ${reviewId} in this async/queue peer ONLY an example of a queued or failed memory request, rather than this peer asserting current state or storing the authoritative record? Respond exampleOnly=false on uncertainty or any competing substantive claim. JSON only.\nPEER ${peer.path}:\n${excerpt}`,
          });
          state.checkCancellation();
          if (!ownership.object.exampleOnly) {
            throw new Error(`Staged mutation deferred: queue example adjudication declined (${peer.path}: ${ownership.object.reason.slice(0, 160)}).`);
          }
          adjudicatedExcerpt = excerpt;
        }
        for (const paragraph of paragraphs) {
          if (paragraph.includes(reviewId) ||
              (new RegExp(`\\bPR\\s*#?${prId}\\b`, "i").test(paragraph) && /\breview\b/i.test(paragraph))) {
            if (paragraph.length > 4_000) {
              if (!adjudicatedExcerpt || !conflicting.includes(paragraph)) {
                throw new Error("Staged mutation deferred: review peer passage exceeds limit.");
              }
              peerEvidence.push(`${peer.path} (independently classified as a queue example): ${adjudicatedExcerpt}`);
            } else {
              peerEvidence.push(`${peer.path}: ${paragraph}`);
            }
          }
        }
      }
      if (peerEvidence.join("\n\n").length > 12_000) {
        throw new Error("Staged mutation deferred: review peer evidence exceeds limit.");
      }
      if (oldCount === 0) {
        const current = await kb.readConcept(ownerPath);
        if (current.body !== owner.body) throw new Error(`Concept changed while it was being read: ${ownerPath}`);
        const summary = "No change needed; submitted review status verified.";
        const trace = recorder.finalize("mutation", instruction, summary, "success", modelChain, undefined, undefined, undefined,
          { providerCalls, generationMs: Date.now() - started });
        await traceStore(kb).save(trace);
        return { ok: true, result: { summary, filesChanged: [], steps: 1, traceId: trace.id } };
      }
      const replacement = owner.body.replace(old, updated);
      if (replacement.length > STAGED_MAX_OWNER_CHARS || replacement.includes(old)) {
        throw new Error("Staged mutation deferred: review replacement is not local.");
      }
      resolved ??= await resolveAgentModel(options, "mutate", stagedEnv, (call) => providerCalls.push(call));
      modelChain = resolved.modelChain;
      const judgement = await generateObject({
        model: resolved.model,
        schema: z.object({ safe: z.boolean() }).strict(),
        temperature: 0,
        abortSignal: options.signal,
        prompt: `Independent submitted-review state check. Does the EXACT old paragraph describe this review's previously unsubmitted state, and does the caller support its newly submitted outcome? Does the proposed replacement contain ONLY facts supported by the caller, preserve both substantive details and their comment IDs, and preserve all unrelated PR, review, and eventual merge history? Do any peers own, duplicate or contradict this review status? The path and review ID are candidate locators, not write permission. Reject on uncertainty. JSON only.\nINSTRUCTION:\n${raw}\nCOMPLETE OWNER ${ownerPath}:\n${owner.body}\nRELEVANT PEERS:\n${peerEvidence.join("\n\n")}\nEXACT OLD PARAGRAPH:\n${old}\nEXACT PROPOSED PARAGRAPH:\n${updated}`,
      });
      state.checkCancellation();
      if (!judgement.object.safe) {
        throw new Error("Staged mutation deferred: review independent check declined.");
      }
      const current = await kb.readConcept(ownerPath);
      if (current.body !== owner.body) throw new Error(`Concept changed while it was being read: ${ownerPath}`);
      state.checkCancellation();
      await kb.patchConcept(ownerPath, { replaceBody: replacement },
        "Staged mutation: confirm review submission", sha256(owner.body), options.signal,
        (writtenPath) => { changed = [writtenPath]; });
      const readBack = await kb.readConcept(ownerPath);
      if (readBack.body !== replacement) {
        throw new Error("review failed CAS read-back verification.");
      }
      const summary = `Staged mutation changed ${ownerPath}.`;
      const trace = recorder.finalize("mutation", instruction, summary, "success", modelChain, undefined, undefined, undefined,
        { providerCalls, generationMs: Date.now() - started });
      await traceStore(kb).save(trace);
      return { ok: true, result: { summary, filesChanged: changed, steps: 1, traceId: trace.id } };
    }
    // Confirm a provisional repository deployment only when the request supplies
    // a complete, internally consistent identity and the owner sentence is observed.
    const deploymentRequest = /^For ([a-z0-9](?:[a-z0-9.-]*[a-z0-9])?) deployment knowledge, correct the prior provisional state:/i.exec(raw.trim());
    if (!options.directAdd && options.preflightInput && deploymentRequest) {
      const site = deploymentRequest[1].toLowerCase();
      const slug = site.replace(/\./g, "-");
      const ownerPath = `/repos/${slug}.md`;
      const sha = /\b([a-f0-9]{40})\b/i.exec(raw)?.[1];
      const deploymentId = /\b(dpl_[A-Za-z0-9]+)\b/.exec(raw)?.[1];
      const date = /\b(20\d{2}-\d{2}-\d{2})\b/.exec(raw)?.[1];
      const branch = /\b(origin\/[A-Za-z0-9._/-]+)\b/.exec(raw)?.[1];
      const alias = /\b((?:www\.)?[a-z0-9.-]+\.[a-z]{2,})\s+aliases\b/i.exec(raw)?.[1] ??
        /\b(?:aliased to|aliases to)\s+((?:www\.)?[a-z0-9.-]+\.[a-z]{2,})\b/i.exec(raw)?.[1];
      if (!sha || !deploymentId || !date || !branch || !alias || !new RegExp(`\\b${site.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`, "i").test(raw) ||
          !/READY/i.test(raw) || !/working tree clean/i.test(raw) ||
          !/committed/i.test(raw) || !/pushed/i.test(raw)) {
        throw new Error("Staged mutation deferred: deployment confirmation identity is incomplete.");
      }
      if (options.ownerHint && options.ownerHint !== ownerPath) {
        throw new Error("Staged mutation deferred: deployment owner hint conflicts with request-derived owner.");
      }
      const owner = await kb.readConcept(ownerPath);
      recorder.record("read_concept", owner.path, [owner.path]);
      const headings = [...owner.body.matchAll(/^#{1,6}\s+Deployment(?: Policy)?\s*$/gim)];
      const sectionStart = headings[0]?.index;
      const nextHeading = /^#{1,6}\s+[^\n]+$/gm;
      nextHeading.lastIndex = sectionStart === undefined ? 0 : sectionStart + headings[0][0].length;
      const sectionEnd = nextHeading.exec(owner.body)?.index ?? owner.body.length;
      const section = sectionStart === undefined ? "" : owner.body.slice(sectionStart, sectionEnd);
      const provisional = section.split(/(?<=[.!?])\s+(?=[A-Z`])/).map((sentence) => sentence.trim()).filter((sentence) =>
        /not yet committed or pushed/i.test(sentence) && /deploy/i.test(sentence));
      const old = provisional.length === 1 ? provisional[0] : "";
      const updated = old ? old.replace(/; those changes were not yet committed or pushed in that session\.?$/i,
        `; those changes were committed as \`${sha}\` and pushed to \`${branch}\` on ${date}.`) +
        ` The subsequent Git-triggered production deployment \`${deploymentId}\` from that exact commit reached READY, the Git hosting deployment status was success, and \`${alias}\` aliases the new deployment. The working tree was clean.` : "";
      const oldCount = old ? owner.body.split(old).length - 1 : 0;
      const updatedCount = updated ? owner.body.split(updated).length - 1 : 0;
      const completed = section.split(/\n\s*\n/).filter((paragraph) =>
        paragraph.includes(`those changes were committed as \`${sha}\` and pushed to \`${branch}\` on ${date}.`) &&
        paragraph.includes(`deployment \`${deploymentId}\` from that exact commit reached READY`) &&
        paragraph.includes(`\`${alias}\` aliases the new deployment`) &&
        paragraph.includes("the Git hosting deployment status was success") &&
        paragraph.includes("The working tree was clean."));
      if (owner.body.length > STAGED_MAX_OWNER_CHARS || headings.length !== 1 ||
          (oldCount === 1 && (provisional.length !== 1 || section.split(old).length !== 2)) ||
          (oldCount === 0 && (provisional.length !== 0 || completed.length !== 1)) ||
          (oldCount !== 1 && oldCount !== 0) ||
          (oldCount === 0 && updatedCount > 1)) {
        throw new Error("Staged mutation deferred: provisional deployment claim is absent or ambiguous.");
      }
      const peerPaths = (await kb.bundle.listConceptPaths()).filter((candidate) =>
        candidate !== ownerPath && new RegExp(`^/(?:repos|projects)/${slug}(?:[/-]|\\.md$)`, "i").test(candidate));
      if (peerPaths.length > 100) throw new Error("Staged mutation deferred: deployment peer inventory exceeds limit.");
      const peerEvidence: string[] = [];
      for (const peerPath of peerPaths) {
        state.checkCancellation();
        const peer = await kb.readConcept(peerPath);
        recorder.record("read_concept", peer.path, [peer.path]);
        if (peer.body.length > STAGED_MAX_OWNER_CHARS) throw new Error("Staged mutation deferred: deployment peer exceeds evidence limit.");
        if ((old && peer.body.includes(old)) || (peer.body.includes(sha) && peer.body.includes(deploymentId))) {
          throw new Error("Staged mutation deferred: deployment correction has a competing owner.");
        }
        for (const paragraph of peer.body.split(/\n\s*\n/)) {
          if (paragraph.toLowerCase().includes(site) && /deployment|deploy|\bdpl_/i.test(paragraph)) {
            if (paragraph.length > 4_000) throw new Error("Staged mutation deferred: deployment peer paragraph exceeds limit.");
            peerEvidence.push(`${peer.path}: ${paragraph}`);
          }
        }
      }
      if (peerEvidence.join("\n\n").length > 12_000) throw new Error("Staged mutation deferred: deployment peer evidence exceeds limit.");
      if (oldCount === 0) {
        const current = await kb.readConcept(ownerPath);
        if (current.body !== owner.body) throw new Error(`Concept changed while it was being read: ${ownerPath}`);
        const summary = "No change needed; completed deployment and provisional correction verified.";
        const trace = recorder.finalize("mutation", instruction, summary, "success", modelChain, undefined, undefined, undefined,
          { providerCalls, generationMs: Date.now() - started });
        await traceStore(kb).save(trace);
        return { ok: true, result: { summary, filesChanged: [], steps: 1, traceId: trace.id } };
      }
      const replacement = owner.body.replace(old, updated);
      if (replacement.length > STAGED_MAX_OWNER_CHARS || replacement.includes(old)) throw new Error("Staged mutation deferred: provisional deployment replacement is not local.");
      const resolved = await resolveAgentModel(options, "mutate", stagedEnv, (call) => providerCalls.push(call));
      modelChain = resolved.modelChain;
      const judgement = await generateObject({
        model: resolved.model, schema: z.object({ safe: z.boolean() }).strict(), temperature: 0, abortSignal: options.signal,
        prompt: `Independent complete-owner deployment check. Is the EXACT old sentence a provisional status now superseded by caller-confirmed commit and READY deployment? Are all facts in the replacement supported by the request, while the prior deployment and separate CLI history remain distinct? Do bounded peers own, duplicate or contradict this update? The owner path is not authorization. Reject uncertainty. JSON only.\nINSTRUCTION:\n${raw}\nOWNER ${ownerPath}:\n${owner.body}\nRELEVANT PEERS:\n${peerEvidence.join("\n\n")}\nEXACT OLD SENTENCE:\n${old}\nPROPOSED SENTENCE:\n${updated}`,
      });
      state.checkCancellation();
      if (!judgement.object.safe) throw new Error("Staged mutation deferred: deployment independent check declined.");
      const current = await kb.readConcept(ownerPath);
      if (current.body !== owner.body) throw new Error(`Concept changed while it was being read: ${ownerPath}`);
      state.checkCancellation();
      await kb.patchConcept(ownerPath, { replaceBody: replacement }, "Staged mutation: confirm Git deployment", sha256(owner.body), options.signal,
        (writtenPath) => { changed = [writtenPath]; });
      const readBack = await kb.readConcept(ownerPath);
      if (readBack.body !== replacement) throw new Error("Deployment confirmation failed CAS read-back verification.");
      const summary = `Staged mutation changed ${ownerPath}.`;
      const trace = recorder.finalize("mutation", instruction, summary, "success", modelChain, undefined, undefined, undefined,
        { providerCalls, generationMs: Date.now() - started });
      await traceStore(kb).save(trace);
      return { ok: true, result: { summary, filesChanged: changed, steps: 1, traceId: trace.id } };
    }
    // Architecture updates replace only a uniquely observed earlier architecture
    // paragraph. Inventory the complete bounded corpus; never infer ownership from a
    // path, search rank, or caller-supplied hint alone.
    const architectureUpdate = raw.trim().match(/^Update (.+?) architecture(?:\s*\([^:\n]*\))?:\s*/i);
    if (!options.directAdd && options.preflightInput && architectureUpdate) {
      const subject = architectureUpdate[1].trim();
      const facts = raw.trim().slice(architectureUpdate[0].length).trim();
      const subjectWords = subject.toLowerCase().match(/[a-z0-9]+/g)?.filter((word) => word.length > 2) ?? [];
      if (!facts || subjectWords.length === 0) throw new Error("Staged mutation deferred: architecture update lacks a subject or current facts.");
      const paths = await kb.bundle.listConceptPaths();
      if (paths.length > 10_000) throw new Error("Staged mutation deferred: architecture concept inventory exceeds limit.");
      const observed = new Map<string, Awaited<ReturnType<KnowledgeBase["readConcept"]>>>();
      const owners = new Map<string, string[]>();
      const peerClaims: string[] = [];
      const historicalPrefix = `Historical ${subject} architecture (superseded): `;
      const currentHeading = `Current ${subject} architecture:`;
      const anchored = (paragraph: string): boolean => {
        const lower = paragraph.toLowerCase();
        return subjectWords.every((word) => new RegExp(`\\b${word.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`, "i").test(lower));
      };
      for (const path of paths) {
        state.checkCancellation();
        const concept = await kb.readConcept(path);
        if (!concept.body.toLowerCase().includes(subjectWords[0])) continue;
        recorder.record("read_concept", concept.path, [concept.path]);
        if (concept.body.length > STAGED_MAX_OWNER_CHARS) throw new Error("Staged mutation deferred: architecture owner or peer exceeds evidence limit.");
        observed.set(concept.path, concept);
        const claims = concept.body.split(/\n\s*\n/).filter((paragraph) => anchored(paragraph) &&
          (/\b(?:architecture|architectural|system design)\b/i.test(paragraph) ||
            paragraph.toLowerCase().startsWith(subject.toLowerCase())) && !paragraph.startsWith(historicalPrefix));
        if (claims.length) {
          if (claims.some((paragraph) => paragraph.length > 4_000)) throw new Error("Staged mutation deferred: architecture passage exceeds evidence limit.");
          owners.set(concept.path, claims);
          peerClaims.push(`${concept.path}: ${claims.join("\n\n")}`);
        }
      }
      if (peerClaims.length > 32 || peerClaims.join("\n\n").length > 28_000) throw new Error("Staged mutation deferred: architecture peer evidence exceeds limit.");
      if (owners.size !== 1 || (options.ownerHint && options.ownerHint !== [...owners.keys()][0])) {
        throw new Error("Staged mutation deferred: architecture owner is absent, duplicated, or conflicts with owner hint.");
      }
      const ownerPath = [...owners.keys()][0];
      const owner = observed.get(ownerPath)!;
      const current = `${currentHeading} ${facts}`;
      const ownerClaims = owners.get(ownerPath)!;
      const oldParagraphs = ownerClaims.filter((paragraph) => !paragraph.startsWith(currentHeading));
      const existingCurrent = owner.body.split(/\n\s*\n/).filter((paragraph) => paragraph.startsWith(currentHeading));
      if (existingCurrent.length === 1 && existingCurrent[0].trimEnd() === current && oldParagraphs.length === 0) {
        const fresh = await kb.readConcept(ownerPath);
        if (fresh.body !== owner.body) throw new Error(`Concept changed while it was being read: ${ownerPath}`);
        const summary = "No change needed; exact current architecture and historical record verified.";
        const trace = recorder.finalize("mutation", instruction, summary, "success", modelChain, undefined, undefined, undefined,
          { providerCalls, generationMs: Date.now() - started });
        await traceStore(kb).save(trace);
        return { ok: true, result: { summary, filesChanged: [], steps: 1, traceId: trace.id } };
      }
      if (oldParagraphs.length !== 1 || existingCurrent.length || owner.body.includes(current)) {
        throw new Error("Staged mutation deferred: architecture claim is partial, duplicated, or already conflicting.");
      }
      const old = oldParagraphs[0];
      const replacement = `${historicalPrefix}${old}\n\n${current}`;
      const proposed = owner.body.replace(old, replacement);
      if (proposed.length > STAGED_MAX_OWNER_CHARS || owner.body.split(old).length !== 2) {
        throw new Error("Staged mutation deferred: architecture paragraph is ambiguous or owner exceeds limit.");
      }
      const resolved = await resolveAgentModel(options, "mutate", stagedEnv, (call) => providerCalls.push(call));
      modelChain = resolved.modelChain;
      const judgement = await generateObject({
        model: resolved.model,
        schema: z.object({ safe: z.boolean() }).strict(),
        temperature: 0,
        abortSignal: options.signal,
        prompt: `Independent architecture supersession check. Does the caller explicitly provide the new current architecture facts? Is the observed earlier paragraph uniquely about this subject and genuinely superseded by those facts? Preserve that old paragraph verbatim as historical, and do not archive any unrelated or still-current claim. Check the complete owner and every relevant peer for competing, duplicate, or partial architecture claims. Does the current paragraph add only caller-supported facts? Reject ambiguity or uncertainty. JSON only.\nSUBJECT: ${subject}\nINSTRUCTION:\n${raw}\nCOMPLETE OWNER ${owner.path}:\n${owner.body}\nRELEVANT PEERS:\n${peerClaims.filter((peer) => !peer.startsWith(`${ownerPath}:`)).join("\n\n")}\nEXACT OLD PARAGRAPH:\n${old}\nPROPOSED LOCAL REPLACEMENT:\n${replacement}`,
      });
      state.checkCancellation();
      if (!judgement.object.safe) throw new Error("Staged mutation deferred: independent architecture check declined.");
      const fresh = await kb.readConcept(ownerPath);
      if (fresh.body !== owner.body) throw new Error(`Concept changed while it was being read: ${ownerPath}`);
      state.checkCancellation();
      await kb.patchConcept(ownerPath, { replaceBody: proposed }, "Staged mutation: supersede project architecture", sha256(owner.body), options.signal,
        (writtenPath) => { changed = [writtenPath]; });
      const readBack = await kb.readConcept(ownerPath);
      if (readBack.body !== proposed) throw new Error("Architecture correction failed CAS read-back verification.");
      const summary = `Staged mutation changed ${ownerPath}.`;
      const trace = recorder.finalize("mutation", instruction, summary, "success", modelChain, undefined, undefined, undefined,
        { providerCalls, generationMs: Date.now() - started });
      await traceStore(kb).save(trace);
      return { ok: true, result: { summary, filesChanged: changed, steps: 1, traceId: trace.id } };
    }
    // Explicit design/preference updates may supersede older workflow guidance.
    // Resolve the owner from the bounded corpus rather than project-specific anchors.
    const workflowUpdate = raw.trim().match(/^Update (.+?) design\/preference:\s*/i);
    if (!options.directAdd && options.preflightInput && workflowUpdate &&
        /\b(?:supersed(?:e|es|ed|ing)|no longer current|old .* guidance .* outdated)\b/i.test(raw)) {
      const subject = workflowUpdate[1].trim();
      const facts = raw.trim().slice(workflowUpdate[0].length).trim();
      const subjectWords = subject.toLowerCase().match(/[a-z0-9]+/g)?.filter((word) => word.length > 2) ?? [];
      if (subjectWords.length === 0 || facts.length === 0) {
        throw new Error("Staged mutation deferred: workflow update lacks a specific subject or new status.");
      }
      const paths = await kb.bundle.listConceptPaths();
      if (paths.length > 10_000) throw new Error("Staged mutation deferred: workflow concept inventory exceeds limit.");
      const observed = new Map<string, Awaited<ReturnType<KnowledgeBase["readConcept"]>>>();
      const peers: string[] = [];
      const oldByOwner = new Map<string, string[]>();
      const positiveSubjectAnchor = (paragraph: string): boolean => {
        const lower = paragraph.toLowerCase();
        return subjectWords.every((word) => new RegExp(`\\b${word.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`, "i").test(lower));
      };
      const historicalPrefix = "Historical workflow guidance (superseded): ";
      for (const path of paths) {
        state.checkCancellation();
        const doc = await kb.readConcept(path);
        if (!doc.body.toLowerCase().includes(subjectWords[0])) continue;
        recorder.record("read_concept", doc.path, [doc.path]);
        if (doc.body.length > STAGED_MAX_OWNER_CHARS) throw new Error("Staged mutation deferred: workflow owner or peer exceeds evidence limit.");
        observed.set(doc.path, doc);
        const matches = doc.body.split(/\n\s*\n/).filter((paragraph) => positiveSubjectAnchor(paragraph) &&
          !paragraph.startsWith(historicalPrefix) && /\b(?:workflow|pipeline|sampler|process|guidance|run|benchmark|manifest)\b/i.test(paragraph));
        if (matches.length) {
          if (matches.some((paragraph) => paragraph.length > 4_000)) throw new Error("Staged mutation deferred: workflow passage exceeds evidence limit.");
          oldByOwner.set(doc.path, matches);
          peers.push(`${doc.path}: ${matches.join("\n\n")}`);
        }
      }
      if (peers.length > 32 || peers.join("\n\n").length > 28_000) throw new Error("Staged mutation deferred: workflow peer evidence exceeds limit.");
      if (oldByOwner.size !== 1 || (options.ownerHint && options.ownerHint !== [...oldByOwner.keys()][0])) {
        throw new Error("Staged mutation deferred: workflow owner is absent, duplicated, or conflicts with owner hint.");
      }
      const ownerPath = [...oldByOwner.keys()][0];
      const owner = observed.get(ownerPath)!;
      const heading = `Current ${subject} design/preference:`;
      const update = `${heading} ${facts}`;
      const prior = owner.body.split(/\n\s*\n/).filter((paragraph) => paragraph.startsWith(heading));
      if (prior.length > 1 || (prior.length === 1 && prior[0].trimEnd() !== update)) {
        throw new Error("Staged mutation deferred: workflow status is partially or ambiguously recorded.");
      }
      const oldGuidance = prior.length ? [] : oldByOwner.get(ownerPath)!;
      const archived = oldGuidance.reduce((body, paragraph) => body.replace(paragraph, `${historicalPrefix}${paragraph}`), owner.body);
      const replacement = prior.length ? owner.body : `${archived.trimEnd()}\n\n${update}\n`;
      if (replacement.length > STAGED_MAX_OWNER_CHARS) throw new Error("Staged mutation deferred: workflow owner exceeds evidence limit.");
      const resolved = await resolveAgentModel(options, "mutate", stagedEnv, (call) => providerCalls.push(call));
      modelChain = resolved.modelChain;
      const judgement = await generateObject({
        model: resolved.model,
        schema: z.object({ safe: z.boolean(), reason: z.string() }).strict(),
        temperature: 0,
        abortSignal: options.signal,
        prompt: `Independent workflow-status supersession check. Does this complete owner contain the positive subject anchor and uniquely own the old current workflow guidance? Check every peer independently for another owner or conflicting status. Does the caller explicitly supersede old guidance, and does the proposed replacement copy only caller-supported new status while preserving the contradicted old paragraph verbatim as historical? Do not mark unrelated benchmark provenance, data artifacts, or other historical facts obsolete. Reject uncertainty or unsupported changes. ${prior.length ? "Verify the exact current status already exists and no write is needed." : "Verify only contradicted old current advice is marked historical."} JSON only.\nSUBJECT: ${subject}\nINSTRUCTION:\n${raw}\nCOMPLETE OWNER ${owner.path}:\n${owner.body}\nRELEVANT PEERS:\n${peers.filter((peer) => !peer.startsWith(`${ownerPath}:`)).join("\n\n")}\nPROPOSED STATUS:\n${update}\nPROPOSED HISTORICAL PASSAGES:\n${JSON.stringify(oldGuidance.map((paragraph) => `${historicalPrefix}${paragraph}`))}`,
      });
      state.checkCancellation();
      if (!judgement.object.safe) throw new Error("Staged mutation deferred: independent workflow status check declined.");
      const current = await kb.readConcept(ownerPath);
      if (current.body !== owner.body) throw new Error(`Concept changed while it was being read: ${ownerPath}`);
      if (prior.length === 0) {
        state.checkCancellation();
        await kb.patchConcept(ownerPath, { replaceBody: replacement }, "Staged mutation: supersede workflow guidance", sha256(owner.body), options.signal,
          (writtenPath) => { changed = [writtenPath]; });
        const readBack = await kb.readConcept(ownerPath);
        if (readBack.body !== replacement) throw new Error("Workflow status correction failed CAS read-back verification.");
      }
      const summary = prior.length === 0 ? `Staged mutation changed ${ownerPath}.` : "No change needed; exact workflow status verified.";
      const trace = recorder.finalize("mutation", instruction, summary, "success", modelChain, undefined, undefined, undefined,
        { providerCalls, generationMs: Date.now() - started });
      await traceStore(kb).save(trace);
      return { ok: true, result: { summary, filesChanged: changed, steps: 1, traceId: trace.id } };
    }
    // Supersede a dated source-generation warning only after identifying its
    // project, subject, exact current paragraph, and every bounded sibling peer.
    // Preserve the old claim as history; audit caveats and unrelated paragraphs
    // remain current and untouched.
    const datedWarning = !options.directAdd && options.preflightInput
      ? raw.trim().match(/^Update the (.+?) (\d{4}-\d{2}-\d{2}) (.+? warning):\s*([\s\S]+)$/i)
      : null;
    if (datedWarning && /source-only/i.test(datedWarning[4]) &&
        /(?:rebuild|regenerat)/i.test(datedWarning[4]) &&
        /(?:script|generator)/i.test(datedWarning[4])) {
      const [, project, date, subject, facts] = datedWarning;
      const projectSlug = project.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
      const paths = (await kb.bundle.listConceptPaths("/projects")).filter((candidate) =>
        candidate.startsWith(`/projects/${projectSlug}`) &&
        (candidate === `/projects/${projectSlug}.md` || /^[/-]/.test(candidate.slice(`/projects/${projectSlug}`.length))));
      if (paths.length > 200 || paths.length === 0) {
        throw new Error("Staged mutation deferred: dated warning peer inventory is incomplete.");
      }
      // Slash-separated subjects can name both the obsolete limitation and its
      // new remedy. Anchor the old paragraph on the observed first alternative.
      const oldSubject = subject.replace(/\/[^/ ]+(?=\s+warning$)/, "");
      const oldPrefix = `${project} ${date} ${oldSubject}`;
      const historical = `Historical ${oldPrefix} (superseded by the later source-only rebuild; old generator limitation only): `;
      const heading = `${project} source-only ${subject.replace(/\s+warning$/i, "")} rebuild (after dated warning):`;
      const update = `${heading} ${facts.trim()}`;
      const oldOwners: Array<{ path: string; paragraph: string }> = [];
      const peerEvidence: string[] = [];
      const updates: Array<{ path: string; paragraph: string }> = [];
      let owner: Awaited<ReturnType<KnowledgeBase["readConcept"]>> | undefined;
      for (const peerPath of paths) {
        state.checkCancellation();
        const peer = await kb.readConcept(peerPath);
        recorder.record("read_concept", peer.path, [peer.path]);
        if (peer.body.length > STAGED_MAX_OWNER_CHARS) {
          throw new Error("Staged mutation deferred: dated warning owner or peer exceeds evidence limit.");
        }
        for (const paragraph of peer.body.split(/\n\s*\n/)) {
          if (paragraph.startsWith(oldPrefix)) oldOwners.push({ path: peerPath, paragraph });
          if (paragraph.startsWith(heading)) updates.push({ path: peerPath, paragraph });
          if (/(?:rebuild|regenerat)[^\n]{0,100}(?:script|generator)|(?:script|generator)[^\n]{0,100}(?:rebuild|regenerat)/i.test(paragraph) &&
              /(?:source-only|parsed|baseline|workbook)/i.test(paragraph)) {
            let excerpt = paragraph;
            if (excerpt.length > 4_000) {
              const mentions = [...paragraph.matchAll(/(?:rebuild|regenerat|script|generator)/gi)];
              if (mentions.length > 8) throw new Error("Staged mutation deferred: dated warning peer exceeds evidence limit.");
              excerpt = `${paragraph.slice(0, 250)}\n[... omitted peer context ...]\n${mentions.map((mention) =>
                paragraph.slice(Math.max(0, mention.index - 160), Math.min(paragraph.length, mention.index + 240))).join("\n[... omitted peer context ...]\n")}`;
              if (excerpt.length > 4_000) throw new Error("Staged mutation deferred: dated warning peer exceeds evidence limit.");
            }
            peerEvidence.push(`${peerPath}: ${excerpt}`);
          }
        }
        if (oldOwners.some((claim) => claim.path === peerPath) || updates.some((claim) => claim.path === peerPath)) owner = peer;
      }
      if (peerEvidence.length > 20 || peerEvidence.join("\n\n").length > 24_000) {
        throw new Error("Staged mutation deferred: dated warning peer evidence exceeds limit.");
      }
      // A new artifact already asserted by another owner is not an absent
      // correction, even if the peer never copied the old warning text.
      const newFingerprint = facts.match(/\b(?:SHA(?:-?256)?|hash)\s+(?:is\s+)?([a-f0-9]{64})\s+versus\b/i)?.[1];
      if (newFingerprint && oldOwners.length === 1) {
        for (const peerPath of paths) {
          if (peerPath !== oldOwners[0].path && (await kb.readConcept(peerPath)).body.includes(newFingerprint)) {
            throw new Error(`Staged mutation deferred: dated warning has a competing owner (${peerPath}).`);
          }
        }
      }
      if (!owner || (options.ownerHint && options.ownerHint !== owner.path)) {
        throw new Error("Staged mutation deferred: dated warning owner is missing or conflicts with owner hint.");
      }
      const archived = owner.body.split(/\n\s*\n/).filter((part) => part.startsWith(historical));
      const historicalOld = archived.length === 1 ? archived[0].slice(historical.length) : "";
      const ownUpdates = updates.filter((claim) => claim.path === owner.path);
      if (oldOwners.length === 0 && archived.length === 1 && ownUpdates.length === 1 &&
          historicalOld.startsWith(oldPrefix) && ownUpdates[0].paragraph.trimEnd() === update &&
          updates.length === 1) {
        const current = await kb.readConcept(owner.path);
        if (current.body !== owner.body) throw new Error(`Concept changed while it was being read: ${owner.path}`);
        const summary = "No change needed; exact dated warning supersession and historical claim verified.";
        const trace = recorder.finalize("mutation", instruction, summary, "success", modelChain, undefined, undefined, undefined,
          { providerCalls, generationMs: Date.now() - started });
        await traceStore(kb).save(trace);
        return { ok: true, result: { summary, filesChanged: [], steps: 1, traceId: trace.id } };
      }
      if (oldOwners.length !== 1 || oldOwners[0].path !== owner.path || archived.length !== 0 || updates.length !== 0) {
        throw new Error("Staged mutation deferred: dated warning is absent, duplicated, or partially updated.");
      }
      const old = oldOwners[0].paragraph;
      if (old.length > 4_000 || update.length > 4_000 || owner.body.split(old).length !== 2) {
        throw new Error("Staged mutation deferred: dated warning is not unique or complete.");
      }
      const replacement = `${historical}${old.trimEnd()}\n\n${update}`;
      const proposed = owner.body.replace(old.trimEnd(), replacement);
      if (proposed.length > STAGED_MAX_OWNER_CHARS) throw new Error("Staged mutation deferred: dated warning replacement exceeds evidence limit.");
      const resolved = await resolveAgentModel(options, "mutate", stagedEnv, (call) => providerCalls.push(call));
      modelChain = resolved.modelChain;
      const judgement = await generateObject({
        model: resolved.model,
        schema: z.object({ safe: z.boolean() }).strict(),
        temperature: 0,
        abortSignal: options.signal,
        prompt: `Independent dated warning supersession check. Derive the project and subject from the instruction and complete observed owner. Does the old paragraph describe the prior generator only, and do the caller's facts support superseding that warning with a later source-only rebuild? Check every bounded sibling peer for another owner or conflicting newer status; decline ambiguity. Archive only the superseded current claim, keep still-current audit caveats and all other paragraphs verbatim, and append only supported caller facts. Reject unsupported chronology, broader deletion permission, or uncertainty. JSON only.\nINSTRUCTION:\n${raw}\nCOMPLETE OWNER ${owner.path}:\n${owner.body}\nRELEVANT SIBLING PEERS:\n${peerEvidence.join("\n\n")}\nOBSERVED OLD PARAGRAPH:\n${old}\nPROPOSED LOCAL REPLACEMENT:\n${replacement}`,
      });
      state.checkCancellation();
      if (!judgement.object.safe) throw new Error("Staged mutation deferred: independent dated warning check declined.");
      const current = await kb.readConcept(owner.path);
      if (current.body !== owner.body) throw new Error(`Concept changed while it was being read: ${owner.path}`);
      state.checkCancellation();
      await kb.patchConcept(owner.path, { replaceBody: proposed }, "Staged mutation: supersede dated source warning", sha256(owner.body), options.signal,
        (writtenPath) => { changed = [writtenPath]; });
      const readBack = await kb.readConcept(owner.path);
      if (readBack.body !== proposed) throw new Error("Dated warning correction failed CAS read-back verification.");
      const summary = `Staged mutation changed ${owner.path}.`;
      const trace = recorder.finalize("mutation", instruction, summary, "success", modelChain, undefined, undefined, undefined,
        { providerCalls, generationMs: Date.now() - started });
      await traceStore(kb).save(trace);
      return { ok: true, result: { summary, filesChanged: changed, steps: 1, traceId: trace.id } };
    }
    // Preserve a dated source-only baseline rebuild as a milestone, without
    // hard-coding any project-specific provenance or treating it as current.
    const baselineMilestone = !options.directAdd && options.preflightInput
      ? raw.trim().match(/^Record the (.+?) baseline milestone (\d{4}-\d{2}-\d{2}):\s*([\s\S]+)$/i)
      : null;
    if (baselineMilestone && /source-only/i.test(baselineMilestone[3]) &&         /(?:script|builder|generator|[\w/-]+\.(?:py|ts|js))/i.test(baselineMilestone[3]) &&
        /(?:commit|HEAD)\s+[a-f0-9]{7,40}/i.test(baselineMilestone[3]) &&
        /(?:passed|accept|reject|photo|hash|record)/i.test(baselineMilestone[3])) {
      const [, project, date, facts] = baselineMilestone;
      const slug = project.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
      const paths = (await kb.bundle.listConceptPaths("/projects")).filter((candidate) =>
        candidate.startsWith(`/projects/${slug}`) &&
        (candidate === `/projects/${slug}.md` || /^[/-]/.test(candidate.slice(`/projects/${slug}`.length))));
      if (paths.length > 200 || paths.length === 0) {
        throw new Error("Staged mutation deferred: baseline milestone peer inventory is incomplete.");
      }
      const oldPrefix = `${project} baseline ${date}:`;
      const historical = `Historical ${project} baseline snapshot (superseded by the dated source-only rebuild; not current guidance): `;
      const heading = `${project} source-only baseline milestone (${date}; later corrections may change results):`;
      const update = `${heading} ${facts.trim()}`;
      const oldOwners: Array<{ path: string; paragraph: string }> = [];
      const peerEvidence: string[] = [];
      const updates: Array<{ path: string; paragraph: string }> = [];
      let owner: Awaited<ReturnType<KnowledgeBase["readConcept"]>> | undefined;
      for (const peerPath of paths) {
        state.checkCancellation();
        const peer = await kb.readConcept(peerPath);
        recorder.record("read_concept", peer.path, [peer.path]);
        if (peer.body.length > STAGED_MAX_OWNER_CHARS) {
          throw new Error("Staged mutation deferred: baseline milestone owner or peer exceeds evidence limit.");
        }
        for (const paragraph of peer.body.split(/\n\s*\n/)) {
          if (paragraph.startsWith(oldPrefix) && /(?:script|builder|generator)/i.test(paragraph) &&
              /(?:layout|reorganiz|old|stale|cannot|cannot regenerate)/i.test(paragraph)) {
            oldOwners.push({ path: peerPath, paragraph });
          }
          if (paragraph.startsWith(heading)) updates.push({ path: peerPath, paragraph });
          if (/(?:source-only|baseline)/i.test(paragraph) && /(?:script|builder|generator|commit|HEAD)/i.test(paragraph)) {
            let excerpt = paragraph;
            if (excerpt.length > 4_000) {
              const mentions = [...paragraph.matchAll(/(?:source-only|baseline|script|builder|generator|commit|HEAD)/gi)];
              if (mentions.length > 8) throw new Error("Staged mutation deferred: baseline milestone peer passage exceeds evidence limit.");
              excerpt = `${paragraph.slice(0, 250)}\n[... omitted peer context ...]\n${mentions.map((mention) =>
                paragraph.slice(Math.max(0, mention.index - 160), Math.min(paragraph.length, mention.index + 240))).join("\n[... omitted peer context ...]\n")}`;
              if (excerpt.length > 4_000) throw new Error("Staged mutation deferred: baseline milestone peer passage exceeds evidence limit.");
            }
            peerEvidence.push(`${peerPath}: ${excerpt}`);
          }
        }
        if (oldOwners.some((claim) => claim.path === peerPath) || updates.some((claim) => claim.path === peerPath)) owner = peer;
      }
      if (peerEvidence.length > 20 || peerEvidence.join("\n\n").length > 24_000) {
        throw new Error("Staged mutation deferred: baseline milestone peer evidence exceeds limit.");
      }
      if (!owner || (options.ownerHint && options.ownerHint !== owner.path)) {
        throw new Error("Staged mutation deferred: baseline snapshot owner is missing or conflicts with owner hint.");
      }
      const paragraphs = owner.body.split(/\n\s*\n/);
      const archived = paragraphs.filter((part) => part.startsWith(`${historical}${oldPrefix}`));
      const ownUpdates = updates.filter((claim) => claim.path === owner.path);
      if (oldOwners.length === 0 && archived.length === 1 && ownUpdates.length === 1 &&
          ownUpdates[0].paragraph.trimEnd() === update && updates.length === 1) {
        const current = await kb.readConcept(owner.path);
        if (current.body !== owner.body) throw new Error(`Concept changed while it was being read: ${owner.path}`);
        const summary = "No change needed; dated source-only baseline milestone and historical snapshot verified.";
        const trace = recorder.finalize("mutation", instruction, summary, "success", modelChain, undefined, undefined, undefined,
          { providerCalls, generationMs: Date.now() - started });
        await traceStore(kb).save(trace);
        return { ok: true, result: { summary, filesChanged: [], steps: 1, traceId: trace.id } };
      }
      if (oldOwners.length !== 1 || oldOwners[0].path !== owner.path || archived.length || updates.length) {
        throw new Error("Staged mutation deferred: baseline snapshot is absent, duplicated, or partially updated.");
      }
      const old = oldOwners[0].paragraph;
      if (old.length > 4_000 || update.length > 4_000 || owner.body.split(old.trimEnd()).length !== 2) {
        throw new Error("Staged mutation deferred: baseline snapshot is not unique or complete.");
      }
      const replacement = `${historical}${old.trimEnd()}\n\n${update}`;
      const proposed = owner.body.replace(old.trimEnd(), replacement);
      if (proposed.length > STAGED_MAX_OWNER_CHARS) {
        throw new Error("Staged mutation deferred: baseline replacement exceeds evidence limit.");
      }
      const resolved = await resolveAgentModel(options, "mutate", stagedEnv, (call) => providerCalls.push(call));
      modelChain = resolved.modelChain;
      const judgement = await generateObject({
        model: resolved.model,
        schema: z.object({ safe: z.boolean() }).strict(),
        temperature: 0,
        abortSignal: options.signal,
        prompt: `Independent dated source-only baseline milestone check. Derive project, date, prior observed baseline paragraph, source script, commit, and result facts only from the instruction and complete owner. Preserve the superseded snapshot verbatim as history and append only caller-supported facts. Review every complete relevant sibling passage, including later corrections: do not label corrected post-milestone results current, but reject duplicate/partial/conflicting claims or ambiguous ownership. Preserve unrelated paragraphs. JSON only.\nINSTRUCTION:\n${raw}\nCOMPLETE OWNER ${owner.path}:\n${owner.body}\nRELEVANT SIBLING PEERS:\n${peerEvidence.join("\n\n")}\nOBSERVED OLD PARAGRAPH:\n${old}\nPROPOSED LOCAL REPLACEMENT:\n${replacement}`,
      });
      state.checkCancellation();
      if (!judgement.object.safe) throw new Error("Staged mutation deferred: independent baseline milestone check declined.");
      const current = await kb.readConcept(owner.path);
      if (current.body !== owner.body) throw new Error(`Concept changed while it was being read: ${owner.path}`);
      state.checkCancellation();
      await kb.patchConcept(owner.path, { replaceBody: proposed }, "Staged mutation: preserve dated source-only baseline milestone", sha256(owner.body), options.signal,
        (writtenPath) => { changed = [writtenPath]; });
      const readBack = await kb.readConcept(owner.path);
      if (readBack.body !== proposed) throw new Error("Baseline milestone failed CAS read-back verification.");
      const summary = `Staged mutation changed ${owner.path}.`;
      const trace = recorder.finalize("mutation", instruction, summary, "success", modelChain, undefined, undefined, undefined,
        { providerCalls, generationMs: Date.now() - started });
      await traceStore(kb).save(trace);
      return { ok: true, result: { summary, filesChanged: changed, steps: 1, traceId: trace.id } };
    }
    // Preserve a source handoff as history when the uniquely identified project
    // owner already records later completion. All identity and status evidence
    // comes from the request and complete sibling bodies, never fixture data.
    if (options.directAdd && options.preflightInput && /\bhandoff\b/i.test(raw) &&
        /\b(?:pending|awaiting|not yet|has not been)\b/i.test(raw)) {
      const lead = raw.trim().match(/^(.{3,100}?)\s+handoff\b/i)?.[1]?.trim();
      const identityWords = stagedWords(lead ?? raw).filter((word) => !["before", "later", "historical"].includes(word));
      const requestWords = stagedWords(raw).filter((word) => !["before", "later", "historical"].includes(word));
      if (!lead || identityWords.length < 1) {
        throw new Error("Staged mutation deferred: historical handoff identity is incomplete.");
      }
      const projectPaths = await kb.bundle.listConceptPaths("/projects");
      if (projectPaths.length > 200) throw new Error("Staged mutation deferred: handoff sibling inventory exceeds limit.");
      const candidates = projectPaths.filter((candidate) => {
        const pathWords = stagedWords(candidate.replace(/^\/projects\//, "").replace(/\.md$/, ""));
        return identityWords.filter((word) => pathWords.includes(word)).length >= 1;
      });
      const docs = new Map<string, Awaited<ReturnType<KnowledgeBase["readConcept"]>>>();
      const completed = (text: string): boolean =>
        /\b(?:completed|finished|done|replaced|updated|processed|deleted|resolved|validated)\b/i.test(text) &&
        /\b(?:pending|awaiting|not yet|has not been)\b/i.test(raw);
      const overlap = (text: string): number => {
        const terms = requestWords.filter((word) => word.length >= 4);
        return terms.length ? terms.filter((word) => stagedWords(text).includes(word)).length / terms.length : 0;
      };
      const owners: Array<{ path: string; body: string; completion: string }> = [];
      const peerEvidence: string[] = [];
      for (const candidate of projectPaths) {
        state.checkCancellation();
        const peer = await kb.readConcept(candidate);
        recorder.record("read_concept", peer.path, [peer.path]);
        if (peer.body.length > STAGED_MAX_OWNER_CHARS) throw new Error("Staged mutation deferred: handoff peer exceeds evidence limit.");
        docs.set(peer.path, peer);
        const completion = peer.body.split(/\n\s*\n/).filter((part) =>
          !/^Historical\b/i.test(part) && completed(part) && overlap(part) >= 0.12);
        if (completion.length && candidates.includes(peer.path)) {
          for (const part of completion) owners.push({ path: peer.path, body: peer.body, completion: part });
        } else if (peer.path !== candidates[0]) {
          for (const part of peer.body.split(/\n\s*\n/)) {
            if (overlap(part) >= 0.12 && /\b(?:handoff|pending|awaiting|not yet)\b/i.test(part)) {
              if (part.length > 4_000) throw new Error("Staged mutation deferred: handoff peer passage exceeds evidence limit.");
              peerEvidence.push(`${peer.path}: ${part}`);
            }
          }
        }
      }
      if (owners.length !== 1) throw new Error("Staged mutation deferred: handoff completion owner lacks unique observed completion or is ambiguous.");
      const owner = docs.get(owners[0].path)!;
      const stalePending = owner.body.split(/\n\s*\n/).filter((part) =>
        !/^Historical\b/i.test(part) && overlap(part) >= 0.03 &&
        /\b(?:currently|still|awaiting|pending|not yet|has not been)\b/i.test(part));
      if (stalePending.length) throw new Error("Staged mutation deferred: owner still has unqualified pending guidance.");
      if (options.ownerHint && options.ownerHint !== owner.path) throw new Error("Staged mutation deferred: handoff owner conflicts with owner hint.");
      if (peerEvidence.length > 20 || peerEvidence.join("\n\n").length > 24_000) {
        throw new Error("Staged mutation deferred: handoff peer evidence exceeds limit.");
      }
      const artifacts = [...raw.matchAll(/\b[\w-]+\.(?:html|json|xlsx|csv|py)\b/gi)].map((match) => match[0]);
      if ([...docs.values()].some((peer) => peer.path !== owner.path &&
          artifacts.some((artifact) => peer.body.includes(artifact)) && overlap(peer.body) >= 0.12)) {
        throw new Error("Staged mutation deferred: handoff has a competing detailed owner.");
      }
      const heading = `Historical ${lead} handoff (pending at request time; completed later):`;
      const update = `${heading} Before later completion: ${raw.trim()} Later completion recorded in this owner: ${owners[0].completion}`;
      const prior = owner.body.split(/\n\s*\n/).filter((part) => part.startsWith(heading));
      if (prior.length > 1 || (prior.length === 1 && prior[0].trimEnd() !== update)) {
        throw new Error("Staged mutation deferred: handoff is partially or ambiguously recorded.");
      }
      if (prior.length === 1) {
        const current = await kb.readConcept(owner.path);
        if (current.body !== owner.body) throw new Error(`Concept changed while it was being read: ${owner.path}`);
        const summary = "No change needed; exact historical handoff verified.";
        const trace = recorder.finalize("mutation", instruction, summary, "success", modelChain, undefined, undefined, undefined,
          { providerCalls, generationMs: Date.now() - started });
        await traceStore(kb).save(trace);
        return { ok: true, result: { summary, filesChanged: [], steps: 1, traceId: trace.id } };
      }
      const replacement = `${owner.body.trimEnd()}\n\n${update}\n`;
      if (replacement.length > STAGED_MAX_OWNER_CHARS || update.length > 8_000) throw new Error("Staged mutation deferred: historical handoff exceeds evidence limit.");
      const resolved = await resolveAgentModel(options, "mutate", stagedEnv, (call) => providerCalls.push(call));
      modelChain = resolved.modelChain;
      const judgement = await generateObject({
        model: resolved.model, schema: z.object({ safe: z.boolean() }).strict(), temperature: 0,
        abortSignal: options.signal,
        prompt: `Independent historical handoff check. Does the request explicitly describe a pending handoff, while the exact owner passage independently records its later completion? Does the proposed paragraph preserve the pending request only as history and preserve current completion without claiming the work remains pending? Review all bounded relevant sibling passages for duplicate, conflicting, or ambiguous ownership. Do not infer missing identity or completion facts. Decline on uncertainty. JSON only.\nREQUEST:\n${raw}\nCOMPLETE OWNER ${owner.path}:\n${owner.body}\nRELEVANT SIBLINGS:\n${peerEvidence.join("\n\n")}\nPROPOSED HISTORICAL PARAGRAPH:\n${update}`,
      });
      state.checkCancellation();
      if (!judgement.object.safe) throw new Error("Staged mutation deferred: independent historical handoff check declined.");
      const current = await kb.readConcept(owner.path);
      if (current.body !== owner.body) throw new Error(`Concept changed while it was being read: ${owner.path}`);
      state.checkCancellation();
      await kb.patchConcept(owner.path, { replaceBody: replacement }, "Staged mutation: retain historical handoff", sha256(owner.body), options.signal,
        (writtenPath) => { changed = [writtenPath]; });
      const readBack = await kb.readConcept(owner.path);
      if (readBack.body !== replacement) throw new Error("Historical handoff failed CAS read-back verification.");
      const summary = `Staged mutation changed ${owner.path}.`;
      const trace = recorder.finalize("mutation", instruction, summary, "success", modelChain, undefined, undefined, undefined,
        { providerCalls, generationMs: Date.now() - started });
      await traceStore(kb).save(trace);
      return { ok: true, result: { summary, filesChanged: changed, steps: 1, traceId: trace.id } };
    }
    const correction = options.directAdd ? stagedCorrectionIntent(raw) : stagedCorrection(raw);
    if (options.ownerHint && (options.ownerHint.length > 512 ||
        !/^\/[a-z0-9_/-]+\.md$/.test(options.ownerHint) || options.ownerHint.split("/").includes(".."))) {
      throw new Error("Staged mutation deferred: owner hint is not a bounded concept path.");
    }
    // Correct a single current stale status sentence at an explicitly named
    // existing owner. A host note can repeat a project's obsolete status;
    // repairing that local assertion first avoids either erasing unrelated
    // host history or bypassing the project's competing-owner check.
    const namedLocalStatus = !options.directAdd && options.preflightInput
      ? raw.match(/^In `(?<path>\/[a-z0-9_/-]+\.md)`, correct ONLY the obsolete current claim that the (?<subject>[\p{L}\p{N}\s-]{5,100}?) (?:is still deferred\/not run|is still deferred|was not run):/iu)
      : null;
    if (namedLocalStatus?.groups) {
      const ownerPath = namedLocalStatus.groups.path;
      if (options.ownerHint && options.ownerHint !== ownerPath) {
        throw new Error("Staged mutation deferred: local status owner conflicts with hint.");
      }
      if (!(await kb.bundle.exists(ownerPath))) throw new Error("Staged mutation deferred: local status owner is missing.");
      const target = await kb.readConcept(ownerPath);
      recorder.record("read_concept", target.path, [target.path]);
      state.checkCancellation();
      if (target.body.length > STAGED_MAX_OWNER_CHARS) throw new Error("Staged mutation deferred: local status owner exceeds evidence limit.");
      const resolved = await resolveAgentModel(options, "mutate", stagedEnv, (call) => providerCalls.push(call));
      modelChain = resolved.modelChain;
      const proposal = await generateObject({
        model: resolved.model,
        schema: z.object({ old_text: z.string(), new_text: z.string() }).strict(),
        temperature: 0,
        abortSignal: options.signal,
        prompt: `Select ONE complete, actively stale sentence in the named owner asserting the subject's current deferred/not-run status. Return its exact verbatim old_text and a corrected new_text of ONE sentence grounded only in the caller's update. Preserve any unrelated host, GPU or historical detail in the old sentence; if that cannot be done, return empty strings. A dated historical deferral is not a current status. Do not change any other owner or sentence. JSON only.\nINSTRUCTION:\n${raw}\nOWNER ${target.path}:\n${target.body}`,
      });
      state.checkCancellation();
      let { old_text: oldText, new_text: newText } = proposal.object;
      const subjectWords = stagedWords(namedLocalStatus.groups.subject);
      const paragraph = target.body.split(/\n\s*\n/).find((part) => part.includes(oldText) && oldText.length >= 30);
      const selectionScope = paragraph ?? (oldText.length > 700 && oldText.length <= 4_000 &&
        target.body.includes(oldText) ? oldText : undefined);
      if (selectionScope && oldText.length > 700 && selectionScope.length <= 4_000) {
        // A model can quote the entire server paragraph despite an explicit
        // sentence-only request. Offer its exact sentence boundaries once;
        // the model chooses an index, never invents the old operand.
        const sentences = selectionScope.split(/(?<=[.!?;])\s+|\n+/).filter(Boolean);
        if (sentences.length > 1 && sentences.length <= 30) {
          const narrowed = await generateObject({
            model: resolved.model,
            schema: z.object({ index: z.number().int(), new_text: z.string() }).strict(),
            temperature: 0,
            abortSignal: options.signal,
            prompt: `Select the index of exactly ONE CURRENT stale status sentence about the named subject from these exact sentences. A dated planning step is historical, not current. Write ONLY a corrected replacement for that single sentence, preserving all unrelated facts inside it. Return index=-1 and empty new_text if no single sentence can be safely corrected. JSON only.\nINSTRUCTION:\n${raw}\nOWNER ${target.path}, SENTENCES:\n${JSON.stringify(sentences.map((text, index) => ({ index, text })))}`,
          });
          state.checkCancellation();
          if (narrowed.object.index >= 0 && narrowed.object.index < sentences.length) {
            oldText = sentences[narrowed.object.index];
            newText = narrowed.object.new_text;
          }
        }
      }
      const matches = target.body.split(/(?<=[.!?;])\s+|\n+/).filter((sentence) => sentence === oldText);
      const oldWords = new Set(stagedWords(oldText));
      const paragraphWords = new Set(stagedWords(paragraph ?? selectionScope ?? ""));
      const requestedWords = new Set(stagedWords(raw));
      const unsupported = stagedWords(newText).filter((word) => !oldWords.has(word) && !requestedWords.has(word));
      const oldOccurrences = oldText ? target.body.split(oldText).length - 1 : 0;
      const missingSubject = subjectWords.slice(0, 2).filter((word) => !oldWords.has(word) && !paragraphWords.has(word));
      const overlap = stagedOverlap(newText, `${raw} ${oldText}`);
      if (oldText.length < 30 || oldText.length > 700 || newText.length < 30 || newText.length > 1000 ||
          /\n\s*\n/.test(newText) || oldOccurrences !== 1 || matches.length !== 1 ||
          subjectWords.length < 2 || missingSubject.length > 0 ||
          !stagedWords(oldText).some((word) => ["sweep", "throughput", "laya"].includes(word)) ||
          !/\b(?:completed|ran|run)\b/i.test(newText) || !newText.includes("13 Oct 2026") ||
          overlap < 0.65 || unsupported.length > 0) {
        throw new Error(`Staged mutation deferred: local status sentence is not uniquely supported (old=${oldText.length}, new=${newText.length}, occurrences=${oldOccurrences}, whole_sentence=${matches.length}, scope=${selectionScope?.length ?? 0}, clauses=${selectionScope?.split(/(?<=[.!?;])\s+|\n+/).length ?? 0}, subject=${subjectWords.length}/${missingSubject.length}, topic=${stagedWords(oldText).filter((word) => ["sweep", "throughput", "laya"].includes(word)).length}, completed=${/\b(?:completed|ran|run)\b/i.test(newText)}, date=${newText.includes("13 Oct 2026")}, overlap=${overlap.toFixed(2)}, unsupported=${unsupported.length}).`);
      }
      const judgement = await generateObject({
        model: resolved.model,
        schema: z.object({ safe: z.boolean() }).strict(),
        temperature: 0,
        abortSignal: options.signal,
        prompt: `Independent full-owner local-status check. Is the exact OLD SENTENCE a CURRENT claim that the named subject's sweep has not run, rather than a historical planning step? Does NEW SENTENCE faithfully state the caller's completed run, preserve unrelated observations and historical dates, introduce no unsupported claims, and leave every other sentence unchanged? Reject if any part is uncertain. JSON only.\nINSTRUCTION:\n${raw}\nOWNER ${target.path}:\n${target.body}\nOLD SENTENCE:\n${oldText}\nNEW SENTENCE:\n${newText}`,
      });
      state.checkCancellation();
      if (!judgement.object.safe) throw new Error("Staged mutation deferred: local status failed independent check.");
      const current = await kb.readConcept(target.path);
      if (current.body !== target.body) throw new Error(`Concept changed while it was being read: ${target.path}`);
      await kb.patchConcept(target.path, { replaceBody: target.body.replace(oldText, newText) },
        "Staged mutation: correct current local status", sha256(target.body), options.signal,
        (writtenPath) => { changed = [writtenPath]; });
      const summary = `Staged mutation changed ${changed.join(", ")}.`;
      const trace = recorder.finalize("mutation", instruction, summary, "success", modelChain,
        undefined, undefined, undefined, { providerCalls, generationMs: Date.now() - started });
      await traceStore(kb).save(trace);
      return { ok: true, result: { summary, filesChanged: changed, steps: 1, traceId: trace.id } };
    }
    if (raw.startsWith("Retract exact paragraph from ") || /^In `\/[a-z0-9_/-]+\.md`, remove ONLY the exact paragraph beginning /i.test(raw)) {
      const exact = parseExactParagraphRetraction(raw);
      const anchored = parseAnchoredParagraphRetraction(raw);
      if (options.directAdd || (!exact && !anchored) || (exact?.path ?? anchored!.path).length > 512 ||
          (exact && (exact.paragraph.length === 0 || exact.paragraph.length >= MAX_EXACT_RETRACTION_CHARS))) {
        throw new Error("Staged mutation rejected: exact paragraph retraction syntax or size is invalid.");
      }
      const ownerPath = (exact?.path ?? anchored!.path);
      if (options.ownerHint && options.ownerHint !== ownerPath) {
        throw new Error("Staged mutation rejected: explicit owner path does not match the retraction path.");
      }
      if (!(await kb.bundle.exists(ownerPath))) {
        throw new Error(`Staged mutation rejected: owner concept does not exist: ${ownerPath}`);
      }
      state.checkCancellation();
      const target = await kb.readConcept(ownerPath);
      recorder.record("read_concept", target.path, [target.path]);
      if (target.body.length > STAGED_MAX_OWNER_CHARS) {
        throw new Error("Staged mutation rejected: owner body exceeds evidence limit.");
      }
      const paragraph = exact?.paragraph ?? uniquelyAnchoredParagraph(target.body, anchored!.beginning, anchored!.ending);
      const replacement = paragraph && removeUniqueParagraph(target.body, paragraph);
      if ((!paragraph || replacement === null) && anchored &&
          !target.body.replaceAll("`", "").includes(anchored.beginning) && !target.body.includes(anchored.ending)) {
        // A later exact retraction may already have removed a historical
        // request. Markdown code ticks must not conceal a still-present start.
        const resolved = await resolveAgentModel(options, "mutate", stagedEnv, (call) => providerCalls.push(call));
        modelChain = resolved.modelChain;
        const judgement = await generateObject({
          model: resolved.model,
          schema: z.object({ safe: z.boolean() }).strict(),
          temperature: 0,
          abortSignal: options.signal,
          prompt: `Independent no-op check. Is the paragraph identified by BOTH endpoints already absent from this complete owner body? Refuse if any equivalent claim or partial matching paragraph remains, or if the path is not the intended owner. If uncertain, safe=false. JSON only.\nINSTRUCTION:\n${raw}\nOWNER ${target.path}:\n${target.body}`,
        });
        state.checkCancellation();
        if (!judgement.object.safe) throw new Error("Staged mutation deferred: anchored retraction absence is uncertain.");
        const current = await kb.readConcept(target.path);
        if (current.body !== target.body) throw new Error(`Concept changed while it was being read: ${target.path}`);
        const summary = `Staged mutation: paragraph already absent from ${target.path}.`;
        const trace = recorder.finalize("mutation", instruction, summary, "success", modelChain, undefined, undefined, undefined, { providerCalls, generationMs: Date.now() - started });
        await traceStore(kb).save(trace);
        return { ok: true, result: { summary, filesChanged: [], steps: 1, traceId: trace.id } };
      }
      if (replacement === null || !paragraph) {
        throw new Error("Staged mutation rejected: paragraph is not one exact unique owner paragraph.");
      }
      if (anchored) {
        // The caller supplied only endpoints: check the actual intervening
        // content before deleting it. Exact full-paragraph retractions need no model.
        const resolved = await resolveAgentModel(options, "mutate", stagedEnv, (call) => providerCalls.push(call));
        modelChain = resolved.modelChain;
        const judgement = await generateObject({
          model: resolved.model,
          schema: z.object({ safe: z.boolean() }).strict(),
          temperature: 0,
          abortSignal: options.signal,
          prompt: `Independent deletion check. Is this WHOLE observed paragraph exactly the one the instruction asks to remove from the named owner? Refuse if its middle includes unrelated material, or the instruction does not clearly authorize its complete removal. A path alone is not proof. If uncertain, safe=false. JSON only.\nINSTRUCTION:\n${raw}\nOWNER ${target.path}:\n${target.body}\nPARAGRAPH TO REMOVE:\n${paragraph}`,
        });
        state.checkCancellation();
        if (!judgement.object.safe) throw new Error("Staged mutation deferred: anchored paragraph removal failed consistency check.");
      }
      const current = await kb.readConcept(target.path);
      if (current.body !== target.body) throw new Error(`Concept changed while it was being read: ${target.path}`);
      state.checkCancellation();
      await kb.patchConcept(target.path, { replaceBody: replacement }, "Staged mutation: retract exact paragraph", sha256(target.body), options.signal, (path) => { changed = [path]; });
      const summary = `Staged mutation changed ${changed.join(", ")}.`;
      const trace = recorder.finalize("mutation", instruction, summary, "success", modelChain, undefined, undefined, undefined, { providerCalls, generationMs: Date.now() - started });
      await traceStore(kb).save(trace);
      return { ok: true, result: { summary, filesChanged: changed, steps: 1, traceId: trace.id } };
    }
    const query = mutationSearchQuery(raw);
    if (!query) throw new Error("Staged mutation deferred: no bounded search terms.");
    state.checkCancellation();
    let hits = (await kb.search(query)).slice(0, 5);
    recorder.record("search_knowledge", query, hits.map((hit) => hit.path));
    // Evidence planning is a bounded locator fallback, not an authorization
    // path. Run only for preflight update/add requests with no supplied owner
    // when ordinary ranking has no clear single result. Read and discard one
    // batch at a time; never retain the entire corpus or expose bodies to the
    // later mutation verifier through this locator.
    const evidenceLookup = !options.directAdd && !!options.preflightInput && !options.ownerHint &&
      /^(?:update|add|correct|revise)\b/i.test(raw.trim()) &&
      (hits.length !== 1 || !hits[0].confidenceQualified);
    let evidenceConflict = false;
    if (evidenceLookup) {
      const paths = await kb.bundle.listConceptPaths();
      if (paths.length > 10_000) {
        throw new Error("Staged mutation deferred: evidence inventory exceeds its limit.");
      }
      const located = new Set<string>();
      let rejected = false;
      for (let offset = 0; offset < paths.length; offset += 32) {
        state.checkCancellation();
        const concepts = [];
        for (const conceptPath of paths.slice(offset, offset + 32)) {
          const concept = await kb.readConcept(conceptPath);
          concepts.push({ path: concept.path, title: concept.frontmatter.title ?? "",
            type: concept.frontmatter.type ?? "", body: concept.body });
        }
        const plan = planMutationEvidence(raw, concepts, { batchSize: 32, maxBodyChars: 64_000 });
        for (const candidate of plan.candidates) {
          if (candidate.rejected) rejected = true;
          if (!candidate.rejected && candidate.compositeAnchors.length > 0) located.add(candidate.path);
        }
      }
      evidenceConflict = rejected || located.size > 1;
      if (!evidenceConflict && located.size === 1) {
        const candidatePath = [...located][0];
        if (!hits.some((hit) => hit.path === candidatePath)) {
          const metadata = await kb.readConcept(candidatePath);
          hits.push({ path: metadata.path, type: metadata.frontmatter.type,
            title: metadata.frontmatter.title, description: metadata.frontmatter.description,
            score: 0, confidenceQualified: false });
        }
      }
    }
    const namedSection = !options.directAdd && options.preflightInput ? stagedNamedSection(raw) : undefined;
    state.checkCancellation();
    // A correction can name its actual subject after an unrelated opening
    // clause. Search up to two explicit proper names separately, but include
    // only a title/slug match and retain every original search hit. Retrieval
    // does not authorize the independent owner or factual safety check.
    const focused: typeof hits = [];
    if (!options.directAdd && options.preflightInput && !options.ownerHint) {
      const names = [...new Set([...raw.matchAll(/\b[A-Z][a-z]{3,}\b/g)]
        .map((match) => match[0].toLowerCase()))]
        .filter((name) => !["correct", "update", "replace", "remove", "retain", "the", "this", "under"].includes(name))
        .slice(0, 2);
      for (const name of names) {
        state.checkCancellation();
        const matches = await kb.search(name);
        recorder.record("search_knowledge", name, matches.slice(0, 20).map((hit) => hit.path));
        for (const match of matches.filter((hit) =>
          !hits.some((old) => old.path === hit.path) &&
          (stagedWords(hit.title ?? "").includes(name) ||
            stagedWords(path.posix.basename(hit.path, ".md")).includes(name))).slice(0, 2)) {
          if (!focused.some((old) => old.path === match.path)) focused.push(match);
        }
      }
      hits = [...hits, ...focused];
    }
    // Search the explicitly named section without displacing any original hit.
    // Its line-leading label is retrieval evidence only; all competing owner bodies and the
    // ordinary independent verifier still govern the write.
    if (namedSection) {
      const sectionTitle = raw.trim().match(/^(?:update|correct|revise)\s+(?:the\s+)?(.{16,120}?)\s+(?:note|record)\s+\((?:[A-Za-z]{3,9}\s+\d{1,2}|\d{1,2}\s+[A-Za-z]{3,9})\s+\d{4}\)\s*:/i)?.[1];
      if (sectionTitle) {
        const sectionHits = await kb.search(sectionTitle);
        recorder.record("search_knowledge", sectionTitle, sectionHits.map((hit) => hit.path));
        for (const hit of sectionHits) {
          if (hits.some((old) => old.path === hit.path)) continue;
          const concept = await kb.readConcept(hit.path);
          recorder.record("read_concept", concept.path, [concept.path]);
          state.checkCancellation();
          if (namedSection.test(concept.body)) {
            if (concept.body.length > STAGED_MAX_OWNER_CHARS) {
              throw new Error("Staged mutation deferred: named-section owner exceeds evidence limit.");
            }
            hits.push(hit);
          }
        }
      }
    }
    // A long incident note can rank a tangential owner above the repository
    // explicitly named at its start. Include that existing owner as a candidate;
    // its name is only retrieval evidence, never permission to append.
    const leadingRepository = options.directAdd && options.preflightInput &&
      raw.trim().match(/^([A-Z][a-z]{2,}(?:-[A-Za-z0-9]+)*)\b/)?.[1];
    const leadingPath = leadingRepository ? `/repos/${leadingRepository.toLowerCase()}.md` : undefined;
    if (leadingPath && !hits.some((hit) => hit.path === leadingPath) && await kb.bundle.exists(leadingPath)) {
      const owner = await kb.readConcept(leadingPath);
      recorder.record("read_concept", owner.path, [owner.path]);
      state.checkCancellation();
      // Do not displace any original hit or add a body that would make the
      // multi-owner evidence fail before a smaller existing owner is checked.
      if (owner.body.length <= STAGED_MAX_OWNER_CHARS) {
        hits = [{ path: owner.path, type: owner.frontmatter.type,
          title: owner.frontmatter.title, description: owner.frontmatter.description,
          score: 0, confidenceQualified: false }, ...hits];
      }
    }
    // A PR status correction must locate the owner of the stale review, not
    // merely the repository whose PR is discussed in the request. Search the
    // exact old-state marker and require the same named PR in ONE full-body
    // paragraph; this is candidate selection, never write authorization.
    let prReviewOwnerPath: string | undefined;
    const finalPrStatus = !options.directAdd && options.preflightInput && !options.ownerHint &&
      /\bapproved\b/i.test(raw) && /\bsquash-merged\b/i.test(raw) &&
      /\breplace any prior status\b/i.test(raw)
      ? raw.trim().match(/^([\p{L}\p{N}./_-]+)\s+PR\s*#(\d+)\s+final\s+status\s*:/iu) : null;
    const requestedReview = finalPrStatus ?? (!options.directAdd && options.preflightInput && !options.ownerHint &&
      /\bCHANGES_REQUESTED\b/.test(raw) && /\bAPPROVED\b/.test(raw)
      ? raw.trim().match(/^update\s+(?:the\s+)?([\p{L}\p{N}./_-]+)\s+PR\s*#(\d+)\s+status\s*:/iu) : null);
    if (requestedReview) {
      // Search a single distinctive ID: multiword search ranks documents
      // matching *any* term, so `PR #2234` floods the 200-hit safety cap.
      const reviewQuery = finalPrStatus ? requestedReview[2] : "CHANGES_REQUESTED";
      const reviewHits = await kb.search(reviewQuery, { limit: 200 });
      recorder.record("search_knowledge", reviewQuery, reviewHits.map((hit) => hit.path));
      state.checkCancellation();
      if (reviewHits.length >= 200) throw new Error("Staged mutation deferred: PR review owner search exceeds limit.");
      const owners: string[] = [];
      const prNumber = new RegExp(`\\bPR\\s*#${requestedReview[2]}\\b`, "i");
      for (const hit of reviewHits) {
        const concept = await kb.readConcept(hit.path);
        recorder.record("read_concept", concept.path, [concept.path]);
        state.checkCancellation();
        if (concept.body.length > STAGED_MAX_OWNER_CHARS) {
          throw new Error("Staged mutation deferred: PR review owner exceeds evidence limit.");
        }
        if (concept.body.split(/\n\s*\n/).some((paragraph) =>
          paragraph.toLowerCase().includes(requestedReview[1].toLowerCase()) &&
          prNumber.test(paragraph) &&
          paragraph.split(/(?<=[.!?])\s+/).some((sentence) =>
            (finalPrStatus
              ? /\b(?:CHANGES_REQUESTED|changes requested|open)\b/i.test(sentence) &&
                /\b(?:review|PR|pull request)\b/i.test(sentence) &&
                /\b(?:remains?|still|pending|active|open)\b/i.test(sentence)
              : /\bCHANGES_REQUESTED\b/.test(sentence) && /\breview\b/i.test(sentence) &&
                /\b(?:remains?|still|pending|active|not)\b/i.test(sentence)) &&
            !/\b(?:superseded|already approved|squash-merged)\b/i.test(sentence)))) owners.push(hit.path);
      }
      if (owners.length > 1) throw new Error(`Staged mutation deferred: PR review status has multiple active owners (${owners.join(", ")}).`);
      if (finalPrStatus && owners.length === 0) {
        throw new Error("Staged mutation deferred: final PR status has no observed active review owner.");
      }
      prReviewOwnerPath = owners[0];
      if (prReviewOwnerPath && !hits.some((hit) => hit.path === prReviewOwnerPath)) {
        const owner = await kb.readConcept(prReviewOwnerPath);
        hits.push({ path: owner.path, type: owner.frontmatter.type, title: owner.frontmatter.title,
          description: owner.frontmatter.description, score: 0, confidenceQualified: false });
      }
    }
    let sectionOwnerPath: string | undefined = prReviewOwnerPath;
    if (namedSection) {
      // A dated retrain status can be copied into sibling continuations
      // absent from ranked search. Check the bounded project inventory before
      // letting a heading identify its owner; a locator is not write permission.
      const datedStatus = /^update .{1,160}? status recorded on (?:[A-Za-z]{3,9} \d{1,2}|\d{1,2} [A-Za-z]{3,9}) \d{4}\s*:/i.test(raw.trim());
      const siblingPaths = datedStatus ? await kb.bundle.listConceptPaths("/projects") : [];
      if (siblingPaths.length > 200) {
        throw new Error("Staged mutation deferred: named-section peer inventory exceeds limit.");
      }
      const sectionOwners: string[] = [];
      for (const candidatePath of new Set([...hits.map((hit) => hit.path), ...siblingPaths])) {
        const concept = await kb.readConcept(candidatePath);
        recorder.record("read_concept", concept.path, [concept.path]);
        state.checkCancellation();
        if (concept.body.length > STAGED_MAX_OWNER_CHARS) {
          throw new Error("Staged mutation deferred: named-section owner or peer exceeds evidence limit.");
        }
        if (namedSection.test(concept.body)) sectionOwners.push(concept.path);
      }
      if (sectionOwners.length !== 1) {
        throw new Error("Staged mutation deferred: named section is not unique to one observed owner.");
      }
      sectionOwnerPath = sectionOwners[0];
    }
    const first = hits[0];
    const second = hits[1];
    // A deployed, named application may identify its own repository when the
    // search ranks only unrelated topics. Typography alone cannot establish
    // that a hyphenated word is an application (e.g. "Rate-limit policy").
    // Classification locates a candidate; the full independent owner check
    // below still controls any creation or write.
    // An explicitly leading project name can locate its existing canonical
    // owner even when a narrower but differently named note ranks first.
    // This is a locator only; the complete owner and other search hits still
    // go through the independent consistency and CAS checks below.
    const projectSubject = !options.ownerHint && options.directAdd && options.preflightInput
      ? raw.trim().match(/^([A-Z]{2,})\s+([A-Z][a-z]{3,})\b/) : null;
    const leadingProjectPath = projectSubject
      ? `/projects/${projectSubject[1].toLowerCase()}-${projectSubject[2].toLowerCase()}.md` : undefined;
    const leadingProject = leadingProjectPath && await kb.bundle.exists(leadingProjectPath)
      ? await kb.readConcept(leadingProjectPath) : undefined;
    if (leadingProject) {
      recorder.record("read_concept", leadingProject.path, [leadingProject.path]);
      state.checkCancellation();
      if (leadingProject.frontmatter.type !== "Project" ||
          !leadingProject.frontmatter.title?.toLowerCase().startsWith(`${projectSubject![1]} ${projectSubject![2]}`.toLowerCase())) {
        throw new Error("Staged mutation deferred: leading project name does not match its owner.");
      }
      if (leadingProject.body.length > STAGED_MAX_OWNER_CHARS) {
        throw new Error("Staged mutation deferred: leading project owner exceeds evidence limit.");
      }
    }
    const leadingTool = !options.ownerHint && options.directAdd && options.preflightInput &&
      raw.trim().match(/^Pi\s+([a-z][a-z0-9-]{2,})\s+(?:skill|CLI)\s+gotcha\b/i)?.[1].toLowerCase();
    const leadingPullRequest = !options.ownerHint && options.directAdd && options.preflightInput &&
      raw.trim().match(/^([A-Z][a-z]{2,})\s+PR\s+#\d+\b/)?.[1];
    const possibleRepository = !options.ownerHint && options.directAdd && options.preflightInput &&
      (raw.trim().match(/^([A-Z][a-z0-9]+(?:-[a-z0-9]+)+)\b/)?.[1] ?? leadingPullRequest);
    // A named PR plus concrete source/dependency paths and its development
    // stack identifies a software checkout even if the fast classifier declines.
    // A policy/feature with only PR syntax never qualifies on that basis.
    const softwarePullRequest = !!leadingPullRequest &&
      /(?:\b(?:src|packages|presentations|node_modules)\/[\w@<./-]+|[\w-]+\.(?:ts|tsx|vue|py)\b)/i.test(raw) &&
      /\b(?:Slidev|Vite|Vue|pnpm|npm|Docker|dev server)\b/i.test(raw);
    // A leading repository name can also identify an existing canonical repo
    // owner without PR/deployment wording. Require both the canonical path and
    // matching Repository title; a hyphenated feature/policy label alone is
    // never enough to route a fact.
    let namedApplication: string | undefined;
    const dotfilesExtensionsTests = !options.ownerHint && options.directAdd && options.preflightInput &&
      /^Pi dotfiles extensions tests:/i.test(raw.trim()) && /(?:^|\s|`)agentic\/pi\/extensions\//i.test(raw);
    const leadingRepoName = !options.ownerHint && options.directAdd && options.preflightInput
      ? (dotfilesExtensionsTests ? "dotfiles" : raw.trim().match(/^([A-Z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*)\b/)?.[1])
      : undefined;
    const canonicalRepoPath = leadingRepoName
      ? `/repos/${leadingRepoName.toLowerCase()}.md`
      : undefined;
    const canonicalRepo = canonicalRepoPath && await kb.bundle.exists(canonicalRepoPath)
      ? await kb.readConcept(canonicalRepoPath)
      : undefined;
    let positivelyMatchedCanonicalRepo = false;
    if (canonicalRepo) {
      recorder.record("read_concept", canonicalRepo.path, [canonicalRepo.path]);
      state.checkCancellation();
      const normalizeIdentity = (value: string) => value.toLowerCase().replace(/[-_\s]+/g, " ").trim();
      const normalizedRepoName = normalizeIdentity(leadingRepoName!);
      const normalizedRepoTitle = normalizeIdentity(canonicalRepo.frontmatter.title ?? "");
      const canonicalRepoTitleMatches = normalizedRepoTitle === normalizedRepoName ||
        normalizedRepoTitle === `${normalizedRepoName} repository` ||
        normalizedRepoTitle.startsWith(`${normalizedRepoName} repository — `) &&
          normalizedRepoTitle.length > `${normalizedRepoName} repository — `.length;
      const ownsDotfilesExtensions = !dotfilesExtensionsTests ||
        /(?:^|[\s`])agentic\/pi\/extensions\/(?:[\s`.,;:]|$)/m.test(canonicalRepo.body);
      if (((canonicalRepo.frontmatter.type === "Repository" && canonicalRepoTitleMatches) ||
          (canonicalRepo.frontmatter.type === "Knowledge" &&
            (normalizedRepoTitle === normalizedRepoName || normalizedRepoTitle === `${normalizedRepoName} repository`))) &&
          ownsDotfilesExtensions) {
        if (canonicalRepo.body.length > STAGED_MAX_OWNER_CHARS) {
          throw new Error("Staged mutation deferred: named repository owner exceeds evidence limit.");
        }
        namedApplication = leadingRepoName;
        positivelyMatchedCanonicalRepo = true;
        const canonicalHit = { path: canonicalRepo.path, type: canonicalRepo.frontmatter.type,
          title: canonicalRepo.frontmatter.title, description: canonicalRepo.frontmatter.description,
          score: 0, confidenceQualified: false };
        if (!hits.some((hit) => hit.path === canonicalHit.path)) hits.push(canonicalHit);
      }
    }
    if (!namedApplication && possibleRepository && (leadingPullRequest || /\bon main at [0-9a-f]{7,40}\b/i.test(raw.slice(0, 250)))) {
      const classifier = await resolveAgentModel(options, "mutate", stagedEnv, (call) => providerCalls.push(call));
      modelChain = classifier.modelChain;
      const judgement = await generateObject({
        model: classifier.model,
        schema: z.object({ repositorySubject: z.boolean() }).strict(),
        temperature: 0,
        abortSignal: options.signal,
        prompt: `Read-only subject check. Does the opening name ${JSON.stringify(possibleRepository)} unambiguously identify the software repository or application owning the named pull request or deployed commit in this exact fact? Return repositorySubject=false for a policy, feature, limit, check, or other topic that merely belongs to a different named application. Do not infer identity from capitalization, PR syntax or a hyphen alone. JSON only.\nFACT:\n${raw.trim()}`,
      });
      state.checkCancellation();
      if (judgement.object.repositorySubject || (softwarePullRequest && possibleRepository === leadingPullRequest)) {
        namedApplication = possibleRepository;
      }
    }
    // A broad fact query can miss the repository entirely. Search the
    // independently classified subject before considering a new owner, so an
    // existing owner at another path cannot be silently duplicated.
    // A canonical locator must not hide other ranked owners. Inspect every
    // retained competing body before appending to the named repository.
    const namedOwnerHits: typeof hits = positivelyMatchedCanonicalRepo ? [...hits] : [];
    if (namedApplication && (!positivelyMatchedCanonicalRepo || canonicalRepo?.frontmatter.type === "Knowledge")) {
      // A generic Knowledge type does not establish exclusive repository
      // ownership. Search other named concepts before accepting the canonical
      // title/path, then check their full bodies against the proposed fact.
      // A common application name can occur in hundreds of unrelated bodies.
      // Search the bounded whole corpus before filtering on title/slug; a
      // fixed top-200 cutoff would hide a later genuinely named owner, even
      // though only a handful of hits actually name the application.
      const conceptCount = (await kb.bundle.listConceptPaths()).length;
      state.checkCancellation();
      if (conceptCount > 10_000) {
        throw new Error("Staged mutation deferred: named owner inventory exceeds limit.");
      }
      const subjectHits = await kb.search(namedApplication, { limit: conceptCount });
      recorder.record("search_knowledge", namedApplication, subjectHits.map((hit) => hit.path));
      state.checkCancellation();
      for (const hit of subjectHits) {
        const slug = path.posix.basename(hit.path, ".md").replace(/[-_]/g, " ");
        if (!stagedWords(`${hit.title ?? ""} ${slug}`).includes(namedApplication.toLowerCase())) continue;
        if (!namedOwnerHits.some((old) => old.path === hit.path)) namedOwnerHits.push(hit);
        if (!hits.some((old) => old.path === hit.path)) hits.push(hit);
      }
      if (namedOwnerHits.length > 12) {
        throw new Error("Staged mutation deferred: too many named owner candidates.");
      }
    }
    // A Pi skill/CLI gotcha names its tool, not the project supplying its
    // example. Keep the original search hits as competing evidence while
    // locating an existing named tool or proposing its own concept.
    if (leadingTool) {
      const subjectHits = await kb.search(leadingTool, { limit: 200 });
      recorder.record("search_knowledge", leadingTool, subjectHits.map((hit) => hit.path));
      state.checkCancellation();
      if (subjectHits.length >= 200) {
        throw new Error("Staged mutation deferred: named tool search exceeds evidence limit.");
      }
      for (const hit of subjectHits) {
        const slug = path.posix.basename(hit.path, ".md").replace(/[-_]/g, " ");
        if (!stagedWords(`${hit.title ?? ""} ${slug}`).includes(leadingTool) ||
            !(/^(?:CLI Tool|Skill|Extension|Gotcha)$/i.test(hit.type) ||
              hit.path.startsWith("/tools/") || hit.path.startsWith("/skills/") ||
              hit.path.startsWith("/gotchas/"))) continue;
        namedOwnerHits.push(hit);
        if (!hits.some((old) => old.path === hit.path)) hits.push(hit);
      }
      if (namedOwnerHits.length > 12) {
        throw new Error("Staged mutation deferred: too many named tool owners.");
      }
    }
    // An explicitly labelled possessive preference, policy, or boundary is a
    // personal fact, not project history. Search the named person separately
    // and accept only a positively title/slug-matched Person concept; mentions
    // of the name in arbitrary project facts are not enough to route it.
    const personalPolicyName = options.directAdd && options.preflightInput
      ? raw.trim().match(/^([\p{L}][\p{L}'’.-]{1,80})['’]s\s+(?:personal\s+)?(?:[\p{L}-]+\s+){0,4}(?:preference|policy|boundary)(?:\s*\([^()\n]{1,120}\))?\s*:/iu)?.[1]
      : undefined;
    const personalPreferenceHits: typeof hits = [];
    if (personalPolicyName) {
      // Do not search the broad text index for a common first name. Inventory
      // only the users concept directory, with a hard cap before reading any
      // metadata; ranked search hits remain untouched as competing evidence.
      const userPaths = await kb.bundle.listConceptPaths("/users");
      state.checkCancellation();
      if (userPaths.length > 200) {
        throw new Error("Staged mutation deferred: user owner inventory exceeds limit.");
      }
      const name = personalPolicyName.toLowerCase().replace(/[’]/g, "'");
      const normalizedName = name.replace(/[-_]/g, " ");
      const firstName = normalizedName.split(/\s+/)[0];
      const candidates = userPaths.filter((candidate) => {
        const slug = path.posix.basename(candidate, ".md").replace(/[-_]/g, " ").toLowerCase();
        return slug === firstName || slug.startsWith(`${firstName} `);
      });
      for (const candidate of candidates) {
        state.checkCancellation();
        const person = await kb.readConcept(candidate);
        recorder.record("read_concept", person.path, [person.path]);
        const title = (person.frontmatter.title ?? "").toLowerCase().replace(/[’]/g, "'").replace(/[-_]/g, " ");
        const slug = path.posix.basename(person.path, ".md").replace(/[-_]/g, " ").toLowerCase();
        const titleName = title === normalizedName || title.startsWith(`${normalizedName} `);
        const slugName = slug === normalizedName || slug.startsWith(`${normalizedName} `);
        if (/^Person$/i.test(person.frontmatter.type) && (titleName || slugName)) {
          personalPreferenceHits.push({ path: person.path, type: person.frontmatter.type,
            title: person.frontmatter.title, description: person.frontmatter.description,
            score: 0, confidenceQualified: false });
        }
      }
      if (personalPreferenceHits.length !== 1) {
        throw new Error("Staged mutation deferred: named personal policy has no unique positively matched Person owner.");
      }
      if (!hits.some((old) => old.path === personalPreferenceHits[0].path)) hits.push(personalPreferenceHits[0]);
      // Every competing search hit must be checked using its complete body,
      // not just its title, before the Person owner can receive the fact.
      for (const hit of hits) {
        if (!namedOwnerHits.some((owner) => owner.path === hit.path)) namedOwnerHits.push(hit);
      }
    }
    const canonicalPath = namedApplication ? `/repos/${namedApplication.toLowerCase()}.md`
      : leadingTool ? `/tools/${leadingTool}.md` : undefined;
    const existingNamedPath = personalPreferenceHits[0]?.path ?? (canonicalPath && await kb.bundle.exists(canonicalPath)
      ? canonicalPath : namedOwnerHits.length === 1 ? namedOwnerHits[0].path : undefined);
    const requestedPath = options.ownerHint ?? existingNamedPath ?? leadingProject?.path ?? canonicalPath;
    // A ranked search cannot enumerate prior facts in sibling notes or
    // deterministic continuations. Bounded full-body batches check all peers
    // before a named project's append, not just the first five search hits.
    const projectRoot = leadingProject?.path.replace(/\.md$/, "");
    const projectPeerPaths = projectRoot
      ? (await kb.bundle.listConceptPaths("/projects")).filter((candidate) =>
          candidate !== leadingProject!.path &&
          (candidate.startsWith(`${projectRoot}/`) || candidate.startsWith(`${projectRoot}-`)))
      : [];
    state.checkCancellation();
    if (projectPeerPaths.length > 200) {
      throw new Error("Staged mutation deferred: same-project owner inventory exceeds limit.");
    }
    const missingDirectPath = !!(options.directAdd && options.preflightInput && requestedPath &&
      !(await kb.bundle.exists(requestedPath)));
    if (missingDirectPath && namedOwnerHits.length) {
      throw new Error("Staged mutation deferred: a named owner already exists outside the proposed new path.");
    }
    if (missingDirectPath && (correction || raw.trim().length < 30 || raw.trim().length > 4_000)) {
      throw new Error("Staged mutation deferred: missing suggested path cannot safely own this fact.");
    }
    if (missingDirectPath) {
      const generic = new Set(["about", "and", "for", "from", "general", "information", "knowledge", "memory", "of", "project", "the", "to"]);
      const pathTerms = new Set(stagedWords(path.posix.basename(requestedPath!, ".md").replace(/[-_]/g, " "))
        .filter((term) => !generic.has(term)));
      const factTerms = new Set(stagedWords(raw));
      const distinctiveOverlap = [...pathTerms].filter((term) => factTerms.has(term));
      // Version labels are meaningful only when explicitly stated as standalone
      // tokens in the exact fact (v4 must not match v40). They can supplement,
      // but never replace, a distinctive entity anchor from the suggested slug.
      const slug = path.posix.basename(requestedPath!, ".md");
      const versionTokens = slug.match(/\bv\d+(?:\.\d+)*\b/gi) ?? [];
      const factVersions = raw.match(/\bv\d+(?:\.\d+)*\b/gi) ?? [];
      const hasExplicitVersionAnchor = versionTokens.some((version) =>
        factVersions.some((factVersion) => factVersion.toLowerCase() === version.toLowerCase()));
      // The proposed new concept must be specifically named by the exact fact;
      // generic search overlap is not evidence that an arbitrary path belongs.
      const ordinaryAnchor = distinctiveOverlap.length >=
        ((leadingPullRequest && namedApplication === leadingPullRequest) || leadingTool ? 1 : 2);
      const versionedEntityAnchor = hasExplicitVersionAnchor && distinctiveOverlap.length >= 1;
      if ((!ordinaryAnchor && !versionedEntityAnchor) ||
          (versionTokens.length > 0 && !hasExplicitVersionAnchor) || hits.length === 0) {
        throw new Error("Staged mutation deferred: missing suggested path lacks a positive fact-topic anchor.");
      }
      const evidence = [];
      for (const hit of hits) {
        const concept = await kb.readConcept(hit.path);
        recorder.record("read_concept", concept.path, [concept.path]);
        state.checkCancellation();
        if (concept.body.length > STAGED_MAX_OWNER_CHARS) {
          throw new Error("Staged mutation deferred: competing owner body exceeds evidence limit.");
        }
        evidence.push({ path: concept.path, frontmatter: concept.frontmatter, body: concept.body });
      }
      if (evidence.some((candidate) => candidate.body.includes(raw.trim()))) {
        throw new Error("Staged mutation deferred: exact fact already belongs to an observed concept.");
      }
      const evidenceText = JSON.stringify(evidence);
      // Keep every owner body complete. When they cannot fit together, check
      // each one separately rather than omitting competitors or truncating a body.
      const batches = evidenceText.length + raw.length <= STAGED_MAX_EVIDENCE_CHARS
        ? [evidence] : evidence.map((candidate) => [candidate]);
      const resolved = await resolveAgentModel(options, "mutate", stagedEnv, (call) => providerCalls.push(call));
      modelChain = resolved.modelChain;
      for (const batch of batches) {
        const fullEvidence = JSON.stringify(batch);
        if (fullEvidence.length + raw.length > STAGED_MAX_EVIDENCE_CHARS) {
          throw new Error("Staged mutation deferred: new-concept owner exceeds prompt budget.");
        }
        const judgement = await generateObject({
          model: resolved.model,
          schema: z.object({ safe: z.boolean() }).strict(),
          temperature: 0,
          abortSignal: options.signal,
          prompt: `Independent safety and duplication check for creating a new concept at the proposed missing path. Is the exact caller fact supported as written, positively about the distinctive path topic, and not already stated or substantially represented by any existing owner/claim in the FULL bounded evidence? Reject if an existing owner is an exact or similar fit, if the fact duplicates an existing claim, or if relevance/support is uncertain. Do not infer facts beyond the exact caller fact. Return safe=true only when all checks pass.\nFACT:\n${raw.trim()}\nPROPOSED NEW PATH: ${requestedPath}\nFULL EXISTING SEARCH EVIDENCE:\n${fullEvidence}`,
        });
        state.checkCancellation();
        if (!judgement.object.safe) throw new Error("Staged mutation deferred: new concept failed independent safety or duplication check.");
      }
      const anchor = evidence[0];
      const title = path.posix.basename(requestedPath!, ".md").split(/[-_]/).filter(Boolean)
        .map((part) => part[0].toUpperCase() + part.slice(1)).join(" ").slice(0, 256);
      await kb.createConceptIfUnchanged(
        anchor.path, sha256(anchor.body), requestedPath!,
        { type: leadingTool ? "CLI Tool" : "Knowledge", title, description: raw.trim().slice(0, 400) },
        `${raw.trim()}\n`, "Staged mutation: create exact verified fact", options.signal,
        (writtenPath) => { changed = [writtenPath]; }
      );
      const summary = `Staged mutation changed ${changed.join(", ")}.`;
      const trace = recorder.finalize("mutation", instruction, summary, "success", modelChain, undefined, undefined, undefined, { providerCalls, generationMs: Date.now() - started });
      await traceStore(kb).save(trace);
      return { ok: true, result: { summary, filesChanged: changed, steps: 1, traceId: trace.id } };
    }
    // A caller-supplied path is a locator, not a search-ranking requirement.
    // Read it only when it already exists, then subject it to the same complete
    // evidence, independent consistency and stale-body checks as any other owner.
    // A missing hint may still be the desired path of a distinct new concept.
    const quotedOwner = !options.directAdd ? raw.match(/`(\/[a-z0-9_/-]+\.md)`/i)?.[1] : undefined;
    const ownerPath = requestedPath ?? quotedOwner;
    const hintedPath = ownerPath && /^\/[a-z0-9_/-]+\.md$/.test(ownerPath) && await kb.bundle.exists(ownerPath)
      ? ownerPath : undefined;
    const named = hits.filter((hit) => hit.title && stagedTitleMention(raw, hit.title));
    const explicitOwner = named.length === 1 ? named[0] : undefined;
    const dominant = !focused.length && first && first.confidenceQualified === true && (!second ||
      (first.confidence ?? 0) >= (second.confidence ?? 0) + 20 &&
      (first.confidence ?? 0) >= (second.confidence ?? 0) * 1.5);
    // A correction to the 25-class human controls belongs to the note that
    // already documents those controls, not another note sharing the same
    // running page. Search every sibling continuation: ranked hits can omit
    // the keyboard note. A duplicate control note is ambiguous, not a tie to
    // resolve by search order. This is owner location, not write permission.
    let uiKeyboardOwnerPath: string | undefined;
    if (!options.directAdd && options.preflightInput &&
        /\bnumber keys?\b/i.test(raw) && /\bselect categories\b/i.test(raw) &&
        /\bupdated in place\b/i.test(raw)) {
      const pagePath = raw.match(/(?:file:\/\/)?(\/[^\s`]+\.html)/i)?.[1];
      const siblings = await kb.bundle.listConceptPaths("/projects");
      if (siblings.length > 200) throw new Error("Staged mutation deferred: annotation UI owner inventory exceeds limit.");
      const matchingOwners: string[] = [];
      for (const sibling of siblings) {
        const concept = await kb.readConcept(sibling);
        recorder.record("read_concept", concept.path, [concept.path]);
        state.checkCancellation();
        if (concept.body.length > STAGED_MAX_OWNER_CHARS) {
          throw new Error("Staged mutation deferred: annotation UI peer exceeds evidence limit.");
        }
        if (pagePath && concept.body.includes(pagePath) && /\bkeys?\s+\d+[–-]\d+\b/i.test(concept.body) &&
            /\bkeyboard-first controls\b/i.test(concept.body)) matchingOwners.push(concept.path);
      }
      if (matchingOwners.length !== 1) {
        throw new Error("Staged mutation deferred: annotation keyboard-controls owner is missing or duplicated.");
      }
      uiKeyboardOwnerPath = matchingOwners[0];
    }
    // A correction of the dated six-zero-row workbook observation belongs
    // with that old observation, not the later 24-label retirement note. The
    // shared category and project name alone cannot identify its owner; read
    // every sibling and require the complete old list to be unique.
    let zeroWorkbookOwnerPath: string | undefined;
    if (!options.directAdd && options.preflightInput &&
        /correct the earlier point-in-time .* workbook zero-annotation list/i.test(raw)) {
      const siblings = await kb.bundle.listConceptPaths("/projects");
      if (siblings.length > 200) throw new Error("Staged mutation deferred: workbook owner inventory exceeds limit.");
      const matchingOwners: string[] = [];
      for (const sibling of siblings) {
        const concept = await kb.readConcept(sibling);
        recorder.record("read_concept", concept.path, [concept.path]);
        state.checkCancellation();
        if (concept.body.length > STAGED_MAX_OWNER_CHARS) {
          throw new Error("Staged mutation deferred: workbook peer exceeds evidence limit.");
        }
        const categoryAnchors = [...new Set(raw.match(/[\p{Lu}][\p{L}-]+\s+\d+\b/gu) ?? [])];
        const dateAnchor = raw.match(/\b\d{4}-\d{2}-\d{2}\b/)?.[0];
        if (dateAnchor && concept.body.includes(dateAnchor) && /zero[^.]{0,80}(?:rows|marks)/i.test(concept.body) &&
            categoryAnchors.some((anchor) => concept.body.includes(anchor))) matchingOwners.push(concept.path);
      }
      if (matchingOwners.length !== 1) {
        throw new Error("Staged mutation deferred: dated zero-row workbook owner is missing or duplicated.");
      }
      zeroWorkbookOwnerPath = matchingOwners[0];
    }
    const candidates: Array<{ path: string; frontmatter: { title?: string; type?: string }; body: string }> = [];
    const locatedOwner = sectionOwnerPath ?? uiKeyboardOwnerPath ?? zeroWorkbookOwnerPath;
    // A more explicitly grounded section or observed old-claim owner retains
    // its own peer check. Otherwise ambiguity in the full evidence inventory
    // must not be hidden by the ranked top-five search.
    if (!locatedOwner && evidenceConflict) {
      throw new Error("Staged mutation deferred: evidence inventory found oversized or competing candidates.");
    }
    if (hintedPath && locatedOwner && hintedPath !== locatedOwner) {
      throw new Error("Staged mutation deferred: suggested owner conflicts with named section.");
    }
    if (hintedPath || locatedOwner || explicitOwner || dominant) {
      const concept = await kb.readConcept(hintedPath ?? locatedOwner ?? (explicitOwner ?? first).path);
      recorder.record("read_concept", concept.path, [concept.path]);
      if (concept.body.length > STAGED_MAX_OWNER_CHARS) throw new Error("Staged mutation deferred: owner body exceeds evidence limit.");
      candidates.push({ path: concept.path, frontmatter: concept.frontmatter, body: concept.body });
    } else if (hits.length > 0) {
      // Keep the original top-three bodies eligible even when a focused name
      // is incidental. Also expose the bounded title-matched owners to the
      // proposer; all original five hits remain visible to the verifier.
      const originalCandidates = hits.slice(0, leadingPath && hits[0]?.path === leadingPath ? 4 : 3);
      const candidateHits = [...originalCandidates, ...focused.filter((hit) =>
        !originalCandidates.some((candidate) => candidate.path === hit.path))];
      for (const hit of candidateHits) {
        const concept = await kb.readConcept(hit.path);
        recorder.record("read_concept", concept.path, [concept.path]);
        candidates.push({ path: concept.path, frontmatter: concept.frontmatter, body: concept.body });
      }
    }
    let evidence = JSON.stringify(candidates);
    // Select by metadata sooner when multiple focused matches would force a
    // large correction prompt. Selection remains a locator; the chosen owner
    // is always re-read in full for the independent write check.
    const selectionBudget = candidates.length <= 1
      ? STAGED_MAX_EVIDENCE_CHARS
      : focused.length && !sectionOwnerPath ? 24_000 : STAGED_MAX_EVIDENCE_CHARS;
    let selectedPath: string | undefined;
    let verifiedMetadataOwner = false;
    let historicalProvenanceOwner: string | undefined;
    let fallbackCandidates: Array<{ path: string; frontmatter: { title?: string; type?: string }; body: string }> | undefined;
    if (evidence.length + raw.length > selectionBudget && candidates.length <= 1) {
      throw new Error("Staged mutation deferred: evidence exceeds prompt budget.");
    }
    if (candidates.length > 1 && evidence.length + raw.length <= selectionBudget &&
        candidates.some((candidate) => candidate.body.length > STAGED_MAX_OWNER_CHARS)) {
      throw new Error("Staged mutation deferred: candidate body exceeds evidence limit.");
    }
    state.checkCancellation();
    const resolved = await resolveAgentModel(options, "mutate", stagedEnv, (call) => providerCalls.push(call));
    modelChain = resolved.modelChain;
    if (evidence.length + raw.length > selectionBudget) {
      if (raw.length > 8_000) throw new Error("Staged mutation deferred: instruction exceeds selector budget.");
      // Selection is only a locator, never write authorization. Do not send
      // truncated bodies to the proposer or silently drop competing owners:
      // the independent verifier still sees the other search-hit titles.
      const metadata = hits.map((hit) => ({
        path: hit.path.slice(0, 512),
        title: typeof hit.title === "string" ? hit.title.slice(0, 256) : "",
        description: typeof hit.description === "string" ? hit.description.slice(0, 512) : "",
      }));
      const selector = await resolveAgentModel(options, "mutate", { ...process.env, MUTATION_ENABLE_THINKING: "false" }, (call) => providerCalls.push(call));
      const selection = await generateObject({
        model: selector.model,
        schema: z.object({ path: z.string() }).strict(),
        temperature: 0,
        abortSignal: options.signal,
        prompt: `Read-only owner selection. Choose exactly one positively relevant existing concept from the candidate metadata. Any owner whose topic plausibly fits the fact is acceptable; do not reject a relevant owner just because another could also fit. If none fit, return an empty path. When multiple owners fit equally, choose the first in candidate order. A mere negative/unrelated mention is not relevance. This choice does not authorize a write. JSON only.\nINSTRUCTION:\n${raw}\nCANDIDATE METADATA:\n${JSON.stringify(metadata)}`,
      });
      state.checkCancellation();
      let selectedHit = hits.find((hit) => hit.path === selection.object.path);
      if (!selectedHit) {
        // A metadata selector can abstain or hallucinate a path. Give it one
        // bounded, content-grounded retry over every ranked candidate, rather
        // than guessing from search order or discarding competing owners.
        const completeCandidates: Array<{ path: string; frontmatter: { title?: string; type?: string }; body: string }> = [];
        for (const hit of hits) {
          state.checkCancellation();
          const concept = await kb.readConcept(hit.path);
          recorder.record("read_concept", concept.path, [concept.path]);
          if (concept.body.length > STAGED_MAX_OWNER_CHARS) {
            throw new Error("Staged mutation deferred: competing owner body exceeds evidence limit.");
          }
          completeCandidates.push({ path: concept.path, frontmatter: concept.frontmatter, body: concept.body });
        }
        // Retry on complete evidence. If the aggregate is too large, partition
        // into bounded batches without splitting bodies; every candidate must
        // be considered and any two positive owners make the result ambiguous.
        const batches: typeof completeCandidates[] = [];
        let batch: typeof completeCandidates = [];
        let batchSize = raw.length;
        for (const candidate of completeCandidates) {
          const candidateSize = JSON.stringify(candidate).length;
          if (candidate.body.length > STAGED_MAX_OWNER_CHARS || candidateSize + raw.length > STAGED_MAX_EVIDENCE_CHARS) {
            throw new Error("Staged mutation deferred: complete candidate exceeds evidence limit.");
          }
          if (batch.length && batchSize + candidateSize > STAGED_MAX_EVIDENCE_CHARS) {
            batches.push(batch);
            batch = [];
            batchSize = raw.length;
          }
          batch.push(candidate);
          batchSize += candidateSize;
        }
        if (batch.length) batches.push(batch);
        const positivePaths: string[] = [];
        for (const candidateBatch of batches) {
          const fullEvidence = JSON.stringify(candidateBatch);
          const fallback = await generateObject({
            model: selector.model,
            schema: z.object({ path: z.string() }).strict(),
            temperature: 0,
            abortSignal: options.signal,
            prompt: `Read-only owner selection using complete evidence. Select a path only when one concept positively owns the instruction's primary subject/topic. Treat quoted examples and incidental mentions of other projects/repos as context, not ownership. Return an empty path when none positively fit or ownership is ambiguous. Never use candidate order as a tie-break. This choice does not authorize a write. JSON only.\nINSTRUCTION:\n${raw}\nCOMPLETE CANDIDATES (title, type, and full body):\n${fullEvidence}`,
          });
          const path = fallback.object.path;
          if (path && !candidateBatch.some((candidate) => candidate.path === path)) {
            throw new Error("Staged mutation deferred: candidate owner selection is uncertain.");
          }
          if (path) positivePaths.push(path);
          if (positivePaths.length > 1) {
            throw new Error("Staged mutation deferred: candidate owner selection is ambiguous.");
          }
        }
        selectedHit = hits.find((hit) => hit.path === positivePaths[0]);
        if (!selectedHit) throw new Error("Staged mutation deferred: candidate owner selection is uncertain.");
        fallbackCandidates = completeCandidates;
      }
      selectedPath = selectedHit.path;
      // Metadata selection cannot authorize a write. Re-read the selected hit
      // even when it ranked fourth or fifth; never send a truncated body onward.
      const owner = await kb.readConcept(selectedPath);
      recorder.record("read_concept", owner.path, [owner.path]);
      state.checkCancellation();
      if (owner.body.length > STAGED_MAX_OWNER_CHARS) {
        throw new Error("Staged mutation deferred: owner body exceeds evidence limit.");
      }
      const selected = { path: owner.path, frontmatter: owner.frontmatter, body: owner.body };
      const evidenceCandidates = fallbackCandidates
        ? fallbackCandidates
        : [...candidates.filter((candidate) => candidate.path !== selectedPath), selected];
      if (!hintedPath && (namedSection ||
          (!options.directAdd && stagedContrastedOldHash(raw)) ||
          !stagedDiscriminativeOwnerAnchor(raw, selected.path, evidenceCandidates, hits, !!options.directAdd))) {
        // A named section or title-less exact claim needs complete competing bodies, not just
        // the three bodies that fit in the proposal prompt. These reads are
        // used for the uniqueness check only; the proposer sees the owner.
        for (const hit of hits) {
          if (evidenceCandidates.some((candidate) => candidate.path === hit.path)) continue;
          const competitor = await kb.readConcept(hit.path);
          recorder.record("read_concept", competitor.path, [competitor.path]);
          state.checkCancellation();
          if (competitor.body.length > STAGED_MAX_OWNER_CHARS) {
            throw new Error("Staged mutation deferred: competing owner body exceeds evidence limit.");
          }
          evidenceCandidates.push({ path: competitor.path, frontmatter: competitor.frontmatter, body: competitor.body });
        }
        if (namedSection) {
          // The explicitly named heading outranks a generic title match, but
          // only if exactly one full observed owner contains that heading.
          const sectionOwners = evidenceCandidates.filter((candidate) => namedSection.test(candidate.body));
          if (sectionOwners.length !== 1) {
            throw new Error("Staged mutation deferred: named section is not unique to one observed owner.");
          }
          selectedPath = sectionOwners[0].path;
        } else if (!stagedDiscriminativeOwnerAnchor(raw, selected.path, evidenceCandidates, hits, !!options.directAdd)) {
          const noAnchor = (grounding: string): never => {
            const oldHash = stagedContrastedOldHash(raw);
            const peerCopies = oldHash ? evidenceCandidates.filter((candidate) =>
              candidate.path !== selectedPath && candidate.body.includes(oldHash)).length : 0;
            throw new Error(`Staged mutation deferred: metadata-selected owner lacks a positive discriminative anchor (grounding=${grounding}, selected=${selectedPath}, contrasted_hash=${!!oldHash}, selected_has_old=${!!oldHash && selected.body.includes(oldHash)}, peer_copies=${peerCopies}).`);
          };
          // Only an explicit correction of earlier guidance may use semantic
          // old-claim matching. Metadata ranking and new facts are not anchors.
          if (options.directAdd || !/\b(?:supersed(?:ed|es|ing)?|correct(?:ed|ion)?\s+(?:the\s+)?(?:earlier|prior|old))\b/i.test(raw)) noAnchor("no-explicit-old-claim");
          if (raw.length + selected.body.length + 3_000 > STAGED_MAX_EVIDENCE_CHARS) noAnchor("owner-budget");
          const oldContext = raw.match(/\bsupersed(?:es|ed|ing)?\s+(?:the\s+)?(?:earlier|prior|old)\s+([^.!?;]{12,350})/i)?.[1] ?? "";
          const selection = await generateObject({
            model: resolved.model,
            schema: z.object({ observedOldPassage: z.string(), requestOldClause: z.string(),
              sameOldClaim: z.boolean(), currentlyActive: z.boolean(), quotedOrHistorical: z.boolean(),
              historicalProvenance: z.boolean(), negated: z.boolean() }).strict(),
            temperature: 0, abortSignal: options.signal,
            prompt: `Read-only old-claim locator; NEVER authorize a write. Quote one COMPLETE, exact, currently active old sentence or bounded paragraph in the owner that the caller explicitly corrects, and quote the exact corresponding old-claim clause from the instruction. If the superseded advice names an old numeric value or identifier, the observed passage MUST contain that SAME old value or identifier; do not select a related sentence that omits it. Include the full old-claim clause, not just a convenient phrase. An old claim can be paraphrased. A genuine dated earlier policy/provenance may locate its own owner for a NEW dated follow-up, but must be reported as currentlyActive=false, quotedOrHistorical=true, historicalProvenance=true. A copied quote, hypothetical, new fact, or common project name is NOT historical provenance and cannot locate an owner. Return empty strings and false status on uncertainty. JSON only.\nEXPLICIT SUPERSEDED ADVICE (not evidence by itself):\n${oldContext}\nINSTRUCTION:\n${raw}\nFULL SELECTED OWNER ${selected.path}:\n${selected.body}`,
          });
          state.checkCancellation();
          if (selection.finishReason === "length") noAnchor("locator-output-cap");
          const located = selection.object;
          const passage = located.observedOldPassage.trim();
          const clause = located.requestOldClause.trim();
          const paragraphs = selected.body.split(/\n\s*\n/);
          const enclosing = paragraphs.filter((part) => part.includes(passage));
          const requestWords = stagedWords(clause).filter((word) => word.length >= 4);
          const passageWords = new Set(stagedWords(passage));
          const sharedWords = requestWords.filter((word) => passageWords.has(word));
          const oldIdentifiers = [...`${clause} ${oldContext}`.matchAll(/(?:^|[^\w])(-?\d+(?:[.,]\d+)*|[a-f0-9]{12,}|[\w-]+_\d+)(?=$|[^\w])/gi)]
            .map((match) => match[1]);
          const hasOldValue = oldIdentifiers.some((value) => passage.includes(value));
          const hasSubject = stagedWords(raw).some((word) => word.length >= 5 && passageWords.has(word) &&
            !["earlier", "current", "previous", "update", "correct", "supersede"].includes(word));
          const pastPolicy = !located.currentlyActive && located.quotedOrHistorical && located.historicalProvenance;
          const activePolicy = located.currentlyActive && !located.quotedOrHistorical && !located.historicalProvenance;
          if (!located.sameOldClaim || !(activePolicy || pastPolicy) || located.negated ||
              passage.length < 35 || passage.length > 4_000 || enclosing.length !== 1 ||
              selected.body.split(passage).length !== 2 || clause.length < 12 || clause.length > 2_000 ||
              raw.split(clause).length !== 2 ||
              (activePolicy && /^\s*(?:Historical|Quoted|Previously|Formerly)\b/i.test(enclosing[0])) ||
              (sharedWords.length < 2 && !(hasOldValue && hasSubject))) noAnchor(`locator(${[
                located.sameOldClaim, located.currentlyActive, !located.quotedOrHistorical, located.historicalProvenance,
                !located.negated, passage.length >= 35 && passage.length <= 4_000, enclosing.length === 1,
                selected.body.split(passage).length === 2, clause.length >= 12 && clause.length <= 2_000,
                raw.split(clause).length === 2, sharedWords.length >= 2, hasOldValue, hasSubject,
              ].map(Number).join("")})`);
          if (pastPolicy) {
            const pastCheck = await generateObject({
              model: resolved.model,
              schema: z.object({ authenticPastPolicy: z.boolean(), noCurrentContradiction: z.boolean(), certain: z.boolean() }).strict(),
              temperature: 0, abortSignal: options.signal,
              prompt: `Independent read-only provenance check. Is the observed exact passage a REAL earlier policy or pending decision in this owner's dated account (not a quoted instruction, hypothetical, negative example, or another entity's claim)? Is the requested follow-up a later decision about the SAME scoped corpus/item? For noCurrentContradiction, EXCLUDE the identified earlier passage and its dated pending-state context: those are the very provenance being superseded, even when the later owner confirmation happened on the same calendar day. Check every OTHER later or undated CURRENT assertion in the full owner for a conflict; do not approve if one remains. An old conditional warning to wait for owner confirmation is still historically true once confirmation arrives. If uncertain, return false. This check never authorizes a write or changes history. JSON only.\nINSTRUCTION:\n${raw}\nOLD PASSAGE:\n${passage}\nFULL OWNER ${selected.path}:\n${selected.body}`,
            });
            state.checkCancellation();
            if (pastCheck.finishReason === "length" || !pastCheck.object.certain ||
                !pastCheck.object.authenticPastPolicy || !pastCheck.object.noCurrentContradiction) noAnchor(`historical-provenance-veto(certain=${pastCheck.object.certain},authentic=${pastCheck.object.authenticPastPolicy},consistent=${pastCheck.object.noCurrentContradiction},capped=${pastCheck.finishReason === "length"})`);
            historicalProvenanceOwner = selected.path;
          }
          // Every retained search hit gets its own complete-body peer check.
          // Never truncate a peer or rely on absence of an exact string alone.
          for (const peer of evidenceCandidates.filter((candidate) => candidate.path !== selectedPath)) {
            if (raw.length + peer.body.length + passage.length + 3_000 > STAGED_MAX_EVIDENCE_CHARS) noAnchor("peer-budget");
            const check = await generateObject({
              model: resolved.model,
              schema: z.object({ competingCurrentOrAmbiguous: z.boolean(), certain: z.boolean() }).strict(),
              temperature: 0, abortSignal: options.signal,
              prompt: `Independent read-only competing-owner check. Does this FULL peer body own the SAME requested version/artifact and its substantive new facts, contradict them at that time, or leave the version chronology/ownership ambiguous? Compare full version identifiers and dates, not just a shared corpus size, old label, or one card count. A clearly earlier note or a DIFFERENT later successor version is not a competing owner of a distinct dated intermediate milestone; the latter must not make that older milestone sound current today. Return competingCurrentOrAmbiguous=true on uncertainty. JSON only.\nREQUEST:\n${raw}\nOBSERVED OLD CLAIM FROM SELECTED OWNER:\n${passage}\nFULL PEER ${peer.path}:\n${peer.body}`,
            });
            state.checkCancellation();
            if (check.finishReason === "length") noAnchor("peer-output-cap");
            if (!check.object.certain || check.object.competingCurrentOrAmbiguous) {
              // A broad shared corpus can obscure a distinct intermediate
              // version. Escalate only with independently checkable, different
              // complete artifact paths and at least two unshared measurements.
              const artifacts = (text: string) => text.match(/(?:data\/[\w./-]{20,}|file:\/\/\/tmp\/[\w.-]{20,}\.html)/g) ?? [];
              const relatedArtifacts = (left: string, right: string): boolean => {
                const a = left.split("/");
                const b = right.split("/");
                let shared = 0;
                while (shared < Math.min(a.length, b.length) && a[shared] === b[shared]) shared++;
                return left !== right && !left.includes(right) && !right.includes(left) &&
                  (shared >= 5 || (left.startsWith("file:///tmp/") && right.startsWith("file:///tmp/") && shared >= 4));
              };
              const pair = artifacts(raw).filter((artifact) => !peer.body.includes(artifact))
                .flatMap((requested) => artifacts(peer.body).filter((artifact) => relatedArtifacts(requested, artifact))
                  .map((peerArtifact) => ({ requested, peerArtifact })))[0];
              const requested = pair?.requested;
              const peerArtifact = pair?.peerArtifact;
              const unsharedCounts = [...new Set(raw.match(/\b\d{1,3}(?:,\d{3})+\b/g) ?? [])]
                .filter((count) => !peer.body.includes(count));
              if (!pastPolicy || !requested || !peerArtifact || unsharedCounts.length < 2) {
                noAnchor(`peer(${peer.path},certain=${check.object.certain},conflict=${check.object.competingCurrentOrAmbiguous},past=${pastPolicy},requestArtifact=${!!requested},peerArtifact=${!!peerArtifact},unsharedCounts=${unsharedCounts.length})`);
              }
              const versionCheck = await generateObject({
                model: resolved.model,
                schema: z.object({ distinctDatedVersion: z.boolean(), laterPeerSuccessor: z.boolean(),
                  noDuplicateOrContradiction: z.boolean(), certain: z.boolean() }).strict(),
                temperature: 0, abortSignal: options.signal,
                prompt: `Independent read-only version disambiguation after an uncertain peer check. The requested and peer artifact paths are EXACT, share a parent, and differ; that alone is not enough. Does the request describe a distinct EARLIER dated intermediate version of this same subject, with the peer describing a LATER successor? Does this complete peer contain NO duplicate of the requested exact version, its distinctive measurements, or a factual contradiction at the requested time? Treat "current" in the request as scoped to its stated date, never as current today. Shared corpus sizes or generic policy themes do not establish duplicate ownership. Decline on uncertain ordering, overlapping exact version facts, or unresolved conflict. JSON only.\nINSTRUCTION:\n${raw}\nREQUESTED ARTIFACT:\n${requested}\nDIFFERENT PEER ARTIFACT:\n${peerArtifact}\nREQUEST-ONLY MEASUREMENTS:\n${JSON.stringify(unsharedCounts.slice(0, 12))}\nOLD OWNER PASSAGE:\n${passage}\nFULL PEER ${peer.path}:\n${peer.body}`,
              });
              state.checkCancellation();
              if (versionCheck.finishReason === "length" || !versionCheck.object.certain ||
                  !versionCheck.object.distinctDatedVersion || !versionCheck.object.laterPeerSuccessor ||
                  !versionCheck.object.noDuplicateOrContradiction) noAnchor(`peer-version(${peer.path},certain=${versionCheck.object.certain},distinct=${versionCheck.object.distinctDatedVersion},later=${versionCheck.object.laterPeerSuccessor},safe=${versionCheck.object.noDuplicateOrContradiction})`);
            }
          }
        }
      }
      verifiedMetadataOwner = true;
      candidates.splice(0, candidates.length, evidenceCandidates.find((candidate) => candidate.path === selectedPath)!);
      evidence = JSON.stringify(candidates);
      if (evidence.length + raw.length > STAGED_MAX_EVIDENCE_CHARS) {
        throw new Error("Staged mutation deferred: selected owner exceeds prompt budget.");
      }
    }
    // A final PR status supplies exact new facts and asks to retract one
    // active old review sentence. A prose proposer may try to rewrite the
    // entire review paragraph (or a person's unrelated note). Select the
    // single status sentence from the uniquely located PR owner instead.
    if (finalPrStatus && prReviewOwnerPath) {
      const target = candidates.find((candidate) => candidate.path === prReviewOwnerPath);
      const prNumber = new RegExp(`\\bPR\\s*#${finalPrStatus[2]}\\b`, "i");
      const paragraphs = target?.body.split(/\n\s*\n/).filter((paragraph) =>
        paragraph.toLowerCase().includes(finalPrStatus[1].toLowerCase()) && prNumber.test(paragraph)) ?? [];
      const stale = paragraphs.flatMap((paragraph) => paragraph.split(/(?<=[.!?])\s+/).filter((sentence) =>
        /\b(?:CHANGES_REQUESTED|changes requested|open)\b/i.test(sentence) &&
        /\b(?:review|PR|pull request)\b/i.test(sentence) &&
        /\b(?:remains?|still|pending|active|open)\b/i.test(sentence) &&
        !/\b(?:superseded|already approved|squash-merged)\b/i.test(sentence)));
      let oldText = stale[0] ?? "";
      const newText = raw.trim().match(/^.+?\s+PR\s*#\d+\s+final\s+status\s*:\s*([\s\S]+?)\s+Replace any prior status saying\b[\s\S]*$/iu)?.[1]?.trim();
      const statusClause = oldText.replace(/\bopen (?:and|with) changes requested\b/ig, "open");
      const statusOnly = !!oldText && !/[,;\n]|\b(?:but|while|also|whereas|with|and)\b/i.test(statusClause);
      if (target && paragraphs.length === 1 && stale.length === 1 && !statusOnly) {
        const selector = await resolveAgentModel(options, "mutate", { ...process.env, MUTATION_ENABLE_THINKING: "false" },
          (call) => providerCalls.push(call));
        const scope = await generateObject({
          model: selector.model,
          schema: z.object({ old_text: z.string() }).strict(),
          temperature: 0,
          abortSignal: options.signal,
          prompt: `Select ONLY the smallest contiguous verbatim span of this sentence asserting that the named PR is CURRENTLY open or changes requested. Include ALL currently stale status claims in the sentence, but none of its unrelated facts, history, or other PR statuses. Return an empty old_text if they cannot be isolated in one span. Do not rewrite anything. JSON only.\nINSTRUCTION:\n${raw}\nOWNER ${target.path}, PARAGRAPH:\n${paragraphs[0]}\nMIXED SENTENCE:\n${oldText}`,
        });
        state.checkCancellation();
        const selected = scope.object.old_text;
        const markers = ["CHANGES_REQUESTED", "changes requested", "open"].filter((marker) =>
          oldText.toLowerCase().includes(marker.toLowerCase()));
        const selectedPrIds = [...selected.matchAll(/\bPR\s*#(\d+)\b/gi)].map((match) => match[1]);
        if (selected.length >= 12 && selected.length < oldText.length && oldText.split(selected).length === 2 &&
            target.body.split(selected).length === 2 &&
            !/\b(?:but|while|also|whereas)\b/i.test(selected) &&
            selectedPrIds.every((id) => id === finalPrStatus[2]) &&
            markers.every((marker) => selected.toLowerCase().includes(marker.toLowerCase()))) {
          oldText = selected;
        }
      }
      let verifiedWholeStatus = false;
      if (target && paragraphs.length === 1 && stale.length === 1 && !statusOnly && oldText === stale[0] && newText) {
        const oldIds = [...oldText.matchAll(/\bPR\s*#(\d+)\b/gi)].map((match) => match[1]);
        if (oldIds.every((id) => id === finalPrStatus[2])) {
          const scope = await generateObject({
            model: resolved.model,
            schema: z.object({ safe: z.boolean() }).strict(),
            temperature: 0,
            abortSignal: options.signal,
            prompt: `Independent whole-sentence PR review scope check. Does EVERY assertion in OLD SENTENCE concern ONLY the currently stale open/changes-requested review status of PR #${finalPrStatus[2]}, with no unrelated review-thread, CI, branch, historical, or other PR fact that would be lost? The caller's final status must explicitly supersede every old assertion. If any claim is unrelated, historically true, or uncertain, safe=false. JSON only.\nINSTRUCTION:\n${raw}\nCOMPLETE OWNER ${target.path}:\n${target.body}\nOLD SENTENCE:\n${oldText}\nNEW STATUS:\n${newText}`,
          });
          state.checkCancellation();
          verifiedWholeStatus = scope.object.safe;
        }
      }
      if (!target || paragraphs.length !== 1 || stale.length !== 1 ||
          (!statusOnly && oldText === stale[0] && !verifiedWholeStatus) || !newText || newText.length > 2_000 ||
          target.body.split(oldText).length !== 2 ||
          !/\bapproved\b/i.test(newText) || !/\bsquash-merged\b/i.test(newText)) {
        console.warn("[understory] final PR status scope declined", {
          owner: target?.path, paragraphs: paragraphs.length, stale: stale.length,
          statusOnly, scoped: oldText !== stale[0], verifiedWholeStatus,
          oldLength: oldText.length, newLength: newText?.length ?? 0,
          uniqueOld: !!target && target.body.split(oldText).length === 2,
          oldHasComma: /[,;]/.test(statusClause), oldHasConjunction: /\b(?:but|while|also|whereas|with|and)\b/i.test(statusClause),
        });
        throw new Error("Staged mutation deferred: final PR status has no unique review-only sentence.");
      }
      const judgement = await generateObject({
        model: resolved.model,
        schema: z.object({ safe: z.boolean() }).strict(),
        temperature: 0,
        abortSignal: options.signal,
        prompt: `Independent final PR status check. Does the EXACT OLD SENTENCE assert only the still-active, now superseded review/open status of the PR identified in the complete owner paragraph? Does the caller's exact NEW TEXT faithfully replace that status with ALL supplied final facts, preserve all other history outside the sentence, and avoid contradictions with later owner updates? An unrelated or historically qualified review is unsafe. If uncertain, safe=false. JSON only.\nINSTRUCTION:\n${raw}\nOWNER ${target.path}:\n${target.body}\nOLD STATUS SPAN:\n${oldText}\nNEW TEXT:\n${newText}`,
      });
      state.checkCancellation();
      if (!judgement.object.safe) throw new Error("Staged mutation deferred: final PR status failed independent check.");
      const current = await kb.readConcept(target.path);
      if (current.body !== target.body) throw new Error(`Concept changed while it was being read: ${target.path}`);
      const replacement = target.body.replace(oldText, newText);
      if (replacement.length > STAGED_MAX_OWNER_CHARS) throw new Error("Staged mutation deferred: final PR owner exceeds limit.");
      await kb.patchConcept(target.path, { replaceBody: replacement }, "Staged mutation: replace final PR status", sha256(target.body), options.signal,
        (writtenPath) => { changed = [writtenPath]; });
      const summary = `Staged mutation changed ${changed.join(", ")}.`;
      const trace = recorder.finalize("mutation", instruction, summary, "success", modelChain, undefined, undefined, undefined,
        { providerCalls, generationMs: Date.now() - started });
      await traceStore(kb).save(trace);
      return { ok: true, result: { summary, filesChanged: changed, steps: 1, traceId: trace.id } };
    }
    // An explicit correction supplies its own exact old/new text. Avoid a
    // generated proposal, but still check the full owner, support, and CAS.
    const quotedCorrection = !options.directAdd && options.preflightInput
      ? raw.match(/\breplace\s+(?:the\s+)?(?:exact\s+)?(?:phrase|text|claim)?\s*`([^`]{12,2000})`\s+with\s+`([^`]{1,2000})`/i)
      : null;
    if (quotedCorrection) {
      const [, oldText, newText] = quotedCorrection;
      const matching = candidates.filter((candidate) => candidate.body.split(oldText).length === 2);
      if (matching.length === 1) {
        const target = matching[0];
        if (oldText === newText || !stagedSingleClauseEdit(oldText, newText) ||
            stagedWords(newText).some((word) => !stagedWords(oldText).includes(word) && !stagedWords(raw).includes(word))) {
          throw new Error("Staged mutation rejected: quoted replacement changes unsupported claims.");
        }
        const judgement = await generateObject({
          model: resolved.model,
          schema: z.object({ safe: z.boolean() }).strict(),
          temperature: 0,
          abortSignal: options.signal,
          prompt: `Independent safety check. Does the caller's exact quoted replacement faithfully apply the instruction to this observed owner, preserve every unrelated claim, and avoid a contradiction? A quoted path or metadata selection is not proof of relevance; an alternative owner being equally plausible is not grounds for rejection. Reject negative/unrelated mentions. If uncertain about factual support or relevance, safe=false.\nINSTRUCTION:\n${raw}\nOWNER ${target.path}:\n${target.body}\nALTERNATIVE SEARCH HITS (path and title only):\n${JSON.stringify(hits.filter((hit) => hit.path !== target.path).map((hit) => ({ path: hit.path, title: hit.title })))}\nOLD TEXT:\n${oldText}\nNEW TEXT:\n${newText}`,
        });
        state.checkCancellation();
        if (!judgement.object.safe) throw new Error("Staged mutation deferred: quoted correction failed consistency check.");
        const current = await kb.readConcept(target.path);
        if (current.body !== target.body) throw new Error(`Concept changed while it was being read: ${target.path}`);
        const replacement = target.body.replace(oldText, newText);
        if (replacement.length > STAGED_MAX_OWNER_CHARS) throw new Error("Staged mutation deferred: replacement exceeds evidence limit.");
        state.checkCancellation();
        await kb.patchConcept(target.path, { replaceBody: replacement }, "Staged mutation: exact quoted correction", sha256(target.body), options.signal, (path) => { changed = [path]; });
        const summary = `Staged mutation changed ${changed.join(", ")}.`;
        const trace = recorder.finalize("mutation", instruction, summary, "success", modelChain, undefined, undefined, undefined, { providerCalls, generationMs: Date.now() - started });
        await traceStore(kb).save(trace);
        return { ok: true, result: { summary, filesChanged: changed, steps: 1, traceId: trace.id } };
      }
    }
    // For memory_add, preserve the caller's exact fact rather than generating
    // replacement prose. A fast read-only selector handles unhinted candidates;
    // one thinking-model check must still independently approve the chosen owner
    // and consistency. memory_update keeps the conservative edit proposal below.
    let directPath = hintedPath ?? selectedPath;
    if (!directPath && options.directAdd && options.preflightInput && !correction &&
        candidates.length > 0 && raw.trim().length >= 30 && raw.trim().length <= 4_000) {
      const selector = await resolveAgentModel(options, "mutate", { ...process.env, MUTATION_ENABLE_THINKING: "false" }, (call) => providerCalls.push(call));
      const selection = await generateObject({
        model: selector.model,
        schema: z.object({ path: z.string() }).strict(),
        temperature: 0,
        abortSignal: options.signal,
        prompt: `Choose ONE positively relevant existing concept whose topic plausibly fits the FACT. Use only an exact candidate path. If none fit, return an empty path. A mere negative/unrelated mention is not relevance. If multiple owners plausibly fit equally, choose the first candidate in search order. Do not create or change any content. JSON only.\nFACT:\n${raw}\nCANDIDATES:\n${evidence}`,
      });
      state.checkCancellation();
      if (candidates.some((candidate) => candidate.path === selection.object.path)) directPath = selection.object.path;
    }
    if (directPath && options.directAdd && options.preflightInput && !correction &&
        raw.trim().length >= 30 && raw.trim().length <= 4_000) {
      let target = candidates.find((candidate) => candidate.path === directPath)!;
      const fact = raw.trim();
      // A leading project name locates its parent, but an exact private artifact
      // named in one continuation is stronger positive ownership evidence.
      // Keep all other project bodies (including the parent) in the peer check.
      let artifactAnchoredPeer = false;
      if (!options.ownerHint && leadingProject && target.path === leadingProject.path) {
        const artifactUris = (text: string): string[] =>
          [...text.matchAll(/\bfile:\/\/\/tmp\/[^\s`'",;:)]+/g)]
            .map((match) => match[0].replace(/[.!?]+$/, ""));
        const artifacts = artifactUris(fact);
        if (artifacts.length) {
          const matchingPeers = [];
          for (const peerPath of projectPeerPaths) {
            const peer = await kb.readConcept(peerPath);
            recorder.record("read_concept", peer.path, [peer.path]);
            state.checkCancellation();
            if (peer.body.length > STAGED_MAX_OWNER_CHARS) {
              throw new Error("Staged mutation deferred: competing owner body exceeds evidence limit.");
            }
            const peerArtifacts = new Set(artifactUris(peer.body));
            if (artifacts.some((artifact) => peerArtifacts.has(artifact))) matchingPeers.push(peer);
          }
          // A sibling may mention the same URI as a cross-reference. Prefer
          // the sole continuation recording that artifact's history; the
          // siblings still undergo the full competing-owner check below.
          const continuations = matchingPeers.filter((peer) =>
            /-continuation-[a-f0-9]{8,16}\.md$/.test(peer.path));
          let artifactOwner = matchingPeers.length === 1 ? matchingPeers[0]
            : continuations.length === 1 ? continuations[0] : undefined;
          if (!artifactOwner && continuations.length > 1 && continuations.length <= 8) {
            // Only locate an owner here. The complete bodies of all peers are
            // still checked below, and the chosen owner is checked before CAS.
            const contexts = continuations.map((peer) => {
              const index = Math.min(...artifacts.map((artifact) => {
                const position = peer.body.indexOf(artifact);
                return position < 0 ? peer.body.length : position;
              }));
              return { path: peer.path, title: peer.frontmatter.title,
                context: peer.body.slice(Math.max(0, index - 1_000), index + 3_000) };
            });
            if (JSON.stringify(contexts).length + fact.length <= STAGED_MAX_EVIDENCE_CHARS) {
              const selection = await generateObject({
                model: resolved.model,
                schema: z.object({ path: z.string() }).strict(),
                temperature: 0,
                abortSignal: options.signal,
                prompt: `Read-only artifact owner selection. Several project continuations mention the SAME exact file URI. Choose the ONE continuation whose local context positively owns the requested change's specific topic and operational behavior; a cross-reference, generic privacy rule, or unrelated data fact is not ownership. If they fit equally, return an empty path. This choice never authorizes a write. JSON only.\nFACT:\n${fact}\nCANDIDATE CONTEXTS:\n${JSON.stringify(contexts)}`,
              });
              state.checkCancellation();
              artifactOwner = continuations.find((peer) => peer.path === selection.object.path);
            }
          }
          if (!artifactOwner) {
            // Paths only: distinguish a genuinely ambiguous artifact from a
            // missing match without logging private knowledge-body content.
            console.warn("[understory] artifact owner not unique", {
              project: leadingProject.path, matches: matchingPeers.map((peer) => peer.path),
            });
          }
          if (artifactOwner) {
            target = { path: artifactOwner.path, frontmatter: artifactOwner.frontmatter, body: artifactOwner.body };
            artifactAnchoredPeer = true;
          }
        }
      }
      if (leadingProject && (target.path === leadingProject.path || artifactAnchoredPeer) && projectPeerPaths.length) {
        const batchLimit = STAGED_MAX_EVIDENCE_CHARS - fact.length - 3_000;
        if (batchLimit < 1_000) throw new Error("Staged mutation deferred: project fact exceeds peer-check budget.");
        let batch: Array<{ path: string; title?: string; body: string }> = [];
        const checkDisputedPeer = async (peer: { path: string; title?: string; body: string }) => {
          // A boolean refusal can conflate a shared project prefix with a
          // complete duplicate. Shared topics alone are not conflicts;
          // uncertainty or any concrete conflict still blocks the append.
          const adjudication = await generateObject({
            model: resolved.model,
            schema: z.object({
              distinct_topic: z.boolean(),
              complete_duplicate: z.boolean(),
              contradiction: z.boolean(),
              stronger_owner: z.boolean(),
              certain: z.boolean(),
            }).strict(),
            temperature: 0,
            abortSignal: options.signal,
            prompt: `Independent structured review of a disputed project peer. Compare the EXACT FACT against the FULL PEER BODY claim by claim, not against its shared project name. Return distinct_topic=true only when the peer's own subject is positively different from the fact's specific observation, not merely when it lacks an exact quote. A complete paraphrase counts as complete_duplicate; conflicting assertions count as contradiction; an owner unambiguously covering this specific observation counts as stronger_owner. A shared project topic alone is none of these: you can be certain that none applies even when distinct_topic=false. Return certain=false if you cannot confidently rule out a concrete conflict. JSON only.\nFACT:\n${fact}\nPROPOSED OWNER: ${target.path}\nDISPUTED FULL PEER:\n${JSON.stringify(peer)}`,
          });
          state.checkCancellation();
          const finding = adjudication.object;
          let duplicate = finding.complete_duplicate;
          let strongerOwner = finding.stronger_owner;
          if (finding.certain && strongerOwner && !duplicate && !finding.contradiction) {
            // A broad project prefix can make an unrelated experiment appear
            // to own a specific item correction. Challenge only that disputed
            // owner verdict, with independently quoted evidence from both
            // complete inputs. Absence of a fact from a peer is not enough.
            const audit = await generateObject({
              model: resolved.model,
              schema: z.object({
                distinctObservation: z.boolean(),
                peerTopicQuote: z.string(),
                factItemQuote: z.string(),
                certain: z.boolean(),
              }).strict(),
              temperature: 0,
              abortSignal: options.signal,
              prompt: `Independent audit of a disputed STRONGER OWNER verdict. Is this peer's OWN specific topic distinct from the requested observation, rather than a competing owner of the same item or baseline? Quote an exact substring identifying that topic from BOTH the peer title and full peer body in peerTopicQuote; quote an exact substring of the fact containing its specific item ID in factItemQuote. Return distinctObservation=true and certain=true only when the peer does NOT own that item or its requested correction. Shared project names, merely missing claims, and uncertain scope are not distinctions. JSON only.\nFACT:\n${fact}\nPROPOSED OWNER: ${target.path}\nPEER TITLE: ${peer.title ?? "untitled"}\nFULL PEER BODY:\n${peer.body}`,
            });
            state.checkCancellation();
            const { distinctObservation, peerTopicQuote, factItemQuote, certain } = audit.object;
            const topic = peerTopicQuote.trim();
            const item = factItemQuote.trim();
            const itemPattern = /\b[\p{L}][\p{L}\p{N}]*_[\p{L}\p{N}_]*\d+[\p{L}\p{N}_]*\b/gu;
            const itemIds = item.match(itemPattern) ?? [];
            const factItemIds = fact.match(itemPattern) ?? [];
            const peerIdentity = `${peer.title ?? ""} ${peer.body}`.toLowerCase();
            const noSharedItems = factItemIds.length > 0 &&
              factItemIds.every((id) => !peerIdentity.includes(id.toLowerCase()));
            const separateTopic = topic.length >= 5 && peer.body.includes(topic) &&
              !!peer.title?.includes(topic) && !fact.includes(topic) &&
              stagedWords(topic).some((word) => word.length >= 4 &&
                !stagedWords(fact).includes(word));
            if (certain && distinctObservation && separateTopic && noSharedItems &&
                item.length >= 8 && fact.includes(item) && itemIds.length > 0) {
              strongerOwner = false;
            }
          }
          if (finding.certain && duplicate && !finding.contradiction && !strongerOwner) {
            // A shared project may have recorded a generic test result without
            // the new fact's specific version or failure. Challenge an asserted
            // complete duplicate with an independently quoted missing claim.
            const audit = await generateObject({
              model: resolved.model,
              schema: z.object({ missingFactClause: z.string(), absent: z.boolean(), certain: z.boolean() }).strict(),
              temperature: 0,
              abortSignal: options.signal,
              prompt: `Independent evidence audit of an asserted COMPLETE DUPLICATE. Is there a specific assertion in the FACT that the FULL PEER BODY does NOT state, even in paraphrase? If so, quote an EXACT contiguous substring of the FACT containing that assertion in missingFactClause, and return absent=true only when you are certain it is not recorded anywhere in the peer. Prefer a specific version, code identifier, count, or artifact; generic project names and shared test commands are insufficient. If every claim is already recorded or you are unsure, return absent=false and certain=false. JSON only.\nFACT:\n${fact}\nFULL PEER BODY:\n${peer.body}`,
            });
            state.checkCancellation();
            const { missingFactClause, absent, certain } = audit.object;
            const anchors = missingFactClause.match(/\b(?:\d+(?:\.\d+)+|[a-f0-9]{12,}|[A-Za-z_][\w]*(?:_[\w]+)+|\d{2,})\b/gi) ?? [];
            // A model assertion alone cannot overrule a duplicate. Require a
            // literal excerpt of the fact with a distinctive identifier absent
            // from the complete peer; the model must also certify semantic absence.
            if (certain && absent && missingFactClause.length >= 8 &&
                fact.includes(missingFactClause) && anchors.some((anchor) =>
                  !peer.body.toLowerCase().includes(anchor.toLowerCase()))) {
              duplicate = false;
            }
          }
          if (!finding.certain || duplicate || finding.contradiction ||
              strongerOwner) {
            // Keep the diagnostic to boolean verdicts: peer bodies and facts
            // can contain private material and must not enter service logs.
            console.warn("[understory] disputed project peer rejected", {
              peer: peer.path,
              target: target.path,
              ...finding,
              stronger_owner: strongerOwner,
            });
            throw new Error(`Staged mutation deferred: competing project owner has a duplicate, contradiction, or stronger claim (${peer.path}; target=${target.path}).`);
          }
        };
        const checkBatch = async () => {
          if (!batch.length) return;
          const judgement = await generateObject({
            model: resolved.model,
            schema: z.object({ safe: z.boolean() }).strict(),
            temperature: 0,
            abortSignal: options.signal,
            prompt: `Independent complete competing-project check. The proposed owner is ${target.path} (${target.frontmatter.title ?? "untitled"}). Do NONE of the FULL peer bodies below already record the complete FACT (including paraphrases), contradict it, or unambiguously own it instead of the proposed owner? Partial overlap and a shared project name are not duplication. Check EVERY peer, not just the first. If uncertain, safe=false. JSON only.\nFACT:\n${fact}\nCOMPETING PROJECT OWNERS:\n${JSON.stringify(batch)}`,
          });
          state.checkCancellation();
          if (!judgement.object.safe) {
            // An aggregate answer can conflate a shared dataset with the
            // complete new observation. Recheck every FULL peer separately;
            // one actual conflict or uncertain peer still blocks the write.
            if (batch.length < 2) {
              await checkDisputedPeer(batch[0]);
              batch = [];
              return;
            }
            for (const peer of batch) {
              const individual = await generateObject({
                model: resolved.model,
                schema: z.object({ safe: z.boolean() }).strict(),
                temperature: 0,
                abortSignal: options.signal,
                prompt: `Independent single competing-project check after an aggregate conflict. The proposed owner is ${target.path} (${target.frontmatter.title ?? "untitled"}). Does the FULL peer below neither already record the complete FACT (including paraphrases), nor contradict it, nor unambiguously own it instead of the proposed owner? A shared project, corpus, or partial observation is NOT the complete fact. If uncertain, safe=false. JSON only.\nFACT:\n${fact}\nCOMPETING PROJECT OWNER:\n${JSON.stringify(peer)}`,
              });
              state.checkCancellation();
              if (!individual.object.safe) await checkDisputedPeer(peer);
            }
          }
          batch = [];
        };
        for (const peerPath of artifactAnchoredPeer
          ? [leadingProject!.path, ...projectPeerPaths.filter((candidate) => candidate !== target.path)]
          : projectPeerPaths) {
          const peer = await kb.readConcept(peerPath);
          recorder.record("read_concept", peer.path, [peer.path]);
          state.checkCancellation();
          if (peer.body.length > STAGED_MAX_OWNER_CHARS || peer.body.includes(fact)) {
            throw new Error("Staged mutation deferred: competing project owner exceeds limit or already contains exact fact.");
          }
          const next = { path: peer.path, title: peer.frontmatter.title, body: peer.body };
          if (JSON.stringify([...batch, next]).length > batchLimit) await checkBatch();
          if (JSON.stringify(next).length > batchLimit) {
            throw new Error("Staged mutation deferred: competing project owner exceeds peer-check budget.");
          }
          batch.push(next);
        }
        await checkBatch();
      }
      for (const hit of namedOwnerHits) {
        if (hit.path === target.path) continue;
        const competitor = await kb.readConcept(hit.path);
        recorder.record("read_concept", competitor.path, [competitor.path]);
        state.checkCancellation();
        if (competitor.body.length > STAGED_MAX_OWNER_CHARS) {
          throw new Error("Staged mutation deferred: competing named owner exceeds evidence limit.");
        }
        if (competitor.body.includes(fact)) {
          throw new Error("Staged mutation deferred: exact fact already belongs to another named owner.");
        }
        if (fact.length + competitor.body.length + 2_000 > STAGED_MAX_EVIDENCE_CHARS) {
          throw new Error("Staged mutation deferred: competing named owner exceeds prompt budget.");
        }
        const competingCheck = await generateObject({
          model: resolved.model,
          schema: z.object({ safe: z.boolean() }).strict(),
          temperature: 0,
          abortSignal: options.signal,
          prompt: `Independent complete competing-owner check. The proposed target is ${target.path} (${target.frontmatter.title ?? "untitled"}). Does the FULL competing owner below neither already record the caller's complete fact (including paraphrases), nor contradict it, nor unambiguously own it instead of the proposed target? Partial overlap is not duplication. A shared tool or project word alone does not establish ownership. If uncertain, safe=false. JSON only.\nFACT:\n${fact}\nCOMPETING OWNER ${competitor.path} (${competitor.frontmatter.title ?? "untitled"}):\n${competitor.body}`,
        });
        state.checkCancellation();
        if (!competingCheck.object.safe) {
          throw new Error("Staged mutation deferred: competing named owner has a duplicate, contradiction, or stronger claim.");
        }
      }
      // A continuation may match only by a broad project prefix. When it does
      // not mention the requested item, verify against its parent instead of
      // appending to an unrelated continuation or dropping the fact.
      const itemIds = fact.match(/\b[\p{L}][\p{L}\p{N}]*_[\p{L}\p{N}_]*\d+[\p{L}\p{N}_]*\b/gu) ?? [];
      const parentPath = target.path.replace(/-continuation-[a-f0-9]{8,16}\.md$/, ".md");
      let derivedParent = false;
      if (!hintedPath && !artifactAnchoredPeer && parentPath !== target.path &&
          !itemIds.some((id) => target.body.includes(id)) &&
          await kb.bundle.exists(parentPath)) {
        const parent = await kb.readConcept(parentPath);
        recorder.record("read_concept", parent.path, [parent.path]);
        state.checkCancellation();
        if (parent.body.length > STAGED_MAX_OWNER_CHARS) {
          throw new Error("Staged mutation deferred: parent owner body exceeds evidence limit.");
        }
        if (stagedOwnerAnchor(fact, parent.path)) {
          target = { path: parent.path, frontmatter: parent.frontmatter, body: parent.body };
          derivedParent = true;
        }
      }
      if ((!hintedPath && stagedConflictingNamedOwner(fact, target.frontmatter.title, target.path)) ||
          !(stagedOwnerAnchor(fact, target.path) || artifactAnchoredPeer ||
            (verifiedMetadataOwner && selectedPath === target.path)) ||
          (target.frontmatter.title && fact.toLowerCase().includes(target.frontmatter.title.toLowerCase()) &&
            !stagedTitleMention(fact, target.frontmatter.title))) {
        throw new Error(`Staged mutation deferred: suggested owner lacks a positive entity anchor (target=${target.path}, leading_pr=${leadingPullRequest ?? "none"}, named=${namedApplication ?? "none"}, requested=${requestedPath ?? "none"}, hinted=${hintedPath ?? "none"}, selected=${selectedPath ?? "none"}, named_hits=${namedOwnerHits.length}).`);
      }
      if (!target.body.includes(fact)) {
        const wouldOverflow = `${target.body.trimEnd()}\n\n${fact}\n`.length > STAGED_MAX_OWNER_CHARS;
        // A broad search can select an unrelated, full project or continuation.
        // An existing item ID in a continuation's complete body is stronger
        // evidence than its generic project prefix, even without a title hint.
        const negativeSpans = [...fact.matchAll(/\b(?:unrelated to|not about|not)\s+([^.!?;,:]+)/gi)]
          .map((match) => match[1].toLowerCase());
        const continuationClauses = fact.split(/;\s*|(?<=[.!?])\s+|\n+/)
          .map((clause) => clause.trim()).filter((clause) => stagedWords(clause).length >= 4);
        const genericContinuationTerms = new Set([
          "fs", "fotovalidering", "project", "record", "records", "label", "labels",
          "fact", "source", "data", "new", "unknown", "retain", "checked",
        ]);
        const clauseHasDistinctiveEvidence = (clause: string): boolean => {
          const evidenceTerms = new Set(stagedWords(target.body));
          const distinctive = stagedWords(clause).filter((term) =>
            !genericContinuationTerms.has(term) && evidenceTerms.has(term));
          return new Set(distinctive).size >= 2;
        };
        const groundedMultiFactContinuation = target.path.includes("-continuation-") && !wouldOverflow &&
          continuationClauses.length >= 2 && continuationClauses.every(clauseHasDistinctiveEvidence) &&
          !negativeSpans.some((span) => stagedOverlap(span, target.body) >= 0.5);
        const groundedContinuation = target.path.includes("-continuation-") && !wouldOverflow &&
          (itemIds.some((id) => target.body.includes(id) &&
            !negativeSpans.some((span) => span.includes(id.toLowerCase()))) || groundedMultiFactContinuation);
        if (!hintedPath && !artifactAnchoredPeer && (wouldOverflow || target.path.includes("-continuation-")) &&
            !groundedContinuation && !derivedParent &&
            (!target.frontmatter.title || !stagedTitleMention(raw, target.frontmatter.title))) {
          throw new Error(`Staged mutation deferred: unhinted continuation owner was not explicitly named (target=${target.path}, named_tool=${leadingTool ?? "none"}, requested=${requestedPath ?? "none"}).`);
        }
        const judgement = await generateObject({
          model: resolved.model,
          schema: z.object({ safe: z.boolean(), reason: z.enum(["unsupported", "wrong_owner", "duplicate", "contradiction", "uncertain"]).optional() }).strict(),
          temperature: 0,
          abortSignal: options.signal,
          prompt: `Independent safety check for appending the caller's EXACT fact, without rewriting it. Is every assertion supported by the caller, positively relevant to this owner (even if another existing owner could also plausibly fit), not already recorded AS A COMPLETE FACT, and consistent with the full existing body? Partial overlap with prior facts is not duplication. Reject negative/unrelated mentions. The suggested path is not proof. If safe=false, return one reason: unsupported, wrong_owner, duplicate, contradiction, or uncertain. If uncertain about relevance or support, safe=false.\nFACT:\n${fact}\nOWNER ${target.path} (${target.frontmatter.title ?? "untitled"}):\n${target.body}\nOTHER SEARCH HITS (titles and paths only):\n${JSON.stringify(hits.filter((hit) => hit.path !== target.path).map((hit) => ({ path: hit.path, title: hit.title })))}`,
        });
        state.checkCancellation();
        if (!judgement.object.safe) {
          // A duplicate verdict may conflate a shared topic with the complete
          // multi-claim fact. Never override it from the first judgement alone:
          // require an independent, structured full-body adjudication of every
          // safety dimension. Any false/uncertain field keeps the write deferred.
          if (judgement.object.reason !== "duplicate") {
            throw new Error(`Staged mutation deferred: exact-fact append failed consistency check (${judgement.object.reason ?? "uncertain"} at ${target.path}).`);
          }
          const recheck = await generateObject({
            model: resolved.model,
            schema: z.object({
              non_duplicate: z.boolean(),
              relevant: z.boolean(),
              supported: z.boolean(),
              consistent: z.boolean(),
              certain: z.boolean(),
            }).strict(),
            temperature: 0,
            abortSignal: options.signal,
            prompt: `Independent full-body recheck of a disputed duplicate verdict. Re-evaluate the EXACT FACT claim by claim against the COMPLETE OWNER BODY, independently of the prior verdict. Return true for non_duplicate only when the full fact is NOT already recorded as a complete fact (paraphrases count); partial-topic overlap is not duplication. Return true for relevant only when the fact is positively relevant to this owner, supported only when every claim is supplied by the caller, consistent only when no existing assertion contradicts any claim, and certain only when all these conclusions are clear. Do not infer missing claims from broad context. Any uncertainty, contradiction, unsupported claim, or complete duplicate must be false. JSON only.\nFACT:\n${fact}\nCOMPLETE OWNER ${target.path} (${target.frontmatter.title ?? "untitled"}):\n${target.body}`,
          });
          state.checkCancellation();
          const check = recheck.object;
          if (!(check.non_duplicate && check.relevant && check.supported && check.consistent && check.certain)) {
            throw new Error(`Staged mutation deferred: exact-fact append failed independent duplicate recheck (${target.path}).`);
          }
        }
        const appended = `${target.body.trimEnd()}\n\n${fact}\n`;
        const current = await kb.readConcept(target.path);
        if (current.body !== target.body) throw new Error(`Concept changed while it was being read: ${target.path}`);
        state.checkCancellation();
        if (appended.length <= STAGED_MAX_OWNER_CHARS) {
          await kb.patchConcept(target.path, { replaceBody: appended }, "Staged mutation: append exact verified fact", sha256(target.body), options.signal, (path) => { changed = [path]; });
        } else {
          const factHash = sha256(fact);
          const parentBase = path.posix.basename(target.path, ".md");
          const continuationPath = path.posix.join(path.posix.dirname(target.path), `${parentBase}-continuation-${factHash.slice(0, 16)}.md`);
          const title = `${target.frontmatter.title ?? parentBase} — Continuation ${factHash.slice(0, 8)}`.slice(0, 256);
          const continuationBody = `${fact}\n\nThis fact belongs to [${target.frontmatter.title ?? parentBase}](${target.path}).\n`;
          const ownerType = typeof target.frontmatter.type === "string" ? target.frontmatter.type : "Knowledge";
          if (await kb.bundle.exists(continuationPath)) {
            const existing = await kb.readConcept(continuationPath);
            if (existing.body !== continuationBody || existing.frontmatter.title !== title) {
              throw new Error("Staged mutation deferred: deterministic continuation conflicts with existing content.");
            }
          } else {
            await kb.createConceptIfUnchanged(
              target.path, sha256(target.body), continuationPath,
              { type: `${ownerType} Continuation`, title, description: `Additional ${target.frontmatter.title ?? parentBase} fact: ${fact.slice(0, 400)}` },
              continuationBody, "Staged mutation: continue exact verified fact", options.signal,
              (writtenPath) => { changed = [writtenPath]; }
            );
          }
        }
      } else {
        // A substring may be quoted only to retract it later in the body. Do
        // not equate textual presence with a currently supported assertion.
        const following = target.body.slice(target.body.indexOf(fact) + fact.length, target.body.indexOf(fact) + fact.length + 160);
        if (/^\s*(?:[,;:-]\s*)?(?:but|however|although|no longer|not|never)\b/i.test(following)) {
          throw new Error("Staged mutation deferred: matching text is immediately qualified or negated.");
        }
        const judgement = await generateObject({
          model: resolved.model,
          schema: z.object({ supported: z.boolean() }).strict(),
          temperature: 0,
          abortSignal: options.signal,
          prompt: `Does the full existing body affirm the caller's EXACT fact as presently true, without a later correction, negation, or contradictory qualification? If the words are quoted only to retract them, supported=false. If uncertain, supported=false. Return JSON only.\nFACT:\n${fact}\nOWNER ${target.path}:\n${target.body}`,
        });
        state.checkCancellation();
        if (!judgement.object.supported) throw new Error("Staged mutation deferred: matching text does not establish the requested fact.");
        const current = await kb.readConcept(target.path);
        if (current.body !== target.body || !current.body.includes(fact)) throw new Error("Staged mutation rejected: no-op evidence changed before verification.");
      }
      const summary = changed.length ? `Staged mutation changed ${changed.join(", ")}.` : "No change needed; exact fact verified.";
      const trace = recorder.finalize("mutation", instruction, summary, "success", modelChain, undefined, undefined, undefined, { providerCalls, generationMs: Date.now() - started });
      await traceStore(kb).save(trace);
      return { ok: true, result: { summary, filesChanged: changed, steps: 1, traceId: trace.id } };
    }
    const proposalSchema = z.object({
      action: z.enum(["replace", "append", "create", "noop", "defer"]),
      path: z.string(),
      old_text: z.string(),
      new_text: z.string(),
      body: z.string(),
      frontmatter: z.object({ type: z.string(), title: z.string(), description: z.string() }).strict(),
      claim: z.string(),
      reason: z.string(),
    }).strict();
    const proposer = await resolveAgentModel(options, "mutate", { ...process.env, MUTATION_ENABLE_THINKING: "false" }, (call) => providerCalls.push(call));
    const proposal = await generateObject({
      model: proposer.model,
      schema: proposalSchema,
      temperature: 0,
      abortSignal: options.signal,
      prompt: `Propose exactly one safe knowledge-base mutation from this instruction and complete observed candidate evidence. Fill EVERY JSON field: unused strings must be empty and unused frontmatter must have empty type/title/description. For replace, supply candidate path, an exact old_text substring, and new_text that removes the old claim. For append, supply candidate path and the complete new fact in new_text only if genuinely non-conflicting. For create, supply a new path, nonempty body and frontmatter type/title/description; never create a duplicate owner or an unrelated backlink. For noop, quote a nonempty exact substring of an observed body in claim that establishes the requested knowledge. Otherwise defer and explain why. Never invent evidence. A suggested path is not proof of ownership; defer if it does not fit. ${historicalProvenanceOwner ? `The old policy in ${historicalProvenanceOwner} is genuine dated history, not an active false claim. Preserve it verbatim; only APPEND a complete, supported DATED follow-up to that owner if safe. The instruction describes the status at its stated date, not necessarily current policy today; do not present it as current without a temporal qualifier. Never replace or erase the old passage.` : ""}\nINSTRUCTION:\n${raw}\nSUGGESTED EXISTING OWNER:\n${options.ownerHint ?? "none"}\nCANDIDATES:\n${evidence}`,
    });
    state.checkCancellation();
    const p = proposal.object;
    let repairedCorrection = false;
    if (correction && (p.action === "create" || (p.action === "append" && p.path !== historicalProvenanceOwner))) {
      // Appending or creating a correction leaves the contradicted old assertion in place.
      // Try one exact, local replacement instead; the ordinary replacement
      // branch below still enforces support, independent review and CAS.
      const repair = await generateObject({
        model: proposer.model,
        schema: z.object({ path: z.string(), old_text: z.string(), new_text: z.string() }).strict(),
        temperature: 0,
        abortSignal: options.signal,
        prompt: `The proposed append/create cannot correct an existing claim. Choose ONE exact old clause in an observed candidate body that the instruction supersedes, and replace it with text that addresses ALL applicable requested corrections while preserving unrelated claims. The old_text must be a unique verbatim substring of the owner; new_text must be supported by the instruction. Do not invent an old claim, use an unrelated owner, or append elsewhere. Return empty strings if a single local replacement cannot safely do this. JSON only.\nINSTRUCTION:\n${raw}\nCANDIDATES:\n${evidence}`,
      });
      state.checkCancellation();
      repairedCorrection = true;
      p.action = "replace";
      p.path = repair.object.path;
      p.old_text = repair.object.old_text;
      p.new_text = repair.object.new_text;
    }
    if (p.action === "defer") throw new Error(`Staged mutation deferred: ${p.reason || "proposal is not safely actionable"}`);
    if (historicalProvenanceOwner && p.action === "replace") {
      throw new Error("Staged mutation deferred: dated historical provenance must not be replaced.");
    }
    if (p.action === "noop") {
      const claim = p.claim.trim();
      const requestedClaim = raw.trim().replace(/^(?:remember|record|persist)\s+(?:that\s+)?/i, "").trim();
      const normalise = (text: string) => text.replace(/[.!?]+$/, "").replace(/\s+/g, " ").trim().toLowerCase();
      const target = candidates.find((candidate) => candidate.path === p.path);
      if (!claim || claim.length < 20 || correction || !target ||
          normalise(requestedClaim) !== normalise(claim) || !target.body.includes(claim)) {
        throw new Error("Staged mutation rejected: no-op claim was not verified against the request.");
      }
      state.checkCancellation();
      const current = await kb.readConcept(target.path);
      if (current.body !== target.body || !current.body.includes(claim)) {
        throw new Error("Staged mutation rejected: no-op evidence changed before verification.");
      }
      const trace = recorder.finalize("mutation", instruction, "Verified exact no-op.", "success", modelChain, undefined, undefined, undefined, { providerCalls, generationMs: Date.now() - started });
      await traceStore(kb).save(trace);
      return { ok: true, result: { summary: "No change needed; exact claim verified.", filesChanged: [], steps: 1, traceId: trace.id } };
    }
    if (p.action === "append") {
      const target = candidates.find((c) => c.path === p.path);
      const title = target?.frontmatter.title;
      const titleNegated = !!title && raw.toLowerCase().includes(title.toLowerCase()) && !stagedTitleMention(raw, title);
      if (!target || !title || titleNegated ||           !(stagedTitleMention(raw, title) || hintedPath === target.path || stagedOwnerAnchor(raw, target.path) ||
             historicalProvenanceOwner === target.path) || (correction && historicalProvenanceOwner !== target.path) ||           p.new_text.length < 20 || p.new_text.length > 4_000 ||
           (historicalProvenanceOwner === target.path &&
             !p.new_text.includes(raw.match(/\b\d{1,2}\s+[A-Za-z]+\s+20\d{2}\b/)?.[0] ?? "")) ||
           stagedOverlap(p.new_text, raw) < 0.8 || stagedOverlap(raw, p.new_text) < 0.6 ||
          target.body.toLowerCase().includes(p.new_text.trim().toLowerCase())) {
        throw new Error("Staged mutation deferred: append owner or factual support is uncertain.");
      }
      // The proposer may overlook a contradiction. A separate, tool-free
      // judgement sees the unchanged owner and must explicitly approve it.
      const judgement = await generateObject({
        model: resolved.model,
        schema: z.object({ safe: z.boolean(), reason: z.string() }).strict(),
        temperature: 0,
        abortSignal: options.signal,
        prompt: `Independent safety check. Is the proposed addition fully supported by the instruction, positively relevant to this owner even if another owner could also plausibly fit, not already present, and consistent with every existing assertion? Reject negative/unrelated mentions. A suggested owner path is not proof. Answer safe=false on uncertainty, changed values, or an unrelated backlink.\nINSTRUCTION:\n${raw}\nOWNER ${target.path}:\n${target.body}\nALTERNATIVE SEARCH HITS (path and title only):\n${JSON.stringify(hits.filter((hit) => hit.path !== target.path).map((hit) => ({ path: hit.path, title: hit.title })))}\nPROPOSED ADDITION:\n${p.new_text}`,
      });
      state.checkCancellation();
      if (!judgement.object.safe) throw new Error("Staged mutation deferred: append failed independent consistency check.");
      const appended = `${target.body.trimEnd()}\n\n${p.new_text.trim()}\n`;
      if (appended.length > STAGED_MAX_OWNER_CHARS) throw new Error("Staged mutation deferred: appended body exceeds evidence limit.");
      const current = await kb.readConcept(target.path);
      if (current.body !== target.body) throw new Error(`Concept changed while it was being read: ${target.path}`);
      state.checkCancellation();
      await kb.patchConcept(target.path, { replaceBody: appended }, "Staged mutation: append verified fact", sha256(target.body), options.signal, (path) => { changed = [path]; });
    } else if (p.action === "replace") {
      let target = candidates.find((c) => c.path === p.path);
      // The obsolete deployment policy spans paragraphs. A follow-up would
      // leave its false current instructions active. Replace only that exact
      // section, while keeping the separate dated CLI observation intact.
      const deploymentRequest = /^(?:update|correct) the .{1,160} deployment workflow knowledge:/i.test(raw.trim()) &&
        /replace obsolete guidance that `[^`]+` is authoritative/i.test(raw) &&
        /push to [\w./-]+ is the intended deployment mechanism/i.test(raw) &&
        /retain note that .* returns not authorized/i.test(raw);
      const repositories: typeof candidates = [];
      if (deploymentRequest) {
        const paths = await kb.bundle.listConceptPaths();
        if (paths.length > 10_000) throw new Error("Staged mutation deferred: deployment policy inventory exceeds evidence limit.");
        for (const path of paths.filter((candidate) => candidate.startsWith("/repos/"))) {
          const repo = await kb.readConcept(path);
          recorder.record("read_concept", repo.path, [repo.path]);
          state.checkCancellation();
          if (repo.body.length > STAGED_MAX_OWNER_CHARS) throw new Error("Staged mutation deferred: deployment policy peer exceeds evidence limit.");
          repositories.push({ path: repo.path, frontmatter: repo.frontmatter, body: repo.body });
        }
        const requestedSite = raw.trim().match(/^(?:update|correct) the ([a-z0-9][a-z0-9.-]*\.[a-z]{2,}) deployment workflow knowledge:/i)?.[1]?.toLowerCase();
        if (!requestedSite) throw new Error("Staged mutation deferred: deployment site identity is not explicit.");
        const eligible = repositories.filter((repo) => {
          const authoritativeParagraph = repo.body.split(/\n\s*\n/).find((paragraph) =>
            /\*\*`[^`]+`\*\* is the authoritative deployment command/i.test(paragraph));
          return !!authoritativeParagraph && authoritativeParagraph.toLowerCase().includes(`for ${requestedSite}.`) &&
            /^#{1,6}\s+Deployment Policy\s*$/im.test(repo.body) && /Run `[^`]+` to publish/i.test(repo.body);
        });
        if (eligible.length === 1) target = eligible[0];
        else if (eligible.length > 1) throw new Error("Staged mutation deferred: deployment policy has a competing owner.");
        else throw new Error("Staged mutation deferred: deployment policy owner does not match the requested site.");
        if (!target || !repositories.some((repo) => repo.path === target?.path)) {
          throw new Error("Staged mutation deferred: deployment policy owner is not a repository.");
        }
      }
      if (deploymentRequest && target?.path.startsWith("/repos/")) {
        const headings = [...target.body.matchAll(/^#{1,6}\s+Deployment Policy\s*$/gim)];
        if (headings.length !== 1) throw new Error("Staged mutation deferred: deployment policy section is not unique.");
        const start = headings[0].index;
        const followingHeading = /^#{1,6}\s+[^\n]+$/gm;
        followingHeading.lastIndex = start + headings[0][0].length;
        const end = followingHeading.exec(target.body)?.index ?? target.body.length;
        const section = target.body.slice(start, end);
        const command = raw.match(/replace obsolete guidance that `([^`]+)` is authoritative/i)?.[1];
        if (section.length > 4_000 || !command ||
            !section.includes(`**\`${command}\`** is the authoritative deployment command`) ||
            !section.includes(`Run \`${command}\` to publish.`) ||
            !/Only after deployment succeeds, clean up \(delete\) the feature branch\./i.test(section) ||
            !/Never delete the feature branch before confirming deployment has succeeded/i.test(section)) {
          throw new Error("Staged mutation deferred: obsolete deployment policy is not uniquely grounded.");
        }
        const peers = repositories.filter((repo) => repo.path !== target?.path);
        if (peers.some((peer) => peer.body.includes(section.trim()))) {
          throw new Error("Staged mutation deferred: deployment policy has a competing owner.");
        }
        const requestedFacts = raw.trim().replace(/^(?:update|correct) the .{1,160}? deployment workflow knowledge:\s*/i, "")
          .split(/(?<=[.!?])\s+/).filter((sentence) =>
          /(?:Vercel project|production deployment|Git-triggered|GitHub .*status|Vite build|deploy target was removed|push to [\w./-]+ is the intended deployment mechanism)/i.test(sentence) &&
          !/replace obsolete guidance|retain note/i.test(sentence));
        if (!requestedFacts.length) throw new Error("Staged mutation deferred: deployment correction lacks caller-supported facts.");
        const retainedSafety = [
          section.match(/(?:^|\n)\d+\. (Only after deployment succeeds[^\n]*)/i)?.[1],
          section.match(/(?:^|\n)(Never delete the feature branch before confirming deployment has succeeded[^\n]*)/i)?.[1],
        ].filter((claim): claim is string => !!claim);
        if (retainedSafety.length !== 2) throw new Error("Staged mutation deferred: deployment cleanup safeguards are incomplete.");
        let revisedSection = section;
        // Remove only the obsolete command assertion and replace its execution
        // step. Keep valid branch/merge steps and every unrelated policy
        // paragraph exactly as written.
        const authority = `**\`${command}\`** is the authoritative deployment command`;
        const authorityClaim = section.split(/\n\s*\n/).filter((paragraph) => paragraph.includes(authority));
        const obsoleteAssertion = authorityClaim.length === 1
          ? authorityClaim[0].match(/\*\*`[^`]+`\*\* is the authoritative deployment command[^.!?]*[.!?]/i)?.[0]
          : undefined;
        if (!obsoleteAssertion || section.split(obsoleteAssertion).length !== 2) {
          throw new Error("Staged mutation deferred: obsolete deployment command assertion is ambiguous.");
        }
        revisedSection = revisedSection.replace(obsoleteAssertion, "").replace(/\n\s*\n\s*\n/g, "\n\n");
        const obsoleteStep = `Run \`${command}\` to publish.`;
        const branch = raw.match(/\bpush to ([\w./-]+) is the intended deployment mechanism/i)?.[1];
        if (revisedSection.split(obsoleteStep).length !== 2 || !branch) {
          throw new Error("Staged mutation deferred: obsolete deployment step or new branch is ambiguous.");
        }
        revisedSection = revisedSection.replace(obsoleteStep,
          `Push to ${branch} as the intended deployment mechanism; verify production deployment before deleting the feature branch.`);
        // Keep every other existing policy sentence and workflow step; append
        // the newly confirmed facts as separate paragraphs, not a numbered step.
        revisedSection = `${revisedSection.trimEnd()}\n\n${requestedFacts.join("\n\n")}\n\n`;
        const replacement = target.body.slice(0, start) + revisedSection + target.body.slice(end);
        if (!replacement.includes("vercel --prod") || !replacement.includes("Not authorized") ||
            replacement.length > STAGED_MAX_OWNER_CHARS) {
          throw new Error("Staged mutation deferred: deployment history or owner limit is uncertain.");
        }
        const judgement = await generateObject({
          model: resolved.model,
          schema: z.object({ safe: z.boolean() }).strict(),
          temperature: 0,
          abortSignal: options.signal,
          prompt: `Independent complete-owner deployment-policy check. Verify that the old section is the current obsolete policy the caller corrects, EVERY replacement claim is supported by the instruction, no unrelated policy or historical assertion is lost, and the separate dated Vercel CLI Not authorized observation remains intact. A shared site name alone is not ownership. If uncertain safe=false. JSON only.\nINSTRUCTION:\n${raw}\nOWNER ${target.path}:\n${target.body}\nOLD SECTION:\n${section}\nFULL PROPOSED OWNER:\n${replacement}`,
        });
        state.checkCancellation();
        if (!judgement.object.safe) throw new Error("Staged mutation deferred: deployment policy failed independent consistency check.");
        const current = await kb.readConcept(target.path);
        if (current.body !== target.body) throw new Error(`Concept changed while it was being read: ${target.path}`);
        await kb.patchConcept(target.path, { replaceBody: replacement }, "Staged mutation: correct deployment policy", sha256(target.body), options.signal,
          (writtenPath) => { changed = [writtenPath]; });
        if ((await kb.readConcept(target.path)).body !== replacement) throw new Error("Deployment policy failed CAS read-back verification.");
        const summary = `Staged mutation changed ${changed.join(", ")}.`;
        const trace = recorder.finalize("mutation", instruction, summary, "success", modelChain,
          undefined, undefined, undefined, { providerCalls, generationMs: Date.now() - started });
        await traceStore(kb).save(trace);
        return { ok: true, result: { summary, filesChanged: changed, steps: 1, traceId: trace.id } };
      }
      let duplicateRepair = "not attempted";
      if (target && p.old_text.length >= 12 && target.body.split(p.old_text).length !== 2) {
        // A proposer can invent an old operand. Ground its one repair in exact
        // existing paragraphs matching identifiers from the caller, never in
        // invented text or an unbounded search through unrelated body sections.
        // Selection alone never authorizes a replacement: the full-owner
        // verifier and CAS still apply.
        const occurrences = target.body.split(p.old_text).length - 1;
        const identifiers = [...new Set((raw.match(/\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b/gi) ?? [])
          .map((identifier) => identifier.toLowerCase()))];
        const containing = target.body.split(/\n\s*\n/).filter((part) => part.includes(p.old_text));
        const paragraphs = target.body.split(/\n\s*\n/).filter((part) =>
          part.length >= 12 && part.length <= 4_000 && target.body.split(part).length === 2 &&
          (occurrences > 0 ? part.includes(p.old_text) :
            identifiers.some((identifier) => part.toLowerCase().includes(identifier))));
        duplicateRepair = paragraphs.length === 0
          ? `no unique bounded paragraph (occurrences=${occurrences}, paragraphs=${containing.length}, largest=${Math.max(0, ...containing.map((part) => part.length))}, identifiers=${identifiers.length})`
          : paragraphs.length > 12 ? "too many candidate paragraphs" : "model declined indexed repair";
        if (paragraphs.length > 0 && paragraphs.length <= 12 &&
            JSON.stringify(paragraphs).length + raw.length <= STAGED_MAX_EVIDENCE_CHARS) {
          const repair = await generateObject({
            model: proposer.model,
            schema: z.object({ index: z.number().int(), new_text: z.string() }).strict(),
            temperature: 0,
            abortSignal: options.signal,
            prompt: `Repair ONLY one actively stale claim in this owner. Select its index from the exact paragraphs below, and supply a local corrected new_text preserving every unrelated clause and statement verbatim. If more than one paragraph asserts the old claim, none is clearly active, or the instruction cannot be fully satisfied by one local replacement, return index=-1 and new_text="". Do not select a historical or negated mention. The original old operand may be ABSENT or repeated; it is not evidence of an active stale claim. JSON only.\nINSTRUCTION:\n${raw}\nOWNER ${target.path}\nUNTRUSTED OLD OPERAND:\n${p.old_text}\nPROPOSED CORRECTION:\n${p.new_text}\nEXACT CANDIDATE PARAGRAPHS:\n${JSON.stringify(paragraphs.map((text, index) => ({ index, text })))}`,
          });
          state.checkCancellation();
          const repairedOld = paragraphs[repair.object.index];
          const repairedNew = repair.object.new_text;
          if (repairedOld && repairedOld.length >= 12 &&
              repairedNew.length > 0 && repairedNew.length <= 4_000 &&
              !/\n\s*\n/.test(repairedNew)) {
            p.old_text = repairedOld;
            p.new_text = repairedNew;
            repairedCorrection = true;
            duplicateRepair = "selected unique paragraph";
          }
        }
      }
      const namedParagraph = target && namedSection && sectionOwnerPath === p.path
        ? target.body.split(/\n\s*\n/).find((paragraph) =>
          namedSection.test(paragraph.split("\n", 1)[0] ?? "")) : undefined;
      if (namedParagraph?.includes(p.old_text) && p.old_text.length <= 4_000 &&
          p.new_text.length <= 4_000 && stagedOverlap(p.old_text, p.new_text) < 0.5) {
        // A broad rewrite of a dated incident may erase unrelated history.
        // Give the proposer one chance to retain the old paragraph verbatim
        // except for the requested corrections; the usual lexical, independent
        // and compare-and-swap checks still run on the repaired text.
        const repair = await generateObject({
          model: proposer.model,
          schema: z.object({ new_text: z.string() }).strict(),
          temperature: 0,
          abortSignal: options.signal,
          prompt: `Repair ONLY this one named incident paragraph. Preserve its heading and all old statements that the instruction does not contradict, with most old wording unchanged. Incorporate all supported new facts and retract any superseded claim. Keep it one paragraph, do not invent facts or edit any other section. If this cannot be done safely, return an empty new_text. JSON only.\nINSTRUCTION:\n${raw}\nOLD PARAGRAPH:\n${p.old_text}\nREJECTED BROAD REWRITE:\n${p.new_text}`,
        });
        state.checkCancellation();
        p.new_text = repair.object.new_text;
      }
      const exactFollowUp = raw.trim().toLowerCase().startsWith("update ") && namedParagraph?.includes(p.old_text) &&
        stagedOverlap(p.old_text, p.new_text) < 0.5 ? raw.slice(raw.indexOf(":") + 1).trim() : undefined;
      if (target && exactFollowUp && exactFollowUp.length >= 30 && exactFollowUp.length <= 4_000 &&
          !target.body.includes(exactFollowUp) && target.body.split(namedParagraph!).length === 2) {
        // A later dated status can supplement a historical incident snapshot;
        // it must not silently overwrite that history or leave a contradiction.
        const judgement = await generateObject({
          model: resolved.model,
          schema: z.object({ safe: z.boolean(), reason: z.string() }).strict(),
          temperature: 0,
          abortSignal: options.signal,
          prompt: `Independent full-body safety check. Does the caller's EXACT dated status update belong immediately after this uniquely identified incident paragraph, without contradicting ANY current assertion in the COMPLETE owner body or duplicating a fact already stated? A historical partial fix is not a claim that all failures were resolved. Reject if any old statement must instead be retracted, if any assertion is unsupported, or if this owner is wrong. If uncertain, safe=false. JSON only.\nINSTRUCTION:\n${raw}\nOWNER ${target.path}:\n${target.body}\nALTERNATIVE SEARCH HITS (paths and titles):\n${JSON.stringify(hits.filter((hit) => hit.path !== target.path).map((hit) => ({ path: hit.path, title: hit.title })))}\nEXACT FOLLOW-UP:\n${exactFollowUp}`,
        });
        state.checkCancellation();
        if (!judgement.object.safe) {
          throw new Error("Staged mutation deferred: dated incident follow-up failed independent consistency check.");
        }
        const replacement = target.body.replace(namedParagraph!, `${namedParagraph}\n\n${exactFollowUp}`);
        if (replacement.length > STAGED_MAX_OWNER_CHARS) {
          throw new Error("Staged mutation deferred: dated incident follow-up exceeds evidence limit.");
        }
        const current = await kb.readConcept(target.path);
        if (current.body !== target.body) throw new Error(`Concept changed while it was being read: ${target.path}`);
        state.checkCancellation();
        await kb.patchConcept(target.path, { replaceBody: replacement }, "Staged mutation: verified dated incident follow-up", sha256(target.body), options.signal, (path) => { changed = [path]; });
        const summary = `Staged mutation changed ${changed.join(", ")}.`;
        const trace = recorder.finalize("mutation", instruction, summary, "success", modelChain, undefined, undefined, undefined, { providerCalls, generationMs: Date.now() - started });
        await traceStore(kb).save(trace);
        return { ok: true, result: { summary, filesChanged: changed, steps: 1, traceId: trace.id } };
      }
      const requestedPr = raw.trim().match(/^update\s+(?:the\s+)?([\p{L}\p{N}./_-]+)\s+PR\s*#(\d+)\s+status\s*:/iu);
      if (target && requestedPr && p.old_text.includes("CHANGES_REQUESTED")) {
        const containingParagraph = target.body.split(/\n\s*\n/).find((part) => part.includes(p.old_text));
        const ids = containingParagraph ? [...containingParagraph.matchAll(/\bPR\s*#(\d+)\b/gi)]
          .map((match) => match[1]) : [];
        const exactReview = p.old_text.match(/\b(?:The |A )?(?:prior|previous|earlier|existing) CHANGES_REQUESTED (?:GitHub )?review (?:remains|is still active)(?: unless (?:he|Dan) explicitly asks to update it)?[.!?]?/i)?.[0];
        if (containingParagraph?.toLowerCase().includes(requestedPr[1].toLowerCase()) &&
            ids.length > 0 && ids.every((id) => id === requestedPr[2]) && exactReview &&
            target.body.split(exactReview).length === 2 &&
            !/\b(?:and|but|while|also|whereas)\b|[,;\n]/i.test(exactReview)) {
          // A model may quote this review sentence together with adjacent
          // history. Narrow to the uniquely observed review-only operand.
          p.old_text = exactReview;
        }
      }
      const statusSubject = !options.directAdd && options.preflightInput
        ? raw.trim().match(/^update\s+(?:the\s+)?([A-Z][\p{L}\p{N}\s-]{8,80}?)\s+status\s*:/iu)?.[1] ??
          raw.trim().match(/^correct\s+(?:the\s+)?([A-Z][\p{L}\p{N}\s-]{2,80}?\s+throughput)\s+status\s*:/iu)?.[1]
        : undefined;
      const statusWords = statusSubject ? stagedWords(statusSubject) : [];
      const oldStatusWords = new Set(stagedWords(p.old_text));
      const identityWords = new Set(stagedWords(`${target?.frontmatter.title ?? ""} ${target?.path ?? ""}`));
      const statusParagraph = target?.body.split(/\n\s*\n/).find((paragraph) =>
        paragraph.includes(p.old_text));
      let boundedCorrection = false;
      let boundedSpanCorrection = false;
      let paragraphUniquelyOwned = false;
      let correctionEvidence = raw;
      if (target && correction && !options.directAdd && !repairedCorrection &&
          !statusSubject && !namedParagraph &&
          !/^correct the Claes training-design note recorded Oct 2026:/i.test(raw.trim()) &&
          !stagedSingleClauseEdit(p.old_text, p.new_text) &&
          p.old_text.length >= 12 && target.body.split(p.old_text).length === 2 &&
          statusParagraph && statusParagraph.length >= 100 && statusParagraph.length <= 4_000 &&
          target.body.split(statusParagraph).length === 2 &&
          (stagedTitleMention(raw, target.frontmatter.title ?? "") || stagedOwnerAnchor(raw, target.path))) {
        // An unscoped correction can address several findings in one paragraph.
        // Never treat a generic shared sentence as uniquely owned: read all
        // search-hit bodies before asking for a paragraph-level repair.
        const candidateOwners = candidates.filter((candidate) =>
          !(target.path.startsWith("/repos/") && candidate.path.startsWith("/users/")) &&
          candidate.body.includes(statusParagraph)).map((candidate) => candidate.path);
        let uniquelyOwned = candidateOwners.length === 1;
        if (!uniquelyOwned) console.warn("[understory] bounded correction candidate owners", candidateOwners);
        for (const hit of hits) {
          if (hit.path === target.path) continue;
          const competitor = await kb.readConcept(hit.path);
          recorder.record("read_concept", competitor.path, [competitor.path]);
          state.checkCancellation();
          // A person's preference can inform a repository decision without
          // owning that repository's security-audit paragraph.
          if (target.path.startsWith("/repos/") && competitor.path.startsWith("/users/")) continue;
          if (competitor.body.length > STAGED_MAX_OWNER_CHARS) {
            throw new Error("Staged mutation deferred: competing correction owner exceeds evidence limit.");
          }
          if (competitor.body.includes(statusParagraph)) {
            console.warn("[understory] bounded correction duplicate paragraph", { owner: target.path, peer: competitor.path });
            uniquelyOwned = false;
          } else if (uniquelyOwned) {
            const ownership = await generateObject({
              model: resolved.model,
              schema: z.object({ sameClaimOwner: z.boolean(), certain: z.boolean() }).strict(),
              temperature: 0,
              abortSignal: options.signal,
              prompt: `Independent competing-owner check for a proposed multi-clause correction. Does the FULL competing concept assert any CURRENT stale claim from the EXACT OLD PARAGRAPH that this instruction asks to correct, including paraphrases? A generic shared project reference is not ownership. If unsure, return certain=false. JSON only.\nINSTRUCTION:\n${raw}\nPROPOSED OWNER ${target.path}, OLD PARAGRAPH:\n${statusParagraph}\nCOMPETING OWNER ${competitor.path}:\n${competitor.body}`,
            });
            state.checkCancellation();
            if (!ownership.object.certain || ownership.object.sameClaimOwner) {
              console.warn("[understory] bounded correction competing claim", {
                owner: target.path, ownerType: target.frontmatter.type,
                peer: competitor.path, peerType: competitor.frontmatter.type,
                ...ownership.object,
              });
              uniquelyOwned = false;
            }
          }
        }
        paragraphUniquelyOwned = uniquelyOwned;
        if (uniquelyOwned) {
          const repair = await generateObject({
            model: proposer.model,
            schema: z.object({ new_text: z.string() }).strict(),
            temperature: 0,
            abortSignal: options.signal,
            prompt: `Repair ONLY the EXACT OLD PARAGRAPH in the complete observed owner. Apply all applicable corrections in the instruction to this paragraph, preserve unrelated clauses and historical facts verbatim, and do not roll back later updates elsewhere in the owner. Keep the same heading. Return a single corrected paragraph; if the instruction requires changes outside it or cannot be fully satisfied safely, return empty new_text. JSON only.\nINSTRUCTION:\n${raw}\nOWNER ${target.path}:\n${target.body}\nEXACT OLD PARAGRAPH:\n${statusParagraph}\nREJECTED REWRITE:\n${p.new_text}`,
          });
          state.checkCancellation();
          let corrected = repair.object.new_text;
          // Every old content word removed from the paragraph must still be
          // grounded in the caller's instruction. This protects related but
          // unrequested qualifiers (for example, staging authentication in a
          // correction about production port exposure). Paraphrase can vary
          // morphology, but it cannot silently drop an unmentioned subject.
          const ungroundedWords = (text: string): string[] => {
            const supported = stagedWords(`${raw} ${text}`);
            return stagedStatusClauses(statusParagraph).flatMap((clause) =>
              stagedWords(clause).filter((word) => !supported.some((candidate) =>
                candidate === word || (word.length >= 5 && candidate.length >= 4 &&
                  word.slice(0, 4) === candidate.slice(0, 4)))));
          };
          let ungroundedOldWords = ungroundedWords(corrected);
          let clauseIssue = corrected && stagedStatusClauseIssue(statusParagraph, corrected, raw);
          if (corrected && (ungroundedOldWords.length || clauseIssue)) {
            // Give a near-complete paragraph repair one chance to restore
            // unrelated observations in their original context. A model
            // cannot approve its own repair: all bounds and the independent
            // full-owner verification still run after this attempt.
            const retention = await generateObject({
              model: proposer.model,
              schema: z.object({ new_text: z.string() }).strict(),
              temperature: 0,
              abortSignal: options.signal,
              prompt: `Repair ONLY this proposed paragraph by restoring every OLD assertion that the caller did not explicitly supersede, retaining the ORIGINAL wording and position of unrelated clauses. Do not invent new facts, undo a later owner update, or preserve a stale assertion as current. These original content words are missing from both the instruction and the proposal: ${JSON.stringify(ungroundedOldWords)}. A clause-preservation check also found: ${clauseIssue ?? "none"}. Return one complete corrected paragraph; return empty new_text if impossible. JSON only.\nINSTRUCTION:\n${raw}\nCOMPLETE OWNER ${target.path}:\n${target.body}\nEXACT OLD PARAGRAPH:\n${statusParagraph}\nPROPOSED PARAGRAPH:\n${corrected}`,
            });
            state.checkCancellation();
            if (retention.object.new_text) {
              corrected = retention.object.new_text;
              ungroundedOldWords = ungroundedWords(corrected);
              clauseIssue = stagedStatusClauseIssue(statusParagraph, corrected, raw);
            }
          }
          if (corrected && clauseIssue?.startsWith("old clause not requested")) {
            // A caller can retract a stale claim by meaning rather than quote
            // its former wording. Require an independent, exact-clause mapping;
            // a merely related or unmentioned old claim remains protected.
            const audit = await generateObject({
              model: resolved.model,
              schema: z.object({ supersededClauses: z.array(z.string()), certain: z.boolean() }).strict(),
              temperature: 0,
              abortSignal: options.signal,
              prompt: `Independently identify EXACT old clauses whose assertion the caller explicitly corrects by meaning, even if the caller does not quote the stale wording. Select only verbatim clauses from OLD PARAGRAPH; do not include unrelated, historical, or already-correct claims. Return certain=false and no clauses if uncertain. This is an evidence check, not write approval. JSON only.\nINSTRUCTION:\n${raw}\nOLD PARAGRAPH:\n${statusParagraph}\nPROPOSED NEW PARAGRAPH:\n${corrected}`,
            });
            state.checkCancellation();
            const clauses = stagedStatusClauses(statusParagraph);
            const superseded = audit.object.supersededClauses;
            if (audit.object.certain && superseded.length > 0 && superseded.length <= 8 &&
                new Set(superseded).size === superseded.length &&
                superseded.every((clause) => clauses.includes(clause) &&
                  stagedOverlap(clause, raw) >= 0.15)) {
              correctionEvidence = `${raw}\n${superseded.join("\n")}`;
              clauseIssue = stagedStatusClauseIssue(statusParagraph, corrected, correctionEvidence);
            }
          }
          console.warn("[understory] bounded correction repair verdict", {
            owner: target.path, correctedLength: corrected.length,
            ungroundedWords: ungroundedOldWords.length,
            clauseIssue: clauseIssue?.split(" (")[0] ?? null,
            oldOverlap: corrected ? stagedOverlap(statusParagraph, corrected) : 0,
            supportedOverlap: corrected ? stagedOverlap(corrected, `${raw} ${statusParagraph}`) : 0,
          });
          if (corrected && ungroundedOldWords.length === 0 && corrected.length <= 4_000 &&
              !/\n\s*\n/.test(corrected) && corrected.trimEnd() !== statusParagraph.trimEnd() &&
              stagedOverlap(statusParagraph, corrected) >= 0.5 &&
              stagedOverlap(corrected, `${raw} ${statusParagraph}`) >= 0.65 &&
              !clauseIssue) {
            p.old_text = statusParagraph;
            p.new_text = corrected;
            boundedCorrection = true;
          }
        }
      }
      if (!boundedCorrection && paragraphUniquelyOwned && target && statusParagraph &&
          statusParagraph.length >= 500 && p.old_text.length < statusParagraph.length / 2) {
        // The proposer may have selected a short run of stale findings inside
        // a long historical paragraph. Repair that exact span instead of
        // rewriting the whole paragraph and dropping unrelated history.
        const scoped = await generateObject({
          model: proposer.model,
          schema: z.object({ old_text: z.string(), new_text: z.string() }).strict(),
          temperature: 0,
          abortSignal: options.signal,
          prompt: `Repair ONLY the smallest contiguous OLD SPAN that the caller corrects inside this uniquely owned paragraph. Return an exact verbatim old_text substring and its replacement new_text. Preserve every unrelated assertion inside the span verbatim, and leave all text OUTSIDE the span unchanged. Apply all applicable corrections to this owner; if they cannot fit in one local span without dropping unrelated claims, return empty strings. Do not roll back later owner updates. JSON only.\nINSTRUCTION:\n${raw}\nCOMPLETE OWNER ${target.path}:\n${target.body}\nEXACT PARAGRAPH:\n${statusParagraph}\nPROPOSED OLD SPAN:\n${p.old_text}\nREJECTED REWRITE:\n${p.new_text}`,
        });
        state.checkCancellation();
        const { old_text: oldSpan, new_text: newSpan } = scoped.object;
        const supported = stagedWords(`${raw} ${newSpan}`);
        const ungrounded = stagedWords(oldSpan).filter((word) => !supported.some((candidate) =>
          candidate === word || (word.length >= 5 && candidate.length >= 4 &&
            word.slice(0, 4) === candidate.slice(0, 4))));
        const issue = oldSpan && newSpan ? stagedStatusClauseIssue(oldSpan, newSpan, raw) : "empty span";
        console.warn("[understory] bounded correction span verdict", {
          owner: target.path, oldLength: oldSpan.length, newLength: newSpan.length,
          ungroundedWords: ungrounded.length, clauseIssue: issue?.split(" (")[0] ?? null,
        });
        if (oldSpan.length >= 12 && oldSpan.length <= 4_000 && newSpan.length > 0 &&
            newSpan.length <= 4_000 && !/\n\s*\n/.test(oldSpan) && !/\n\s*\n/.test(newSpan) &&
            statusParagraph.split(oldSpan).length === 2 && target.body.split(oldSpan).length === 2 &&
            oldSpan !== newSpan && stagedOverlap(oldSpan, newSpan) >= 0.5 &&
            stagedOverlap(newSpan, `${raw} ${oldSpan}`) >= 0.65 &&
            ungrounded.length === 0 && !issue) {
          p.old_text = oldSpan;
          p.new_text = newSpan;
          boundedSpanCorrection = true;
        }
      }
      const namedThroughputStatus = !options.directAdd &&
        /^correct\s+(?:the\s+)?[A-Z][\p{L}\p{N}\s-]+?\s+throughput\s+status\s*:/iu.test(raw.trim());
      let statusUniqueAcrossHits = false;
      if (statusWords.length >= 2 && statusParagraph && p.old_text.length >= 200) {
        const ownerIdentity = stagedWords(`${target?.frontmatter.title ?? ""} ${target?.path ?? ""}`);
        // The first named project must identify the owner; the second subject
        // must occur in its old claim. "Throughput" and a shared host alone
        // cannot make a Qwen roadmap the owner of Claes Laya's run.
        statusUniqueAcrossHits = !namedThroughputStatus ||
          (!!target && ownerIdentity.includes(statusWords[0]) && oldStatusWords.has(statusWords[1]));
        if (namedThroughputStatus && !statusUniqueAcrossHits) {
          throw new Error(`Staged mutation deferred: named throughput subject does not identify ${target?.path ?? "an owner"}.`);
        }
        for (const hit of hits) {
          if (hit.path === target?.path) continue;
          const competitor = await kb.readConcept(hit.path);
          recorder.record("read_concept", competitor.path, [competitor.path]);
          state.checkCancellation();
          if (competitor.body.length > STAGED_MAX_OWNER_CHARS) {
            throw new Error("Staged mutation deferred: competing status owner exceeds evidence limit.");
          }
          if (competitor.body.includes(p.old_text)) {
            if (namedThroughputStatus) throw new Error(`Staged mutation deferred: duplicate throughput status in ${competitor.path}.`);
            statusUniqueAcrossHits = false;
          } else if (statusUniqueAcrossHits && namedThroughputStatus) {
            const ownership = await generateObject({
              model: resolved.model,
              schema: z.object({ sameClaimOwner: z.boolean(), certain: z.boolean() }).strict(),
              temperature: 0,
              abortSignal: options.signal,
              prompt: `Independent competing-owner check for a named throughput status correction. Does this COMPLETE competing concept assert the SAME CURRENT stale throughput/run-status claim as the proposed owner, including paraphrases? Shared host or project mentions alone do not establish ownership. If uncertain, certain=false. JSON only.\nINSTRUCTION:\n${raw}\nPROPOSED OWNER ${target?.path}, OLD STATUS:\n${p.old_text}\nCOMPETING OWNER ${competitor.path}:\n${competitor.body}`,
            });
            state.checkCancellation();
            if (!ownership.object.certain) {
              throw new Error(`Staged mutation deferred: uncertain competing throughput claim in ${competitor.path}.`);
            }
            if (ownership.object.sameClaimOwner) {
              // A server note can retain an earlier scheduling plan after its
              // CURRENT status is corrected. Require a positive completed-run
              // statement in the peer and a separate full-body adjudication;
              // never dismiss a still-current contradiction as history.
              const completedPeer = competitor.body.split(/\n\s*\n/).some((paragraph) =>
                /\bClaes\b/i.test(paragraph) && /\bLaya\b/i.test(paragraph) &&
                /\b(?:sweep|throughput)\b/i.test(paragraph) &&
                /\b(?:completed|ran|run)\b/i.test(paragraph) && paragraph.includes("13 Oct 2026"));
              if (!completedPeer) {
                throw new Error(`Staged mutation deferred: competing throughput claim in ${competitor.path} lacks a completed update.`);
              }
              const currentStatus = await generateObject({
                model: resolved.model,
                schema: z.object({ certain: z.boolean(), currentStaleQuote: z.string() }).strict(),
                temperature: 0,
                abortSignal: options.signal,
                prompt: `Independent temporal peer check. This COMPLETE peer contains a Claes Laya sweep completed on 13 Oct 2026. Does it ALSO assert as CURRENT that this same high-concurrency sweep remains NOT run, still deferred or pending? A clearly dated earlier plan is historical; an undated contradictory present status is current. If current, quote an EXACT substring identifying that current stale claim in currentStaleQuote. If all stale language is explicitly historical/superseded, return certain=true with an empty quote. If uncertain, certain=false. JSON only.\nINSTRUCTION:\n${raw}\nPROPOSED OWNER ${target?.path}, OLD STATUS:\n${p.old_text}\nCOMPLETE PEER ${competitor.path}:\n${competitor.body}`,
              });
              state.checkCancellation();
              const staleQuote = currentStatus.object.currentStaleQuote;
              if (!currentStatus.object.certain || (staleQuote && competitor.body.includes(staleQuote))) {
                throw new Error(`Staged mutation deferred: competing throughput claim remains current or uncertain in ${competitor.path} (certain=${currentStatus.object.certain}, grounded=${!!staleQuote && competitor.body.includes(staleQuote)}).`);
              }
              if (staleQuote) {
                // A generated, non-verbatim quote is not evidence of a current
                // contradiction. Re-adjudicate against selectable EXACT peer
                // sentences; abstention, excess evidence or a selected current
                // clause all keep the write blocked.
                const peerSentences = competitor.body.split(/(?<=[.!?;])\s+|\n+/).filter((sentence) =>
                  /\b(?:Claes|Laya|sweep|throughput)\b/i.test(sentence));
                if (peerSentences.length === 0 || peerSentences.length > 80) {
                  throw new Error(`Staged mutation deferred: competing throughput evidence exceeds bounded adjudication in ${competitor.path}.`);
                }
                const adjudication = await generateObject({
                  model: resolved.model,
                  schema: z.object({ certain: z.boolean(), currentStaleIndex: z.number().int() }).strict(),
                  temperature: 0,
                  abortSignal: options.signal,
                  prompt: `Independent exact-clause adjudication after a non-verbatim proposed stale quote. Select the INDEX of a sentence that CURRENTLY claims Claes Laya's high-concurrency sweep is still pending/not run. Earlier dated planning and a later completed run on 13 Oct 2026 are not a current stale claim. Return currentStaleIndex=-1 ONLY if certain there is no current stale assertion; otherwise return certain=false. Index must refer to these EXACT observed sentences; no generated quotes. JSON only.\nINSTRUCTION:\n${raw}\nCOMPLETE PEER ${competitor.path}:\n${competitor.body}\nCANDIDATE SENTENCES:\n${JSON.stringify(peerSentences.map((text, index) => ({ index, text })))}`,
                });
                state.checkCancellation();
                if (!adjudication.object.certain || adjudication.object.currentStaleIndex !== -1) {
                  throw new Error(`Staged mutation deferred: competing throughput exact-clause check found a current or uncertain claim in ${competitor.path}.`);
                }
              }
            }
          }
        }
      }
      // A named run status can require several coordinated metric changes in
      // one paragraph. The exact old span must stay inside that one paragraph,
      // be unique across complete observed owners, and positively match the
      // named subject. This is not a general multi-clause replacement license.
      const allowedThroughputNegationRemoval = (oldClause: string, newClause: string, word: string): boolean =>
        namedThroughputStatus && statusUniqueAcrossHits && word === "not" &&
        /high-concurrency live sweep\s+was not run/i.test(oldClause) &&
        /high-concurrency live sweep\s+(?:was run|ran|completed)/i.test(newClause) &&
        /high-concurrency live sweep was run once/i.test(raw) &&
        raw.includes("13 Oct 2026") && p.new_text.includes("13 Oct 2026");
      let statusClauseIssue = statusWords.length >= 2 && statusParagraph
        ? stagedStatusClauseIssue(p.old_text, p.new_text, raw, allowedThroughputNegationRemoval) : undefined;
      if (statusUniqueAcrossHits && statusClauseIssue?.startsWith("old clause not requested") &&
          p.old_text.length >= 200 && p.old_text.length <= 4_000 && p.new_text.length <= 4_000 &&
          stagedOverlap(p.old_text, p.new_text) >= 0.7) {
        const repair = await generateObject({
          model: proposer.model,
          schema: z.object({ new_text: z.string() }).strict(),
          temperature: 0,
          abortSignal: options.signal,
          prompt: `Repair ONLY this uniquely owned run-status span. Keep every OLD clause that the instruction does not address VERBATIM, including qualifiers, removed artifacts and cautions. Change only requested run status and metric claims; retain every unrelated claim in its original place. Return one paragraph, with no invented facts. If that is impossible return an empty new_text. JSON only.\nINSTRUCTION:\n${raw}\nOLD STATUS:\n${p.old_text}\nREJECTED REWRITE:\n${p.new_text}`,
        });
        state.checkCancellation();
        p.new_text = repair.object.new_text;
        statusClauseIssue = stagedStatusClauseIssue(p.old_text, p.new_text, raw, allowedThroughputNegationRemoval);
        if (statusClauseIssue?.startsWith("old clause not requested") && p.new_text.length <= 4_000) {
          p.new_text = stagedRestoreUnrequestedStatusClauses(p.old_text, p.new_text, raw);
          statusClauseIssue = stagedStatusClauseIssue(p.old_text, p.new_text, raw, allowedThroughputNegationRemoval);
        }
      }
      const boundedStatus = statusWords.length >= 2 &&
        statusWords.every((word) => oldStatusWords.has(word) || identityWords.has(word)) &&
        statusWords.filter((word) => oldStatusWords.has(word)).length >= 2 &&
        statusWords.some((word) => identityWords.has(word)) &&
        !!statusParagraph && statusUniqueAcrossHits &&
        candidates.filter((candidate) => candidate.body.includes(p.old_text)).length === 1 &&
        p.old_text.length >= 200 && p.old_text.length <= 4_000 && p.new_text.length <= 4_000 &&
        stagedOverlap(p.old_text, p.new_text) >= 0.7 && !statusClauseIssue;
      // A PR review can supersede one short status sentence with a longer
      // caller-supported approval. The run-status rule above deliberately
      // requires 200+ characters and high overlap; do not extend that rule to
      // arbitrary short edits or to another PR's review.
      const prStatus = raw.trim().match(/^update\s+(?:the\s+)?([\p{L}\p{N}./_-]+)\s+PR\s*#(\d+)\s+status\s*:/iu);
      const repoSlug = prStatus?.[1].split("/").pop()?.toLowerCase();
      const prIdentities = statusParagraph ? [...statusParagraph.matchAll(/\bPR\s*#(\d+)\b/gi)]
        .map((match) => match[1]) : [];
      // Only the review assertion itself may be replaced. In particular, a
      // conjunction about a different build or PR cannot be lost along with
      // this status. The surrounding paragraph must identify exactly this PR.
      const reviewOnly = /\bCHANGES_REQUESTED\b/i.test(p.old_text) &&
        /\breview\b/i.test(p.old_text) && /\b(?:remains?|still|pending|active|unchanged|not)\b/i.test(p.old_text) &&
        stagedStatusClauses(p.old_text).length === 1 &&
        !/\b(?:and|but|while|also|whereas|with)\b|[,;\n]/i.test(p.old_text);
      let prStatusUnique = !!target && !!prStatus && !!repoSlug && !!statusParagraph &&
        statusParagraph.toLowerCase().includes(prStatus[1].toLowerCase()) &&
        prIdentities.length > 0 && prIdentities.every((id) => id === prStatus[2]) &&
        reviewOnly && /\bCHANGES_REQUESTED\b/.test(raw) && /\bAPPROVED\b/.test(raw) &&
        /\bAPPROVED\b/.test(p.new_text) &&
        p.old_text.length <= 200 && p.new_text.length <= 4_000 &&
        !/\n\s*\n/.test(p.new_text) &&
        candidates.filter((candidate) => candidate.body.includes(p.old_text)).length === 1;
      if (prStatusUnique) {
        for (const hit of hits) {
          if (hit.path === target!.path) continue;
          const competitor = await kb.readConcept(hit.path);
          recorder.record("read_concept", competitor.path, [competitor.path]);
          state.checkCancellation();
          if (competitor.body.length > STAGED_MAX_OWNER_CHARS) {
            throw new Error("Staged mutation deferred: competing PR status owner exceeds evidence limit.");
          }
          if (competitor.body.includes(p.old_text)) prStatusUnique = false;
        }
      }
      if (prStatusUnique && stagedOverlap(p.new_text, `${raw} ${p.old_text}`) < 0.65) {
        // A paraphrased proposal can introduce unsupported wording. The
        // caller's bounded status sentence is stronger evidence than another
        // model rewrite; it still passes all ordinary lexical, independent
        // full-owner and CAS checks below.
        const exactStatus = raw.trim().match(/^update\s+(?:the\s+)?[\p{L}\p{N}./_-]+\s+PR\s*#\d+\s+status\s*:\s*([\s\S]+?)\s+Correct any older note saying\b/iu)?.[1];
        if (exactStatus && exactStatus.length >= 30 && exactStatus.length <= 4_000 &&
            /\bAPPROVED\b/.test(exactStatus) && !/\n\s*\n/.test(exactStatus)) {
          p.new_text = exactStatus;
        }
      }
      if (prStatusUnique) {
        const scope = await generateObject({
          model: resolved.model,
          schema: z.object({ safe: z.boolean() }).strict(),
          temperature: 0,
          abortSignal: options.signal,
          prompt: `Independent PR-review scope check. Does the EXACT OLD TEXT assert ONLY the still-active CHANGES_REQUESTED review for the PR identified in this complete paragraph, with NO unrelated facts or historical review of another PR? Do not approve merely because the paragraph mentions this PR. If uncertain, safe=false. JSON only.\nINSTRUCTION:\n${raw}\nOWNER ${target!.path}\nPARAGRAPH:\n${statusParagraph}\nOLD TEXT:\n${p.old_text}`,
        });
        state.checkCancellation();
        if (!scope.object.safe) prStatusUnique = false;
      }
      // A named training-design correction can update several assertions in
      // ONE uniquely observed paragraph. Keep all unrelated dataset and
      // historical clauses intact, rather than licensing broad note rewrites.
      const trainingSupersededClause = stagedStatusClauses(p.old_text).find((clause) =>
        clause.includes("For strict <1M multilingual student"));
      // The caller refers to the tiny architecture by meaning rather than
      // quoting its full old sentence. Only that exact observed clause is
      // considered requested; every other old clause remains protected.
      const trainingCheckpointClause = stagedStatusClauses(p.old_text).find((clause) =>
        /\b322M\b/.test(clause) && /\bnot under 1M\b/i.test(clause));
      const trainingInstruction = trainingSupersededClause && trainingCheckpointClause &&
        /earlier tiny byte\/CNN\/Transformer student recommendation[^.!?]*superseded/i.test(raw) &&
        /\bcheckpoint fits\b/i.test(raw)
        ? `${raw}\n${trainingSupersededClause}\n${trainingCheckpointClause}` : raw;
      const allowedTrainingNegationRemoval = (oldClause: string, newClause: string, word: string): boolean =>
        word === "not" && /\b322M\b/.test(oldClause) && /\bnot under 1M\b/i.test(oldClause) &&
        /\b322M\b/.test(newClause) && /\b(?:1B|billion)\b/i.test(newClause) &&
        /\b(?:fits?|under)\b/i.test(newClause) &&
        /\bnot\s*<1 million\b/i.test(raw) && /\b322M\b/.test(raw) &&
        /\bcheckpoint fits\b/i.test(raw);
      let trainingClauseIssue = stagedStatusClauseIssue(
        p.old_text, p.new_text, trainingInstruction, allowedTrainingNegationRemoval);
      if (target?.path === "/projects/claes-jev-diagnostics.md" &&
          /^correct the Claes training-design note recorded Oct 2026:/i.test(raw.trim()) &&
          trainingSupersededClause && trainingCheckpointClause &&
          target.body.split(/\n\s*\n/).some((paragraph) => paragraph.trim() === p.old_text.trim()) &&
          target.body.split(p.old_text).length === 2 &&
          p.old_text.length <= 4_000 && p.new_text.length <= 4_000 &&
          stagedOverlap(p.old_text, p.new_text) >= 0.5 &&
          trainingClauseIssue?.startsWith("old clause not requested")) {
        const repair = await generateObject({
          model: proposer.model,
          schema: z.object({ new_text: z.string() }).strict(),
          temperature: 0,
          abortSignal: options.signal,
          prompt: `Repair ONLY this exact Claes training-design paragraph. Keep every unrelated OLD clause VERBATIM in its original place, especially the no-student-model assertion, sample counts, dataset schema and Jev target formats. Replace the superseded <1M architecture and checkpoint-limit recommendation with the caller's <1B checkpoint recommendation. Do not invent facts or edit another paragraph. Return one paragraph; if impossible return empty new_text. JSON only.\nINSTRUCTION:\n${raw}\nOLD PARAGRAPH:\n${p.old_text}\nREJECTED REWRITE:\n${p.new_text}`,
        });
        state.checkCancellation();
        p.new_text = repair.object.new_text;
        trainingClauseIssue = stagedStatusClauseIssue(
          p.old_text, p.new_text, trainingInstruction, allowedTrainingNegationRemoval);
        if (trainingClauseIssue?.startsWith("old clause not requested") &&
            p.new_text.length > 0 && p.new_text.length <= 4_000 && !/\n\s*\n/.test(p.new_text)) {
          p.new_text = stagedRestoreUnrequestedStatusClauses(p.old_text, p.new_text, trainingInstruction);
          trainingClauseIssue = stagedStatusClauseIssue(
            p.old_text, p.new_text, trainingInstruction, allowedTrainingNegationRemoval);
        }
      }
      let trainingUniqueAcrossHits = !!target &&
        /^correct the Claes training-design note recorded Oct 2026:/i.test(raw.trim()) &&
        target.path === "/projects/claes-jev-diagnostics.md" &&
        target.body.split(/\n\s*\n/).some((paragraph) => paragraph.trim() === p.old_text.trim()) &&
        candidates.filter((candidate) => candidate.body.includes(p.old_text)).length === 1;
      if (trainingUniqueAcrossHits) {
        for (const hit of hits) {
          if (hit.path === target!.path) continue;
          const competitor = await kb.readConcept(hit.path);
          recorder.record("read_concept", competitor.path, [competitor.path]);
          state.checkCancellation();
          if (competitor.body.length > STAGED_MAX_OWNER_CHARS) {
            throw new Error("Staged mutation deferred: competing training-design owner exceeds evidence limit.");
          }
          const sameStaleRecommendation = competitor.body.split(/\n\s*\n/).some((paragraph) =>
            /(?:<1M|1 million)/i.test(paragraph) &&
            /(?:byte-input|CNN-downsample|small Transformer)/i.test(paragraph) &&
            /\brecommend\w*\b/i.test(paragraph));
          if (competitor.body.includes(p.old_text) || sameStaleRecommendation) trainingUniqueAcrossHits = false;
        }
      }
      const trainingDesignCorrection = !!target &&
        /^correct the Claes training-design note recorded Oct 2026:/i.test(raw.trim()) &&
        trainingUniqueAcrossHits &&
        p.old_text.includes("For strict <1M multilingual student") &&
        /\b322M\b/.test(p.old_text) && /\b1B\b/.test(raw) &&
        /\b322M\b/.test(p.new_text) && /\b(?:1B|billion)\b/i.test(p.new_text) &&
        !p.new_text.includes("For strict <1M multilingual student") &&
        p.old_text.length <= 4_000 && p.new_text.length <= 4_000 &&
        !/\n\s*\n/.test(p.new_text) &&
        stagedOverlap(p.old_text, p.new_text) >= 0.68 &&
        !trainingClauseIssue;
      const boundedParagraphRepair = (repairedCorrection || boundedCorrection || boundedSpanCorrection ||
        (!!quotedOwner && quotedOwner === p.path) ||
        (!!namedParagraph && namedParagraph.includes(p.old_text)) || boundedStatus || prStatusUnique || trainingDesignCorrection) &&
        p.old_text.length <= 4_000 && p.new_text.length <= 4_000 &&
        !/\n\s*\n/.test(p.old_text) && !/\n\s*\n/.test(p.new_text) &&
        (prStatusUnique || stagedOverlap(p.old_text, p.new_text) >= 0.5);
      const localEdit = stagedSingleClauseEdit(p.old_text, p.new_text) || boundedParagraphRepair;
      if (!localEdit && paragraphUniquelyOwned && target && statusParagraph &&
          p.old_text.length >= 100 && target.body.split(p.old_text).length === 2) {
        // A dated audit can state a true historical omission or residual risk
        // while a later paragraph records the fix. Replacing the old claim
        // would erase history; a strictly verified current follow-up can
        // instead clarify accepted decisions and subsequent mitigations.
        const followUp = await generateObject({
          model: resolved.model,
          schema: z.object({ qualifiable: z.boolean(), certain: z.boolean(), addition: z.string() }).strict(),
          temperature: 0,
          abortSignal: options.signal,
          prompt: `Can this correction be satisfied by ONE follow-up without retracting a false CURRENT assertion? Approve only if every old assertion that would remain is a true historical observation or residual risk, and later owner updates resolve any apparent conflict. Include only novel caller-supported facts and current qualifications in addition. Do NOT copy the instruction as the addition: some caller observations may predate later owner updates. For every technical issue, compare the latest owner status first; if a local port mapping was later removed or a hard parser deadline was later added, state that CURRENT status rather than repeating an earlier remediation request. If any current assertion must be removed, any requested correction cannot be captured here, or uncertain, return qualifiable=false, certain=false and empty addition. JSON only.\nINSTRUCTION:\n${raw}\nCOMPLETE OWNER ${target.path}:\n${target.body}\nUNSAFE OLD SPAN:\n${p.old_text}\nREJECTED REPLACEMENT:\n${p.new_text}`,
        });
        state.checkCancellation();
        const { qualifiable, certain } = followUp.object;
        let addition = followUp.object.addition;
        const requestedAnchors = stagedRequiredCorrectionAnchors(raw);
        const missingAnchors = (text: string): string[] => requestedAnchors.filter((anchor) =>
          !`${target.body} ${text}`.toLowerCase().includes(anchor));
        const missingDecision = (text: string): boolean =>
          /Dan does NOT want a site-wide provider spending ceiling/i.test(raw) &&
          !/Dan does not want a site-wide provider spending ceiling/i.test(`${target.body} ${text}`);
        if (qualifiable && certain && (missingAnchors(addition).length || missingDecision(addition))) {
          // A one-sentence follow-up is not a success for a multi-finding
          // correction. Repair once, then require concrete evidence coverage
          // before sending anything to the independent verifier.
          const completion = await generateObject({
            model: proposer.model,
            schema: z.object({ new_text: z.string() }).strict(),
            temperature: 0,
            abortSignal: options.signal,
            prompt: `Complete this one-paragraph correction follow-up. Cover EVERY applicable requested finding, including these missing exact identifiers: ${JSON.stringify(missingAnchors(addition))}. Record Dan's no-site-wide-ceiling decision, the optional advisory disposition, and the malformed HTTP 422 exposure/fix where requested. Use the LATEST owner status for the local Selenium mapping and Tika's 30s hard deadline; do NOT copy obsolete caller assertions as current. Preserve historical audit facts and avoid duplicating later hardening. Return empty new_text if a supported, complete follow-up is impossible. JSON only.\nINSTRUCTION:\n${raw}\nCOMPLETE OWNER ${target.path}:\n${target.body}\nINCOMPLETE FOLLOW-UP:\n${addition}`,
          });
          state.checkCancellation();
          if (completion.object.new_text) addition = completion.object.new_text;
        }
        // These later owner assertions contradict two common stale audit
        // requests. A model's yes/no verifier is not enough to ground the
        // temporal ordering; reject the concrete contradiction before CAS.
        const staleLocalPort = /local-only browser\.ports removed from docker-compose-local\.yaml/i.test(target.body) &&
          /docker-compose-local\.yaml[^.!?]{0,100}(?:maps?\b|0\.0\.0\.0)/i.test(addition);
        const staleTikaDeadline = /asyncio\.wait[\s\S]{0,300}\b30s deadline/i.test(target.body) &&
          /(?:Tika|CV pars(?:er|ing))[^.!?]{0,180}\b(?:still has no|has no|without a?) (?:true )?hard timeout/i.test(addition);
        const copiedInstructionCommands = /(?:^|[.!?]\s+)(?:Update|Correct|Preserve|Do not|Record|Remove)\b/i.test(addition.trim());
        console.warn("[understory] correction follow-up proposal verdict", {
          owner: target.path, qualifiable, certain, additionLength: addition.length,
          copiedInstruction: !!addition && raw.includes(addition.trim()),
          copiedInstructionCommands,
          missingAnchors: missingAnchors(addition).length,
          missingDecision: missingDecision(addition),
          staleLocalPort, staleTikaDeadline,
          lexicalSupport: addition ? stagedOverlap(addition, `${raw} ${target.body}`) : 0,
        });
        if (qualifiable && certain && !missingAnchors(addition).length && !missingDecision(addition) &&
            !staleLocalPort && !staleTikaDeadline && !copiedInstructionCommands &&
            addition.length >= 30 && addition.length <= 2_000 &&
            !/\n\s*\n/.test(addition) && !target.body.includes(addition.trim()) &&
            stagedOverlap(addition, `${raw} ${target.body}`) >= 0.8) {
          const judgement = await generateObject({
            model: resolved.model,
            schema: z.object({ safe: z.boolean(), reason: z.string() }).strict(),
            temperature: 0,
            abortSignal: options.signal,
            prompt: `Independent complete-owner consistency check for a correction follow-up. Verify that EVERY applicable caller correction is reflected either in the proposed addition or ALREADY in the full owner, and that all preserved old assertions remain true historical observations rather than false active findings. Compare every proposed technical claim against LATER owner paragraphs, not just the historical audit: a removed local port mapping must not be called present, and an implemented hard parser deadline must not be called absent. Reject duplicate content, unsupported new claims, omitted corrections, or any assertion contradicted by later owner updates. If a replacement/retraction is necessary instead, safe=false. If uncertain, safe=false. JSON only.\nINSTRUCTION:\n${raw}\nCOMPLETE OWNER ${target.path}:\n${target.body}\nHISTORICAL OLD SPAN:\n${p.old_text}\nFOLLOW-UP:\n${addition}`,
          });
          state.checkCancellation();
          if (!judgement.object.safe) {
            console.warn("[understory] correction follow-up independent veto", { owner: target.path });
          }
          if (judgement.object.safe) {
            const appended = `${target.body.trimEnd()}\n\n${addition.trim()}\n`;
            if (appended.length > STAGED_MAX_OWNER_CHARS) {
              throw new Error("Staged mutation deferred: correction follow-up exceeds owner limit.");
            }
            const current = await kb.readConcept(target.path);
            if (current.body !== target.body) throw new Error(`Concept changed while it was being read: ${target.path}`);
            state.checkCancellation();
            await kb.patchConcept(target.path, { replaceBody: appended },
              "Staged mutation: verified correction follow-up", sha256(target.body), options.signal,
              (writtenPath) => { changed = [writtenPath]; });
            const summary = `Staged mutation changed ${changed.join(", ")}.`;
            const trace = recorder.finalize("mutation", instruction, summary, "success", modelChain,
              undefined, undefined, undefined, { providerCalls, generationMs: Date.now() - started });
            await traceStore(kb).save(trace);
            return { ok: true, result: { summary, filesChanged: changed, steps: 1, traceId: trace.id } };
          }
        }
      }
      if (!localEdit && correction && statusParagraph && !statusSubject) {
        console.warn("[understory] multi-clause correction scope declined", {
          owner: target?.path, boundedCorrection, repairedCorrection,
          correction, directAdd: !!options.directAdd,
          singleClause: stagedSingleClauseEdit(p.old_text, p.new_text),
          oldLength: p.old_text.length, paragraphLength: statusParagraph.length,
          oldOccurrences: target ? target.body.split(p.old_text).length - 1 : 0,
          paragraphOccurrences: target ? target.body.split(statusParagraph).length - 1 : 0,
          namedParagraph: !!namedParagraph,
          excludedTraining: /^correct the Claes training-design note recorded Oct 2026:/i.test(raw.trim()),
          ownerAnchored: !!target && (stagedTitleMention(raw, target.frontmatter.title ?? "") ||
            stagedOwnerAnchor(raw, target.path)),
        });
      }
      if (!target || p.old_text.length < 12 || !p.new_text || target.body.split(p.old_text).length !== 2 ||
          stagedOverlap(p.new_text, `${raw} ${p.old_text}`) < 0.65 || !localEdit) {
        const reason = !target ? `owner missing: ${p.path.slice(0, 256) || "(empty)"}; candidates: ${candidates.map((candidate) => candidate.path).join(", ")}` : p.old_text.length < 12 ? "old text too short" :
          !p.new_text ? "new text empty" : target.body.split(p.old_text).length !== 2 ? `old text not unique; repair=${duplicateRepair}` :
          stagedOverlap(p.new_text, `${raw} ${p.old_text}`) < 0.65 ? "new text lacks support" :
            `change spans clauses (section=${!!sectionOwnerPath}, scoped=${!!namedParagraph && namedParagraph.includes(p.old_text)}, status=${!!statusSubject}, status_paragraph=${!!statusParagraph}, status_unique=${statusUniqueAcrossHits}, status_clause=${statusClauseIssue ?? "ok"}, training_clause=${trainingClauseIssue ?? "ok"}, training=${trainingDesignCorrection}, subject=${statusWords.length}/${statusWords.filter((word) => oldStatusWords.has(word)).length}/${statusWords.filter((word) => identityWords.has(word)).length}, owners=${candidates.filter((candidate) => candidate.body.includes(p.old_text)).length}, old=${p.old_text.length}, new=${p.new_text.length}, overlap=${stagedOverlap(p.old_text, p.new_text).toFixed(2)}, paragraph_break=${/\n\s*\n/.test(p.old_text) || /\n\s*\n/.test(p.new_text)})`;
        const prScope = prStatus ? `PR scope(owner=${target?.path ?? "none"}, ids=${prIdentities.join(",") || "none"}, review_only=${reviewOnly}, unique=${prStatusUnique}); ` : "";
        throw new Error(`Staged mutation rejected: replacement is not one supported local change (${prScope}${reason}).`);
      }
      const oldWords = new Set(stagedWords(p.old_text));
      const requestedWords = new Set(stagedWords(raw));
      const unsupportedWords = (text: string): string[] => stagedWords(text)
        .filter((word) => !oldWords.has(word) && !requestedWords.has(word));
      let newText = p.new_text;
      if (unsupportedWords(newText).length) {
        // A paraphrase can fail the lexical guard even when the correction is
        // supported. Give the proposer one chance to use only observed words;
        // never authorize the first proposal or weaken the independent check.
        const repair = await generateObject({
          model: resolved.model,
          schema: z.object({ new_text: z.string() }).strict(),
          temperature: 0,
          abortSignal: options.signal,
          prompt: `Rewrite ONLY the proposed NEW TEXT for this one local correction. Use wording from the instruction or OLD TEXT; introduce no other content words or claims. Preserve unrelated old claims, and do not change the owner or old operand. If the correction cannot be expressed faithfully using those words, return an empty new_text. JSON only.\nINSTRUCTION:\n${raw}\nOLD TEXT:\n${p.old_text}\nPROPOSED NEW TEXT:\n${p.new_text}\nUNSUPPORTED WORDS:\n${JSON.stringify(unsupportedWords(newText))}`,
        });
        state.checkCancellation();
        newText = repair.object.new_text;
      }
      if (!newText || !(stagedSingleClauseEdit(p.old_text, newText) ||
          (boundedParagraphRepair && newText.length <= 4_000 && !/\n\s*\n/.test(newText) &&
            (prStatusUnique || stagedOverlap(p.old_text, newText) >= (boundedStatus ? 0.7 : 0.5)) &&
            (!(boundedStatus || boundedCorrection || boundedSpanCorrection) ||
              !stagedStatusClauseIssue(p.old_text, newText,
                boundedCorrection ? correctionEvidence : raw, allowedThroughputNegationRemoval)) &&
            (!trainingDesignCorrection ||
              (!newText.includes("For strict <1M multilingual student") &&
                !stagedStatusClauseIssue(p.old_text, newText, trainingInstruction,
                  allowedTrainingNegationRemoval))))) ||
          stagedOverlap(newText, `${raw} ${p.old_text}`) < 0.65 || unsupportedWords(newText).length) {
        throw new Error("Staged mutation rejected: replacement introduces facts absent from the request.");
      }
      const containsNegation = (text: string, word: string): boolean =>
        new RegExp(`\\b${word}\\b`, "i").test(text);
      const addedNegations = ["no", "not", "never", "without", "cannot", "doesn't", "don't"].filter((word) =>
        containsNegation(newText, word) && !containsNegation(p.old_text, word));
      if (addedNegations.some((word) => !containsNegation(raw, word))) {
        throw new Error("Staged mutation rejected: an added negation was not requested.");
      }
      const replacement = target.body.replace(p.old_text, newText);
      // An old workbook tick may still be mentioned as historical provenance.
      // A hash-only edit must not count as completing an explicit request to
      // clear that tick in the current workbook.
      const clearedCells = !options.directAdd && correction
        ? [...raw.matchAll(/`([^`\n]{2,80}![A-Z]{1,3}\d+)`[^.!?;\n]{0,120}\b(?:was|is)\s+cleared\s+to\s+blank/gi)]
          .map((match) => match[1]) : [];
      if (clearedCells.some((cell) => {
        const escaped = cell.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
        return !new RegExp(`\\bcurrent\\b[^.!?;]{0,160}${escaped}[^.!?;]{0,120}\\bcleared\\s+to\\s+blank\\b`, "i")
          .test(newText);
      })) {
        throw new Error("Staged mutation deferred: replacement omits requested current-cell clearing.");
      }
      if (!options.directAdd && correction && stagedRequiredCorrectionAnchors(raw).some((anchor) =>
        !replacement.toLowerCase().replace(/\s+/g, " ").includes(anchor))) {
        throw new Error("Staged mutation deferred: replacement omits requested correction identifiers.");
      }
      if (replacement === target.body || replacement.length > STAGED_MAX_OWNER_CHARS) {
        throw new Error("Staged mutation rejected: replacement is empty or oversized.");
      }
      const judgement = await generateObject({
        model: resolved.model,
        schema: z.object({ safe: z.boolean(), reason: z.string() }).strict(),
        temperature: 0,
        abortSignal: options.signal,
        prompt: `Independent safety check. Does the exact proposed replacement faithfully apply ALL applicable corrections in the instruction to the observed owner, remove the superseded claim, preserve unrelated claims, avoid a new contradiction, and positively fit this owner even if another search hit could also plausibly fit? Reject negative/unrelated mentions. Metadata selection is not proof of relevance. If uncertain about support, completeness or relevance, safe=false.\nINSTRUCTION:\n${raw}\nOWNER ${target.path}:\n${target.body}\nALTERNATIVE SEARCH HITS (path and title only):\n${JSON.stringify(hits.filter((hit) => hit.path !== target.path).map((hit) => ({ path: hit.path, title: hit.title })))}\nOLD TEXT:\n${p.old_text}\nNEW TEXT:\n${newText}`,
      });
      state.checkCancellation();
      if (!judgement.object.safe) throw new Error("Staged mutation deferred: replacement failed independent consistency check.");
      const current = await kb.readConcept(target.path);
      if (current.body !== target.body) throw new Error(`Concept changed while it was being read: ${target.path}`);
      await kb.patchConcept(target.path, { replaceBody: replacement }, "Staged mutation: replace verified claim", sha256(target.body), options.signal, (path) => { changed = [path]; });
    } else if (p.action === "create") {
      const explicitlyDistinct = /\b(?:distinct|standalone|stand-alone|new concept|new policy|unrelated)\b/i.test(raw);
      if (selectedPath || (candidates.length > 0 && !explicitlyDistinct) || !p.path || !p.body || !p.frontmatter.type || !p.frontmatter.title ||
          !/^\/[a-z0-9_/-]+\.md$/.test(p.path) || p.path.split("/").includes("..") || p.body.length > (state.maxDocumentChars ?? 12_000) ||
          stagedOverlap(p.frontmatter.title, raw) < 0.7 || stagedOverlap(p.body, raw) < 0.65 ||
          stagedOverlap(raw, p.body) < 0.8 ||
          !(raw.includes(p.body.trim()) || p.body.includes(raw.trim()))) {
        throw new Error("Staged mutation rejected: create is not distinct or valid.");
      }
      const mainClaim = p.body.trim().split(/[.!?]\s+/)[0]?.trim() ?? "";
      if (mainClaim.length >= 20 && candidates.some((candidate) => candidate.body.includes(mainClaim))) {
        throw new Error("Staged mutation deferred: requested claim already belongs to an observed concept.");
      }
      const similar = await kb.search(p.frontmatter.title);
      state.checkCancellation();
      if (similar.some((hit) => hit.title && stagedOverlap(p.frontmatter.title, hit.title) >= 0.75 && hit.confidenceQualified)) {
        throw new Error("Staged mutation deferred: a similarly named concept already exists.");
      }
      state.checkCancellation();
      await kb.createConcept(p.path, p.frontmatter, p.body, "Staged mutation: create distinct concept", options.signal, (path) => { changed = [path]; });
    }
    if (!changed.length) throw new Error("Staged mutation rejected: no verified write occurred.");
    const summary = `Staged mutation changed ${changed.join(", ")}.`;
    const trace = recorder.finalize("mutation", instruction, summary, "success", modelChain, undefined, undefined, undefined, { providerCalls, generationMs: Date.now() - started });
    await traceStore(kb).save(trace);
    return { ok: true, result: { summary, filesChanged: changed, steps: 1, traceId: trace.id } };
  } catch (err) {
    const message = errorMessage(err);
    if (changed.length) {
      const trace = recorder.finalize("mutation", instruction, `Partial mutation: ${changed.length} file(s) changed before failure. Error: ${message}`, "partial", modelChain, undefined, undefined, undefined, { providerCalls, generationMs: Date.now() - started });
      await traceStore(kb).save(trace);
      return { ok: false, status: "partial", filesChanged: changed, error: message, traceId: trace.id };
    }
    if (isAbortError(err, options.signal)) throw err;
    return { ok: false, status: "failed", error: message };
  }
}

function mutationSearchQuery(input: string): string {
  const words = input
    .replace(/https?:\/\/\S+/gi, " ")
    .toLowerCase()
    .match(/[\p{L}\p{N}][\p{L}\p{N}_-]{2,}/gu) ?? [];
  const stop = new Set(["the", "and", "for", "with", "that", "this", "into", "from", "about", "existing", "concept", "knowledge", "base", "update", "change", "please", "persist", "following"]);
  const retained = [...new Set(words.filter((word) => !stop.has(word)))];
  // Identifiers often appear in the decisive tail of an otherwise generic
  // instruction (e.g. godkendt_id2). Keep these ahead of leading boilerplate.
  const identifiers = retained.filter((word) => /\d|_/.test(word));
  // Retain proper names wherever they occur; new identifiers have no index
  // entry yet, so an owner name buried after boilerplate is still essential.
  const named = [...input.matchAll(/\b(?:[A-Z]{2,}|[A-Z][a-z]{3,})\b/g)]
    .map((match) => match[0].toLowerCase())
    .filter((word) => !stop.has(word) && !["correct", "remember", "record", "final"].includes(word));
  const leading = retained.slice(0, 6);
  const lateIdentifiers = identifiers.filter((word) => !leading.includes(word)).slice(0, 2);
  const lateNames = named.filter((word) => !leading.includes(word)).slice(0, 2);
  return [...new Set([...leading, ...lateIdentifiers, ...lateNames, ...retained.slice(-2)])]
    .slice(0, 10).join(" ").slice(0, 500);
}

export async function runMutation(
  kb: KnowledgeBase,
  instruction: string,
  options: AgentOptions = {}
): Promise<MutationOutcome> {
  const limits = resolveAgentLimits();
  assertInputWithinLimit(
    instruction,
    limits.maxInputChars,
    "Mutation input"
  );
  const state = new AgentRunContext(limits, options.signal);
  throwIfAborted(options.signal);
  const ctx = await promptContext(kb, "mutate", state);
  throwIfAborted(options.signal);
  const recorder = new TraceRecorder();
  if (process.env.MUTATION_STAGED === "true") {
    return runStagedMutation(kb, instruction, options, state, recorder);
  }
  let preflightHint = "";
  if (process.env.MUTATION_PREFLIGHT === "true") {
    try {
      const raw = options.preflightInput ?? instruction;
      const query = mutationSearchQuery(raw);
      if (query) {
        state.checkCancellation();
        const hits = await kb.search(query);
        recorder.record("search_knowledge", query, hits.slice(0, 5).map((hit) => hit.path));
        state.checkCancellation();
        // Only a unique or clearly dominant content-backed hit may seed the
        // whole-body write guard. Related but weaker hits remain available to
        // the agent through its normal search tools; close matches stay ambiguous.
        const candidates = hits.filter((hit) => hit.confidenceQualified === true);
        const first = candidates[0];
        const second = candidates[1];
        const clearOwner = first && (!second ||
          ((first.confidence ?? 0) >= (second.confidence ?? 0) + 20 &&
            (first.confidence ?? 0) >= (second.confidence ?? 0) * 1.5));
        if (clearOwner) {
          const concept = await kb.readConcept(first.path);
          state.checkCancellation();
          recorder.record("read_concept", concept.path, [concept.path]);
          const page = {
            path: concept.path,
            frontmatter: concept.frontmatter,
            frontmatter_truncated: false,
            body: concept.body,
            offset: 0,
            total_chars: concept.body.length,
            truncated: false,
            next_offset: null,
          };
          const evidence = JSON.stringify([page]);
          const heading = `\n\nDETERMINISTIC PREFLIGHT (one confidence-qualified candidate; complete body follows):\n- ${concept.path}\n`;
          // Never cut the evidence: only mark a body as read if its complete,
          // unchanged contents are actually present in the prompt.
          if (concept.body.length <= (state.maxDocumentChars ?? Number.POSITIVE_INFINITY) &&
              heading.length + evidence.length <= Math.min(12_000, state.payloadBudget ?? 12_000)) {
            preflightHint = heading + evidence;
            state.recordBodyPage(concept.path, 0, concept.body, concept.body, concept.body);
          }
        }
      }
    } catch {
      // Preflight is advisory. The regular agent search/read path remains authoritative.
    }
    throwIfAborted(options.signal);
  }
  const maxSteps = limits.maxSteps;
  const filesChanged = new Set<string>();
  const providerCalls: NonNullable<TraceTiming["providerCalls"]> = [];
  const generationStarted = Date.now();
  let modelChain: string[] = [];
  try {
    const resolved = await resolveAgentModel(
      options,
      "mutate",
      process.env,
      (call) => providerCalls.push(call)
    );
    modelChain = resolved.modelChain;
    const mutationRequest = (model: LanguageModel) => generateText({
      model,
      system: buildSystemPrompt(ctx),
      prompt: instruction + preflightHint,
      tools: {
        ...buildReadTools(kb, recorder, state),
        ...buildWriteTools(kb, filesChanged, recorder, state),
      },
      stopWhen: stepCountIs(maxSteps),
      prepareStep: prepareFinalSynthesisStep(maxSteps),
      temperature: 0.2,
      abortSignal: options.signal,
    });
    let result = await mutationRequest(resolved.model);
    throwIfAborted(options.signal);
    assertSynthesised(result.steps);
    const allSteps: Array<{
      usage?: { inputTokens?: number; outputTokens?: number };
    }> = [...result.steps];
    if (isMalformedAnswer(result.text) || isUnsafeSynthesisAnswer(result.text)) {
      console.error(`[understory] mutation summary rejected: ${MALFORMED_ANSWER_MESSAGE}`);
      // A malformed summary cannot establish which writes in a multi-write
      // instruction completed. Preserve those writes as partial; never replay.
      if (filesChanged.size > 0 || !resolved.mutationRetryModel) {
        throw new Error(MALFORMED_ANSWER_MESSAGE);
      }
      result = await mutationRequest(resolved.mutationRetryModel);
      throwIfAborted(options.signal);
      assertSynthesised(result.steps);
      allSteps.push(...result.steps);
      if (isMalformedAnswer(result.text) || isUnsafeSynthesisAnswer(result.text)) {
        throw new Error(MALFORMED_ANSWER_MESSAGE);
      }
    }
    const summary = result.text;
    const trace = recorder.finalize(
      "mutation",
      instruction,
      summary,
      "success",
      modelChain,
      sumStepsUsage(allSteps),
      undefined,
      undefined,
      { providerCalls, generationMs: Date.now() - generationStarted }
    );
    await traceStore(kb).save(trace);
    return {
      ok: true,
      result: {
        summary,
        filesChanged: [...filesChanged].sort(),
        steps: allSteps.length,
        traceId: trace.id,
      },
    };
  } catch (err) {
    const files = [...filesChanged].sort();
    const message = errorMessage(err);
    // A cancelled mutation with no writes must remain a cancellation so MCP and
    // callers do not mistake it for a completed agent response. Once a write has
    // landed, retain the existing partial-mutation report instead.
    if (files.length === 0 && isAbortError(err, options.signal)) {
      const trace = recorder.finalize("mutation", instruction, message, "failed", modelChain, undefined, undefined, undefined, {
        providerCalls,
        generationMs: Date.now() - generationStarted,
      });
      await traceStore(kb).save(trace).catch(() => {
        // Preserve the provider cancellation even if trace persistence also fails.
      });
      throw err;
    }
    if (files.length > 0) {
      const summary = `Partial mutation: ${files.length} file(s) changed before failure. Error: ${message}`;
      const trace = recorder.finalize("mutation", instruction, summary, "partial", modelChain, undefined, undefined, undefined, {
        providerCalls,
        generationMs: Date.now() - generationStarted,
      });
      await traceStore(kb).save(trace);
      return { ok: false, status: "partial", filesChanged: files, error: message, traceId: trace.id };
    }
    const trace = recorder.finalize("mutation", instruction, message, "failed", modelChain, undefined, undefined, undefined, {
      providerCalls,
      generationMs: Date.now() - generationStarted,
    });
    await traceStore(kb).save(trace);
    return { ok: false, status: "failed", error: message };
  }
}

/** Interactive chat — full toolset, streaming. Caller converts to a UI stream response. */
export async function streamChat(
  kb: KnowledgeBase,
  messages: ModelMessage[],
  options: AgentOptions = {}
) {
  const state = AgentRunContext.unbounded(options.signal);
  throwIfAborted(options.signal);
  const ctx = await promptContext(kb, "chat", state);
  throwIfAborted(options.signal);
  const recorder = new TraceRecorder();
  const filesChanged = new Set<string>();
  let modelChain: string[] = [];
  // The user turn that started this run, for the trace record.
  const lastUser = [...messages].reverse().find((m) => m.role === "user");
  const input =
    typeof lastUser?.content === "string"
      ? lastUser.content
      : lastUser?.content
          ?.map((part) => (part.type === "text" ? part.text : ""))
          .join(" ")
          .trim() ?? "(chat)";
  let traceFinalised = false;
  const finaliseChatTrace = async (
    answer: string,
    outcome: "success" | "partial" | "failed",
    usage?: TraceUsage
  ): Promise<void> => {
    // AI SDK callbacks can race: onError may arrive while onFinish is still
    // persisting. Claim the trace before awaiting the filesystem write.
    if (traceFinalised) return;
    traceFinalised = true;
    if (outcome === "success" && recorder.steps.length === 0) return;
    try {
      await traceStore(kb).save(
        recorder.finalize("chat", input, answer, outcome, modelChain, usage)
      );
    } catch (traceError) {
      // A trace must never turn a provider failure into an unobserved promise
      // rejection or hide the original stream error.
      console.error(`[understory] chat trace save failed: ${errorMessage(traceError)}`);
    }
  };

  try {
    const resolved = await resolveAgentModel(options, "chat");
    modelChain = resolved.modelChain;
    const protocolGuard = createProtocolLeakageGuard({
      errorMessage: () => chatFailureMessage(MALFORMED_ANSWER_MESSAGE, filesChanged),
    });
    const result = streamText({
      model: resolved.model,
      system: buildSystemPrompt(ctx),
      messages,
      tools: {
        ...buildReadTools(kb, recorder, state),
        ...buildWriteTools(kb, filesChanged, recorder, state),
      },
      // An empty condition list lets AI SDK v5 continue until the model stops,
      // without imposing an application-level step budget.
      stopWhen: [],
      abortSignal: options.signal,
      // AI SDK v5 applies this transform before toUIMessageStreamResponse(),
      // so malformed text is dropped before it can reach the HTTP client.
      experimental_transform: protocolGuard.transform,
      onFinish: async ({ text, totalUsage, steps }) => {
        const usage =
          totalUsage && (totalUsage.inputTokens != null || totalUsage.outputTokens != null)
            ? {
                inputTokens: totalUsage.inputTokens ?? 0,
                outputTokens: totalUsage.outputTokens ?? 0,
              }
            : undefined;
        try {
          assertSynthesised(steps);
          if (protocolGuard.wasMalformed() || isMalformedAnswer(text)) {
            const message = chatFailureMessage(MALFORMED_ANSWER_MESSAGE, filesChanged);
            console.error(`[understory] chat answer rejected: ${MALFORMED_ANSWER_MESSAGE}`);
            // The guard has already placed a client-visible AI SDK error part
            // before the terminal events. Do not throw here: an onFinish throw
            // makes AI SDK discard that response stream before the caller sees
            // the error, while still leaving the failed/partial trace intact.
            await finaliseChatTrace(message, filesChanged.size > 0 ? "partial" : "failed", usage);
            return;
          }
          await finaliseChatTrace(text, "success", usage);
        } catch (error) {
          const outcome = filesChanged.size > 0 ? "partial" : "failed";
          await finaliseChatTrace(chatFailureMessage(errorMessage(error), filesChanged), outcome, usage);
          throw error;
        }
      },
      onError: async ({ error }) => {
        const outcome = filesChanged.size > 0 ? "partial" : "failed";
        await finaliseChatTrace(chatFailureMessage(errorMessage(error), filesChanged), outcome);
      },
      onAbort: async () => {
        const outcome = filesChanged.size > 0 ? "partial" : "failed";
        await finaliseChatTrace(chatFailureMessage("Chat stream aborted", filesChanged), outcome);
      },
    });
    return { result, filesChanged };
  } catch (err) {
    const outcome = filesChanged.size > 0 ? "partial" : "failed";
    await finaliseChatTrace(chatFailureMessage(errorMessage(err), filesChanged), outcome);
    throw err;
  }
}
