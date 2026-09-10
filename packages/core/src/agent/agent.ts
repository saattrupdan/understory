import {
  generateText,
  modelMessageSchema,
  streamText,
  stepCountIs,
  type LanguageModel,
  type ModelMessage,
} from "ai";
import type { KnowledgeBase } from "../okf/index.js";
import {
  createModel,
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
import { TraceRecorder, TraceStore, type TraceUsage } from "./trace.js";
import {
  createProtocolLeakageGuard,
  isMalformedAnswer,
  isUnsafeSynthesisAnswer,
  MALFORMED_ANSWER_MESSAGE,
} from "./answer-validation.js";

export interface AgentOptions {
  model?: string;
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

async function resolveAgentModel(
  options: AgentOptions,
  mode: "query" | "mutate" | "chat",
  env: NodeJS.ProcessEnv = process.env
): Promise<ResolvedAgentModel> {
  const primaryConfig = withModelOverride(resolveModelConfig(env), options.model);
  const primary = await createModel(primaryConfig);
  const fallbackConfig = resolveFallbackConfig(env);

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

  const fallback = await createModel(fallbackConfig);
  return {
    // The initial loop keeps the existing transport-only fallback behaviour.
    model: withFallback(primary, fallback, {
      retry429: env.LLM_FALLBACK_RETRY_429 === "true",
    }),
    // A malformed answer is a successful transport response, so the wrapper
    // cannot help. Repair directly on the configured fallback model instead.
    synthesisModel: mode === "query" ? fallback : primary,
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

function assertSynthesised(steps: ReadonlyArray<{ toolCalls?: unknown[] }>): void {
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
  const state = new AgentRunContext(limits);
  const ctx = await promptContext(kb, "query", state);
  const recorder = new TraceRecorder();
  const maxSteps = limits.maxSteps;
  let modelChain: string[] = [];
  try {
    const resolved = await resolveAgentModel(options, "query");
    modelChain = resolved.modelChain;
    const result = await generateText({
      model: resolved.model,
      system: buildSystemPrompt(ctx),
      prompt: question,
      tools: buildReadTools(kb, recorder, state),
      stopWhen: stepCountIs(maxSteps),
      prepareStep: prepareFinalSynthesisStep(maxSteps),
    });
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
        throw new Error(MALFORMED_ANSWER_MESSAGE);
      }
      const repair = await generateText({
        model: resolved.synthesisModel,
        system: buildQuerySynthesisPrompt(),
        messages: repairMessages,
        tools: {},
      });
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
        });
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

    const trace = recorder.finalize(
      "query",
      question,
      finalText,
      "success",
      modelChain,
      sumStepsUsage(allSteps)
    );
    await traceStore(kb).save(trace);
    return { answer: finalText, steps: allSteps.length, traceId: trace.id };
  } catch (err) {
    const trace = recorder.finalize("query", question, errorMessage(err), "failed", modelChain);
    await traceStore(kb).save(trace);
    throw err;
  }
}

/** Knowledge add/update — full toolset, low temperature. */
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
  const state = new AgentRunContext(limits);
  const ctx = await promptContext(kb, "mutate", state);
  const recorder = new TraceRecorder();
  const maxSteps = limits.maxSteps;
  const filesChanged = new Set<string>();
  let modelChain: string[] = [];
  try {
    const resolved = await resolveAgentModel(options, "mutate");
    modelChain = resolved.modelChain;
    const result = await generateText({
      model: resolved.model,
      system: buildSystemPrompt(ctx),
      prompt: instruction,
      tools: {
        ...buildReadTools(kb, recorder, state),
        ...buildWriteTools(kb, filesChanged, recorder, state),
      },
      stopWhen: stepCountIs(maxSteps),
      prepareStep: prepareFinalSynthesisStep(maxSteps),
      temperature: 0.2,
    });
    assertSynthesised(result.steps);
    if (isMalformedAnswer(result.text)) {
      console.error(`[understory] mutation summary rejected: ${MALFORMED_ANSWER_MESSAGE}`);
      throw new Error(MALFORMED_ANSWER_MESSAGE);
    }
    const trace = recorder.finalize("mutation", instruction, result.text, "success", modelChain, sumStepsUsage(result.steps));
    await traceStore(kb).save(trace);
    return {
      ok: true,
      result: {
        summary: result.text,
        filesChanged: [...filesChanged].sort(),
        steps: result.steps.length,
        traceId: trace.id,
      },
    };
  } catch (err) {
    const files = [...filesChanged].sort();
    const message = errorMessage(err);
    if (files.length > 0) {
      const summary = `Partial mutation: ${files.length} file(s) changed before failure. Error: ${message}`;
      const trace = recorder.finalize("mutation", instruction, summary, "partial", modelChain);
      await traceStore(kb).save(trace);
      return { ok: false, status: "partial", filesChanged: files, error: message, traceId: trace.id };
    }
    const trace = recorder.finalize("mutation", instruction, message, "failed", modelChain);
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
  const limits = resolveAgentLimits();
  const state = new AgentRunContext(limits);
  const ctx = await promptContext(kb, "chat", state);
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
