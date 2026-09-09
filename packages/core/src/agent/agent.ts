import { generateText, streamText, stepCountIs, type LanguageModel, type ModelMessage } from "ai";
import type { KnowledgeBase } from "../okf/index.js";
import {
  createModel,
  resolveFallbackConfig,
  resolveModelConfig,
  type ModelConfig,
} from "../providers/index.js";
import { withFallback } from "../providers/fallback.js";
import { buildSystemPrompt } from "./system-prompt.js";
import { buildReadTools, buildWriteTools, formatTree } from "./tools.js";
import {
  assertInputWithinLimit,
  resolveAgentLimits,
} from "./limits.js";
import { AgentRunContext } from "./run-context.js";
import { TraceRecorder, TraceStore, type TraceUsage } from "./trace.js";
import { isMalformedAnswer, MALFORMED_ANSWER_MESSAGE } from "./answer-validation.js";

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
  model: LanguageModel;
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
    return { model: primary, modelChain: [modelLabel(primaryConfig)] };
  }

  const allowFor = resolveAllowFor(env.LLM_FALLBACK_ALLOW_FOR);
  if (allowFor && !allowFor.has(mode)) {
    return { model: primary, modelChain: [modelLabel(primaryConfig)] };
  }

  const fallback = await createModel(fallbackConfig);
  return {
    model: withFallback(primary, fallback, {
      retry429: env.LLM_FALLBACK_RETRY_429 === "true",
    }),
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
    if (isMalformedAnswer(finalText)) {
      console.error(`[understory] query answer rejected: ${MALFORMED_ANSWER_MESSAGE}`);
      const repairMessages: ModelMessage[] = [
        { role: "user", content: question },
        ...(result.response?.messages ?? []),
      ];
      const repair = await generateText({
        model: resolved.model,
        system:
          `${buildSystemPrompt(ctx)}\n\n` +
          "SYNTHESIS ONLY: Answer the user's question from the supplied conversation " +
          "and tool results. Do not call tools and never emit tool-call markers or " +
          "tool syntax; return ordinary user-facing prose only.",
        messages: repairMessages,
        tools: {},
      });
      assertSynthesised(repair.steps);
      if (isMalformedAnswer(repair.text)) {
        throw new Error(MALFORMED_ANSWER_MESSAGE);
      }
      finalText = repair.text;
      allSteps.push(...repair.steps);
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
  assertInputWithinLimit(
    messages,
    limits.maxInputChars,
    "Chat history"
  );
  const state = new AgentRunContext(limits);
  const ctx = await promptContext(kb, "chat", state);
  const recorder = new TraceRecorder();
  const maxSteps = limits.maxSteps;
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
    const result = streamText({
      model: resolved.model,
      system: buildSystemPrompt(ctx),
      messages,
      tools: {
        ...buildReadTools(kb, recorder, state),
        ...buildWriteTools(kb, filesChanged, recorder, state),
      },
      stopWhen: stepCountIs(maxSteps),
      prepareStep: prepareFinalSynthesisStep(maxSteps),
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
          if (isMalformedAnswer(text)) {
            console.error(`[understory] chat answer rejected: ${MALFORMED_ANSWER_MESSAGE}`);
            throw new Error(MALFORMED_ANSWER_MESSAGE);
          }
          await finaliseChatTrace(text, "success", usage);
        } catch (error) {
          const outcome = filesChanged.size > 0 ? "partial" : "failed";
          await finaliseChatTrace(errorMessage(error), outcome, usage);
          throw error;
        }
      },
      onError: async ({ error }) => {
        const outcome = filesChanged.size > 0 ? "partial" : "failed";
        await finaliseChatTrace(errorMessage(error), outcome);
      },
      onAbort: async () => {
        const outcome = filesChanged.size > 0 ? "partial" : "failed";
        await finaliseChatTrace("Chat stream aborted", outcome);
      },
    });
    return { result, filesChanged };
  } catch (err) {
    const outcome = filesChanged.size > 0 ? "partial" : "failed";
    await finaliseChatTrace(errorMessage(err), outcome);
    throw err;
  }
}
