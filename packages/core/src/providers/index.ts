import { createAnthropic } from "@ai-sdk/anthropic";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import type { LanguageModel } from "ai";

type ResolvedLanguageModel = Extract<LanguageModel, { doGenerate: unknown }>;

export type ApiFormat = "openai" | "anthropic";

export interface ModelConfig {
  baseURL: string;
  apiKey: string;
  format: ApiFormat;
  model: string;
  /**
   * Extra fields merged into the chat-completion request body. Needed for
   * server-specific knobs the AI SDK has no first-class option for — notably
   * vLLM's `chat_template_kwargs`, which is how a reasoning budget is asked
   * of a thinking model (see `thinkingBudgetBody`). Ignored by Anthropic.
   */
  extraBody?: Record<string, unknown>;
}

/**
 * Request body asking a reasoning model to think within a token budget instead
 * of at its default length. Long reasoning traces are the dominant generation
 * cost of a short, grounded answer, so bounded thinking is much faster without
 * dropping the grounding. Endpoints that do not implement it ignore the field.
 */
export function thinkingBudgetBody(budget: number): Record<string, unknown> {
  if (!Number.isFinite(budget) || budget < 0) return {};
  return { chat_template_kwargs: { thinking: true, thinking_budget: budget } };
}

const LEGACY_NOTICE =
  "[understory] using legacy env vars. Migrate to LLM_API_BASE_URL + LLM_API_KEY + LLM_API_FORMAT.";

/** Ensure the URL ends in /v1 — llama-server serves the OpenAI API there. */
function normalizeV1(baseURL: string): string {
  const trimmed = baseURL.replace(/\/+$/, "");
  return trimmed.endsWith("/v1") ? trimmed : `${trimmed}/v1`;
}

function parseFormat(value: string | undefined, fallback: ApiFormat, envName: string): ApiFormat {
  const format = value ?? fallback;
  if (format !== "openai" && format !== "anthropic") {
    throw new Error(`${envName} must be "openai" or "anthropic"`);
  }
  return format;
}

let legacyNoticed = false;

function legacyNotice(): void {
  if (legacyNoticed) return;
  legacyNoticed = true;
  console.error(LEGACY_NOTICE);
}

function legacyConfig(env: NodeJS.ProcessEnv): ModelConfig | null {
  const provider = env.LLM_PROVIDER;
  if (provider) {
    switch (provider) {
      case "anthropic":
        if (!env.ANTHROPIC_API_KEY) throw new Error("ANTHROPIC_API_KEY is required for legacy anthropic provider");
        legacyNotice();
        return {
          baseURL: "https://api.anthropic.com/v1",
          apiKey: env.ANTHROPIC_API_KEY,
          format: "anthropic",
          model: env.LLM_MODEL ?? "claude-sonnet-5",
        };
      case "openrouter":
        if (!env.OPENROUTER_API_KEY) throw new Error("OPENROUTER_API_KEY is required for legacy openrouter provider");
        legacyNotice();
        return {
          baseURL: "https://openrouter.ai/api/v1",
          apiKey: env.OPENROUTER_API_KEY,
          format: "openai",
          model: env.LLM_MODEL ?? "anthropic/claude-sonnet-5",
        };
      case "llamacpp":
        if (!env.LLAMACPP_BASE_URL) throw new Error("LLAMACPP_BASE_URL is required for legacy llamacpp provider");
        legacyNotice();
        return {
          baseURL: env.LLAMACPP_BASE_URL,
          apiKey: env.LLAMACPP_API_KEY ?? "not-needed",
          format: "openai",
          model: env.LLM_MODEL ?? "",
        };
      case "deepseek":
        if (!env.DEEPSEEK_API_KEY) throw new Error("DEEPSEEK_API_KEY is required for legacy deepseek provider");
        legacyNotice();
        return {
          baseURL: "https://api.deepseek.com/v1",
          apiKey: env.DEEPSEEK_API_KEY,
          format: "openai",
          model: env.LLM_MODEL ?? "deepseek-chat",
        };
      case "local":
        if (!env.LOCAL_BASE_URL) throw new Error("LOCAL_BASE_URL is required for legacy local provider");
        legacyNotice();
        return {
          baseURL: env.LOCAL_BASE_URL,
          apiKey: env.LOCAL_API_KEY ?? "not-needed",
          format: "openai",
          model: env.LLM_MODEL ?? "local-model",
        };
      default:
        throw new Error(`Unknown legacy LLM_PROVIDER "${provider}" (anthropic|openrouter|llamacpp|deepseek|local)`);
    }
  }

  const configured = [
    env.ANTHROPIC_API_KEY ? "ANTHROPIC_API_KEY" : null,
    env.OPENROUTER_API_KEY ? "OPENROUTER_API_KEY" : null,
    env.LLAMACPP_BASE_URL ? "LLAMACPP_BASE_URL" : null,
    env.DEEPSEEK_API_KEY ? "DEEPSEEK_API_KEY" : null,
    env.LOCAL_BASE_URL ? "LOCAL_BASE_URL" : null,
  ].filter(Boolean) as string[];

  if (configured.length === 0) return null;
  if (configured.length === 1 && configured[0] === "ANTHROPIC_API_KEY") {
    legacyNotice();
    return {
      baseURL: "https://api.anthropic.com/v1",
      apiKey: env.ANTHROPIC_API_KEY!,
      format: "anthropic",
      model: env.LLM_MODEL ?? "claude-sonnet-5",
    };
  }

  throw new Error(
    `Ambiguous legacy LLM configuration (${configured.join(", ")}). Set LLM_API_BASE_URL + LLM_API_KEY + LLM_API_FORMAT, or set LLM_PROVIDER explicitly.`
  );
}

/**
 * Request-body knobs shared by the primary and fallback configs:
 * - `<p>THINKING_BUDGET`: cap the reasoning tokens of a thinking model.
 * - `<p>MAX_OUTPUT_TOKENS`: hard ceiling per generation, so a model that loops
 *   on its thinking channel cannot run an agent step for minutes.
 */
function envExtras(env: NodeJS.ProcessEnv, prefix = "LLM_"): Pick<ModelConfig, "extraBody"> | object {
  const extra: Record<string, unknown> = {};
  const budget = Number.parseInt(env[`${prefix}THINKING_BUDGET`] ?? "", 10);
  if (Number.isFinite(budget) && budget >= 0) Object.assign(extra, thinkingBudgetBody(budget));
  const cap = Number.parseInt(env[`${prefix}MAX_OUTPUT_TOKENS`] ?? "", 10);
  if (Number.isFinite(cap) && cap > 0) extra.max_tokens = cap;
  return Object.keys(extra).length ? { extraBody: extra } : {};
}

export function resolveModelConfig(env: NodeJS.ProcessEnv = process.env): ModelConfig {
  if (env.LLM_API_BASE_URL) {
    return {
      baseURL: env.LLM_API_BASE_URL,
      apiKey: env.LLM_API_KEY ?? "not-needed",
      format: parseFormat(env.LLM_API_FORMAT, "openai", "LLM_API_FORMAT"),
      model: env.LLM_MODEL ?? "",
      ...envExtras(env),
    };
  }

  const legacy = legacyConfig(env);
  if (legacy) return legacy;

  throw new Error(
    "No LLM configured. Set LLM_API_BASE_URL + LLM_API_KEY + LLM_API_FORMAT + LLM_MODEL."
  );
}

export function resolveFallbackConfig(env: NodeJS.ProcessEnv = process.env): ModelConfig | null {
  if (!env.LLM_FALLBACK_API_BASE_URL) return null;
  return {
    baseURL: env.LLM_FALLBACK_API_BASE_URL,
    apiKey: env.LLM_FALLBACK_API_KEY ?? "not-needed",
    format: parseFormat(env.LLM_FALLBACK_API_FORMAT, "openai", "LLM_FALLBACK_API_FORMAT"),
    model: env.LLM_FALLBACK_MODEL ?? "",
    ...envExtras(env, "LLM_FALLBACK_"),
  };
}

// Any OpenAI-compatible endpoint exposes GET /v1/models.
// Cache discovery per base URL for a short TTL — avoids a discovery
// round-trip on every single agent turn, while still noticing within a
// session that the user swapped which model (e.g. via llama-swap) has
// loaded (a process-lifetime cache would never see that again).
const DISCOVERY_TTL_MS = 60_000;
const discoveryCache = new Map<string, { promise: Promise<string>; expiresAt: number }>();

/**
 * Auto-discover the model id from an OpenAI-compatible /v1/models endpoint.
 * Prefers a model reported as "loaded" (e.g. by llama-swap); falls back to
 * the first listed. Results are cached per URL with a 60s TTL so model
 * swaps are noticed within a session.
 */
export async function discoverLlamaCppModel(baseURL: string): Promise<string> {
  const url = normalizeV1(baseURL);
  const cached = discoveryCache.get(url);
  if (cached && cached.expiresAt > Date.now()) {
    return cached.promise;
  }
  const promise = (async () => {
    const res = await fetch(`${url}/models`);
    if (!res.ok) {
      throw new Error(`Model discovery failed: ${res.status} at ${url}/models`);
    }
    const body = (await res.json()) as {
      data?: { id: string; status?: { value?: string } }[];
    };
    const models = body.data ?? [];
    if (models.length === 0) {
      throw new Error(`No models listed at ${url}/models`);
    }
    const loaded = models.find((m) => m.status?.value === "loaded");
    return (loaded ?? models[0]).id;
  })();
  discoveryCache.set(url, { promise, expiresAt: Date.now() + DISCOVERY_TTL_MS });
  // Don't cache failures — the server may just be starting up.
  promise.catch(() => discoveryCache.delete(url));
  return promise;
}

export async function createModel(cfg: ModelConfig): Promise<ResolvedLanguageModel> {
  let model = cfg.model;
  if (!model) {
    if (cfg.format === "openai") {
      try {
        model = await discoverLlamaCppModel(cfg.baseURL);
      } catch {
        throw new Error("LLM_MODEL is required for this endpoint.");
      }
    } else {
      throw new Error("LLM_MODEL is required for this endpoint.");
    }
  }

  switch (cfg.format) {
    case "anthropic":
      return createAnthropic({ baseURL: cfg.baseURL, apiKey: cfg.apiKey })(model) as ResolvedLanguageModel;
    case "openai":
      return createOpenAICompatible({
        name: "custom",
        baseURL: normalizeV1(cfg.baseURL),
        apiKey: cfg.apiKey,
        ...(cfg.extraBody && Object.keys(cfg.extraBody).length
          ? {
              transformRequestBody: (args: Record<string, unknown>) => ({
                ...args,
                ...cfg.extraBody,
              }),
            }
          : {}),
      })(model) as ResolvedLanguageModel;
  }
}
