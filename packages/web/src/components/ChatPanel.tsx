import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type KeyboardEvent,
  type RefObject,
} from "react";
import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { MarkdownRenderer } from "../components/MarkdownRenderer";
import { authHeaders } from "../api";
import type { AppConfig } from "../api";
import {
  createChatScrollState,
  followLatestContent,
  reactivateChatScroll,
  updateChatScrollState,
} from "./chatScroll";

const WRITE_TOOLS = new Set(["write_concept", "patch_concept", "delete_concept"]);

function chatErrorMessage(error: Error): string {
  try {
    const parsed: unknown = JSON.parse(error.message);
    if (parsed && typeof parsed === "object" && "error" in parsed) {
      const message = (parsed as { error?: unknown }).error;
      if (typeof message === "string") return message;
    }
  } catch {
    // The AI SDK may provide a plain provider or network error.
  }
  return error.message || "The chat request failed.";
}

/**
 * Chat with the same agent the MCP server runs. Tool calls render inline —
 * watching which tools fire on which files is how we test the agent.
 */
export function ChatPanel({
  config,
  onMutation,
  onOpenConcept,
  onCollapse,
  collapseButtonRef,
}: {
  config: AppConfig | null;
  onMutation: () => void;
  onOpenConcept: (path: string) => void;
  onCollapse: () => void;
  collapseButtonRef: RefObject<HTMLButtonElement>;
}) {
  const [input, setInput] = useState("");
  const [model, setModel] = useState("");
  const chatScrollRef = useRef<HTMLDivElement>(null);
  const chatContentRef = useRef<HTMLDivElement>(null);
  const chatScrollStateRef = useRef(createChatScrollState());

  const handleChatScroll = () => {
    const element = chatScrollRef.current;
    if (element) updateChatScrollState(chatScrollStateRef.current, element);
  };

  const followLatestChatContent = () => {
    const element = chatScrollRef.current;
    if (element) followLatestContent(chatScrollStateRef.current, element);
  };

  const reactivateChat = () => reactivateChatScroll(chatScrollStateRef.current);
  const { messages, sendMessage, setMessages, status, error, clearError } = useChat({
    transport: new DefaultChatTransport({
      api: "/api/chat",
      headers: () => authHeaders(),
      body: () => ({ model: model || undefined }),
    }),
    onFinish: () => onMutation(), // refresh browse pane; agent may have written files
  });

  const busy = status === "submitted" || status === "streaming";

  const handleInputKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (
      event.key !== "Enter" ||
      event.shiftKey ||
      event.nativeEvent.isComposing
    ) {
      return;
    }

    event.preventDefault();
    event.currentTarget.form?.requestSubmit();
  };

  // Keep following streamed text and tool updates while the user is already at
  // the bottom. Once they scroll up, leave the viewport where they put it.
  useLayoutEffect(() => {
    followLatestChatContent();
  }, [messages, status, error]);

  // Markdown rendering can change the content height after the message update
  // (for example when syntax highlighting finishes), so message dependencies
  // alone are not enough to keep the latest content visible.
  useEffect(() => {
    const content = chatContentRef.current;
    if (!content || typeof ResizeObserver === "undefined") return;

    const observer = new ResizeObserver(followLatestChatContent);
    observer.observe(content);
    return () => observer.disconnect();
  }, []);

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-2 border-b border-zinc-800 px-3 py-2">
        <h2 className="text-sm font-semibold text-zinc-300">Agent chat</h2>
        <div className="ml-auto flex items-center gap-1.5">
          <button
            ref={collapseButtonRef}
            type="button"
            onClick={onCollapse}
            aria-label="Collapse chat sidebar"
            aria-expanded={true}
            aria-controls="chat-sidebar"
            title="Collapse chat sidebar"
            className="rounded border border-zinc-700 px-2 py-0.5 text-xs text-zinc-300 hover:bg-zinc-800 hover:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-cyan-500"
          >
            <span aria-hidden="true">→</span>
          </button>
          {config && (
            <>
              {config.fallbackConfigured && (
                <span title="Fallback model configured" className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
              )}
              <input
                value={model}
                onChange={(e) => setModel(e.target.value)}
                placeholder={config.model}
                className="w-32 rounded border border-zinc-700 bg-zinc-900 px-1.5 py-0.5 text-xs text-zinc-300 outline-none focus:border-cyan-600"
              />
            </>
          )}
          <button
            type="button"
            onClick={() => {
              reactivateChat();
              setMessages([]);
              clearError();
            }}
            disabled={busy}
            className="rounded border border-zinc-700 px-2 py-0.5 text-xs text-zinc-300 hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-50"
          >
            Reset chat
          </button>
        </div>
      </div>

      <div ref={chatScrollRef} onScroll={handleChatScroll} className="flex-1 overflow-y-auto">
        <div ref={chatContentRef} className="space-y-3 p-3">
          {error && (
          <div role="alert" className="rounded-lg border border-red-800/70 bg-red-950/40 px-3 py-2 text-sm text-red-200">
            <p>{chatErrorMessage(error)}</p>
            <div className="mt-2 flex gap-2 text-xs">
              <button
                type="button"
                onClick={() => {
                  reactivateChat();
                  clearError();
                  void sendMessage();
                }}
                className="rounded border border-red-700 px-2 py-1 font-semibold hover:bg-red-900/50"
              >
                Retry
              </button>
              <button
                type="button"
                onClick={clearError}
                className="rounded border border-zinc-700 px-2 py-1 hover:bg-zinc-800"
              >
                Dismiss
              </button>
            </div>
          </div>
        )}
        {messages.length === 0 && (
          <p className="p-4 text-sm text-zinc-500">
            Test the knowledge agent here — ask a question, or tell it something worth
            remembering. Tool calls show inline so you can watch it work.
          </p>
        )}
        {messages.map((m) => (
          <div key={m.id} className={m.role === "user" ? "text-right" : ""}>
            {m.parts.map((part, i) => {
              if (part.type === "text") {
                return (
                  <div
                    key={i}
                    className={`markdown inline-block max-w-[95%] rounded-xl px-3 py-2 text-left text-sm ${
                      m.role === "user" ? "bg-cyan-900/50" : "bg-zinc-900 border border-zinc-800"
                    }`}
                  >
                    <MarkdownRenderer>{part.text}</MarkdownRenderer>
                  </div>
                );
              }
              if (part.type.startsWith("tool-")) {
                const toolName = part.type.slice(5);
                const p = part as unknown as {
                  state: string;
                  input?: Record<string, unknown>;
                  output?: unknown;
                };
                const filePath =
                  typeof p.input?.path === "string" ? (p.input.path as string) : undefined;
                return (
                  <div
                    key={i}
                    className={`my-1 flex items-center gap-2 rounded-lg border px-2 py-1 font-mono text-xs ${
                      WRITE_TOOLS.has(toolName)
                        ? "border-amber-800/60 bg-amber-950/30 text-amber-300"
                        : "border-zinc-800 bg-zinc-900/60 text-zinc-400"
                    }`}
                  >
                    <span>{p.state === "output-available" ? "✓" : "…"}</span>
                    <span className="font-semibold">{toolName}</span>
                    {filePath && (
                      <button
                        onClick={() => onOpenConcept(filePath)}
                        className="truncate text-cyan-400 hover:underline"
                      >
                        {filePath}
                      </button>
                    )}
                    {!filePath && typeof p.input?.query === "string" && (
                      <span className="truncate text-zinc-500">"{String(p.input.query)}"</span>
                    )}
                  </div>
                );
              }
              return null;
            })}
          </div>
        ))}
          {busy && <div className="animate-pulse text-xs text-zinc-500">agent working…</div>}
        </div>
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (!input.trim() || busy) return;
          reactivateChat();
          sendMessage({ text: input });
          setInput("");
        }}
        className="border-t border-zinc-800 p-3"
      >
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleInputKeyDown}
          placeholder="Ask or teach the knowledge base…"
          rows={3}
          className="w-full resize-y rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm outline-none focus:border-cyan-600"
        />
      </form>
    </div>
  );
}
