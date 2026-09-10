import { useCallback, useEffect, useRef, useState } from "react";
import {
  api,
  ApiError,
  setAuthToken,
  type AppConfig,
  type Concept,
  type ConformanceReport,
  type LogEntry,
  type SearchHit,
  type TreeNode,
} from "./api";
import { Tree } from "./components/Tree";
import { ConceptView } from "./components/ConceptView";
import { LogView } from "./components/LogView";
import { ChatPanel } from "./components/ChatPanel";
import { GraphView } from "./components/GraphView";
import { QUERY_PATHS_DEFAULT_OPEN } from "./components/queryPathsLayout";

type View =
  | { kind: "concept"; path: string }
  | { kind: "log" }
  | { kind: "graph" }
  | { kind: "empty" };

const isNarrowViewport = () =>
  typeof window !== "undefined" && window.matchMedia("(max-width: 1023px)").matches;

export default function App() {
  const [tree, setTree] = useState<TreeNode | null>(null);
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [report, setReport] = useState<ConformanceReport | null>(null);
  const [log, setLog] = useState<LogEntry[]>([]);
  const [view, setView] = useState<View>({ kind: "graph" });
  const [concept, setConcept] = useState<Concept | null>(null);
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<SearchHit[] | null>(null);
  const [memoryOpen, setMemoryOpen] = useState(false);
  const [chatOpen, setChatOpen] = useState(false);
  const [queryPathsOpen, setQueryPathsOpen] = useState(QUERY_PATHS_DEFAULT_OPEN);
  const memoryExpandRef = useRef<HTMLButtonElement>(null);
  const memoryCollapseRef = useRef<HTMLButtonElement>(null);
  const pendingMemoryFocusRef = useRef<"expand" | "collapse" | null>(null);
  const chatExpandRef = useRef<HTMLButtonElement>(null);
  const chatCollapseRef = useRef<HTMLButtonElement>(null);
  const pendingChatFocusRef = useRef<"expand" | "collapse" | null>(null);
  const queryExpandRef = useRef<HTMLButtonElement>(null);
  const queryCollapseRef = useRef<HTMLButtonElement>(null);
  const pendingQueryFocusRef = useRef<"expand" | "collapse" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [needsToken, setNeedsToken] = useState(false);
  const [tokenInput, setTokenInput] = useState("");

  const refresh = useCallback(() => {
    api
      .tree()
      .then((t) => {
        setTree(t);
        setNeedsToken(false);
      })
      .catch((e) => {
        if (e instanceof ApiError && e.status === 401) setNeedsToken(true);
        else setError(String(e));
      });
    api.validate().then(setReport).catch(() => {});
    api.log().then(setLog).catch(() => {});
  }, []);

  useEffect(() => {
    refresh();
    api.config().then(setConfig).catch(() => {});
  }, [refresh]);

  useEffect(() => {
    if (view.kind === "concept") {
      api.concept(view.path).then(setConcept).catch((e) => setError(String(e)));
    }
  }, [view]);

  useEffect(() => {
    if (view.kind !== "graph") setQueryPathsOpen(QUERY_PATHS_DEFAULT_OPEN);
  }, [view.kind]);

  useEffect(() => {
    const focusTarget = pendingMemoryFocusRef.current;
    if (!focusTarget) return;
    pendingMemoryFocusRef.current = null;
    if (focusTarget === "collapse") memoryCollapseRef.current?.focus();
    else memoryExpandRef.current?.focus();
  }, [memoryOpen]);

  useEffect(() => {
    const focusTarget = pendingChatFocusRef.current;
    if (!focusTarget) return;
    pendingChatFocusRef.current = null;
    if (focusTarget === "collapse") chatCollapseRef.current?.focus();
    else chatExpandRef.current?.focus();
  }, [chatOpen]);

  useEffect(() => {
    const focusTarget = pendingQueryFocusRef.current;
    if (!focusTarget) return;
    pendingQueryFocusRef.current = null;
    if (focusTarget === "collapse") queryCollapseRef.current?.focus();
    else queryExpandRef.current?.focus();
  }, [queryPathsOpen]);

  const openMemory = () => {
    pendingMemoryFocusRef.current = "collapse";
    if (isNarrowViewport()) {
      pendingChatFocusRef.current = null;
      pendingQueryFocusRef.current = null;
      setChatOpen(false);
      setQueryPathsOpen(false);
    }
    setMemoryOpen(true);
  };

  const collapseMemory = () => {
    pendingMemoryFocusRef.current = "expand";
    setMemoryOpen(false);
  };

  const openChat = () => {
    pendingChatFocusRef.current = "collapse";
    if (isNarrowViewport()) {
      pendingMemoryFocusRef.current = null;
      pendingQueryFocusRef.current = null;
      setMemoryOpen(false);
      setQueryPathsOpen(false);
    }
    setChatOpen(true);
  };

  const collapseChat = () => {
    pendingChatFocusRef.current = "expand";
    setChatOpen(false);
  };

  const setQueryPathsVisibility = (open: boolean) => {
    pendingQueryFocusRef.current = open ? "collapse" : "expand";
    if (open && isNarrowViewport()) {
      pendingChatFocusRef.current = null;
      pendingMemoryFocusRef.current = null;
      setChatOpen(false);
      setMemoryOpen(false);
    }
    setQueryPathsOpen(open);
  };

  // Re-load the open concept (and graph) after chat mutations.
  const [graphRefreshKey, setGraphRefreshKey] = useState(0);
  const onMutation = useCallback(() => {
    refresh();
    setGraphRefreshKey((k) => k + 1);
    setView((v) => ({ ...v }));
  }, [refresh]);

  useEffect(() => {
    if (!query.trim()) {
      setHits(null);
      return;
    }
    const t = setTimeout(() => api.search(query).then(setHits).catch(() => {}), 200);
    return () => clearTimeout(t);
  }, [query]);

  const openConcept = useCallback((path: string) => {
    if (path === "/log.md") {
      setView({ kind: "log" });
    } else {
      setView({ kind: "concept", path });
    }
    setError(null);
    setQuery("");
  }, []);

  if (needsToken) {
    return (
      <div className="flex h-screen items-center justify-center">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            setAuthToken(tokenInput.trim());
            setTokenInput("");
            refresh();
            api.config().then(setConfig).catch(() => {});
          }}
          className="w-80 rounded-xl border border-zinc-800 bg-zinc-900/60 p-5"
        >
          <h1 className="text-sm font-bold tracking-wide text-cyan-300">understory 🌱</h1>
          <p className="mt-2 text-sm text-zinc-400">
            This server requires an access token (its <code className="text-zinc-300">AUTH_TOKEN</code>).
          </p>
          <input
            type="password"
            value={tokenInput}
            onChange={(e) => setTokenInput(e.target.value)}
            placeholder="Access token"
            autoFocus
            className="mt-3 w-full rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm outline-none focus:border-cyan-600"
          />
          <button
            type="submit"
            className="mt-3 w-full rounded-lg bg-cyan-700 px-3 py-2 text-sm font-medium text-white hover:bg-cyan-600"
          >
            Unlock
          </button>
        </form>
      </div>
    );
  }

  return (
    <div data-testid="app-shell" className="relative flex h-screen min-w-0 overflow-hidden">
      {!memoryOpen && (
        <button
          ref={memoryExpandRef}
          data-testid="memory-sidebar-toggle"
          type="button"
          onClick={openMemory}
          aria-label="Expand memory sidebar"
          aria-expanded={false}
          aria-controls="memory-sidebar"
          title="Expand memory sidebar"
          className="absolute left-0 top-1/2 z-50 h-10 w-9 -translate-y-1/2 rounded-r-lg border border-l-0 border-zinc-700 bg-zinc-900/95 text-sm text-zinc-300 shadow-lg hover:bg-zinc-800 hover:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-cyan-500"
        >
          <span aria-hidden="true">→</span>
        </button>
      )}

      <aside
        id="memory-sidebar"
        data-testid="memory-sidebar"
        aria-label="Memory navigation sidebar"
        aria-hidden={!memoryOpen}
        hidden={!memoryOpen}
        className={
          memoryOpen
            ? "fixed inset-y-0 left-0 z-40 flex w-[min(18rem,calc(100vw-3rem))] flex-col border-r border-zinc-800 bg-zinc-950 shadow-2xl lg:static lg:z-auto lg:w-72 lg:shrink-0 lg:shadow-none"
            : undefined
        }
      >
        <div data-testid="navigation-header" className="border-b border-zinc-800 p-3">
          <div className="flex items-center gap-2">
            <h1 className="text-sm font-bold tracking-wide text-cyan-300">understory 🌱</h1>
            {report && (
              <span
                title={`${report.conceptCount} concepts, ${report.issues.length} issues`}
                className={`ml-auto rounded px-1.5 py-0.5 text-[10px] font-medium ${
                  report.conformant
                    ? "bg-emerald-900/60 text-emerald-300"
                    : "bg-red-900/60 text-red-300"
                }`}
              >
                {report.conformant ? "conformant" : "non-conformant"}
              </span>
            )}
            <button
              ref={memoryCollapseRef}
              data-testid="memory-sidebar-collapse"
              type="button"
              onClick={collapseMemory}
              aria-label="Collapse memory sidebar"
              aria-expanded={true}
              aria-controls="memory-sidebar"
              title="Collapse memory sidebar"
              className="shrink-0 rounded border border-zinc-700 px-2 py-0.5 text-xs text-zinc-300 hover:bg-zinc-800 hover:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-cyan-500"
            >
              <span aria-hidden="true">←</span>
            </button>
          </div>
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search…"
            className="mt-2 w-full rounded-lg border border-zinc-700 bg-zinc-900 px-2 py-1 text-sm outline-none focus:border-cyan-600"
          />
        </div>

        <div className="flex-1 overflow-y-auto p-2">
          {hits ? (
            <div className="space-y-1">
              {hits.length === 0 && <p className="px-2 text-xs text-zinc-500">No matches.</p>}
              {hits.map((h) => (
                <button
                  key={h.path}
                  onClick={() => openConcept(h.path)}
                  className="block w-full rounded px-2 py-1.5 text-left hover:bg-zinc-800"
                >
                  <div className="text-sm text-zinc-200">{h.title ?? h.path}</div>
                  {h.snippet && <div className="truncate text-xs text-zinc-500">{h.snippet}</div>}
                </button>
              ))}
            </div>
          ) : tree ? (
            <Tree
              node={tree}
              selected={view.kind === "concept" ? view.path : null}
              onSelect={openConcept}
            />
          ) : (
            <p className="px-2 text-xs text-zinc-500">Loading…</p>
          )}
        </div>

        <div className="flex border-t border-zinc-800 text-xs">
          <button
            onClick={() => setView({ kind: "log" })}
            className={`flex-1 px-3 py-2 hover:bg-zinc-800 ${view.kind === "log" ? "text-cyan-300" : "text-zinc-400"}`}
          >
            Log
          </button>
          <button
            onClick={() => setView({ kind: "graph" })}
            className={`flex-1 border-l border-zinc-800 px-3 py-2 hover:bg-zinc-800 ${view.kind === "graph" ? "text-cyan-300" : "text-zinc-400"}`}
          >
            Graph
          </button>
        </div>
      </aside>

      <main
        className={`relative min-w-0 flex-1 ${view.kind === "graph" ? "overflow-hidden" : "overflow-y-auto"}`}
      >
        {error && <p className="p-6 text-sm text-red-400">{error}</p>}
        {!error && view.kind === "empty" && (
          <div className="flex h-full items-center justify-center text-zinc-600">
            <p>Select a concept, or ask the agent →</p>
          </div>
        )}
        {!error && view.kind === "concept" && concept && (
          <ConceptView concept={concept} onNavigate={openConcept} />
        )}
        {!error && view.kind === "log" && <LogView entries={log} onNavigate={openConcept} />}
        {!error && view.kind === "graph" && (
          <GraphView
            refreshKey={graphRefreshKey}
            onNavigate={openConcept}
            pathsOpen={queryPathsOpen}
            onPathsOpenChange={setQueryPathsVisibility}
            expandButtonRef={queryExpandRef}
            collapseButtonRef={queryCollapseRef}
          />
        )}
      </main>

      {!chatOpen && (
        <button
          ref={chatExpandRef}
          data-testid="chat-sidebar-toggle"
          type="button"
          onClick={openChat}
          aria-label="Expand chat sidebar"
          aria-expanded={false}
          aria-controls="chat-sidebar"
          title="Expand chat sidebar"
          className="absolute right-0 top-1/2 z-50 h-10 w-9 -translate-y-1/2 rounded-l-lg border border-r-0 border-zinc-700 bg-zinc-900/95 text-sm text-zinc-300 shadow-lg hover:bg-zinc-800 hover:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-cyan-500"
        >
          <span aria-hidden="true">←</span>
        </button>
      )}

      <aside
        id="chat-sidebar"
        data-testid="chat-sidebar"
        aria-label="Chat sidebar"
        aria-hidden={!chatOpen}
        hidden={!chatOpen}
        className={
          chatOpen
            ? "fixed inset-y-0 right-0 z-40 w-[min(24rem,calc(100vw-3rem))] border-l border-zinc-800 bg-zinc-950 shadow-2xl lg:static lg:z-auto lg:w-96 lg:max-w-[35vw] lg:shrink-0 lg:shadow-none"
            : undefined
        }
      >
        <ChatPanel
          config={config}
          onMutation={onMutation}
          onOpenConcept={openConcept}
          onCollapse={collapseChat}
          collapseButtonRef={chatCollapseRef}
        />
      </aside>
    </div>
  );
}
