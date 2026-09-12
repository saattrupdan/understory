import { randomUUID } from "node:crypto";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import type { Request, Response, Router } from "express";
import express from "express";
import type { KnowledgeBase } from "@understory/core";
import { buildMcpServer } from "./server.js";

type RequestId = string | number;
type RegistryKey = string;
type CancellationMessage = {
  jsonrpc: "2.0";
  method: "notifications/cancelled";
  params: { requestId: RequestId; reason?: string };
};

type ActiveRequest = {
  transport: StreamableHTTPServerTransport;
  handlingStarted: boolean;
  closeResponse: () => void;
  pendingCancellation?: CancellationMessage;
};

// Stateless MCP HTTP has no server-side session store, but the standard
// Mcp-Session-Id header still gives a client a stable cancellation identity. A
// token is issued on initialize below and is used only by this registry; the
// transport/server remain fresh per request. Keep matching requests in a set so
// duplicate IDs from one client are also treated as ambiguous.
const activeRequests = new Map<RegistryKey, Set<ActiveRequest>>();

function registryKey(clientId: string | undefined, id: RequestId): RegistryKey | undefined {
  if (!clientId) return undefined;
  return `${clientId.length}:${clientId}:${typeof id}:${String(id)}`;
}

function messagesIn(body: unknown): unknown[] {
  return Array.isArray(body) ? body : body === undefined ? [] : [body];
}

function requestIds(body: unknown): RequestId[] {
  return messagesIn(body).flatMap((message) => {
    if (!message || typeof message !== "object") return [];
    const candidate = message as { method?: unknown; id?: unknown };
    return candidate.method !== undefined && (typeof candidate.id === "string" || typeof candidate.id === "number")
      ? [candidate.id]
      : [];
  });
}

function cancellationMessages(body: unknown): CancellationMessage[] {
  return messagesIn(body).flatMap((message) => {
    if (!message || typeof message !== "object") return [];
    const candidate = message as { jsonrpc?: unknown; method?: unknown; params?: unknown };
    if (candidate.jsonrpc !== "2.0" || candidate.method !== "notifications/cancelled" || !candidate.params) return [];
    const params = candidate.params as { requestId?: unknown; reason?: unknown };
    if (typeof params.requestId !== "string" && typeof params.requestId !== "number") return [];
    return [
      {
        jsonrpc: "2.0",
        method: "notifications/cancelled",
        params: {
          requestId: params.requestId,
          ...(typeof params.reason === "string" ? { reason: params.reason } : {}),
        },
      },
    ];
  });
}

function addActiveRequest(key: RegistryKey, entry: ActiveRequest): void {
  let entries = activeRequests.get(key);
  if (!entries) {
    entries = new Set();
    activeRequests.set(key, entries);
  }
  entries.add(entry);
}

function removeActiveRequest(key: RegistryKey, entry: ActiveRequest): void {
  const entries = activeRequests.get(key);
  if (!entries) return;
  entries.delete(entry);
  if (entries.size === 0) activeRequests.delete(key);
}

function removeActiveTransport(transport: StreamableHTTPServerTransport): void {
  for (const [id, entries] of activeRequests) {
    for (const entry of entries) {
      if (entry.transport === transport) entries.delete(entry);
    }
    if (entries.size === 0) activeRequests.delete(id);
  }
}

function cancelActiveRequest(clientId: string | undefined, message: CancellationMessage): void {
  const key = registryKey(clientId, message.params.requestId);
  if (!key) return;
  const entries = activeRequests.get(key);
  // Never guess when duplicate requests from one client share an id.
  if (!entries || entries.size !== 1) return;
  const entry = entries.values().next().value as ActiveRequest;
  if (!entry.handlingStarted || !entry.transport.onmessage) {
    entry.pendingCancellation = message;
    return;
  }
  entry.transport.onmessage(message);
  // JSON-response mode intentionally emits no response for a cancelled MCP
  // request, so close the originating HTTP response explicitly. Otherwise the
  // SDK's JSON response promise remains open forever after its handler aborts.
  entry.closeResponse();
  removeActiveTransport(entry.transport);
}

/**
 * MCP streamable-HTTP at /mcp. Stateless: a fresh McpServer + transport per
 * request (no session store) — the KB itself serializes mutations. Express
 * hands the SDK transport the raw Node req/res directly, so there is no
 * hijack/lifecycle glue and CORS is handled by the app-level cors() middleware.
 */
export function mcpRouter(kb: KnowledgeBase): Router {
  const router = express.Router();

  const handle = async (req: Request, res: Response) => {
    const suppliedClientId = req.get("Mcp-Session-Id");
    const isInitialization = messagesIn(req.body).some(
      (message) =>
        message &&
        typeof message === "object" &&
        (message as { method?: unknown }).method === "initialize"
    );
    // Stateless transports do not retain this token or use it for protocol
    // validation. It is returned only so SDK cancellation POSTs can be tied to
    // the originating client without allowing another client to cancel it.
    const clientId = suppliedClientId ?? (isInitialization ? randomUUID() : undefined);
    if (isInitialization && !suppliedClientId) res.setHeader("Mcp-Session-Id", clientId!);

    const cancellations = cancellationMessages(req.body);
    for (const cancellation of cancellations) cancelActiveRequest(clientId, cancellation);
    // A cancellation notification is itself a complete MCP POST. Do not build
    // a throwaway server for it; the originating transport already owns it.
    if (cancellations.length > 0 && requestIds(req.body).length === 0) {
      res.status(202).end();
      return;
    }

    const transport = new StreamableHTTPServerTransport({
      sessionIdGenerator: undefined, // stateless
      enableJsonResponse: true, // one JSON reply per request — no long-lived SSE
    });
    const ids = requestIds(req.body);
    const entries = ids.flatMap((id) => {
      const key = registryKey(clientId, id);
      if (!key) return [];
      const entry: ActiveRequest = {
        transport,
        handlingStarted: false,
        closeResponse: () => {
          if (!res.destroyed) res.destroy();
        },
      };
      addActiveRequest(key, entry);
      return [{ key, entry }];
    });

    try {
      const server = await buildMcpServer(kb);
      res.on("close", () => {
        removeActiveTransport(transport);
        transport.close();
        server.close();
      });
      await server.connect(transport);

      // Mark the request before entering the SDK transport. A cancellation
      // arriving while the server was being built is dispatched only after the
      // SDK has installed its request handler and signal map.
      for (const { entry } of entries) entry.handlingStarted = true;
      // express.json() already parsed the body; pass it so the transport
      // doesn't try to re-read the consumed stream.
      const handling = transport.handleRequest(req, res, req.body);
      for (const { entry } of entries) {
        if (entry.pendingCancellation) {
          const cancellation = entry.pendingCancellation;
          entry.pendingCancellation = undefined;
          transport.onmessage?.(cancellation);
          entry.closeResponse();
          removeActiveTransport(entry.transport);
        }
      }
      await handling;
    } finally {
      for (const { key, entry } of entries) removeActiveRequest(key, entry);
    }
  };

  router.post("/", handle);
  router.get("/", handle);
  router.delete("/", handle);
  return router;
}
