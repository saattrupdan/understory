import express, { type Router } from "express";
import { convertToModelMessages, type UIMessage } from "ai";
import { streamChat, type KnowledgeBase } from "@understory/core";

interface ChatBody {
  messages: UIMessage[];
  model?: string;
}

/**
 * Streaming chat endpoint for the web UI (`useChat`). Full agent toolset —
 * the chat exists to exercise the same agent the MCP server uses.
 */
export function chatRouter(kb: KnowledgeBase): Router {
  const router = express.Router();

  router.post("/chat", async (req, res) => {
    try {
      const { messages, model } = req.body as ChatBody;
      const { result } = await streamChat(kb, convertToModelMessages(messages), { model });
      const response = result.toUIMessageStreamResponse({
        // AI SDK's default deliberately hides server errors. Chat failures carry
        // the partial-mutation/file list, which is part of this endpoint's safety
        // contract and must be visible to the caller.
        onError: (error) => (error instanceof Error ? error.message : String(error)),
      });
      res.status(response.status);
      response.headers.forEach((value, key) => res.setHeader(key, value));
      if (response.body) {
        for await (const chunk of response.body as unknown as AsyncIterable<Uint8Array>) {
          res.write(chunk);
        }
      }
      res.end();
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      // Errors before a stream exists otherwise become an empty client request.
      // Keep the response machine-readable; the web client extracts `error`.
      if (!res.headersSent) {
        res.status(500).json({ error: message });
      } else {
        res.end();
      }
    }
  });

  return router;
}
