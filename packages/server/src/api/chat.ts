import express, { type Router } from "express";
import {
  createUIMessageStream,
  createUIMessageStreamResponse,
  convertToModelMessages,
  type UIMessage,
} from "ai";
import { streamChat, type KnowledgeBase } from "@understory/core";

interface ChatBody {
  messages: UIMessage[];
  model?: string;
}

function chatErrorMessage(error: unknown, filesChanged: Set<string>): string {
  const message = error instanceof Error ? error.message : String(error);
  // streamChat includes this suffix in trace/error messages too. Avoid adding it
  // twice when an AI SDK error part is passed through the wrapper below.
  if (filesChanged.size === 0 || message.includes("⚠ Partial mutation:")) return message;
  const files = [...filesChanged].sort();
  return `${message}\n\n⚠ Partial mutation: ${files.length} file(s) changed before failure.\nFiles changed:\n${files.map((file) => `- ${file}`).join("\n")}`;
}

/**
 * Streaming chat endpoint for the web UI (`useChat`). Full agent toolset —
 * the chat exists to exercise the same agent the MCP server uses.
 */
export function chatRouter(kb: KnowledgeBase): Router {
  const router = express.Router();

  // This is intentionally route-specific: chat history has no application-level
  // size ceiling. Other API/MCP requests use the app's 4 MiB parser.
  router.post("/chat", express.json({ limit: Infinity }), async (req, res) => {
    let responseStarted = false;
    try {
      const { messages, model } = req.body as ChatBody;
      const { result, filesChanged } = await streamChat(
        kb,
        convertToModelMessages(messages),
        { model }
      );
      const onError = (error: unknown) => chatErrorMessage(error, filesChanged);

      // streamText emits its `finish` chunk before awaiting stream callbacks such
      // as onFinish. Wrap its UI stream in a second AI SDK v5 stream so a later
      // callback rejection is encoded as an `error` part followed by [DONE],
      // rather than looking like a clean response to DefaultChatTransport.
      const stream = createUIMessageStream({
        execute: ({ writer }) => {
          writer.merge(result.toUIMessageStream({ onError }));
        },
        onError,
      });
      responseStarted = true;
      const response = createUIMessageStreamResponse({ stream });
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
      if (!responseStarted && !res.headersSent) {
        res.status(500).json({ error: message });
      } else {
        // The AI SDK wrapper handles stream failures. Anything reaching here is
        // an unexpected transport/write failure; do not turn it into a clean EOF.
        res.destroy(error instanceof Error ? error : new Error(message));
      }
    }
  });

  return router;
}
