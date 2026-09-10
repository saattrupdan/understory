import http from "node:http";
import express from "express";
import { afterEach, describe, expect, it, vi } from "vitest";

const streamChatMock = vi.hoisted(() => vi.fn());
vi.mock("@understory/core", () => ({ streamChat: streamChatMock }));

import { chatRouter } from "../src/api/chat.js";

let server: http.Server | undefined;

afterEach(async () => {
  streamChatMock.mockReset();
  if (server) {
    await new Promise<void>((resolve, reject) =>
      server!.close((error) => (error ? reject(error) : resolve()))
    );
    server = undefined;
  }
});

describe("chat endpoint", () => {
  it("returns a visible HTTP error when streaming cannot start", async () => {
    streamChatMock.mockRejectedValueOnce(new Error("Chat history exceeds AGENT_CHAT_MAX_INPUT_CHARS"));
    const app = express();
    app.use(express.json());
    app.use(chatRouter({} as never));
    server = http.createServer(app);
    await new Promise<void>((resolve) => server!.listen(0, "127.0.0.1", () => resolve()));
    const address = server.address();
    if (!address || typeof address === "string") throw new Error("test server did not start");

    const response = await fetch(`http://127.0.0.1:${address.port}/chat`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ messages: [] }),
    });

    expect(response.status).toBe(500);
    expect(await response.json()).toEqual({
      error: "Chat history exceeds AGENT_CHAT_MAX_INPUT_CHARS",
    });
  });
});
