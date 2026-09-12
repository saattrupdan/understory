import http from "node:http";
import express from "express";
import { DefaultChatTransport } from "ai";
import { afterEach, describe, expect, it, vi } from "vitest";

const streamChatMock = vi.hoisted(() => vi.fn());
vi.mock("@understory/core", async () => {
  const actual = await vi.importActual<typeof import("@understory/core")>("@understory/core");
  return { ...actual, streamChat: streamChatMock };
});

import { chatRouter } from "../src/api/chat.js";
import { createApp } from "../src/index.js";

let server: http.Server | undefined;

afterEach(async () => {
  vi.unstubAllEnvs();
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
    streamChatMock.mockRejectedValueOnce(new Error("The model provider is unavailable"));
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
      error: "The model provider is unavailable",
    });
    const options = streamChatMock.mock.calls[0]?.[2] as { signal: AbortSignal };
    expect(options.signal).toBeInstanceOf(AbortSignal);
    expect(options.signal.aborted).toBe(false);
  });

  it("aborts the server-side stream when the client disconnects", async () => {
    let started!: () => void;
    let capturedSignal!: AbortSignal;
    const streamStarted = new Promise<void>((resolve) => {
      started = resolve;
    });
    streamChatMock.mockImplementationOnce(
      (_kb: unknown, _messages: unknown, options: { signal: AbortSignal }) => {
        capturedSignal = options.signal;
        started();
        return new Promise((_resolve, reject) => {
          options.signal.addEventListener("abort", () => reject(options.signal.reason), {
            once: true,
          });
        });
      }
    );
    const app = express();
    app.use(express.json());
    app.use(chatRouter({} as never));
    server = http.createServer(app);
    await new Promise<void>((resolve) => server!.listen(0, "127.0.0.1", () => resolve()));
    const address = server.address();
    if (!address || typeof address === "string") throw new Error("test server did not start");

    const request = http.request({
      host: "127.0.0.1",
      port: address.port,
      path: "/chat",
      method: "POST",
      headers: { "content-type": "application/json" },
    });
    request.on("error", () => {});
    request.end(JSON.stringify({ messages: [] }));
    await streamStarted;
    request.destroy();
    await vi.waitFor(() => expect(capturedSignal.aborted).toBe(true));
  });

  it("encodes a post-header failure as an AI SDK error part", async () => {
    const failure = new Error("synthesis failed");
    const chunksBeforeFailure = [
      { type: "start" },
      { type: "text-start", id: "text-1" },
      { type: "text-delta", id: "text-1", delta: "partial answer" },
      { type: "text-end", id: "text-1" },
      { type: "finish", finishReason: "stop" },
    ];
    let chunkIndex = 0;
    const failingStream = new ReadableStream({
      pull(controller) {
        if (chunkIndex < chunksBeforeFailure.length) {
          controller.enqueue(chunksBeforeFailure[chunkIndex++]);
        } else {
          controller.error(failure);
        }
      },
    });
    streamChatMock.mockResolvedValueOnce({
      result: { toUIMessageStream: vi.fn(() => failingStream) },
      filesChanged: new Set(["concepts/example.md"]),
    });
    vi.stubEnv("AUTH_TOKEN", "");
    const app = createApp({ bundle: { root: process.cwd() } } as never);
    server = http.createServer(app);
    await new Promise<void>((resolve) => server!.listen(0, "127.0.0.1", () => resolve()));
    const address = server.address();
    if (!address || typeof address === "string") throw new Error("test server did not start");

    const transport = new DefaultChatTransport({ api: `http://127.0.0.1:${address.port}/api/chat` });
    const stream = await transport.sendMessages({
      trigger: "submit-message",
      chatId: "test-chat",
      messageId: undefined,
      messages: [{ id: "user-1", role: "user", parts: [{ type: "text", text: "hello" }] }],
      abortSignal: undefined,
    });
    const chunks: Array<{ type: string; errorText?: string }> = [];
    for await (const chunk of stream) chunks.push(chunk as { type: string; errorText?: string });

    expect(chunks.some((chunk) => chunk.type === "finish")).toBe(true);
    expect(chunks).toContainEqual({
      type: "error",
      errorText:
        "synthesis failed\n\n⚠ Partial mutation: 1 file(s) changed before failure.\nFiles changed:\n- concepts/example.md",
    });
  });

  it("passes chat histories larger than the bounded API parser", async () => {
    streamChatMock.mockResolvedValueOnce({
      result: {
        toUIMessageStream: vi.fn(
          () => new ReadableStream({ start(controller) { controller.close(); } })
        ),
      },
      filesChanged: new Set<string>(),
    });
    vi.stubEnv("AUTH_TOKEN", "");
    const app = createApp({ bundle: { root: process.cwd() } } as never);
    server = http.createServer(app);
    await new Promise<void>((resolve) => server!.listen(0, "127.0.0.1", () => resolve()));
    const address = server.address();
    if (!address || typeof address === "string") throw new Error("test server did not start");

    const response = await fetch(`http://127.0.0.1:${address.port}/api/chat`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        messages: [{ id: "user-1", role: "user", parts: [{ type: "text", text: "x".repeat(4_500_000) }] }],
      }),
    });

    expect(response.status).toBe(200);
    await response.text();
    expect(streamChatMock).toHaveBeenCalledOnce();
  });
});
