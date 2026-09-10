import { describe, expect, it } from "vitest";
import {
  createChatScrollState,
  followLatestContent,
  reactivateChatScroll,
  updateChatScrollState,
} from "../src/components/chatScroll";

type FakeScrollElement = {
  scrollHeight: number;
  scrollTop: number;
  clientHeight: number;
};

describe("chat scroll follow state", () => {
  it("follows content initially", () => {
    const state = createChatScrollState();
    const element: FakeScrollElement = {
      scrollHeight: 500,
      scrollTop: 0,
      clientHeight: 300,
    };

    expect(state.shouldFollow).toBe(true);
    followLatestContent(state, element);
    expect(element.scrollTop).toBe(500);
  });

  it("opts out after scrolling up and ignores delayed content growth", () => {
    const state = createChatScrollState();
    const element: FakeScrollElement = {
      scrollHeight: 500,
      scrollTop: 100,
      clientHeight: 300,
    };

    updateChatScrollState(state, element);
    expect(state.shouldFollow).toBe(false);

    element.scrollHeight = 900;
    followLatestContent(state, element);
    expect(element.scrollTop).toBe(100);
  });

  it("resumes at the bottom, including within the bottom tolerance", () => {
    const state = createChatScrollState();
    const element: FakeScrollElement = {
      scrollHeight: 900,
      scrollTop: 576,
      clientHeight: 300,
    };

    updateChatScrollState(state, element);
    expect(state.shouldFollow).toBe(true);
    followLatestContent(state, element);
    expect(element.scrollTop).toBe(900);
  });

  it("reactivates following for a new turn after an intentional scroll-up", () => {
    const state = createChatScrollState();
    const element: FakeScrollElement = {
      scrollHeight: 500,
      scrollTop: 100,
      clientHeight: 300,
    };

    updateChatScrollState(state, element);
    reactivateChatScroll(state);
    expect(state.shouldFollow).toBe(true);

    element.scrollHeight = 700;
    followLatestContent(state, element);
    expect(element.scrollTop).toBe(700);
  });
});
