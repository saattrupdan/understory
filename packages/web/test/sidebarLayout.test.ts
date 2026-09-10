import { describe, expect, it } from "vitest";
import { sidebarControlLayout } from "../src/components/sidebarLayout";

describe("integrated sidebar controls", () => {
  it.each([
    [false, false, true, false, false],
    [false, true, false, true, true],
    [true, false, false, false, false],
    [true, true, false, false, true],
  ])(
    "coordinates chat and query controls (chat=%s, query=%s)",
    (
      chatOpen,
      queryPathsOpen,
      showChatLauncher,
      showMobileChatLauncher,
      queryToggleInHeader
    ) => {
      expect(
        sidebarControlLayout({
          chatOpen,
          graphVisible: true,
          queryPathsOpen,
        })
      ).toEqual({
        showChatLauncher,
        showMobileChatLauncher,
        queryToggleInSidebarHeader: queryToggleInHeader,
      });
    }
  );

  it("keeps the chat route available outside the graph", () => {
    expect(
      sidebarControlLayout({ chatOpen: false, graphVisible: false, queryPathsOpen: true })
        .showChatLauncher
    ).toBe(true);
    expect(
      sidebarControlLayout({ chatOpen: false, graphVisible: false, queryPathsOpen: true })
        .showMobileChatLauncher
    ).toBe(false);
  });
});
