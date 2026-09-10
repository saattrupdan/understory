const BOTTOM_TOLERANCE_PX = 24;

type ScrollMetrics = Pick<HTMLElement, "scrollHeight" | "scrollTop" | "clientHeight">;

export type ChatScrollState = {
  shouldFollow: boolean;
};

export function createChatScrollState(): ChatScrollState {
  return { shouldFollow: true };
}

/** Whether a scroll container is close enough to its end to keep following it. */
export function isNearBottom(element: ScrollMetrics): boolean {
  return element.scrollHeight - element.scrollTop - element.clientHeight <= BOTTOM_TOLERANCE_PX;
}

/** Record whether the user has opted out of following new content. */
export function updateChatScrollState(state: ChatScrollState, element: ScrollMetrics): void {
  state.shouldFollow = isNearBottom(element);
}

/** Explicit actions start a new turn at the latest content. */
export function reactivateChatScroll(state: ChatScrollState): void {
  state.shouldFollow = true;
}

/** Move to the latest content only while following is enabled. */
export function followLatestContent(
  state: ChatScrollState,
  element: Pick<HTMLElement, "scrollHeight" | "scrollTop">,
): void {
  if (state.shouldFollow) scrollToBottom(element);
}

/** Move a scroll container to its latest content without animated scrolling. */
export function scrollToBottom(element: Pick<HTMLElement, "scrollHeight" | "scrollTop">): void {
  element.scrollTop = element.scrollHeight;
}
