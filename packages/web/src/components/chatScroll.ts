const BOTTOM_TOLERANCE_PX = 24;

type ScrollMetrics = Pick<HTMLElement, "scrollHeight" | "scrollTop" | "clientHeight">;

/** Whether a scroll container is close enough to its end to keep following it. */
export function isNearBottom(element: ScrollMetrics): boolean {
  return element.scrollHeight - element.scrollTop - element.clientHeight <= BOTTOM_TOLERANCE_PX;
}

/** Move a scroll container to its latest content without animated scrolling. */
export function scrollToBottom(element: Pick<HTMLElement, "scrollHeight" | "scrollTop">): void {
  element.scrollTop = element.scrollHeight;
}
