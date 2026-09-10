import { describe, expect, it } from "vitest";
import {
  QUERY_PATHS_DEFAULT_OPEN,
  QUERY_PATHS_LAYOUT,
  queryPathsListTop,
  queryPathsToggleBottom,
  queryPathsToggleLabel,
} from "../src/components/queryPathsLayout";

describe("query paths sidebar layout", () => {
  it("keeps the first row below the toggle hit box", () => {
    expect(queryPathsListTop()).toBeGreaterThan(queryPathsToggleBottom());
  });

  it("starts collapsed with matching accessible control labels", () => {
    expect(QUERY_PATHS_DEFAULT_OPEN).toBe(false);
    expect(queryPathsToggleLabel(QUERY_PATHS_DEFAULT_OPEN)).toBe(
      "Expand query paths sidebar"
    );
    expect(queryPathsToggleLabel(true)).toBe("Collapse query paths sidebar");
  });

  it("leaves a clear gap after the toggle hit box", () => {
    expect(queryPathsListTop() - queryPathsToggleBottom()).toBeGreaterThanOrEqual(12);
    expect(QUERY_PATHS_LAYOUT.headerHeight).toBeGreaterThan(
      QUERY_PATHS_LAYOUT.toggleTop + QUERY_PATHS_LAYOUT.toggleHeight
    );
  });
});
