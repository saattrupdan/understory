export const QUERY_PATHS_DEFAULT_OPEN = false;

// Keep the list below the persistent toggle in both the static and overlay layouts.
export const QUERY_PATHS_LAYOUT = {
  toggleTop: 12,
  toggleHeight: 40,
  headerHeight: 64,
  listPaddingTop: 8,
} as const;

export function queryPathsToggleLabel(open: boolean): string {
  return `${open ? "Collapse" : "Expand"} query paths sidebar`;
}

export function queryPathsToggleBottom(): number {
  return QUERY_PATHS_LAYOUT.toggleTop + QUERY_PATHS_LAYOUT.toggleHeight;
}

export function queryPathsListTop(): number {
  return QUERY_PATHS_LAYOUT.headerHeight + QUERY_PATHS_LAYOUT.listPaddingTop;
}
