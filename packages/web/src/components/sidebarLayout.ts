export interface SidebarLayoutState {
  chatOpen: boolean;
  graphVisible: boolean;
  queryPathsOpen: boolean;
}

export interface SidebarControlLayout {
  /** The main-area launcher is redundant while the query sidebar is open. */
  showChatLauncher: boolean;
  /** On narrow screens the query overlay hides the left sidebar route. */
  showMobileChatLauncher: boolean;
  /** An open query sidebar owns its toggle in the shared header. */
  queryToggleInSidebarHeader: boolean;
}

/** Keep controls from occupying the other sidebar's content area. */
export function sidebarControlLayout({
  chatOpen,
  graphVisible,
  queryPathsOpen,
}: SidebarLayoutState): SidebarControlLayout {
  return {
    showChatLauncher: !chatOpen && !(graphVisible && queryPathsOpen),
    showMobileChatLauncher: !chatOpen && graphVisible && queryPathsOpen,
    queryToggleInSidebarHeader: graphVisible && queryPathsOpen,
  };
}
