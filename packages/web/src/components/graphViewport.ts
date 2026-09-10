export interface GraphViewTransform {
  x: number;
  y: number;
  k: number;
}

export interface GraphViewportSize {
  width: number;
  height: number;
}

/** Keep the rendered graph centered without changing its settled layout. */
export function recenterGraphAfterResize(
  view: GraphViewTransform,
  previous: GraphViewportSize,
  next: GraphViewportSize
): GraphViewTransform {
  return {
    ...view,
    x: view.x + (next.width - previous.width) / 2,
    y: view.y + (next.height - previous.height) / 2,
  };
}
