import { describe, expect, it } from "vitest";
import { recenterGraphAfterResize } from "../src/components/graphViewport";

describe("recenterGraphAfterResize", () => {
  it("translates the camera by half the viewport size change", () => {
    expect(
      recenterGraphAfterResize(
        { x: 20, y: -10, k: 1.5 },
        { width: 900, height: 600 },
        { width: 600, height: 700 }
      )
    ).toEqual({ x: -130, y: 40, k: 1.5 });
  });

  it("preserves the relative positions of rendered nodes", () => {
    const view = { x: 20, y: -10, k: 2 };
    const next = recenterGraphAfterResize(
      view,
      { width: 900, height: 600 },
      { width: 600, height: 700 }
    );
    const nodes = [
      { x: 100, y: 200 },
      { x: 350, y: 125 },
    ];
    const before = nodes.map((node) => ({
      x: view.x + node.x * view.k,
      y: view.y + node.y * view.k,
    }));
    const after = nodes.map((node) => ({
      x: next.x + node.x * next.k,
      y: next.y + node.y * next.k,
    }));

    expect(after[1].x - after[0].x).toBe(before[1].x - before[0].x);
    expect(after[1].y - after[0].y).toBe(before[1].y - before[0].y);
  });
});
