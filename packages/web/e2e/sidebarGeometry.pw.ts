import { expect, test, type Locator, type Page } from "@playwright/test";

const VIEWPORTS = [
  { width: 390, height: 844 },
  { width: 1024, height: 768 },
  { width: 1280, height: 800 },
] as const;

interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

const graph = {
  nodes: [
    { path: "/one.md", title: "One", type: "Long concept type", links: 1 },
    { path: "/two.md", title: "Two", type: "Another type", links: 1 },
    { path: "/orphan.md", title: "Orphan", type: "Orphan type", links: 0 },
  ],
  edges: [{ source: "/one.md", target: "/two.md" }],
};

const traces = [
  {
    id: "trace-1",
    kind: "query",
    input: "Where is the first concept?",
    startedAt: "2026-01-01T00:00:00Z",
    durationMs: 10,
    notation: "one -> two",
    stepCount: 2,
  },
  {
    id: "trace-2",
    kind: "chat",
    input: "Show another path",
    startedAt: "2026-01-01T00:00:01Z",
    durationMs: 12,
    notation: "two",
    stepCount: 1,
  },
];

async function mockApi(page: Page) {
  await page.route("**/api/**", async (route) => {
    const pathname = new URL(route.request().url()).pathname;
    const bodies: Record<string, unknown> = {
      "/api/tree": {
        name: "root",
        path: "/",
        kind: "directory",
        children: [],
      },
      "/api/validate": {
        conformant: true,
        conceptCount: 3,
        directoryCount: 1,
        issues: [],
      },
      "/api/log": [],
      "/api/config": {
        model: "test-model",
        format: "openai",
        fallbackConfigured: false,
      },
      "/api/graph": graph,
      "/api/traces": traces,
    };
    await route.fulfill({ json: bodies[pathname] ?? [] });
  });
}

async function rect(locator: Locator): Promise<Rect> {
  await expect(locator).toBeVisible();
  const box = await locator.boundingBox();
  expect(box).not.toBeNull();
  return box!;
}

function overlaps(a: Rect, b: Rect): boolean {
  return (
    a.x < b.x + b.width &&
    a.x + a.width > b.x &&
    a.y < b.y + b.height &&
    a.y + a.height > b.y
  );
}

async function expectNoOverlap(a: Locator, b: Locator) {
  expect(overlaps(await rect(a), await rect(b))).toBe(false);
}

async function expectContained(child: Locator, parent: Locator) {
  const childRect = await rect(child);
  const parentRect = await rect(parent);
  expect(childRect.x).toBeGreaterThanOrEqual(parentRect.x);
  expect(childRect.y).toBeGreaterThanOrEqual(parentRect.y);
  expect(childRect.x + childRect.width).toBeLessThanOrEqual(
    parentRect.x + parentRect.width
  );
  expect(childRect.y + childRect.height).toBeLessThanOrEqual(
    parentRect.y + parentRect.height
  );
}

async function expectHitTarget(locator: Locator) {
  const hit = await locator.evaluate((element) => {
    const box = element.getBoundingClientRect();
    const target = document.elementFromPoint(
      box.left + box.width / 2,
      box.top + box.height / 2
    );
    return target === element || element.contains(target);
  });
  expect(hit).toBe(true);
}

type GraphNodeGeometry = Record<string, { x: number; y: number }>;

async function graphNodeGeometry(page: Page): Promise<GraphNodeGeometry> {
  return page.getByTestId("graph-node").evaluateAll((nodes) =>
    Object.fromEntries(
      nodes.map((node) => {
        const box = node.getBoundingClientRect();
        return [
          node.getAttribute("data-node-path") ?? "",
          { x: box.left + box.width / 2, y: box.top + box.height / 2 },
        ];
      })
    )
  );
}

async function settledGraphNodeGeometry(page: Page): Promise<GraphNodeGeometry> {
  let previous = await graphNodeGeometry(page);
  let stableSamples = 0;

  for (let sample = 0; sample < 80; sample += 1) {
    await page.waitForTimeout(100);
    const current = await graphNodeGeometry(page);
    const stable = Object.keys(previous).every(
      (path) =>
        Math.abs(current[path].x - previous[path].x) <= 0.25 &&
        Math.abs(current[path].y - previous[path].y) <= 0.25
    );
    stableSamples = stable ? stableSamples + 1 : 0;
    if (stableSamples >= 3) return current;
    previous = current;
  }

  throw new Error("Graph nodes did not settle");
}

function expectRelativeGraphPositions(
  before: GraphNodeGeometry,
  after: GraphNodeGeometry,
  tolerance = 3
) {
  const paths = Object.keys(before).sort();
  expect(Object.keys(after).sort()).toEqual(paths);
  for (let i = 0; i < paths.length; i += 1) {
    for (let j = i + 1; j < paths.length; j += 1) {
      const first = paths[i];
      const second = paths[j];
      expect(
        Math.abs(
          after[first].x - after[second].x - (before[first].x - before[second].x)
        )
      ).toBeLessThanOrEqual(tolerance);
      expect(
        Math.abs(
          after[first].y - after[second].y - (before[first].y - before[second].y)
        )
      ).toBeLessThanOrEqual(tolerance);
    }
  }
}

async function expectCanvasWidth(canvas: Locator, expected: number) {
  expect(Math.abs((await rect(canvas)).width - expected)).toBeLessThanOrEqual(1);
}

for (const viewport of VIEWPORTS) {
  test(`sidebar geometry at ${viewport.width}x${viewport.height}`, async ({ page }) => {
    await page.setViewportSize(viewport);
    await mockApi(page);
    await page.goto("/");
    await page.getByRole("button", { name: "Graph", exact: true }).click();

    const mobile = viewport.width === 390;
    const rail = page.getByTestId("mobile-control-rail");
    const mobileQueryToggle = page.getByTestId("mobile-query-toggle");
    const mobileChatToggle = page.getByTestId("mobile-chat-toggle");
    const navHeader = page.getByTestId("navigation-header");
    const legend = page.getByTestId("graph-legend");
    const canvas = page.getByTestId("graph-canvas");
    const querySidebar = page.locator("#query-paths-sidebar");
    const chatSidebar = page.locator("#chat-sidebar");
    const queryLauncher = mobile
      ? mobileQueryToggle
      : page.getByRole("button", { name: "Expand query paths sidebar" });
    const chatLauncher = mobile
      ? mobileChatToggle
      : page.getByRole("button", { name: "Expand chat sidebar" });
    const chatRoute = mobile
      ? mobileChatToggle
      : page.getByTestId("navigation-chat-toggle");

    await expect(legend.getByText("Long concept type")).toBeVisible();
    await expect(page.getByTestId("graph-node")).toHaveCount(3);
    const initialNodeGeometry = await settledGraphNodeGeometry(page);
    const initialCanvasWidth = (await rect(canvas)).width;

    await expect(queryLauncher).toHaveAttribute("aria-expanded", "false");
    await expect(chatLauncher).toHaveAttribute("aria-expanded", "false");
    await expect(querySidebar).toBeHidden();
    await expect(chatSidebar).toBeHidden();
    await expectNoOverlap(chatLauncher, queryLauncher);
    await expectNoOverlap(legend, chatLauncher);
    await expectNoOverlap(legend, queryLauncher);

    if (mobile) {
      await expect(rail).toBeVisible();
      await expectContained(chatLauncher, rail);
      await expectContained(queryLauncher, rail);
      await expectNoOverlap(rail, navHeader);
      await expectNoOverlap(rail, legend);
      await expectHitTarget(queryLauncher);
      await expectHitTarget(chatLauncher);
    } else {
      await expect(rail).toBeHidden();
      await expectHitTarget(queryLauncher);
      await expectHitTarget(chatLauncher);
    }

    await queryLauncher.click();
    const queryHeading = querySidebar.getByRole("heading", { name: "Query paths" });
    const queryCollapse = querySidebar.getByRole("button", {
      name: "Collapse query paths sidebar",
    });
    const firstQueryRow = page.getByTestId("query-path-row").first();
    await expect(querySidebar).toBeVisible();
    await expect(querySidebar).toHaveAttribute("aria-hidden", "false");
    await expect(queryCollapse).toHaveAttribute("aria-expanded", "true");
    await expect(queryCollapse).toBeFocused();
    await expect(queryHeading).toBeVisible();
    await expect(firstQueryRow).toBeVisible();
    await expectContained(queryHeading, querySidebar);
    await expectContained(queryCollapse, querySidebar);
    await expectContained(firstQueryRow, querySidebar);
    await expectNoOverlap(queryCollapse, firstQueryRow);
    await expectNoOverlap(chatRoute, firstQueryRow);
    const queryOpenNodeGeometry = await settledGraphNodeGeometry(page);
    const queryOpenCanvasWidth = (await rect(canvas)).width;
    expectRelativeGraphPositions(initialNodeGeometry, queryOpenNodeGeometry);

    if (mobile) {
      await expect(legend).toBeHidden();
      await expectNoOverlap(rail, querySidebar);
      await expectContained(mobileChatToggle, rail);
      await expect(mobileQueryToggle).toHaveAttribute("aria-expanded", "true");
      await expectHitTarget(mobileQueryToggle);
      await expectCanvasWidth(canvas, initialCanvasWidth);
    } else {
      expect(queryOpenCanvasWidth).toBeLessThan(initialCanvasWidth);
      await expectHitTarget(queryCollapse);
    }

    await chatRoute.click();
    const chatHeading = chatSidebar.getByRole("heading", { name: "Agent chat" });
    const chatCollapse = chatSidebar.getByRole("button", {
      name: "Collapse chat sidebar",
    });
    const chatInput = chatSidebar.locator("textarea");
    await expect(chatSidebar).toBeVisible();
    await expect(chatSidebar).toHaveAttribute("aria-hidden", "false");
    if (mobile) {
      await expect(mobileChatToggle).toBeFocused();
    } else {
      await expect(chatCollapse).toBeFocused();
    }
    await expect(chatHeading).toBeVisible();
    await expect(chatCollapse).toHaveAttribute("aria-expanded", "true");
    await expect(chatInput).toBeVisible();
    await expectContained(chatHeading, chatSidebar);
    await expectContained(chatCollapse, chatSidebar);
    await expectContained(chatInput, chatSidebar);
    const bothOpenNodeGeometry = await settledGraphNodeGeometry(page);
    const bothOpenCanvasWidth = (await rect(canvas)).width;
    expectRelativeGraphPositions(queryOpenNodeGeometry, bothOpenNodeGeometry);

    if (mobile) {
      // Both panels stay mounted in the both-open state, but the rail is above
      // the full-screen chat overlay and is the unambiguous panel switch.
      await expect(mobileQueryToggle).toHaveAttribute("aria-expanded", "true");
      await expect(mobileChatToggle).toHaveAttribute("aria-expanded", "true");
      await expectNoOverlap(rail, chatHeading);
      await expectNoOverlap(rail, chatCollapse);
      await expectHitTarget(mobileQueryToggle);
      await expectHitTarget(mobileChatToggle);
      await expectHitTarget(chatCollapse);
      const queryIsOccluded = await firstQueryRow.evaluate((element) => {
        const box = element.getBoundingClientRect();
        const target = document.elementFromPoint(
          box.left + box.width / 2,
          box.top + box.height / 2
        );
        const chat = document.querySelector("#chat-sidebar");
        return Boolean(target && chat && (target === chat || chat.contains(target)));
      });
      expect(queryIsOccluded).toBe(true);

      // Switching from both-open never depends on the obscured query panel.
      await mobileChatToggle.click();
      await expect(chatSidebar).toBeHidden();
      await expect(mobileChatToggle).toBeFocused();
      await expect(querySidebar).toBeVisible();
      await expectHitTarget(firstQueryRow);
      const queryAfterChatCollapse = await settledGraphNodeGeometry(page);
      expectRelativeGraphPositions(bothOpenNodeGeometry, queryAfterChatCollapse);
      await expectCanvasWidth(canvas, queryOpenCanvasWidth);

      await mobileQueryToggle.click();
      await expect(querySidebar).toBeHidden();
      await expect(mobileQueryToggle).toHaveAttribute("aria-expanded", "false");
      await expect(mobileQueryToggle).toBeFocused();
      const closedAfterQueryCollapse = await settledGraphNodeGeometry(page);
      expectRelativeGraphPositions(queryAfterChatCollapse, closedAfterQueryCollapse);
      await expectCanvasWidth(canvas, initialCanvasWidth);

      // The chat-only overlay remains switchable from the same rail.
      await mobileChatToggle.click();
      await expect(chatSidebar).toBeVisible();
      await expect(mobileChatToggle).toBeFocused();
      await expectHitTarget(mobileChatToggle);
      await expectNoOverlap(rail, chatHeading);
      await expectNoOverlap(rail, chatCollapse);
      await expectHitTarget(chatCollapse);
      const chatOnlyNodeGeometry = await settledGraphNodeGeometry(page);
      expectRelativeGraphPositions(closedAfterQueryCollapse, chatOnlyNodeGeometry);
      await mobileChatToggle.click();
      await expect(chatSidebar).toBeHidden();
      await expect(mobileChatToggle).toBeFocused();
      const closedAfterChatCollapse = await settledGraphNodeGeometry(page);
      expectRelativeGraphPositions(chatOnlyNodeGeometry, closedAfterChatCollapse);
      await expectCanvasWidth(canvas, initialCanvasWidth);
    } else {
      await expectNoOverlap(querySidebar, chatSidebar);
      expect(bothOpenCanvasWidth).toBeLessThan(initialCanvasWidth);
      await expectHitTarget(chatCollapse);
      await chatCollapse.click();
      await expect(chatSidebar).toBeHidden();
      await expect(chatRoute).toBeFocused();
      const queryAfterChatCollapse = await settledGraphNodeGeometry(page);
      expectRelativeGraphPositions(bothOpenNodeGeometry, queryAfterChatCollapse);
      await expectCanvasWidth(canvas, queryOpenCanvasWidth);

      await queryCollapse.click();
      await expect(querySidebar).toBeHidden();
      await expect(queryLauncher).toBeFocused();
      await expect(legend).toBeVisible();
      const closedAfterQueryCollapse = await settledGraphNodeGeometry(page);
      expectRelativeGraphPositions(queryAfterChatCollapse, closedAfterQueryCollapse);
      await expectCanvasWidth(canvas, initialCanvasWidth);

      await chatLauncher.click();
      await expect(chatSidebar).toBeVisible();
      const chatOnlyNodeGeometry = await settledGraphNodeGeometry(page);
      expectRelativeGraphPositions(closedAfterQueryCollapse, chatOnlyNodeGeometry);
      expect((await rect(canvas)).width).toBeLessThan(initialCanvasWidth);
      await chatSidebar.getByRole("button", { name: "Collapse chat sidebar" }).click();
      await expect(chatSidebar).toBeHidden();
      const closedAfterChatCollapse = await settledGraphNodeGeometry(page);
      expectRelativeGraphPositions(chatOnlyNodeGeometry, closedAfterChatCollapse);
      await expectCanvasWidth(canvas, initialCanvasWidth);
    }

    expectRelativeGraphPositions(initialNodeGeometry, await graphNodeGeometry(page));
    await expectCanvasWidth(canvas, initialCanvasWidth);
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(
      viewport.width
    );
  });
}
