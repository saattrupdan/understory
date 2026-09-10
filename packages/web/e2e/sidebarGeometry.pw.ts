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

async function graphNodePaths(page: Page): Promise<string[]> {
  return page.getByTestId("graph-node").evaluateAll((nodes) =>
    nodes.map((node) => node.getAttribute("data-node-path") ?? "")
  );
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
    const initialNodePaths = await graphNodePaths(page);
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

    if (mobile) {
      await expect(legend).toBeHidden();
      await expectNoOverlap(rail, querySidebar);
      await expectContained(mobileChatToggle, rail);
      await expect(mobileQueryToggle).toHaveAttribute("aria-expanded", "true");
      await expectHitTarget(mobileQueryToggle);
    } else {
      expect((await rect(canvas)).width).toBeLessThan(initialCanvasWidth);
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

    if (mobile) {
      // Both panels stay mounted in the both-open state, but the rail is above
      // the full-screen chat overlay and is the unambiguous panel switch.
      await expect(mobileQueryToggle).toHaveAttribute("aria-expanded", "true");
      await expect(mobileChatToggle).toHaveAttribute("aria-expanded", "true");
      await expectHitTarget(mobileQueryToggle);
      await expectHitTarget(mobileChatToggle);
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
      await mobileQueryToggle.click();
      await expect(querySidebar).toBeHidden();
      await expect(mobileQueryToggle).toHaveAttribute("aria-expanded", "false");
      await expect(mobileQueryToggle).toBeFocused();

      // The chat-only overlay remains switchable from the same rail.
      await mobileChatToggle.click();
      await expect(chatSidebar).toBeVisible();
      await expect(mobileChatToggle).toBeFocused();
      await expectHitTarget(mobileChatToggle);
      await mobileChatToggle.click();
      await expect(chatSidebar).toBeHidden();
      await expect(mobileChatToggle).toBeFocused();
    } else {
      await expectNoOverlap(querySidebar, chatSidebar);
      expect((await rect(canvas)).width).toBeLessThan(initialCanvasWidth);
      await expectHitTarget(chatCollapse);
      await chatCollapse.click();
      await expect(chatSidebar).toBeHidden();
      await expect(chatRoute).toBeFocused();
      await queryCollapse.click();
      await expect(querySidebar).toBeHidden();
      await expect(queryLauncher).toBeFocused();
      await expect(legend).toBeVisible();

      await chatLauncher.click();
      await expect(chatSidebar).toBeVisible();
      expect((await rect(canvas)).width).toBeLessThan(initialCanvasWidth);
      await chatSidebar.getByRole("button", { name: "Collapse chat sidebar" }).click();
      await expect(chatSidebar).toBeHidden();
    }

    expect(await graphNodePaths(page)).toEqual(initialNodePaths);
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(
      viewport.width
    );
  });
}
