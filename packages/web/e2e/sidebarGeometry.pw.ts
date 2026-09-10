import { expect, test, type Locator, type Page } from "@playwright/test";

const VIEWPORTS = [
  { width: 390, height: 844 },
  { width: 1024, height: 800 },
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

for (const viewport of VIEWPORTS) {
  test(`sidebar geometry at ${viewport.width}px`, async ({ page }) => {
    await page.setViewportSize(viewport);
    await mockApi(page);
    await page.goto("/");
    await page.getByRole("button", { name: "Graph", exact: true }).click();

    const rail = page.getByTestId("mobile-control-rail");
    const navHeader = page.getByTestId("navigation-header");
    const legend = page.getByTestId("graph-legend");
    const canvas = page.getByTestId("graph-canvas");
    const chatLauncher = page.getByRole("button", {
      name: "Expand chat sidebar",
    });
    const queryLauncher = page.getByRole("button", {
      name: "Expand query paths sidebar",
    });
    await expect(legend.getByText("Long concept type")).toBeVisible();
    const initialCanvasWidth = (await rect(canvas)).width;

    await expect(chatLauncher).toHaveCount(1);
    await expect(queryLauncher).toHaveCount(1);
    await expectNoOverlap(chatLauncher, queryLauncher);
    await expectNoOverlap(legend, chatLauncher);
    await expectNoOverlap(legend, queryLauncher);

    if (viewport.width === 390) {
      await expectContained(chatLauncher, rail);
      await expectContained(queryLauncher, rail);
      await expectNoOverlap(rail, navHeader);
      await expectNoOverlap(rail, legend);
    } else {
      await expect(rail).toBeHidden();
    }

    await queryLauncher.click();
    const querySidebar = page.locator("#query-paths-sidebar");
    const queryCollapse = page.getByRole("button", {
      name: "Collapse query paths sidebar",
    });
    const firstQueryRow = page.getByTestId("query-path-row").first();
    const chatRoute =
      viewport.width === 390
        ? chatLauncher
        : page.getByRole("button", { name: "Chat", exact: true });
    await expect(querySidebar).toBeVisible();
    await expect(queryCollapse).toBeFocused();
    await expectNoOverlap(queryCollapse, firstQueryRow);
    await expectNoOverlap(chatRoute, firstQueryRow);

    if (viewport.width === 390) {
      await expect(legend).toBeHidden();
      await expectNoOverlap(rail, querySidebar);
      await expectContained(chatLauncher, rail);
    } else {
      expect((await rect(canvas)).width).toBeLessThan(initialCanvasWidth);
    }

    await chatRoute.click();
    const chatSidebar = page.locator("#chat-sidebar");
    const chatCollapse = page.getByRole("button", {
      name: "Collapse chat sidebar",
    });
    await expect(chatSidebar).toBeVisible();
    await expect(chatCollapse).toBeFocused();
    if (viewport.width > 390) {
      await expectNoOverlap(querySidebar, chatSidebar);
      expect((await rect(canvas)).width).toBeLessThan(initialCanvasWidth);
    }

    await chatCollapse.click();
    await expect(chatRoute).toBeFocused();
    await queryCollapse.click();
    await expect(queryLauncher).toBeFocused();
    await expect(legend).toBeVisible();
    expect((await rect(canvas)).width).toBe(initialCanvasWidth);

    await chatLauncher.click();
    await expect(chatSidebar).toBeVisible();
    if (viewport.width === 390) {
      expect((await rect(chatSidebar)).width).toBe(viewport.width);
    } else {
      expect((await rect(canvas)).width).toBeLessThan(initialCanvasWidth);
    }

    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(
      viewport.width
    );
  });
}
