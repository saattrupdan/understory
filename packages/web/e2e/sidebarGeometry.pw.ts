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

async function expectWidth(locator: Locator, expected: number) {
  const box = await rect(locator);
  expect(Math.abs(box.width - expected)).toBeLessThanOrEqual(1);
}

for (const viewport of VIEWPORTS) {
  test(`app shell geometry at ${viewport.width}x${viewport.height}`, async ({ page }) => {
    await page.setViewportSize(viewport);
    await mockApi(page);
    await page.goto("/");

    const memorySidebar = page.locator("#memory-sidebar");
    const chatSidebar = page.locator("#chat-sidebar");
    const canvas = page.getByTestId("graph-canvas");
    const memoryLauncher = page.getByRole("button", {
      name: "Expand memory sidebar",
    });
    const chatLauncher = page.getByRole("button", {
      name: "Expand chat sidebar",
    });
    const queryLauncher = page.getByRole("button", {
      name: "Expand query paths sidebar",
    });

    await expect(page.getByTestId("graph-node")).toHaveCount(3);
    await expect(page.getByTestId("graph-legend")).toBeVisible();
    await expect(memorySidebar).toBeHidden();
    await expect(chatSidebar).toBeHidden();
    await expect(memoryLauncher).toHaveAttribute("aria-expanded", "false");
    await expect(chatLauncher).toHaveAttribute("aria-expanded", "false");
    await expectHitTarget(memoryLauncher);
    await expectHitTarget(chatLauncher);
    await expectHitTarget(queryLauncher);
    await expectNoOverlap(memoryLauncher, queryLauncher);
    await expectNoOverlap(chatLauncher, queryLauncher);
    await expectNoOverlap(memoryLauncher, page.getByTestId("graph-legend"));
    await expectNoOverlap(chatLauncher, page.getByTestId("graph-legend"));

    const initialWidth = (await rect(canvas)).width;
    const desktop = viewport.width >= 1024;
    const memoryWidth = desktop ? 288 : initialWidth;
    const chatWidth = desktop ? Math.min(384, viewport.width * 0.35) : initialWidth;

    await memoryLauncher.click();
    const memoryCollapse = page.getByRole("button", {
      name: "Collapse memory sidebar",
    });
    await expect(memorySidebar).toBeVisible();
    await expect(memorySidebar).toHaveAttribute("aria-hidden", "false");
    await expect(memoryCollapse).toBeFocused();
    await expect(page.getByTestId("navigation-header")).toBeVisible();
    await expect(page.getByPlaceholder("Search…")).toBeVisible();
    await expect(page.getByRole("button", { name: "Log", exact: true })).toBeVisible();
    await expect(page.getByRole("button", { name: "Graph", exact: true })).toBeVisible();
    if (desktop) {
      await expectWidth(canvas, initialWidth - memoryWidth);
    } else {
      await expectWidth(canvas, initialWidth);
    }
    await expectHitTarget(memoryCollapse);
    await expectHitTarget(chatLauncher);

    await memoryCollapse.click();
    await expect(memorySidebar).toBeHidden();
    await expect(memoryLauncher).toBeFocused();
    await expectWidth(canvas, initialWidth);

    await chatLauncher.click();
    const chatCollapse = page.getByRole("button", {
      name: "Collapse chat sidebar",
    });
    await expect(chatSidebar).toBeVisible();
    await expect(chatSidebar).toHaveAttribute("aria-hidden", "false");
    await expect(chatCollapse).toBeFocused();
    await expect(chatSidebar.getByRole("heading", { name: "Agent chat" })).toBeVisible();
    await expect(chatSidebar.locator("textarea")).toBeVisible();
    if (desktop) {
      await expectWidth(canvas, initialWidth - chatWidth);
    } else {
      await expectWidth(canvas, initialWidth);
    }
    await expectHitTarget(chatCollapse);
    await expectHitTarget(memoryLauncher);

    await chatCollapse.click();
    await expect(chatSidebar).toBeHidden();
    await expect(chatLauncher).toBeFocused();
    await expectWidth(canvas, initialWidth);

    await queryLauncher.click();
    const querySidebar = page.locator("#query-paths-sidebar");
    const queryCollapse = querySidebar.getByRole("button", {
      name: "Collapse query paths sidebar",
    });
    await expect(querySidebar).toBeVisible();
    await expect(queryCollapse).toBeFocused();
    await expectNoOverlap(queryCollapse, memoryLauncher);
    await expectNoOverlap(queryCollapse, chatLauncher);
    await expectNoOverlap(querySidebar, chatLauncher);
    await expectHitTarget(queryCollapse);
    if (desktop) {
      expect((await rect(canvas)).width).toBeLessThan(initialWidth);
    } else {
      await expectWidth(canvas, initialWidth);
    }

    await queryCollapse.click();
    await expect(querySidebar).toBeHidden();
    await expect(queryLauncher).toBeFocused();
    await expectWidth(canvas, initialWidth);
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(
      viewport.width
    );
  });
}
