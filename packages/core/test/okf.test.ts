import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  Bundle,
  BundleError,
  KnowledgeBase,
  parseDoc,
  replaceSection,
  validateBundle,
  regenerateIndex,
  readLog,
  searchBundle,
  lintBundle,
} from "../src/okf/index.js";
import { sha256 } from "../src/util/hash.js";

let root: string;
let kb: KnowledgeBase;

beforeEach(async () => {
  root = await fs.mkdtemp(path.join(os.tmpdir(), "okf-test-"));
  kb = new KnowledgeBase(root);
});

afterEach(async () => {
  await fs.rm(root, { recursive: true, force: true });
});

describe("frontmatter round-trip", () => {
  it("writes and reads a concept preserving fields, stamping timestamp", async () => {
    const written = await kb.writeConcept(
      "/tables/customers.md",
      { type: "BigQuery Table", title: "Customers", description: "Core customer table", tags: ["crm"], custom_key: 42 },
      "# Schema\n\nid, name, email",
      "Added customers table concept."
    );
    expect(written.frontmatter.timestamp).toMatch(/^\d{4}-\d{2}-\d{2}T/);

    const read = await kb.readConcept("/tables/customers.md");
    expect(read.frontmatter.type).toBe("BigQuery Table");
    expect(read.frontmatter.custom_key).toBe(42);
    expect(read.body).toContain("# Schema");
  });

  it("rejects concepts without a type", async () => {
    await expect(
      kb.writeConcept("/x.md", { type: "" } as never, "body", "log")
    ).rejects.toMatchObject({ code: "INVALID_FRONTMATTER" });
  });

  it("rejects reserved filenames as concepts", async () => {
    await expect(
      kb.writeConcept("/index.md", { type: "T" }, "body", "log")
    ).rejects.toMatchObject({ code: "RESERVED_NAME" });
    await expect(
      kb.writeConcept("/sub/log.md", { type: "T" }, "body", "log")
    ).rejects.toMatchObject({ code: "RESERVED_NAME" });
  });

  it("is permissive reading unknown keys and types", () => {
    const { frontmatter } = parseDoc(`---\ntype: Alien Format\nweird: [1, 2]\n---\nbody`);
    expect(frontmatter.type).toBe("Alien Format");
    expect(frontmatter.weird).toEqual([1, 2]);
  });
});

describe("sandbox", () => {
  it("rejects .. escapes", () => {
    const bundle = new Bundle(root);
    expect(() => bundle.resolve("/../../etc/passwd")).toThrow(BundleError);
    expect(() => bundle.resolve("../outside.md")).toThrow(BundleError);
  });

  it("allows normal nested paths", () => {
    const bundle = new Bundle(root);
    expect(bundle.resolve("/a/b/c.md")).toBe(path.join(root, "a/b/c.md"));
  });
});

describe("index regeneration (spec §6)", () => {
  it("generates bullet lists with titles and descriptions, root gets okf_version", async () => {
    await kb.writeConcept(
      "/tables/customers.md",
      { type: "Table", title: "Customers", description: "Customer records" },
      "body",
      "add"
    );
    const rootIndex = await fs.readFile(path.join(root, "index.md"), "utf-8");
    expect(rootIndex).toContain('okf_version: "0.1"');
    expect(rootIndex).toContain("## Memory Segments");
    // Segment lines summarize contents: count, types, titles — not just "subdirectory".
    expect(rootIndex).toContain("* [tables](tables/) - 1 concept (Table): Customers");

    const dirIndex = await fs.readFile(path.join(root, "tables/index.md"), "utf-8");
    expect(dirIndex).toContain("* [Customers](customers.md) - Customer records");
    // index.md must not have frontmatter outside root
    expect(dirIndex.startsWith("---")).toBe(false);
  });

  it("regenerates the whole ancestor chain after nested writes", async () => {
    await kb.writeConcept("/a/b/deep.md", { type: "T", title: "Deep" }, "x", "add deep");
    for (const p of ["index.md", "a/index.md", "a/b/index.md"]) {
      await expect(fs.access(path.join(root, p))).resolves.toBeUndefined();
    }
  });
});

describe("log (spec §7)", () => {
  it("appends newest-first with action bullets under ISO date headings", async () => {
    await kb.writeConcept("/one.md", { type: "T", title: "One" }, "x", "Created one.");
    await kb.writeConcept("/one.md", { type: "T", title: "One" }, "y", "Updated one.");
    await kb.deleteConcept("/one.md", "Removed one.");

    const log = await fs.readFile(path.join(root, "log.md"), "utf-8");
    expect(log).toMatch(/^# Directory Update Log/);
    expect(log).toMatch(/## \d{4}-\d{2}-\d{2}/);

    const entries = await readLog(kb.bundle);
    expect(entries.map((e) => e.action)).toEqual(["Deletion", "Update", "Creation"]);
    expect(entries[0].summary).toBe("Removed one.");
  });
});

describe("patch", () => {
  it("merges frontmatter and replaces a named section only", async () => {
    await kb.writeConcept(
      "/doc.md",
      { type: "T", title: "Doc", tags: ["a"] },
      "intro text\n\n# Schema\n\nold schema\n\n# Examples\n\nkeep me",
      "add"
    );
    const patched = await kb.patchConcept(
      "/doc.md",
      { frontmatter: { tags: ["a", "b"] }, replaceSection: { heading: "Schema", content: "new schema" } },
      "Updated schema section."
    );
    expect(patched.frontmatter.tags).toEqual(["a", "b"]);
    expect(patched.body).toContain("new schema");
    expect(patched.body).not.toContain("old schema");
    expect(patched.body).toContain("keep me");
    expect(patched.body).toContain("intro text");
  });

  it("appends the section when the heading is absent", () => {
    const out = replaceSection("just a body", "Citations", "[1] [X](https://x.com)");
    expect(out).toContain("# Citations");
    expect(out).toContain("[1] [X](https://x.com)");
    expect(out).toContain("just a body");
  });
});

describe("search", () => {
  beforeEach(async () => {
    await kb.writeConcept(
      "/tables/customers.md",
      { type: "Table", title: "Customers", description: "CRM customer records", tags: ["crm"] },
      "Contains emails and billing country.",
      "add"
    );
    await kb.writeConcept(
      "/apis/billing.md",
      { type: "API Endpoint", title: "Billing API", tags: ["billing"] },
      "Charges customers monthly.",
      "add"
    );
  });

  it("ranks title matches above body matches", async () => {
    const hits = await searchBundle(kb.bundle, "customers");
    expect(hits[0].path).toBe("/tables/customers.md");
    expect(hits.length).toBe(2); // body match on billing too
  });

  it("filters by type and tags", async () => {
    const byType = await searchBundle(kb.bundle, "customers", { type: "API Endpoint" });
    expect(byType.map((h) => h.path)).toEqual(["/apis/billing.md"]);
    const byTag = await searchBundle(kb.bundle, "", { tags: ["crm"] });
    expect(byTag.map((h) => h.path)).toEqual(["/tables/customers.md"]);
  });

  it("ranks compound installation and icon questions above generic distractors", async () => {
    await kb.writeConcept(
      "/gotchas/ptr-ms-analysis-pipx-installation.md",
      { type: "Gotcha", title: "PTR-MS/Sniff pipx installation" },
      "Use the branch/install command for ptr-ms/sniff. The packaging/make_icons.py tool is unrelated.",
      "add"
    );
    await kb.writeConcept(
      "/decisions/ptr-ms-analysis-work-on-main.md",
      { type: "Gotcha", title: "PTR-MS analysis work on main" },
      "Sniff work on main is documented here, including the installation branch workflow.",
      "add"
    );
    await kb.writeConcept(
      "/sources/vm-isolation.md",
      { type: "Source", title: "VM isolation installation notes" },
      "General installation notes for the VM runner and its source files.",
      "add"
    );
    await kb.writeConcept(
      "/notes/ptr-ms-analysis-icons.md",
      { type: "Note", title: "PTR-MS analysis icon and rename artwork — packaging/make_icons.py" },
      "The icon and rename artwork concept documents the logo assets and make_icons.py packaging helper.",
      "add"
    );
    await kb.writeConcept(
      "/sources/logo-overview.md",
      { type: "Source", title: "Logo overview" },
      "A general overview of installation documentation and packaging.",
      "add"
    );

    const installation = await searchBundle(
      kb.bundle,
      "Where is the ptr-ms/sniff installation documentation, and what command should I use?"
    );
    expect(installation[0].path).toBe("/gotchas/ptr-ms-analysis-pipx-installation.md");
    expect(installation[0].confidence).toBeGreaterThanOrEqual(20);
    expect(installation.slice(0, 6).map((hit) => hit.path)).toContain(
      "/decisions/ptr-ms-analysis-work-on-main.md"
    );

    const icons = await searchBundle(
      kb.bundle,
      "Where is the icon rename artwork for the project logo?"
    );
    expect(icons[0].path).toBe("/notes/ptr-ms-analysis-icons.md");
    expect(icons[0].confidence).toBeGreaterThanOrEqual(20);
  });

  it("bounds generic accumulation for a production installation question", async () => {
    await kb.writeConcept(
      "/gotchas/ptr-ms-analysis-pipx-installation.md",
      { type: "Gotcha", title: "PTR-MS Analysis — pipx Stale Launcher Gotcha" },
      "The pipx 0.1.0 launcher is stale and still imports the obsolete flat analyze module. Install the editable local checkout so Dan can open Sniff and test changes.",
      "add"
    );
    await kb.writeConcept(
      "/decisions/ptr-ms-analysis-work-on-main.md",
      { type: "Gotcha", title: "PTR-MS Analysis — Work Directly on Main" },
      "Work directly on main when testing Sniff changes; this repository uses no long-lived feature branches. The branch and install conventions are documented here.",
      "add"
    );
    await kb.writeConcept(
      "/notes/jottacloud-desktop.md",
      { type: "Note", title: "Jottacloud desktop application" },
      "The current desktop application opens local files and tests changes from a repository. Installation and branch conventions are general operational notes.",
      "add"
    );
    await kb.writeConcept(
      "/notes/dotfiles-install.md",
      { type: "Note", title: "Dotfiles installation conventions" },
      "This documents installing a local application, opening it for testing, and keeping changes on the current branch.",
      "add"
    );
    await kb.writeConcept(
      "/notes/local-desktop-testing.md",
      { type: "Note", title: "Local desktop testing workflow" },
      "A desktop application can be installed locally and opened to test current changes from a repository.",
      "add"
    );

    const hits = await searchBundle(
      kb.bundle,
      "How is the current Sniff desktop application installed locally from the ptr-ms/sniff repository so Dan can test changes by opening Sniff, and what branch/install conventions have been used?",
      { limit: 10 }
    );
    const topThree = hits.slice(0, 3).map((hit) => hit.path);
    expect(topThree).toContain("/gotchas/ptr-ms-analysis-pipx-installation.md");
    expect(topThree).toContain("/decisions/ptr-ms-analysis-work-on-main.md");
  });

  it("canonicalises morphological query variants as one evidence group", async () => {
    await kb.writeConcept(
      "/gotchas/install.md",
      { type: "Gotcha", title: "Install the local application" },
      "The editable checkout is installable from the local repository.",
      "add"
    );
    await kb.writeConcept(
      "/notes/unrelated.md",
      { type: "Note", title: "Unrelated note" },
      "This document has no installation instructions.",
      "add"
    );

    const hits = await searchBundle(kb.bundle, "installed installing installation");
    expect(hits[0].path).toBe("/gotchas/install.md");
    expect(hits[0].matchedGroups).toBe(1);
    expect(hits[0].confidence).toBeLessThan(20);
    expect(hits[0].confidenceQualified).toBe(false);

    const stemmed = await searchBundle(kb.bundle, "installation");
    expect(stemmed[0].path).toBe("/gotchas/install.md");
  });

  it("decomposes punctuation without double-weighting duplicate terms", async () => {
    await kb.writeConcept(
      "/docs/branch-install.md",
      { type: "Guide", title: "Branch install" },
      "The branch/install guide covers packaging/make_icons.py.",
      "add"
    );
    const once = await searchBundle(kb.bundle, "branch/install packaging/make_icons.py", { limit: 1 });
    const repeated = await searchBundle(
      kb.bundle,
      "branch/install branch/install packaging/make_icons.py packaging/make_icons.py",
      { limit: 1 }
    );
    expect(once[0].path).toBe("/docs/branch-install.md");
    expect(repeated[0].path).toBe(once[0].path);
    expect(repeated[0].score).toBe(once[0].score);

    // Existing broad substring matching remains available for ordinary prose.
    const broad = await searchBundle(kb.bundle, "customer");
    expect(broad.map((hit) => hit.path)).toContain("/tables/customers.md");
  });

  it("normalises and searches Unicode text without entering browse mode", async () => {
    await kb.writeConcept(
      "/steder/soeen.md",
      { type: "Sted", title: "Sø og café" },
      "Crème brûlée ved søen.",
      "add"
    );
    await kb.writeConcept(
      "/sprog/kyrillisk.md",
      { type: "Note", title: "Память проекта" },
      "Сведения хранятся здесь.",
      "add"
    );
    await kb.writeConcept(
      "/sprog/cjk.md",
      { type: "Note", title: "记忆系统" },
      "这里保存知识。",
      "add"
    );

    expect((await searchBundle(kb.bundle, "sø")).map((hit) => hit.path)).toEqual([
      "/steder/soeen.md",
    ]);
    expect((await searchBundle(kb.bundle, "cafe\u0301"))[0].path).toBe("/steder/soeen.md");
    expect((await searchBundle(kb.bundle, "память"))[0].path).toBe("/sprog/kyrillisk.md");
    expect((await searchBundle(kb.bundle, "记"))[0].path).toBe("/sprog/cjk.md");

    const browse = await searchBundle(kb.bundle, "  ... -- /  ");
    expect(browse.length).toBe(5);
    expect(browse.every((hit) => hit.score === 1 && hit.confidence === 0)).toBe(true);
  });

  it("searches symbol-only input instead of treating it as browse", async () => {
    await kb.writeConcept(
      "/symbols/gear.md",
      { type: "Note", title: "⚙️ workflow marker" },
      "The gear symbol marks an operational workflow.",
      "add"
    );

    const hits = await searchBundle(kb.bundle, "⚙️");
    expect(hits[0].path).toBe("/symbols/gear.md");
    expect(hits[0].score).toBeGreaterThan(1);
  });

  it("keeps absent filenames and common path-like queries below recall confidence", async () => {
    for (const directory of ["repositories", "gotchas", "notes"]) {
      for (const name of ["alpha", "beta", "gamma", "delta"]) {
        await kb.writeConcept(
          `/${directory}/${name}.md`,
          { type: "Note", title: `Unrelated ${name}` },
          "No matching filename is documented here.",
          "add"
        );
      }
    }

    const absent = await searchBundle(kb.bundle, "Where is completely-absent.md?");
    expect(absent.length).toBeGreaterThan(0); // `md` remains useful for ranking.
    expect(absent[0].confidence).toBeLessThan(20);

    for (const query of ["repositories/md", "gotchas/md", "notes md"]) {
      const common = await searchBundle(kb.bundle, query);
      expect(common.length).toBeGreaterThan(0);
      expect(common[0].confidence).toBeLessThan(20);
    }
  });

  it("uses corroborated groups against a large distractor corpus", async () => {
    await kb.writeConcept(
      "/gotchas/ptr-ms-analysis-pipx-installation.md",
      { type: "Gotcha", title: "PTR-MS Analysis pipx stale launcher" },
      "The installed pipx launcher still imports the obsolete flat analyze module. Reinstalling the checkout fixes the launcher.",
      "add"
    );
    await kb.writeConcept(
      "/notes/euroeval-visual-identity.md",
      { type: "Note", title: "EuroEval visual identity" },
      "The official EuroEval logo artwork is gfx/euroeval.png.",
      "add"
    );

    await Promise.all(
      Array.from({ length: 90 }, async (_, index) => {
        const directory = ["repositories", "gotchas", "notes"][index % 3];
        const filename = `${directory}/archive-entry-${index}.md`;
        const absolute = path.join(root, filename);
        await fs.mkdir(path.dirname(absolute), { recursive: true });
        await fs.writeFile(
          absolute,
          `---\ntype: Note\ntitle: Archive entry ${index}\n---\nRoutine documentation and project records for archive entry ${index}.\n`
        );
      })
    );

    const installation = await searchBundle(
      kb.bundle,
      "What is the PTR-MS pipx installation fix?"
    );
    expect(installation.slice(0, 3).map((hit) => hit.path)).toContain(
      "/gotchas/ptr-ms-analysis-pipx-installation.md"
    );
    expect(installation[0].confidence).toBeGreaterThanOrEqual(20);
    expect(installation[0].confidenceQualified).toBe(true);

    const logo = await searchBundle(kb.bundle, "Where is the EuroEval logo artwork?");
    expect(logo.slice(0, 3).map((hit) => hit.path)).toContain(
      "/notes/euroeval-visual-identity.md"
    );
    expect(logo[0].confidence).toBeGreaterThanOrEqual(20);
    expect(logo[0].confidenceQualified).toBe(true);

    const absent = await searchBundle(kb.bundle, "completely-absent.md");
    expect(absent[0].confidenceQualified).toBe(false);
    expect(absent[0].confidence).toBeLessThan(20);

    for (const query of ["repositories/md", "gotchas/md", "notes md"]) {
      const lowCoverage = await searchBundle(kb.bundle, query);
      expect(lowCoverage.length).toBeGreaterThan(0);
      expect(lowCoverage[0].confidenceQualified).toBe(false);
      expect(lowCoverage[0].confidence).toBeLessThan(20);
    }

    const broad = await searchBundle(kb.bundle, "documentation");
    expect(broad.length).toBeGreaterThan(0);
    expect(broad[0].confidence).toBeLessThan(20);

    const once = await searchBundle(kb.bundle, "PTR-MS pipx installation", { limit: 1 });
    const repeated = await searchBundle(kb.bundle, "PTR-MS pipx installation PTR-MS pipx installation", {
      limit: 1,
    });
    expect(repeated[0].path).toBe(once[0].path);
    expect(repeated[0].score).toBe(once[0].score);
  });
});

describe("conformance (spec §9)", () => {
  it("valid bundle passes; missing type is an error; broken link is only a warning", async () => {
    await kb.writeConcept(
      "/good.md",
      { type: "T", title: "Good", description: "fine" },
      "See [missing](/nope.md).",
      "add"
    );
    // Write a malformed concept behind the KB's back.
    await fs.writeFile(path.join(root, "bad.md"), `---\ntitle: No Type\n---\nbody\n`);

    const report = await validateBundle(kb.bundle);
    expect(report.conformant).toBe(false);
    expect(report.issues.some((i) => i.severity === "error" && i.path === "/bad.md")).toBe(true);
    const linkIssue = report.issues.find((i) => i.message.includes("/nope.md"));
    expect(linkIssue?.severity).toBe("warning");

    await fs.rm(path.join(root, "bad.md"));
    const clean = await validateBundle(kb.bundle);
    expect(clean.conformant).toBe(true);
  });
});

describe("lint (graph health)", () => {
  it("flags orphans (no inbound links) and treats index catalogs as non-sources", async () => {
    // hub is linked-to; spoke links out but nothing links back to it; lonely is isolated.
    await kb.writeConcept("/hub.md", { type: "T", title: "Hub" }, "The central concept.", "add");
    await kb.writeConcept("/spoke.md", { type: "T", title: "Spoke" }, "See [Hub](/hub.md).", "add");
    await kb.writeConcept("/lonely.md", { type: "T", title: "Lonely" }, "Nothing here links out.", "add");

    const report = await lintBundle(kb.bundle);
    const orphanPaths = report.orphans.map((o) => o.path).sort();
    // hub has an inbound link → not orphan. spoke + lonely have none → orphans.
    // The generated index.md files must NOT count as inbound links.
    expect(orphanPaths).toEqual(["/lonely.md", "/spoke.md"]);
    expect(report.linkCount).toBe(1);
    expect(report.healthy).toBe(false);
  });

  it("flags broken links and reports healthy when fully wired", async () => {
    await kb.writeConcept("/a.md", { type: "T", title: "A" }, "Link to [B](/b.md).", "add");
    let report = await lintBundle(kb.bundle);
    expect(report.brokenLinks).toEqual([{ path: "/a.md", target: "/b.md" }]);

    // Add B and a back-link so nothing is orphaned or broken.
    await kb.writeConcept("/b.md", { type: "T", title: "B" }, "Back to [A](/a.md).", "add");
    report = await lintBundle(kb.bundle);
    expect(report.brokenLinks).toEqual([]);
    expect(report.orphans).toEqual([]);
    expect(report.healthy).toBe(true);
  });
});

describe("graph export", () => {
  it("returns nodes with metadata/degree and deduped edges", async () => {
    await kb.writeConcept("/a.md", { type: "T", title: "A", description: "alpha" }, "See [B](/b.md) and again [B](/b.md).", "add");
    await kb.writeConcept("/b.md", { type: "U", title: "B" }, "Back to [A](/a.md).", "add");
    await kb.writeConcept("/c.md", { type: "T", title: "C" }, "island", "add");

    const graph = await kb.graph();
    expect(graph.nodes.length).toBe(3);
    const a = graph.nodes.find((n) => n.path === "/a.md")!;
    expect(a.title).toBe("A");
    expect(a.description).toBe("alpha");
    expect(a.links).toBe(2); // one out (deduped), one in
    expect(graph.nodes.find((n) => n.path === "/c.md")!.links).toBe(0);
    expect(graph.edges).toEqual(
      expect.arrayContaining([
        { source: "/a.md", target: "/b.md" },
        { source: "/b.md", target: "/a.md" },
      ])
    );
    expect(graph.edges.length).toBe(2); // duplicate A→B link counted once
  });
});

describe("mutation serialization", () => {
  it("serializes instances sharing one bundle root", async () => {
    const first = new KnowledgeBase(root);
    const second = new KnowledgeBase(path.join(root, "."));
    const results = await Promise.allSettled([
      first.createConcept("/same.md", { type: "T" }, "first", "first"),
      second.createConcept("/same.md", { type: "T" }, "second", "second"),
    ]);

    expect(results.filter((result) => result.status === "fulfilled")).toHaveLength(1);
    expect(results.filter((result) => result.status === "rejected")).toHaveLength(1);
    await expect(first.readConcept("/same.md")).resolves.toMatchObject({
      body: expect.stringMatching(/^(first|second)\n$/),
    });
  });

  it("checks a replace precondition inside the shared mutation queue", async () => {
    const first = new KnowledgeBase(root);
    const second = new KnowledgeBase(root);
    await first.createConcept("/stale.md", { type: "T" }, "original", "initial");
    const expected = sha256("original\n");

    const results = await Promise.allSettled([
      first.patchConcept("/stale.md", { replaceBody: "first" }, "first", expected),
      second.patchConcept("/stale.md", { replaceBody: "second" }, "second", expected),
    ]);

    expect(results.filter((result) => result.status === "fulfilled")).toHaveLength(1);
    expect(results.filter((result) => result.status === "rejected")).toHaveLength(1);
    await expect(first.readConcept("/stale.md")).resolves.toMatchObject({
      body: expect.stringMatching(/^(first|second)\n$/),
    });
  });

  it("concurrent writes all land and log all entries", async () => {
    await Promise.all(
      Array.from({ length: 8 }, (_, i) =>
        kb.writeConcept(`/c${i}.md`, { type: "T", title: `C${i}` }, "x", `Added C${i}.`)
      )
    );
    const entries = await readLog(kb.bundle);
    expect(entries.length).toBe(8);
    const tree = await kb.listTree();
    const concepts = tree.children!.filter((c) => c.kind === "concept");
    expect(concepts.length).toBe(8);
  });
});

describe("empty directory pruning (#10)", () => {
  it("prunes a directory left holding only its index.md after the last concept is deleted", async () => {
    await kb.writeConcept("/family/children/kid.md", { type: "Person", title: "Kid" }, "x", "add");
    await kb.writeConcept("/family/parent.md", { type: "Person", title: "Parent" }, "x", "add");
    await kb.deleteConcept("/family/children/kid.md", "moved elsewhere");

    // /family/children held only index.md → gone; /family still has parent.md → kept.
    await expect(fs.access(path.join(root, "family/children"))).rejects.toThrow();
    await expect(fs.access(path.join(root, "family/parent.md"))).resolves.toBeUndefined();
    // Parent index no longer lists the pruned subdirectory.
    const familyIndex = await fs.readFile(path.join(root, "family/index.md"), "utf-8");
    expect(familyIndex).not.toContain("children");
  });

  it("prunes emptied ancestor chains but never the root", async () => {
    await kb.writeConcept("/a/b/c/deep.md", { type: "T", title: "Deep" }, "x", "add");
    await kb.writeConcept("/other/keep.md", { type: "T", title: "Keep" }, "x", "add");
    await kb.deleteConcept("/a/b/c/deep.md", "gone");

    for (const dir of ["a/b/c", "a/b", "a"]) {
      await expect(fs.access(path.join(root, dir))).rejects.toThrow();
    }
    await expect(fs.access(root)).resolves.toBeUndefined();
    const rootIndex = await fs.readFile(path.join(root, "index.md"), "utf-8");
    expect(rootIndex).not.toContain("[a](a/)");
    expect(rootIndex).toContain("other");
  });

  it("heals pre-existing husks on the next unrelated mutation and spares dot-dirs", async () => {
    // Simulate an old husk + a .traces dir behind the KB's back.
    await fs.mkdir(path.join(root, "husk"), { recursive: true });
    await fs.writeFile(path.join(root, "husk/index.md"), "# Husk\n");
    await fs.mkdir(path.join(root, ".traces"), { recursive: true });
    await fs.writeFile(path.join(root, ".traces/t.json"), "{}");

    await kb.writeConcept("/fresh.md", { type: "T", title: "Fresh" }, "x", "add");

    await expect(fs.access(path.join(root, "husk"))).rejects.toThrow();
    await expect(fs.access(path.join(root, ".traces/t.json"))).resolves.toBeUndefined();
  });
});
