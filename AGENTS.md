# Understory

Understory is an OKF knowledge-base service with an MCP memory API, an agent,
CLI tools, and a web UI. It is a Node.js 22 / TypeScript pnpm monorepo.

## Layout

- `packages/core` — knowledge-base access, retrieval, agents, providers, and tests.
- `packages/server` — MCP and HTTP APIs; serves the built web UI.
- `packages/web` — React UI and browser geometry tests.
- `sample-bundle` — public fixture for development and grounded-answer tests.

## Commands

- Install dependencies: `pnpm install --frozen-lockfile` (pnpm 10).
- Run checks: `pnpm test` and `pnpm build`.
- Start the local stack: configure `.env` from `.env.example`, then run
  `docker compose up -d --build`.
- See `README.md` for local model and CLI configuration.

## Git workflow

Work on `main`, commit changes there, and push to `origin/main` after passing
checks. Do not leave completed work only on a feature branch. If an agent runtime
requires an isolated worktree, integrate its commits into `main` and push before
finishing. Verify the working tree and remote branch before merging or pushing;
never fold unrelated dirty changes into a commit.

## Gotchas

- `.env` contains deployment settings and may be symlinked into a worktree.
  Never commit it or copy its contents into logs. Keep the live bundle outside
  the checkout; `sample-bundle` is only a fixture.
- Docker builds must exclude local dependency/build symlinks and `.env`; keep
  `.dockerignore` current.
- `RECALL_ENABLE_THINKING=false` disables reasoning only for fast recall, not
  the deep agent. Llama.cpp's `RECALL_TOKENIZER_URL` is opt-in and receives
  transient reasoning text for counting; traces store numeric counts, not text.
- A recall generation that hits its output cap must decline, not cache a
  truncated answer. Keep fallback and trace behavior covered by tests.
