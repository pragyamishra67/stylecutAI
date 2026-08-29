# StyleCut AI

StyleCut AI is a frontend-only cinematic video editing workspace demo for short-form content creators.

## Run & Operate

- `pnpm --filter @workspace/api-server run dev` — run the API server (port 5000)
- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from the OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- Required env: `DATABASE_URL` — Postgres connection string

## Stack

- pnpm workspaces, Node.js 24, TypeScript 5.9
- API: Express 5
- DB: PostgreSQL + Drizzle ORM
- Validation: Zod (`zod/v4`), `drizzle-zod`
- API codegen: Orval (from OpenAPI spec)
- Build: esbuild (CJS bundle)

## Where things live

- `artifacts/stylecut-ai/src/App.tsx` — routed application shell, mock data, and interactive product flows
- `artifacts/stylecut-ai/src/index.css` — global theme, glass surfaces, gradients, motion, and responsive styling
- `artifacts/stylecut-ai/.replit-artifact/artifact.toml` — app artifact and managed web workflow configuration

## Architecture decisions

- The first release is deliberately frontend-only; all editing, upload, assistant, and processing states are simulated with local React state.
- The app uses a single routed shell so creator workflow state can remain visible while moving between dashboard, editor, library, projects, and settings.
- Mock content is fictional and designed for product-demo clarity rather than representing real uploads or external media.

## Product

Creators can review their editing workspace, upload mock raw footage, choose a visual style, customize the edit with an AI assistant, simulate processing, preview a before/after result, and browse projects, preferences, profile, and settings.

## User preferences

The requested product direction is premium, dark, cinematic, glass-forward, and suitable for a polished demo or investor presentation.

## Gotchas

- The API and database workspace packages are scaffolded but are intentionally not used by StyleCut AI.
- Use the managed `artifacts/stylecut-ai: web` workflow to run the app so the artifact base path and preview routing are injected correctly.

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
