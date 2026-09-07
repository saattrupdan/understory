/**
 * Parse an env var that is an output-token cap.
 *
 * A cap has to be strictly positive, which a plain "any non-negative integer"
 * parse is not: 0 parses fine and asks the endpoint for zero completion
 * tokens, so every query would burn one doomed generation and get nothing —
 * the layer switched off with extra latency on top. Same guard the provider env
 * cap uses in providers/index.ts (`Number.isFinite(cap) && cap > 0`), shared so
 * the fast-path layers cannot drift apart.
 */
export function capEnv(value: string | undefined, fallback: number): number {
  const n = value === undefined ? NaN : Number.parseInt(value, 10);
  return Number.isFinite(n) && n > 0 ? n : fallback;
}
