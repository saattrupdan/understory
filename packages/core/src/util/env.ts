/** Parse a strictly positive, safe integer, or return the fallback. */
export function positiveIntegerEnv(value: string | undefined, fallback: number): number {
  const trimmed = value?.trim();
  if (!trimmed || !/^[1-9]\d*$/.test(trimmed)) return fallback;
  const parsed = Number(trimmed);
  return Number.isSafeInteger(parsed) ? parsed : fallback;
}

/**
 * Parse an env var that is an output-token cap.
 *
 * Zero is intentionally invalid: requesting zero completion tokens still
 * performs a doomed generation and adds latency while returning no answer.
 */
export function capEnv(value: string | undefined, fallback: number): number {
  return positiveIntegerEnv(value, fallback);
}
