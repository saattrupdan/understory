/** Return whether an error represents cancellation of an operation. */
export function isAbortError(error: unknown, signal?: AbortSignal): boolean {
  if (signal?.aborted) return true;
  if (error === null || typeof error !== "object") return false;
  const name = (error as { name?: unknown }).name;
  return name === "AbortError" || name === "TimeoutError";
}

/** Throw the signal's reason when the operation has been cancelled. */
export function throwIfAborted(signal?: AbortSignal): void {
  if (!signal?.aborted) return;
  throw signal.reason ?? new DOMException("The operation was aborted", "AbortError");
}
