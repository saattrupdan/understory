import { createHash } from "node:crypto";

/** Return the hexadecimal SHA-256 digest of a UTF-8 string. */
export function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}
