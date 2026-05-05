#!/usr/bin/env node
/**
 * Swap `packages/omnivad/models` between two states around `npm publish`.
 *
 *   real     — replace the dev-time symlink (-> ../../models) with a real
 *              copy of the model files. `npm pack` cannot follow symlinks
 *              for entries listed in package.json's `files` array, so we
 *              materialise the directory before publishing.
 *   symlink  — restore the symlink pointer so the dev tree behaves as
 *              before (single source of truth at OmniVAD-Kit/models/).
 *
 * Wired into `prepublishOnly` (real) and `postpublish` (symlink) — see
 * package.json. No-op when the directory is already in the requested
 * state, so re-running by hand is safe.
 *
 * Usage:
 *   node scripts/swap-models.mjs real
 *   node scripts/swap-models.mjs symlink
 */
import {
  cpSync,
  existsSync,
  lstatSync,
  rmSync,
  symlinkSync,
  readdirSync,
} from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const PKG_ROOT = resolve(__dirname, "..");
const MODELS_DIR = join(PKG_ROOT, "models");
// Path the dev-time symlink targets, *relative to PKG_ROOT*. Keeping it
// relative makes the symlink portable when the worktree is moved.
const SYMLINK_TARGET = join("..", "..", "models");
// Resolved absolute path used when copying real files for publish.
const REAL_SOURCE = resolve(PKG_ROOT, SYMLINK_TARGET);

const mode = process.argv[2];
if (mode !== "real" && mode !== "symlink") {
  console.error(
    "[swap-models] usage: swap-models.mjs <real|symlink>",
  );
  process.exit(2);
}

function isSymlink(p) {
  try {
    return lstatSync(p).isSymbolicLink();
  } catch {
    return false;
  }
}

function isRealDir(p) {
  try {
    const s = lstatSync(p);
    return s.isDirectory() && !s.isSymbolicLink();
  } catch {
    return false;
  }
}

if (mode === "real") {
  if (isRealDir(MODELS_DIR)) {
    console.log("[swap-models] already real — no-op");
    process.exit(0);
  }
  if (!existsSync(REAL_SOURCE)) {
    console.error(`[swap-models] source not found: ${REAL_SOURCE}`);
    process.exit(1);
  }
  if (isSymlink(MODELS_DIR)) rmSync(MODELS_DIR);
  cpSync(REAL_SOURCE, MODELS_DIR, { recursive: true, dereference: true });
  const count = readdirSync(MODELS_DIR).length;
  console.log(
    `[swap-models] materialised ${count} entries from ${REAL_SOURCE} → ${MODELS_DIR}`,
  );
}

if (mode === "symlink") {
  if (isSymlink(MODELS_DIR)) {
    console.log("[swap-models] already symlink — no-op");
    process.exit(0);
  }
  if (isRealDir(MODELS_DIR)) {
    rmSync(MODELS_DIR, { recursive: true, force: true });
  }
  symlinkSync(SYMLINK_TARGET, MODELS_DIR, "dir");
  console.log(`[swap-models] restored symlink ${MODELS_DIR} → ${SYMLINK_TARGET}`);
}
