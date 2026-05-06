import { defineConfig } from "tsup";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

// Inject package.json version at build time so the runtime VERSION/CDN URL
// can never drift from the published npm version.
const here = dirname(fileURLToPath(import.meta.url));
const pkg = JSON.parse(readFileSync(join(here, "package.json"), "utf8"));

export default defineConfig({
  entry: ["src/index.ts"],
  format: ["esm", "cjs"],
  dts: true,
  sourcemap: true,
  clean: false, // Don't clean dist/wasm/ built by Emscripten
  splitting: false,
  treeshake: true,
  noExternal: [],
  // Don't bundle the Emscripten WASM glue — loaded at runtime
  external: [/dist\/wasm/],
  define: {
    __OMNIVAD_VERSION__: JSON.stringify(pkg.version),
  },
});
