import { defineConfig } from "tsup";

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
});
