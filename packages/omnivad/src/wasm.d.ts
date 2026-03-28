// Type declarations for Emscripten-generated WASM modules
declare module "../dist/wasm/omnivad.js" {
  const createOmniVAD: (opts?: Record<string, unknown>) => Promise<unknown>;
  export default createOmniVAD;
}

declare module "../dist/wasm/omnivad.cjs" {
  const createOmniVAD: (opts?: Record<string, unknown>) => Promise<unknown>;
  export = createOmniVAD;
}
