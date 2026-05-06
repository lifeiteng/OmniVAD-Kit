// Module Web Worker that runs OmniStreamVAD off the main thread.
//
// Uses the high-level class via ES module import. This requires the
// loader fix in wasm-binding.ts (fetch+eval fallback when importScripts
// is unavailable) — older builds of dist/ would fail here with
//   "Module scripts don't support importScripts()".

import { OmniStreamVAD } from "../../packages/omnivad/dist/index.js";

self.postMessage({ type: "ready" });

self.onmessage = async (e) => {
  const m = e.data;
  if (m.type !== "run") return;

  try {
    const vad = await OmniStreamVAD.create({ modelUrl: m.modelUrl });
    const audio = new Float32Array(m.audio);

    // Warm-up
    vad.reset();
    vad.detectFull(audio);

    const times = [];
    let segs = 0;
    for (let i = 0; i < m.repeats; i++) {
      vad.reset();
      const t0 = performance.now();
      const r = vad.detectFull(audio);
      times.push(performance.now() - t0);
      segs = r.numFrames;
    }

    const sorted = [...times].sort((a, b) => a - b);
    const medianMs = sorted[Math.floor(times.length / 2)];

    vad.dispose();
    self.postMessage({ type: "result", times, medianMs, segs });
  } catch (err) {
    self.postMessage({
      type: "error",
      message: String(err && err.message ? err.message : err),
    });
  }
};
