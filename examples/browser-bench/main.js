// Browser-side bench for OmniVAD streaming VAD.
// Compares main-thread vs Worker execution. Main-thread run also measures
// dropped animation frames so we know whether a 0.05–0.15 RTF actually
// stalls the UI.
//
// Serve from the repo root: `python3 -m http.server` then open
// http://localhost:8000/examples/browser-bench/index.html

import { OmniStreamVAD } from "../../packages/omnivad/dist/index.js";

const REPO_ROOT = "../..";
const MODEL_URL = `${REPO_ROOT}/models/stream-vad.omnivad`;
const AUDIO_URL = `${REPO_ROOT}/tests/data/zh_medium.wav`;
const REPEATS = 10;

// ---------- helpers ----------

const $ = (id) => document.getElementById(id);
const log = (msg) => {
  $("log").textContent += msg + "\n";
};
const setStatus = (s) => ($("status").textContent = s);

function median(xs) {
  const sorted = [...xs].sort((a, b) => a - b);
  return sorted[Math.floor(xs.length / 2)];
}

async function fetchPCM(url) {
  const buf = await (await fetch(url)).arrayBuffer();
  // Assume canonical 44-byte WAV header, 16-bit PCM mono @ 16 kHz
  const i16 = new Int16Array(buf, 44);
  const f32 = new Float32Array(i16.length);
  for (let i = 0; i < i16.length; i++) f32[i] = i16[i] / 32768;
  return f32;
}

// Stall monitor via the Long Tasks API. PerformanceObserver fires on every
// task that ran > 50 ms on the main thread — exactly the "UI is stuck"
// definition. Sum of durations is the total time the page was unresponsive
// during the monitoring window.
//
// Why not rAF: in background tabs (and in headless Chrome) rAF is heavily
// throttled, so its inter-frame gaps are no longer a reliable proxy for
// "main thread blocked". longtask is reported by the scheduler regardless
// of tab visibility.
function startFrameMonitor() {
  let stallMs = 0;
  let count = 0;
  let observer = null;
  const startedAt = performance.now();
  if (typeof PerformanceObserver === "function") {
    try {
      observer = new PerformanceObserver((list) => {
        for (const e of list.getEntries()) {
          stallMs += e.duration;
          count++;
        }
      });
      observer.observe({ entryTypes: ["longtask"] });
    } catch {
      observer = null;
    }
  }
  return () => {
    if (observer) observer.disconnect();
    return {
      stallMs: Math.round(stallMs),
      longTaskCount: count,
      wallMs: Math.round(performance.now() - startedAt),
    };
  };
}

// ---------- main thread bench ----------

async function benchMain(audio) {
  setStatus("Loading model + wasm in main thread…");
  const vad = await OmniStreamVAD.create({ modelUrl: MODEL_URL });
  const dur = audio.length / 16000;

  // Warm-up
  vad.reset();
  vad.detectFull(audio);

  const stop = startFrameMonitor();
  const ms = [];
  let segCount = 0;
  for (let i = 0; i < REPEATS; i++) {
    vad.reset();
    const t0 = performance.now();
    const r = vad.detectFull(audio);
    ms.push(performance.now() - t0);
    segCount = r.numFrames; // detectFull returns probs only; segments require per-frame loop
  }
  const frameStats = stop();

  vad.dispose();

  const med = median(ms);
  log(`[main] times(ms): ${ms.map((x) => x.toFixed(1)).join(", ")}`);
  // Fallback for envs that don't report longtask (e.g. headless Chrome):
  // synchronous main-thread compute IS the stall. ms.sum() is the floor.
  const computeSum = Math.round(ms.reduce((a, b) => a + b, 0));
  return {
    rtf: med / 1000 / dur,
    ms: med,
    stallMs: Math.max(frameStats.stallMs, computeSum),
    wallMs: frameStats.wallMs,
    segs: segCount,
  };
}

// ---------- worker bench ----------

function benchWorker(audio) {
  return new Promise((resolve, reject) => {
    setStatus("Spawning worker…");
    // Module worker — uses high-level OmniStreamVAD class. Requires the
    // wasm-binding loader's module-worker fallback (fetch+eval); see
    // packages/omnivad/src/wasm-binding.ts:loadScript().
    const w = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    const modelAbs = new URL(MODEL_URL, location.href).href;
    const stop = startFrameMonitor();
    w.onerror = (e) => {
      stop();
      w.terminate();
      reject(e.error || new Error(e.message || "worker error"));
    };
    w.onmessage = (e) => {
      const m = e.data;
      if (m.type === "ready") {
        // Transfer audio buffer to worker (zero-copy)
        const buf = audio.buffer.slice(0);
        w.postMessage(
          { type: "run", audio: buf, repeats: REPEATS, modelUrl: modelAbs },
          [buf],
        );
        return;
      }
      if (m.type === "result") {
        const frameStats = stop();
        w.terminate();
        log(`[worker] times(ms): ${m.times.map((x) => x.toFixed(1)).join(", ")}`);
        resolve({
          rtf: m.medianMs / 1000 / (audio.length / 16000),
          ms: m.medianMs,
          stallMs: frameStats.stallMs,
          wallMs: frameStats.wallMs,
          segs: m.segs,
        });
        return;
      }
      if (m.type === "error") {
        stop();
        w.terminate();
        reject(new Error(m.message));
      }
    };
  });
}

// ---------- UI plumbing ----------

function fillEnv() {
  $("env").textContent = [
    `userAgent: ${navigator.userAgent}`,
    `hardwareConcurrency: ${navigator.hardwareConcurrency}`,
    `deviceMemory: ${navigator.deviceMemory ?? "n/a"} GB`,
    `repeats: ${REPEATS}`,
  ].join("\n");
}

function paint(prefix, r) {
  $(`${prefix}-rtf`).textContent = r.rtf.toFixed(4);
  $(`${prefix}-ms`).textContent = r.ms.toFixed(1);
  $(`${prefix}-stall`).textContent = `${r.stallMs} ms / ${r.wallMs} ms wall`;
  $(`${prefix}-segs`).textContent = r.segs;
}

async function runAll() {
  $("log").textContent = "";
  setStatus("Loading audio…");
  const audio = await fetchPCM(AUDIO_URL);
  log(`audio: ${(audio.length / 16000).toFixed(2)} s, ${audio.length} samples`);

  setStatus("Running main thread…");
  const m = await benchMain(audio);
  paint("m", m);

  setStatus("Running worker…");
  const w = await benchWorker(audio);
  paint("w", w);

  setStatus("Done.");
}

async function runWorkerOnly() {
  $("log").textContent = "";
  setStatus("Loading audio…");
  const audio = await fetchPCM(AUDIO_URL);
  log(`audio: ${(audio.length / 16000).toFixed(2)} s, ${audio.length} samples`);
  setStatus("Running worker…");
  const w = await benchWorker(audio);
  paint("w", w);
  setStatus("Done.");
}

fillEnv();
$("run").addEventListener("click", () => {
  runAll().catch((e) => {
    setStatus("Error: " + (e.message || e));
    console.error(e);
  });
});
$("run-worker").addEventListener("click", () => {
  runWorkerOnly().catch((e) => {
    setStatus("Error: " + (e.message || e));
    console.error(e);
  });
});

// Headless / CI mode: ?auto=1 auto-runs the full bench and reports via
// document.title (DOM scraping) AND console.log (chrome --enable-logging
// captures stderr, useful when there's no human-driven scrape step).
if (new URLSearchParams(location.search).get("auto") === "1") {
  runAll()
    .then(() => {
      const dump = {
        m: { rtf: $("m-rtf").textContent, ms: $("m-ms").textContent, stall: $("m-stall").textContent },
        w: { rtf: $("w-rtf").textContent, ms: $("w-ms").textContent, stall: $("w-stall").textContent },
      };
      const line = "BENCH_DONE " + JSON.stringify(dump);
      document.title = line;
      console.log(line);
    })
    .catch((e) => {
      const line = "BENCH_ERROR " + (e.message || String(e));
      document.title = line;
      console.error(line);
    });
}
