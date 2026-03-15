// 80-dim Mel Fbank feature extraction
// Ported from FireRedVAD (upstream) C++ runtime/common/frontend/fbank.h
//
// Key parameters (matching Kaldi defaults):
//   - Sample rate: 16000 Hz
//   - Frame length: 400 samples (25ms)
//   - Frame shift: 160 samples (10ms)
//   - FFT size: 512 (next power of 2 from 400)
//   - Mel bins: 80
//   - Mel range: 20 Hz - 8000 Hz
//   - Window: Povey (NOT Hanning)
//   - Pre-emphasis coefficient: 0.97 (streaming, with state)
//   - DC offset removal: per frame
//   - Log energy floor: 1e-20
//   - snip_edges: true

import { FFTComputer } from "./fft.js";

const M_2PI = 6.283185307179586476925286766559005;

/**
 * Convert frequency in Hz to mel scale.
 * Uses the standard HTK formula: mel = 1127 * ln(1 + f/700)
 */
function melScale(freq: number): number {
  return 1127.0 * Math.log(1.0 + freq / 700.0);
}

/**
 * Compute the next power of two >= n.
 */
function upperPowerOfTwo(n: number): number {
  return Math.pow(2, Math.ceil(Math.log(n) / Math.log(2)));
}

/** A mel filter bin: starting FFT index and filter weights */
interface MelBin {
  firstIndex: number;
  weights: Float32Array;
}

export class Fbank {
  readonly numBins: number;
  readonly sampleRate: number;
  readonly frameLength: number;
  readonly frameShift: number;
  readonly fftSize: number;

  private readonly window: Float32Array;
  private readonly bins: MelBin[];
  private readonly fftComputer: FFTComputer;
  private preEmphasisState: number;

  constructor(
    numBins: number = 80,
    sampleRate: number = 16000,
    frameLength: number = 400,
    frameShift: number = 160,
  ) {
    this.numBins = numBins;
    this.sampleRate = sampleRate;
    this.frameLength = frameLength;
    this.frameShift = frameShift;
    this.fftSize = upperPowerOfTwo(frameLength);
    this.preEmphasisState = 0.0;

    this.fftComputer = new FFTComputer(this.fftSize);
    this.window = this.initWindow();
    this.bins = this.initMelFilters();
  }

  /**
   * Initialize the Povey window.
   * Povey window = pow(0.5 - 0.5 * cos(2*PI*i/(N-1)), 0.85)
   */
  private initWindow(): Float32Array {
    const win = new Float32Array(this.frameLength);
    const a = M_2PI / (this.frameLength - 1);
    for (let i = 0; i < this.frameLength; ++i) {
      win[i] = Math.pow(0.5 - 0.5 * Math.cos(a * i), 0.85);
    }
    return win;
  }

  /**
   * Initialize triangular mel filterbank weights.
   * Matches the C++ Fbank::InitMelFilters() exactly.
   */
  private initMelFilters(): MelBin[] {
    const numFftBins = this.fftSize / 2;
    const fftBinWidth = this.sampleRate / this.fftSize;
    const melLowFreq = melScale(20.0); // low_freq = 20 Hz
    const melHighFreq = melScale(this.sampleRate / 2.0); // Nyquist
    const melFreqDelta = (melHighFreq - melLowFreq) / (this.numBins + 1);

    const bins: MelBin[] = [];

    for (let bin = 0; bin < this.numBins; ++bin) {
      const leftMel = melLowFreq + bin * melFreqDelta;
      const centerMel = melLowFreq + (bin + 1) * melFreqDelta;
      const rightMel = melLowFreq + (bin + 2) * melFreqDelta;

      const thisBin = new Float32Array(numFftBins);
      let firstIndex = -1;
      let lastIndex = -1;

      for (let i = 0; i < numFftBins; ++i) {
        const freq = fftBinWidth * i;
        const mel = melScale(freq);
        if (mel > leftMel && mel < rightMel) {
          let weight: number;
          if (mel <= centerMel) {
            weight = (mel - leftMel) / (centerMel - leftMel);
          } else {
            weight = (rightMel - mel) / (rightMel - centerMel);
          }
          thisBin[i] = weight;
          if (firstIndex === -1) firstIndex = i;
          lastIndex = i;
        }
      }

      if (firstIndex === -1 || lastIndex < firstIndex) {
        // Invalid mel filter, push empty bin
        bins.push({ firstIndex: 0, weights: new Float32Array(0) });
        continue;
      }

      const size = lastIndex + 1 - firstIndex;
      const weights = new Float32Array(size);
      for (let i = 0; i < size; ++i) {
        weights[i] = thisBin[firstIndex + i];
      }
      bins.push({ firstIndex, weights });
    }

    return bins;
  }

  /**
   * Reset the streaming pre-emphasis state. Call before processing a new audio.
   */
  reset(): void {
    this.preEmphasisState = 0.0;
  }

  /**
   * Compute fbank features from raw PCM samples.
   *
   * @param wave - PCM samples as float32 (already normalized to int16 range or raw int16 values).
   *               If Int16Array is provided, it will be converted to Float32Array.
   * @returns Float32Array of shape [numFrames * numBins], row-major.
   *          Returns empty array if audio is too short.
   */
  compute(wave: Float32Array | Int16Array): Float32Array {
    // Convert Int16Array to Float32Array if needed
    let samples: Float32Array;
    if (wave instanceof Int16Array) {
      samples = new Float32Array(wave.length);
      for (let i = 0; i < wave.length; ++i) {
        samples[i] = wave[i];
      }
    } else {
      samples = wave;
    }

    const numSamples = samples.length;
    if (numSamples < this.frameLength) {
      return new Float32Array(0);
    }

    // snip_edges = true: number of frames
    const numFrames =
      1 + Math.floor((numSamples - this.frameLength) / this.frameShift);
    const feat = new Float32Array(numFrames * this.numBins);

    // Reusable buffers for FFT
    const fftReal = new Float32Array(this.fftSize);
    const fftImag = new Float32Array(this.fftSize);
    const power = new Float32Array(this.fftSize / 2);

    for (let i = 0; i < numFrames; ++i) {
      const frameStart = i * this.frameShift;

      // Extract frame data
      const data = new Float32Array(this.frameLength);
      for (let j = 0; j < this.frameLength; ++j) {
        data[j] = samples[frameStart + j];
      }

      // DC offset removal
      let mean = 0.0;
      for (let j = 0; j < this.frameLength; ++j) {
        mean += data[j];
      }
      mean /= this.frameLength;
      for (let j = 0; j < this.frameLength; ++j) {
        data[j] -= mean;
      }

      // Pre-emphasis with streaming state: y[n] = x[n] - 0.97 * x[n-1]
      let prev = this.preEmphasisState;
      for (let j = 0; j < this.frameLength; ++j) {
        const curr = data[j];
        data[j] = curr - 0.97 * prev;
        prev = curr;
      }
      this.preEmphasisState = data[this.frameLength - 1];

      // Apply Povey window
      for (let j = 0; j < this.frameLength; ++j) {
        data[j] *= this.window[j];
      }

      // Prepare FFT buffers: zero-pad data to fftSize
      fftImag.fill(0);
      fftReal.fill(0);
      for (let j = 0; j < this.frameLength; ++j) {
        fftReal[j] = data[j];
      }

      // FFT
      this.fftComputer.compute(fftReal, fftImag);

      // Power spectrum
      for (let j = 0; j < this.fftSize / 2; ++j) {
        power[j] = fftReal[j] * fftReal[j] + fftImag[j] * fftImag[j];
      }

      // Mel filterbank
      for (let j = 0; j < this.numBins; ++j) {
        let melEnergy = 0.0;
        const bin = this.bins[j];
        const s = bin.firstIndex;
        for (let k = 0; k < bin.weights.length; ++k) {
          melEnergy += bin.weights[k] * power[s + k];
        }

        // Log with floor
        if (melEnergy < 1e-20) melEnergy = 1e-20;
        melEnergy = Math.log(melEnergy);

        feat[i * this.numBins + j] = melEnergy;
      }
    }

    return feat;
  }

  /**
   * Compute fbank features and return as a 2D structure.
   *
   * @param wave - PCM samples
   * @returns { features: Float32Array, numFrames: number, numBins: number }
   */
  extract(
    wave: Float32Array | Int16Array,
  ): { features: Float32Array; numFrames: number; numBins: number } {
    const feat = this.compute(wave);
    const numFrames = feat.length / this.numBins;
    return { features: feat, numFrames, numBins: this.numBins };
  }

  /**
   * Compute fbank features for exactly one frame (frameLength samples).
   * This is designed for streaming use where you feed 400 samples and
   * get one 80-dim feature vector back. The pre-emphasis state is
   * maintained across calls.
   *
   * IMPORTANT: The caller must provide the correct 400-sample window.
   * For proper streaming, use a sliding buffer that shifts by frameShift
   * (160 samples) each time, and call this method once per shift.
   *
   * @param frame - Exactly frameLength (400) PCM samples
   * @returns Float32Array of numBins (80) features, or null if input size is wrong
   */
  computeFrame(frame: Float32Array | Int16Array): Float32Array | null {
    if (frame.length !== this.frameLength) {
      return null;
    }

    // Convert Int16Array if needed
    const data = new Float32Array(this.frameLength);
    for (let j = 0; j < this.frameLength; ++j) {
      data[j] = frame[j];
    }

    // DC offset removal
    let mean = 0.0;
    for (let j = 0; j < this.frameLength; ++j) {
      mean += data[j];
    }
    mean /= this.frameLength;
    for (let j = 0; j < this.frameLength; ++j) {
      data[j] -= mean;
    }

    // Pre-emphasis with streaming state
    let prev = this.preEmphasisState;
    for (let j = 0; j < this.frameLength; ++j) {
      const curr = data[j];
      data[j] = curr - 0.97 * prev;
      prev = curr;
    }
    this.preEmphasisState = data[this.frameLength - 1];

    // Apply Povey window
    for (let j = 0; j < this.frameLength; ++j) {
      data[j] *= this.window[j];
    }

    // FFT
    const fftReal = new Float32Array(this.fftSize);
    const fftImag = new Float32Array(this.fftSize);
    for (let j = 0; j < this.frameLength; ++j) {
      fftReal[j] = data[j];
    }
    this.fftComputer.compute(fftReal, fftImag);

    // Power spectrum
    const power = new Float32Array(this.fftSize / 2);
    for (let j = 0; j < this.fftSize / 2; ++j) {
      power[j] = fftReal[j] * fftReal[j] + fftImag[j] * fftImag[j];
    }

    // Mel filterbank + log
    const feat = new Float32Array(this.numBins);
    for (let j = 0; j < this.numBins; ++j) {
      let melEnergy = 0.0;
      const bin = this.bins[j];
      const s = bin.firstIndex;
      for (let k = 0; k < bin.weights.length; ++k) {
        melEnergy += bin.weights[k] * power[s + k];
      }
      if (melEnergy < 1e-20) melEnergy = 1e-20;
      feat[j] = Math.log(melEnergy);
    }

    return feat;
  }
}
