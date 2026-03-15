// Radix-2 Cooley-Tukey FFT implementation
// Ported from FireRedVAD (upstream) C++ runtime/common/frontend/fft.cc
// Uses precomputed sin table and bit-reversal table for performance.

const M_PI = 3.1415926535897932384626433832795;

/**
 * Build the trigonometric function table used by the FFT.
 * Table layout: sintbl[0..n/4-1] holds sin/cos values,
 * sintbl[n/4..n+n/4-1] holds mirrored/negated copies.
 */
export function makeSinTable(n: number): Float32Array {
  const sintbl = new Float32Array(n + n / 4);
  const n2 = n / 2;
  const n4 = n / 4;
  const n8 = n / 8;
  const t0 = Math.sin(M_PI / n);
  let dc = 2 * t0 * t0;
  let ds = Math.sqrt(dc * (2 - dc));
  const t = 2 * dc;
  let c = 1;
  let s = 0;
  sintbl[n4] = 1;
  sintbl[0] = 0;

  for (let i = 1; i < n8; ++i) {
    c -= dc;
    dc += t * c;
    s += ds;
    ds -= t * s;
    sintbl[i] = s;
    sintbl[n4 - i] = c;
  }

  if (n8 !== 0) {
    sintbl[n8] = Math.sqrt(0.5);
  }

  for (let i = 0; i < n4; ++i) {
    sintbl[n2 - i] = sintbl[i];
  }

  for (let i = 0; i < n2 + n4; ++i) {
    sintbl[i + n2] = -sintbl[i];
  }

  return sintbl;
}

/**
 * Build the bit-reversal table for an FFT of length n.
 */
export function makeBitReversal(n: number): Int32Array {
  const bitrev = new Int32Array(n);
  const n2 = n / 2;
  let i = 0;
  let j = 0;

  for (;;) {
    bitrev[i] = j;
    if (++i >= n) break;
    let k = n2;
    while (k <= j) {
      j -= k;
      k /= 2;
    }
    j += k;
  }

  return bitrev;
}

/**
 * In-place FFT (forward transform).
 * @param bitrev - bit reversal table from makeBitReversal(n)
 * @param sintbl - sin table from makeSinTable(n)
 * @param x - real part array (length n), modified in place
 * @param y - imaginary part array (length n), modified in place
 * @param n - FFT length (must be power of 2)
 */
export function fft(
  bitrev: Int32Array,
  sintbl: Float32Array,
  x: Float32Array,
  y: Float32Array,
  n: number,
): void {
  const n4 = n / 4;

  // Bit reversal
  for (let i = 0; i < n; ++i) {
    const j = bitrev[i];
    if (i < j) {
      let t = x[i];
      x[i] = x[j];
      x[j] = t;
      t = y[i];
      y[i] = y[j];
      y[j] = t;
    }
  }

  // Butterfly computation
  let k = 1;
  while (k < n) {
    let h = 0;
    const k2 = k + k;
    const d = n / k2;
    for (let j = 0; j < k; ++j) {
      const c = sintbl[h + n4];
      const s = sintbl[h];
      for (let i = j; i < n; i += k2) {
        const ik = i + k;
        const dx = s * y[ik] + c * x[ik];
        const dy = c * y[ik] - s * x[ik];
        x[ik] = x[i] - dx;
        x[i] += dx;
        y[ik] = y[i] - dy;
        y[i] += dy;
      }
      h += d;
    }
    k = k2;
  }
}

/**
 * Precomputed FFT tables for a given size.
 * Reuse across frames to avoid recomputation.
 */
export class FFTComputer {
  readonly size: number;
  private readonly bitrev: Int32Array;
  private readonly sintbl: Float32Array;

  constructor(size: number) {
    this.size = size;
    this.bitrev = makeBitReversal(size);
    this.sintbl = makeSinTable(size);
  }

  /**
   * Compute in-place FFT on the given real and imaginary arrays.
   */
  compute(real: Float32Array, imag: Float32Array): void {
    fft(this.bitrev, this.sintbl, real, imag, this.size);
  }
}
