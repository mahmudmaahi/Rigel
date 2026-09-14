/**
 * Colour palette utilities for Rigel DSP visualisations.
 */

// dBFS display range typically used across visualisations
export const DB_MIN = -90;   // maps to darkest colour
export const DB_MAX = 0;     // maps to brightest colour

/**
 * Map a normalised value [0, 1] to an RGB colour array using a
 * perceptually-ordered aurora gradient: navy → indigo → purple → teal → white.
 *
 *   0.00 → deep navy    (#0d0d18)
 *   0.20 → dark indigo  (#312e78)
 *   0.45 → indigo-500   (#6366f1)
 *   0.70 → purple-400   (#c084fc)
 *   0.90 → teal-400     (#2dd4bf)
 *   1.00 → teal-50      (#f0fdfa)
 */
export function getRgbForT(t: number): [number, number, number] {
  const v = Math.max(0, Math.min(1, t));

  const stops: [number, number, number, number][] = [
    [0.00,  13,  13,  24],
    [0.20,  49,  46, 120],
    [0.45,  99, 102, 241],
    [0.70, 192, 132, 252],
    [0.90,  45, 212, 191],
    [1.00, 240, 253, 250],
  ];

  for (let i = 0; i < stops.length - 1; i++) {
    const [t0, r0, g0, b0] = stops[i];
    const [t1, r1, g1, b1] = stops[i + 1];
    if (v >= t0 && v <= t1) {
      const f = (v - t0) / (t1 - t0);
      return [
        Math.round(r0 + f * (r1 - r0)),
        Math.round(g0 + f * (g1 - g0)),
        Math.round(b0 + f * (b1 - b0)),
      ];
    }
  }
  return [255, 255, 255];
}

/**
 * Convert a dBFS value to a normalised [0, 1] range based on DB_MIN and DB_MAX.
 */
export function dbToT(db: number, min = DB_MIN, max = DB_MAX): number {
  return (Math.max(min, Math.min(max, db)) - min) / (max - min);
}

/**
 * Create a canvas linear gradient matching the perceptual colour map.
 * Assumes (x0, y0) is the "top" (0 dBFS, max magnitude) and (x1, y1) is the "bottom".
 */
export function createColorMapGradient(
  ctx: CanvasRenderingContext2D,
  x0: number, y0: number,
  x1: number, y1: number
): CanvasGradient {
  const gradient = ctx.createLinearGradient(x0, y0, x1, y1);
  const stops = [0.00, 0.20, 0.45, 0.70, 0.90, 1.00];
  for (const t of stops) {
    // t=0 is the top (y0), which should be bright (color map value 1)
    // t=1 is the bottom (y1), which should be dark (color map value 0)
    const [r, g, b] = getRgbForT(1 - t);
    gradient.addColorStop(t, `rgb(${r},${g},${b})`);
  }
  return gradient;
}

/**
 * Create a symmetric canvas linear gradient for waveforms.
 * Center (t=0.5) is dark, edges (t=0.0 and t=1.0) are bright.
 */
export function createSymmetricColorMapGradient(
  ctx: CanvasRenderingContext2D,
  x0: number, y0: number,
  x1: number, y1: number
): CanvasGradient {
  const gradient = ctx.createLinearGradient(x0, y0, x1, y1);
  const stops = [0.0, 0.1, 0.275, 0.4, 0.5, 0.6, 0.725, 0.9, 1.0];
  for (const t of stops) {
    // Distance from center (0 to 0.5) mapped to (0 to 1)
    const distFromCenter = Math.abs(t - 0.5) * 2;
    const [r, g, b] = getRgbForT(distFromCenter);
    gradient.addColorStop(t, `rgb(${r},${g},${b})`);
  }
  return gradient;
}
