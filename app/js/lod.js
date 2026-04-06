// ============================================================
// LOD — Per-sphere level-of-detail blend factors
// ============================================================
//
// 3 LOD levels per sphere, driven by camera distance to sphere center:
//
//   LOD 2 (near):  dist < LOD_MID_DIST        — full detail: tags + thoughts
//   LOD 1 (mid):   LOD_MID_DIST < dist < LOD_FAR_DIST  — tags only, thoughts fade out
//   LOD 0 (far):   dist > LOD_FAR_DIST        — proxy dot only, tags+thoughts invisible
//
// Two smooth blend values per sphere:
//   tagFade[si]     : 0 = tags invisible (LOD 0), 1 = fully visible (LOD 1-2)
//   thoughtFade[si] : 0 = thoughts invisible (LOD 1-2), 1 = fully visible (LOD 2)
//
// Both are smoothly animated toward their targets with a configurable speed.

import { sphereCenters } from './geometry.js';
import { specialists } from './data.js';

// Distance thresholds (from camera to sphere center).
// These are in world units. GALAXY_RADIUS = 15, SPHERE_RADIUS = 2.
// LOD_FAR_DIST:  single dot only. Roughly 3x sphere diameter away.
// LOD_MID_DIST:  tags only. Roughly 1.5x sphere diameter away.
export const LOD_FAR_DIST  = 20.0;  // dist > this → proxy dot only
export const LOD_MID_DIST  =  9.0;  // dist > this → tags only (no thoughts)

// Transition band width: distance over which the fade happens
const FADE_BAND = 3.5;

// Per-sphere smooth blend factors — updated every frame by updateLOD()
export const tagFade     = new Float32Array(specialists.length); // 0=hidden,1=visible
export const thoughtFade = new Float32Array(specialists.length); // 0=hidden,1=visible

// Speed of the smooth transition (higher = snappier)
const LOD_LERP = 6.0;

// Per-sphere current distance from camera (updated each frame)
export const sphereDist = new Float32Array(specialists.length);

export function updateLOD(eyeX, eyeY, eyeZ, dt) {
  const lerpFactor = Math.min(dt * LOD_LERP, 0.92);

  for (let si = 0; si < specialists.length; si++) {
    const c = sphereCenters[si];
    const dx = c.x - eyeX, dy = c.y - eyeY, dz = c.z - eyeZ;
    const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
    sphereDist[si] = dist;

    // tagFade: 0 when dist > LOD_FAR_DIST, 1 when dist < LOD_FAR_DIST - FADE_BAND
    const targetTag = 1.0 - smoothstep01(LOD_FAR_DIST - FADE_BAND, LOD_FAR_DIST, dist);
    tagFade[si] += (targetTag - tagFade[si]) * lerpFactor;

    // thoughtFade: 0 when dist > LOD_MID_DIST, 1 when dist < LOD_MID_DIST - FADE_BAND
    const targetThought = 1.0 - smoothstep01(LOD_MID_DIST - FADE_BAND, LOD_MID_DIST, dist);
    thoughtFade[si] += (targetThought - thoughtFade[si]) * lerpFactor;
  }
}

function smoothstep01(lo, hi, x) {
  const t = Math.max(0, Math.min(1, (x - lo) / (hi - lo)));
  return t * t * (3 - 2 * t);
}
