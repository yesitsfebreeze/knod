// ============================================================
// POINT BUFFERS
// ============================================================

import { gl } from './webgl.js';
import { totalNodes, displaced, restPositions, thoughtPrimaryTag, sphereInfo, bridgeNodes } from './geometry.js';
import { specialists, GLOBAL_TAGS, allThoughts } from './data.js';

// Build combined point data: all nodes in the unified array
export const STRIDE = 8; // x,y,z, r,g,b,a, size
export const totalPoints = totalNodes;
export const pointData = new Float32Array(totalPoints * STRIDE);
// Depth sort removed — dots use additive blending, no back-to-front order needed

// Initialize point positions
let idx = 0;
for (let i = 0; i < totalNodes; i++) {
  const p = restPositions[i];
  pointData[idx++] = p.x;
  pointData[idx++] = p.y;
  pointData[idx++] = p.z;
  // Default colors — will be overwritten each frame by render loop
  if (p.isTag) {
    pointData[idx++] = 0.27;
    pointData[idx++] = 0.53;
    pointData[idx++] = 0.53;
    pointData[idx++] = 1.0;
    pointData[idx++] = 0.9; // size
  } else {
    pointData[idx++] = 0.50;
    pointData[idx++] = 0.40;
    pointData[idx++] = 0.40;
    pointData[idx++] = 1.0;
    pointData[idx++] = 0.35; // size
  }
}

export let pointBuf;
export function initPointBuf() {
  pointBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, pointBuf);
  gl.bufferData(gl.ARRAY_BUFFER, pointData, gl.DYNAMIC_DRAW);
}

// Glow buffer — for tag points (larger, softer halos)
// Count all tag nodes across all spheres + bridge nodes
let tagCount = 0;
for (let si = 0; si < specialists.length; si++) {
  tagCount += sphereInfo[si].tagCount;
}
// Add bridge nodes (which are also tags)
tagCount += bridgeNodes.length;

export const glowTagCount = tagCount;
export const glowData = new Float32Array(tagCount * STRIDE);
export let glowBuf;
export function initGlowBuf() {
  glowBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, glowBuf);
  gl.bufferData(gl.ARRAY_BUFFER, glowData.byteLength, gl.DYNAMIC_DRAW);
}

// Pre-built index list of tag node indices — avoids full scan each frame
export const glowNodeIndices = new Int32Array(tagCount);
let _gi = 0;
for (let i = 0; i < totalNodes; i++) {
  if (restPositions[i].isTag) glowNodeIndices[_gi++] = i;
}

// --- Pre-baked per-node base colors & sizes (palette-dependent) ---
// Layout: [r, g, b, size] per node — stride 4
// Call rebakeBaseColors(palette, pOld, pNew) whenever palette changes.
export const BASE_STRIDE = 4;
export const baseNodeColors = new Float32Array(totalNodes * BASE_STRIDE);

export function rebakeBaseColors(palette, pOld, pNew) {
  const bridgeColor = palette.base0D || [0.40, 0.53, 0.67];
  for (let i = 0; i < totalNodes; i++) {
    const off = i * BASE_STRIDE;
    const node = restPositions[i];
    if (node.isTag) {
      if (node.isBridge) {
        baseNodeColors[off]     = bridgeColor[0];
        baseNodeColors[off + 1] = bridgeColor[1];
        baseNodeColors[off + 2] = bridgeColor[2];
      } else {
        const si = node.sphereIdx;
        const localIdx = i - sphereInfo[si].tagStart;
        const spread = localIdx / sphereInfo[si].tagCount;
        const ageFactor = 0.05 + spread * 0.15;
        baseNodeColors[off]     = pOld[0] + ageFactor * (pNew[0] - pOld[0]);
        baseNodeColors[off + 1] = pOld[1] + ageFactor * (pNew[1] - pOld[1]);
        baseNodeColors[off + 2] = pOld[2] + ageFactor * (pNew[2] - pOld[2]);
      }
      baseNodeColors[off + 3] = 0.9; // size
    } else {
      const si = node.sphereIdx;
      const localIdx = i - sphereInfo[si].thoughtStart;
      const th = specialists[si].thoughts[localIdx];
      if (th) {
        const brightness = 0.5 + 0.5 * Math.min(th.access_count / 30, 1);
        const ageFactor = 1.0 - th.age;
        baseNodeColors[off]     = (pOld[0] + ageFactor * (pNew[0] - pOld[0])) * brightness;
        baseNodeColors[off + 1] = (pOld[1] + ageFactor * (pNew[1] - pOld[1])) * brightness;
        baseNodeColors[off + 2] = (pOld[2] + ageFactor * (pNew[2] - pOld[2])) * brightness;
        baseNodeColors[off + 3] = 0.25 + 0.2 * (th.access_count / 50); // size
      } else {
        baseNodeColors[off]     = 0.5;
        baseNodeColors[off + 1] = 0.4;
        baseNodeColors[off + 2] = 0.4;
        baseNodeColors[off + 3] = 0.35;
      }
    }
  }
}
