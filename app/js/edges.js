// ============================================================
// INTER-THOUGHT EDGE GEOMETRY (straight lines between connected nodes)
// ============================================================

import { gl } from './webgl.js';
import { allEdges, allThoughts, GLOBAL_TAGS } from './data.js';
import { displaced, thoughtToGlobalIdx, restPositions } from './geometry.js';
import { opacitySettings } from './physics.js';
import { getOldColor, getRecentColor } from './themes.js';

const LINE_STRIDE = 7; // x,y,z, r,g,b,a

// Each edge = 1 straight segment = 2 verts
const maxEdgeLineVerts = allEdges.length * 2;
export const edgeLineData = new Float32Array(maxEdgeLineVerts * LINE_STRIDE);

export let edgeLineBuf;
export function initEdgeBuf() {
  edgeLineBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, edgeLineBuf);
  gl.bufferData(gl.ARRAY_BUFFER, edgeLineData.byteLength, gl.DYNAMIC_DRAW);
}

// Map thought id -> index in displaced[] array
// In multi-sphere mode, we use thoughtToGlobalIdx from geometry.js
// For backward compat, also support old TAGS.length-based indexing
const thoughtIdToDisplacedIdx = thoughtToGlobalIdx;

export function buildEdgeLines(eyeX, eyeY, eyeZ, hoverAnim, hoverActiveAnim, thoughtFade) {
  let vi = 0;

  // Palette colors for edge gradient: old/recent from color map
  const eOld = getOldColor();
  const eNew = getRecentColor();
  const edgeOp = opacitySettings.edges;
  const bias = opacitySettings.recencyBias; // 0.5 = neutral, >0.5 fades old, <0.5 fades recent

  for (let e = 0; e < allEdges.length; e++) {
    const edge = allEdges[e];
    const srcIdx = thoughtIdToDisplacedIdx.get(edge.source_id);
    const tgtIdx = thoughtIdToDisplacedIdx.get(edge.target_id);
    if (srcIdx === undefined || tgtIdx === undefined) continue;

    const src = displaced[srcIdx];
    const tgt = displaced[tgtIdx];

    // Endpoints with breathing
    const x0 = src.x, y0 = src.y, z0 = src.z;
    const x1 = tgt.x, y1 = tgt.y, z1 = tgt.z;

    // Age-based color: old → new from palette
    const ageFactor = 1.0 - edge.age; // 0=old, 1=new
    // Smooth hover: use max hoverAnim of source/target
    const hSrc = hoverAnim[srcIdx] || 0;
    const hTgt = hoverAnim[tgtIdx] || 0;
    const h = Math.max(hSrc, hTgt);
    // Alpha > 1.0 signals hover factor to line shader
    // Non-hovered edges dim to 50% when something else is hovered
    const dimFactor = 1.0 - (1.0 - h) * hoverActiveAnim * 0.5;
    // Recency bias opacity: at 0.5 all edges full alpha,
    // >0.5 fades old edges (recent stay full), <0.5 fades recent edges (old stay full)
    let biasOp = 1.0;
    if (bias > 0.5) {
      // Fade old edges: ageFactor=1 (new) stays 1.0, ageFactor=0 (old) fades toward 0
      const strength = (bias - 0.5) * 2.0; // 0..1
      biasOp = 1.0 - strength * (1.0 - ageFactor);
    } else if (bias < 0.5) {
      // Fade recent edges: ageFactor=0 (old) stays 1.0, ageFactor=1 (new) fades toward 0
      const strength = (0.5 - bias) * 2.0; // 0..1
      biasOp = 1.0 - strength * ageFactor;
    }
    // Recent edges are more visible; old edges fade down
    const ageAlphaBoost = 0.35 + ageFactor * 0.45; // 0.35 for oldest, 0.80 for newest
    // LOD: edge is hidden when either endpoint's sphere is at far LOD
    let lodFade = 1.0;
    if (thoughtFade) {
      const srcSi = restPositions[srcIdx] ? restPositions[srcIdx].sphereIdx : -1;
      const tgtSi = restPositions[tgtIdx] ? restPositions[tgtIdx].sphereIdx : -1;
      const srcFade = srcSi >= 0 ? thoughtFade[srcSi] : 1.0;
      const tgtFade = tgtSi >= 0 ? thoughtFade[tgtSi] : 1.0;
      lodFade = Math.min(srcFade, tgtFade);
    }
    const baseAlpha = (edge.weight * (ageAlphaBoost + h * (1.0 - ageAlphaBoost)) * dimFactor + h) * edgeOp * biasOp * lodFade;

    // Interpolate with a steeper curve so the gradient is more visible
    const colorT = ageFactor * ageFactor; // quadratic: pushes more edges toward the old color, making recent ones pop
    let r = eOld[0] + (eNew[0] - eOld[0]) * colorT;
    let g = eOld[1] + (eNew[1] - eOld[1]) * colorT;
    let b = eOld[2] + (eNew[2] - eOld[2]) * colorT;

    // Source vertex
    edgeLineData[vi++] = x0; edgeLineData[vi++] = y0; edgeLineData[vi++] = z0;
    edgeLineData[vi++] = r; edgeLineData[vi++] = g; edgeLineData[vi++] = b;
    edgeLineData[vi++] = baseAlpha;
    // Target vertex
    edgeLineData[vi++] = x1; edgeLineData[vi++] = y1; edgeLineData[vi++] = z1;
    edgeLineData[vi++] = r; edgeLineData[vi++] = g; edgeLineData[vi++] = b;
    edgeLineData[vi++] = baseAlpha;
  }

  return vi / LINE_STRIDE;
}
