// ============================================================
// STRAIGHT-LINE GRAPH WITH INERTIAL PHYSICS
// ============================================================

import { gl } from './webgl.js';
import { totalNodes, restPositions, displaced, thoughtPrimaryTag, SPHERE_RADIUS, sphereInfo, sphereCenters } from './geometry.js';
import { STORE, specialists, GLOBAL_TAGS, allThoughts } from './data.js';
import { quatMul } from './math.js';
import { getStemColor } from './themes.js';

// Physics state — angular velocity from camera rotation
let angVelY = 0;
let angVelX = 0;
let prevQuat = null; // set on first frame

// Stiffness per node — tags (surface, longer stems) are more springy,
// thoughts (interior, shorter) are stiffer
export let STIFFNESS_TAG = 2.6;
export let STIFFNESS_THOUGHT = 4.4;
export let DAMPING = 0.926;        // derived from reactivity=0.70: 0.80 + 0.70*0.18
export let INERTIA_STRENGTH = 3.5; // derived from reactivity=0.70: 0.70*5.0

export function setStiffness(tag, thought) {
  STIFFNESS_TAG = tag;
  STIFFNESS_THOUGHT = thought;
}

export function setDamping(d) { DAMPING = d; }
export function setInertiaStrength(s) { INERTIA_STRENGTH = s; }

// Opacity multipliers (0..1) — controlled by settings panel
// recencyBias: 0.0 = boost old/fade recent, 0.5 = neutral, 1.0 = boost recent/fade old
export const opacitySettings = {
  edges: 1.0,
  recencyBias: 0.5,
  stems: 1.0,
  glow: 1.0,
  labels: 1.0,
};

// Dot size multiplier (0..1) — 0 hides dots entirely
export let dotSizeMul = 1.0;
export function setDotSizeMul(v) { dotSizeMul = v; }

// Stem line data
const LINE_STRIDE = 7; // x,y,z, r,g,b,a
// For multi-sphere: each specialist has up to 200 thoughts => 200 stems each
const maxStemVerts = allThoughts.length * 2;
export const lineData = new Float32Array(maxStemVerts * LINE_STRIDE);
export const LINE_STRIDE_EXPORT = LINE_STRIDE;

export let lineBuf;
export function initLineBuf() {
  lineBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, lineBuf);
  gl.bufferData(gl.ARRAY_BUFFER, lineData.byteLength, gl.DYNAMIC_DRAW);
}

export function buildStemLines(eyeX, eyeY, eyeZ, hoverAnim, hoverActiveAnim, thoughtFade) {
  const stemOp = opacitySettings.stems * opacitySettings.edges;
  // Hoist stem color lookup — constant for entire frame
  const sc = getStemColor();
  const lineR = sc[0], lineG = sc[1], lineB = sc[2];
  let vi = 0;

  // Only thoughts get stem lines: from their primary tag to the thought
  let thoughtGlobalIdx = 0;
  for (let si = 0; si < specialists.length; si++) {
    const sp = specialists[si];
    // LOD: stems fade with thoughts
    const lodFade = thoughtFade ? thoughtFade[si] : 1.0;

    for (let i = 0; i < sp.thoughts.length; i++) {
      const n = findThoughtGlobalNodeIdx(si, i);
      const tagIdx = thoughtPrimaryTag[thoughtGlobalIdx];
      const disp = displaced[n];
      const tagDisp = displaced[tagIdx];

      const h = hoverAnim[n]; // smooth 0→1 hover factor

      // Tag endpoint (on sphere surface)
      const tx = tagDisp.x, ty = tagDisp.y, tz = tagDisp.z;
      // Thought endpoint (inside sphere)
      const px = disp.x, py = disp.y, pz = disp.z;

      // Alpha: subtle by default, full opacity on hover
      const dimFactor = 1.0 - (1.0 - h) * hoverActiveAnim * 0.5;
      const alphaTag = (0.03 * dimFactor + h * 1.97) * stemOp * lodFade;
      const alphaThought = (0.20 * dimFactor + h * 1.80) * stemOp * lodFade;

      // Vertex 0: tag end
      lineData[vi++] = tx; lineData[vi++] = ty; lineData[vi++] = tz;
      lineData[vi++] = lineR; lineData[vi++] = lineG; lineData[vi++] = lineB;
      lineData[vi++] = alphaTag;
      // Vertex 1: thought end
      lineData[vi++] = px; lineData[vi++] = py; lineData[vi++] = pz;
      lineData[vi++] = lineR; lineData[vi++] = lineG; lineData[vi++] = lineB;
      lineData[vi++] = alphaThought;

      thoughtGlobalIdx++;
    }
  }

  return vi / LINE_STRIDE; // total vertices
}

// Helper: find the global node index for a thought given specialist index and local thought index
function findThoughtGlobalNodeIdx(si, localThoughtIdx) {
  return sphereInfo[si].thoughtStart + localThoughtIdx;
}

// Pre-computed rest distances from origin — eliminates sqrt per node per frame
const restDist = new Float32Array(totalNodes);
for (let n = 0; n < totalNodes; n++) {
  const r = restPositions[n];
  restDist[n] = Math.sqrt(r.x * r.x + r.y * r.y + r.z * r.z);
}

export function updatePhysics(dt, camQuat) {
  // First frame: initialize prevQuat, no delta
  if (!prevQuat) {
    prevQuat = camQuat.slice();
    return;
  }

  // Derive angular velocity from quaternion delta
  // delta = camQuat * conjugate(prevQuat) gives the incremental rotation
  const pConj = [-prevQuat[0], -prevQuat[1], -prevQuat[2], prevQuat[3]];
  const dq = quatMul(camQuat, pConj);
  prevQuat = camQuat.slice();

  // Extract approximate angular velocity vector (world-space)
  // For small rotations: axis * angle ≈ 2 * [x, y, z] of the delta quat
  const dAngX = 2 * dq[0];
  const dAngY = 2 * dq[1];
  const dAngZ = 2 * dq[2];

  // Accumulate angular velocity with inertia
  angVelX += dAngX * INERTIA_STRENGTH;
  angVelY += dAngY * INERTIA_STRENGTH;
  let angVelZ_inst = dAngZ * INERTIA_STRENGTH;

  // Damp
  angVelX *= DAMPING;
  angVelY *= DAMPING;

  // Full angular velocity vector in world space
  const wX = angVelX;
  const wY = angVelY;
  const wZ = angVelZ_inst;

  // Apply displacement: each node's tip lags behind the rotation
  // Proper cross product: offset = omega × position (tangential lag)
  for (let n = 0; n < totalNodes; n++) {
    const rest = restPositions[n];
    const dist = restDist[n];
    const stiffness = rest.isTag ? STIFFNESS_TAG : STIFFNESS_THOUGHT;

    // cross(omega, rest) gives the tangential displacement direction
    const crossX = wY * rest.z - wZ * rest.y;
    const crossY = wZ * rest.x - wX * rest.z;
    const crossZ = wX * rest.y - wY * rest.x;

    // Scale offset by distance from center — farther nodes sway more
    const scale = dist * 0.525;
    const offsetX = -crossX * scale;
    const offsetY = -crossY * scale;
    const offsetZ = -crossZ * scale;

    // Spring back toward rest position
    const targetX = rest.x + offsetX;
    const targetY = rest.y + offsetY;
    const targetZ = rest.z + offsetZ;

    const springDt = Math.min(dt * stiffness, 0.95);
    displaced[n].x += (targetX - displaced[n].x) * springDt;
    displaced[n].y += (targetY - displaced[n].y) * springDt;
    displaced[n].z += (targetZ - displaced[n].z) * springDt;

    // Clamp: don't let nodes stretch beyond the sphere radius
    // In multi-sphere layout, nodes are offset by sphere centers,
    // so we clamp relative to the sphere center (not global origin)
    const node = restPositions[n];
    let cx = 0, cy = 0, cz = 0;
    if (node.sphereIdx >= 0) {
      const center = sphereCenters[node.sphereIdx];
      cx = center.x; cy = center.y; cz = center.z;
    } else if (node.isBridge) {
      // Bridge nodes: use rest position as their own "center" (no clamping needed)
      cx = rest.x; cy = rest.y; cz = rest.z;
    }
    const lx = displaced[n].x - cx, ly = displaced[n].y - cy, lz = displaced[n].z - cz;
    const localDist = Math.sqrt(lx * lx + ly * ly + lz * lz);
    if (localDist > SPHERE_RADIUS) {
      const clampScale = SPHERE_RADIUS / localDist;
      displaced[n].x = cx + lx * clampScale;
      displaced[n].y = cy + ly * clampScale;
      displaced[n].z = cz + lz * clampScale;
    }
  }
}
