// ============================================================
// GEOMETRY: MULTI-SPHERE GALAXY LAYOUT
// ============================================================

import { specialists, sharedTags } from './data.js';

export const SPHERE_RADIUS = 2.0;        // radius of each specialist sphere
export const GALAXY_RADIUS = 15.0;       // radius for placing sphere centers
const GOLDEN_RATIO = (1 + Math.sqrt(5)) / 2;

// Generate Fibonacci sphere points
export function fibonacciSphere(n, radius) {
  const points = [];
  for (let i = 0; i < n; i++) {
    const theta = 2 * Math.PI * i / GOLDEN_RATIO;
    const phi = Math.acos(1 - 2 * (i + 0.5) / n);
    points.push({
      x: radius * Math.sin(phi) * Math.cos(theta),
      y: radius * Math.sin(phi) * Math.sin(theta),
      z: radius * Math.cos(phi),
    });
  }
  return points;
}

// --- Galaxy-level: place 25 sphere centers ---
export const sphereCenters = fibonacciSphere(specialists.length, GALAXY_RADIUS);

// --- Per-sphere: place tags on surface, thoughts inside ---
// Build unified node arrays across all spheres + bridge nodes
//
// Node layout in the unified arrays:
//   [sphere0_tags | sphere0_thoughts | sphere1_tags | sphere1_thoughts | ... | bridge_tags]
//
// We track offsets so we can identify which sphere a node belongs to.

export const sphereInfo = []; // per specialist: { tagStart, tagCount, thoughtStart, thoughtCount, centerIdx }
export const restPositions = [];   // all nodes: { x, y, z, isTag, isBridge, sphereIdx }
export const thoughtPrimaryTag = []; // for each thought (globally), index into restPositions of its primary tag
export const thoughtToGlobalIdx = new Map(); // thought.id -> index in restPositions

let nodeIdx = 0;
for (let si = 0; si < specialists.length; si++) {
  const sp = specialists[si];
  const cx = sphereCenters[si].x;
  const cy = sphereCenters[si].y;
  const cz = sphereCenters[si].z;

  const tagStart = nodeIdx;
  const localTagPositions = fibonacciSphere(sp.tags.length, SPHERE_RADIUS);

  // Tags: on sphere surface, offset by sphere center
  for (let ti = 0; ti < sp.tags.length; ti++) {
    const lp = localTagPositions[ti];
    restPositions.push({
      x: cx + lp.x, y: cy + lp.y, z: cz + lp.z,
      isTag: true, isBridge: false, sphereIdx: si,
    });
    nodeIdx++;
  }

  const thoughtStart = nodeIdx;

  // Thoughts: projected inside sphere, offset by sphere center
  for (let i = 0; i < sp.thoughts.length; i++) {
    const thought = sp.thoughts[i];
    // Weighted centroid of local tag positions
    let x = 0, y = 0, z = 0, wsum = 0;
    for (let t = 0; t < sp.tags.length; t++) {
      const w = thought.affinities[t];
      x += localTagPositions[t].x * w;
      y += localTagPositions[t].y * w;
      z += localTagPositions[t].z * w;
      wsum += w;
    }
    if (wsum > 0) { x /= wsum; y /= wsum; z /= wsum; }
    const maxAff = Math.max(...thought.affinities);
    const spread = 0.3 + maxAff * 0.65;
    const len = Math.sqrt(x*x + y*y + z*z) || 1;
    const r = SPHERE_RADIUS * spread;

    restPositions.push({
      x: cx + x/len * r, y: cy + y/len * r, z: cz + z/len * r,
      isTag: false, isBridge: false, sphereIdx: si,
    });

    // Primary tag for this thought (index in restPositions)
    let best = 0;
    for (let t = 1; t < thought.affinities.length; t++) {
      if (thought.affinities[t] > thought.affinities[best]) best = t;
    }
    thoughtPrimaryTag.push(tagStart + best);
    thoughtToGlobalIdx.set(thought.id, nodeIdx);
    nodeIdx++;
  }

  sphereInfo.push({
    tagStart, tagCount: sp.tags.length,
    thoughtStart, thoughtCount: sp.thoughts.length,
    centerIdx: si,
  });
}

// --- Bridge nodes: shared tags positioned at midpoint between sharing spheres ---
export const bridgeStart = nodeIdx;
export const bridgeNodes = []; // { label, owners, nodeIndex, positions... }
for (const st of sharedTags) {
  // Compute centroid of all owning sphere centers
  let bx = 0, by = 0, bz = 0;
  for (const si of st.owners) {
    bx += sphereCenters[si].x;
    by += sphereCenters[si].y;
    bz += sphereCenters[si].z;
  }
  bx /= st.owners.length;
  by /= st.owners.length;
  bz /= st.owners.length;

  restPositions.push({
    x: bx, y: by, z: bz,
    isTag: true, isBridge: true, sphereIdx: -1,
  });
  bridgeNodes.push({ label: st.label, owners: st.owners, nodeIndex: nodeIdx });
  nodeIdx++;
}

export const totalNodes = restPositions.length;
export const displaced = restPositions.map(p => ({ x: p.x, y: p.y, z: p.z }));
