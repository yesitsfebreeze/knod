// ============================================================
// HOVER / TOOLTIP
// ============================================================

import { totalNodes, displaced, restPositions, sphereInfo, SPHERE_RADIUS, GALAXY_RADIUS, bridgeNodes, thoughtPrimaryTag } from './geometry.js';
import { specialists, allEdges, allThoughts, adjacency } from './data.js';
import { mouseX, mouseY, camDist } from './camera.js';

export let hoveredNodeIndex = -1; // index into restPositions (-1 = none)
// Smooth hover animation: per-node blend factor 0..1
export const hoverAnim = new Float32Array(totalNodes); // all start at 0
export let hoverActiveAnim = 0; // smooth 0→1 when anything is hovered

const tooltip = document.getElementById('tooltip');
const tooltipLabel = tooltip.querySelector('.label');
const tooltipDetail = tooltip.querySelector('.detail');
const tooltipSource = tooltip.querySelector('.source');
const tooltipConnections = tooltip.querySelector('.connections');

export function project3Dto2D(x, y, z, mvp, w, h) {
  const cx = mvp[0]*x + mvp[4]*y + mvp[8]*z + mvp[12];
  const cy = mvp[1]*x + mvp[5]*y + mvp[9]*z + mvp[13];
  const cw = mvp[3]*x + mvp[7]*y + mvp[11]*z + mvp[15];
  if (cw <= 0) return null;
  return {
    x: (cx/cw * 0.5 + 0.5) * w,
    y: (1 - (cy/cw * 0.5 + 0.5)) * h,
    depth: cw,
  };
}

function findBridgeLabel(nodeIndex) {
  for (const bn of bridgeNodes) {
    if (bn.nodeIndex === nodeIndex) return bn.label;
  }
  return 'bridge';
}

// Pre-built Map: thought id → thought object — O(1) lookup in tooltip
const thoughtById = new Map();
for (const t of allThoughts) thoughtById.set(t.id, t);

// Cache: last mouse position and result — skip full search when mouse is static
let _lastMouseX = -1, _lastMouseY = -1;
let _lastBestItem = null;

export function updateTooltip(mvp, eyeX, eyeY, eyeZ) {
  const W = window.innerWidth;
  const H = window.innerHeight;

  // Skip expensive search if mouse hasn't moved
  const mouseMoved = (mouseX !== _lastMouseX || mouseY !== _lastMouseY);
  _lastMouseX = mouseX;
  _lastMouseY = mouseY;

  let bestItem = _lastBestItem; // reuse last result if mouse static

  if (mouseMoved) {
    let bestScore = Infinity;
    bestItem = null;

    // Depth-weighted selection: score = pixelDist * depthPenalty
    function score(screenDistSq, px, py, pz) {
      const dx = px - eyeX, dy = py - eyeY, dz = pz - eyeZ;
      const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
      // Use galaxy-scale range for depth penalty
      const nearDist = camDist - GALAXY_RADIUS - SPHERE_RADIUS;
      const farDist = camDist + GALAXY_RADIUS + SPHERE_RADIUS;
      const range = farDist - nearDist || 1;
      const depthFactor = (dist - nearDist) / range;
      const clamped = Math.max(0, Math.min(1, depthFactor));
      const penalty = 1 + clamped * clamped * 15;
      return Math.sqrt(screenDistSq) * penalty;
    }

    // Check all nodes — squared-dist pre-filter avoids sqrt for far nodes
    const threshold = 40 * 40; // 40px radius squared
    for (let i = 0; i < totalNodes; i++) {
      const p = displaced[i];
      const s = project3Dto2D(p.x, p.y, p.z, mvp, W, H);
      if (!s) continue;
      const dx = s.x - mouseX, dy = s.y - mouseY;
      const d2 = dx * dx + dy * dy;
      if (d2 > threshold) continue;
      const sc = score(d2, p.x, p.y, p.z);
      if (sc < bestScore) {
        bestScore = sc;
        const nodeInfo = restPositions[i];
        bestItem = { type: nodeInfo.isTag ? 'tag' : 'thought', nodeIndex: i, screen: s, sphereIdx: nodeInfo.sphereIdx, isBridge: nodeInfo.isBridge };
      }
    }
    _lastBestItem = bestItem;
  }

  if (bestItem) {
    tooltip.style.opacity = '1';
    tooltip.style.left = (bestItem.screen.x + 12) + 'px';
    tooltip.style.top = (bestItem.screen.y - 8) + 'px';

    if (bestItem.type === 'tag') {
      hoveredNodeIndex = bestItem.nodeIndex;
      let tagLabel = 'tag';
      if (bestItem.isBridge) {
        tagLabel = findBridgeLabel(bestItem.nodeIndex);
      } else if (bestItem.sphereIdx >= 0) {
        const si = sphereInfo[bestItem.sphereIdx];
        const localIdx = bestItem.nodeIndex - si.tagStart;
        if (localIdx >= 0 && localIdx < si.tagCount) {
          tagLabel = specialists[bestItem.sphereIdx].tags[localIdx].label;
        }
      }
      tooltipLabel.textContent = tagLabel;
      const sphereName = bestItem.sphereIdx >= 0 ? specialists[bestItem.sphereIdx].name : 'bridge';
      tooltipDetail.textContent = bestItem.isBridge ? 'bridge tag \u00b7 shared between spheres' : `sphere: ${sphereName}`;
      tooltipSource.textContent = '';
      tooltipConnections.innerHTML = '';
    } else {
      hoveredNodeIndex = bestItem.nodeIndex;
      const si = bestItem.sphereIdx;
      const localIdx = bestItem.nodeIndex - sphereInfo[si].thoughtStart;
      const t = specialists[si].thoughts[localIdx];
      if (!t) {
        tooltipLabel.textContent = 'thought';
        tooltipDetail.textContent = '';
        tooltipSource.textContent = '';
        tooltipConnections.innerHTML = '';
      } else {
        const topAffIdx = t.affinities.indexOf(Math.max(...t.affinities));
        const topTag = specialists[si].tags[topAffIdx];

        const daysAgo = Math.floor((Date.now() - t.created_at) / 86400000);
        const recencyStr = daysAgo === 0 ? 'today' : daysAgo === 1 ? '1 day ago' : `${daysAgo}d ago`;

        tooltipLabel.textContent = t.text;
        tooltipDetail.textContent = `access: ${t.access_count} \u00b7 primary: ${topTag ? topTag.label : '?'} \u00b7 ${recencyStr} \u00b7 sphere: ${specialists[si].name}`;
        tooltipSource.textContent = `source: ${t.source_id}`;

        const neighbors = adjacency.get(t.id) || [];
        if (neighbors.length > 0) {
          const maxShow = 3;
          let html = `<div style="color:#8a7a60">${neighbors.length} connection${neighbors.length > 1 ? 's' : ''}:</div>`;
          for (let c = 0; c < Math.min(neighbors.length, maxShow); c++) {
            const edge = allEdges[neighbors[c].edgeIdx];
            const targetThought = thoughtById.get(neighbors[c].target);
            const targetName = targetThought ? targetThought.text : `thought-${neighbors[c].target}`;
            const w = (edge.weight * 100).toFixed(0);
            const reason = edge.reasoning.length > 50 ? edge.reasoning.slice(0, 47) + '...' : edge.reasoning;
            const name = targetName.length > 35 ? targetName.slice(0, 32) + '...' : targetName;
            html += `<div class="edge-item">\u2192 ${name} <span style="color:#6a8a6a">(${w}%)</span></div>`;
            html += `<div style="color:#706858;font-style:italic;margin-left:10px">${reason}</div>`;
          }
          if (neighbors.length > maxShow) {
            html += `<div style="color:#606060">+${neighbors.length - maxShow} more</div>`;
          }
          tooltipConnections.innerHTML = html;
        } else {
          tooltipConnections.innerHTML = '';
        }
      }
    }
  } else {
    hoveredNodeIndex = -1;
    tooltip.style.opacity = '0';
  }
}

// Pre-computed prefix sum: sphereThoughtOffset[si] = sum of thought counts for spheres 0..si-1
// Eliminates the O(n×si) inner accumulation loop in updateHoverAnim
const sphereThoughtOffset = new Int32Array(specialists.length);
for (let s = 1; s < specialists.length; s++) {
  sphereThoughtOffset[s] = sphereThoughtOffset[s - 1] + specialists[s - 1].thoughts.length;
}

export function updateHoverAnim(dt) {
  const hoverLerpSpeed = Math.min(dt * 10, 0.9);
  const hoverActiveTarget = (hoveredNodeIndex !== -1) ? 1.0 : 0.0;
  hoverActiveAnim += (hoverActiveTarget - hoverActiveAnim) * hoverLerpSpeed;

  for (let i = 0; i < totalNodes; i++) {
    let target = 0.0;
    if (i === hoveredNodeIndex) {
      target = 1.0;
    } else if (hoveredNodeIndex !== -1 && restPositions[hoveredNodeIndex] && restPositions[hoveredNodeIndex].isTag && !restPositions[i].isTag) {
      // Hovered node is a tag — check if this thought's primary tag matches
      const si = restPositions[i].sphereIdx;
      if (si >= 0) {
        const localThoughtIdx = i - sphereInfo[si].thoughtStart;
        const globalThoughtIdx = sphereThoughtOffset[si] + localThoughtIdx;
        if (thoughtPrimaryTag[globalThoughtIdx] === hoveredNodeIndex) target = 1.0;
      }
    }
    hoverAnim[i] += (target - hoverAnim[i]) * hoverLerpSpeed;
  }
}
