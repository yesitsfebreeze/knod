// ============================================================
// LABEL RENDERING (HTML overlay for tag labels)
// ============================================================

import { specialists, GLOBAL_TAGS } from './data.js';
import { displaced, sphereInfo, bridgeNodes, totalNodes, restPositions } from './geometry.js';
import { opacitySettings } from './physics.js';
import { project3Dto2D } from './hover.js';

const labelContainer = document.createElement('div');
labelContainer.id = 'label-container';
labelContainer.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:500;';
document.body.appendChild(labelContainer);

// Create labels for all tags across all spheres + bridge nodes
export const labelEls = [];
const labelNodeIndices = []; // which node index each label corresponds to

for (let si = 0; si < specialists.length; si++) {
  const sp = specialists[si];
  const info = sphereInfo[si];
  for (let ti = 0; ti < info.tagCount; ti++) {
    const el = document.createElement('div');
    el.style.cssText = `
      position: absolute;
      font-size: 9px;
      color: rgba(160, 195, 255, 0.9);
      white-space: nowrap;
      transform: translate(-50%, -50%);
      padding: 2px 8px;
      border-radius: 10px;
      background: rgba(8, 10, 22, 0.55);
      backdrop-filter: blur(8px);
      -webkit-backdrop-filter: blur(8px);
      border: 1px solid rgba(100, 150, 255, 0.12);
      box-shadow: 0 0 8px rgba(60, 100, 200, 0.08), inset 0 0 6px rgba(80, 120, 255, 0.04);
      letter-spacing: 0.3px;
    `;
    el.textContent = sp.tags[ti].label;
    labelContainer.appendChild(el);
    labelEls.push(el);
    labelNodeIndices.push(info.tagStart + ti);
  }
}

// Bridge tag labels
for (const bn of bridgeNodes) {
  const el = document.createElement('div');
  el.style.cssText = `
    position: absolute;
    font-size: 9px;
    color: rgba(160, 195, 255, 0.9);
    white-space: nowrap;
    transform: translate(-50%, -50%);
    padding: 2px 8px;
    border-radius: 10px;
    background: rgba(8, 10, 22, 0.55);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid rgba(100, 150, 255, 0.12);
    box-shadow: 0 0 8px rgba(60, 100, 200, 0.08), inset 0 0 6px rgba(80, 120, 255, 0.04);
    letter-spacing: 0.3px;
  `;
  el.textContent = bn.label;
  labelContainer.appendChild(el);
  labelEls.push(el);
  labelNodeIndices.push(bn.nodeIndex);
}

// Per-label last-written value cache — avoids DOM writes when unchanged
const _labelCache = [];
for (let i = 0; i < labelEls.length; i++) _labelCache.push({ left: '', top: '', opacity: '', display: '' });

export function updateLabels(mvp, eyeX, eyeY, eyeZ, tagFade) {
  const W = window.innerWidth;
  const H = window.innerHeight;
  const eLen = Math.sqrt(eyeX*eyeX + eyeY*eyeY + eyeZ*eyeZ) || 1;
  const ex = eyeX/eLen, ey = eyeY/eLen, ez = eyeZ/eLen;
  for (let i = 0; i < labelEls.length; i++) {
    const nodeIdx = labelNodeIndices[i];
    const p = displaced[nodeIdx]; // follow displaced position
    const s = project3Dto2D(p.x, p.y, p.z, mvp, W, H);
    const el = labelEls[i];
    const cache = _labelCache[i];
    if (s && s.x > -50 && s.x < W + 50 && s.y > -50 && s.y < H + 50) {
      const leftVal = s.x + 'px';
      const topVal = (s.y - 14) + 'px';
      const pLen = Math.sqrt(p.x*p.x + p.y*p.y + p.z*p.z) || 1;
      const facing = (p.x*ex + p.y*ey + p.z*ez) / pLen;
      const facingFade = Math.max(0, Math.min(1, (facing + 1) / 1.6)) * 0.92 + 0.08;
      const node = restPositions[nodeIdx];
      const si = node ? node.sphereIdx : -1;
      const lodFade = (tagFade && si >= 0) ? tagFade[si] : 1.0;
      const opVal = (facingFade * lodFade * opacitySettings.labels).toFixed(3);
      if (cache.display !== 'block') { el.style.display = 'block'; cache.display = 'block'; }
      if (cache.left !== leftVal)    { el.style.left = leftVal;     cache.left = leftVal; }
      if (cache.top !== topVal)      { el.style.top = topVal;       cache.top = topVal; }
      if (cache.opacity !== opVal)   { el.style.opacity = opVal;    cache.opacity = opVal; }
    } else {
      if (cache.display !== 'none') { el.style.display = 'none'; cache.display = 'none'; }
    }
  }
}
