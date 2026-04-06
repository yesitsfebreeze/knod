// ============================================================
// LOD PROXY — One dot per specialist shown when sphere is far away
// ============================================================

import { gl } from './webgl.js';
import { sphereCenters } from './geometry.js';
import { specialists } from './data.js';
import { tagFade } from './lod.js';

export const PROXY_STRIDE = 8; // x,y,z, r,g,b,a, size
export const proxyData = new Float32Array(specialists.length * PROXY_STRIDE);
export let proxyBuf;

export function initProxyBuf() {
  proxyBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, proxyBuf);
  gl.bufferData(gl.ARRAY_BUFFER, proxyData.byteLength, gl.DYNAMIC_DRAW);
}

// Update proxy buffer: position = sphere center, alpha = 1 - tagFade (visible only when tags invisible)
export function updateProxyBuf(palette) {
  const proxyColor = palette.base0C || [0.52, 0.76, 0.72];

  for (let si = 0; si < specialists.length; si++) {
    const off = si * PROXY_STRIDE;
    const c = sphereCenters[si];
    proxyData[off]     = c.x;
    proxyData[off + 1] = c.y;
    proxyData[off + 2] = c.z;
    proxyData[off + 3] = proxyColor[0];
    proxyData[off + 4] = proxyColor[1];
    proxyData[off + 5] = proxyColor[2];
    // Alpha = inverse of tagFade — proxy appears as tags fade out
    proxyData[off + 6] = 1.0 - tagFade[si];
    // Size: slightly larger than a tag dot so it's clearly a summary node
    proxyData[off + 7] = 1.4;
  }
}

// --- Proxy labels (specialist names) ---
// These appear when the sphere is at LOD 0 (far away).
const proxyLabelContainer = document.createElement('div');
proxyLabelContainer.id = 'proxy-label-container';
proxyLabelContainer.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:490;';
document.body.appendChild(proxyLabelContainer);

export const proxyLabelEls = [];

for (const sp of specialists) {
  const el = document.createElement('div');
  el.style.cssText = `
    position: absolute;
    font-size: 10px;
    font-weight: 500;
    color: rgba(180, 210, 255, 0.95);
    white-space: nowrap;
    transform: translate(-50%, -50%);
    padding: 3px 10px;
    border-radius: 12px;
    background: rgba(8, 10, 22, 0.70);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(120, 170, 255, 0.20);
    letter-spacing: 0.5px;
    pointer-events: none;
  `;
  el.textContent = sp.name;
  proxyLabelContainer.appendChild(el);
  proxyLabelEls.push(el);
}

// Project a 3D point to 2D screen coords
function project(x, y, z, mvp, W, H) {
  const cx = mvp[0]*x + mvp[4]*y + mvp[8]*z  + mvp[12];
  const cy = mvp[1]*x + mvp[5]*y + mvp[9]*z  + mvp[13];
  const cw = mvp[3]*x + mvp[7]*y + mvp[11]*z + mvp[15];
  if (cw <= 0) return null;
  return {
    x: (cx/cw * 0.5 + 0.5) * W,
    y: (1 - (cy/cw * 0.5 + 0.5)) * H,
  };
}

export function updateProxyLabels(mvp, palette) {
  const W = window.innerWidth, H = window.innerHeight;
  // Derive text color from palette
  const b0D = palette.base0D;
  const colorStr = b0D
    ? `rgba(${Math.round(b0D[0]*255)},${Math.round(b0D[1]*255)},${Math.round(b0D[2]*255)},0.95)`
    : 'rgba(180,210,255,0.95)';

  for (let si = 0; si < specialists.length; si++) {
    const el = proxyLabelEls[si];
    const c = sphereCenters[si];
    const fade = 1.0 - tagFade[si]; // 0 = sphere is near (hidden), 1 = sphere is far (visible)

    if (fade < 0.01) {
      el.style.display = 'none';
      continue;
    }

    const s = project(c.x, c.y, c.z, mvp, W, H);
    if (!s || s.x < -60 || s.x > W + 60 || s.y < -20 || s.y > H + 20) {
      el.style.display = 'none';
      continue;
    }

    el.style.display = 'block';
    el.style.left = s.x + 'px';
    el.style.top = (s.y + 16) + 'px'; // just below the dot
    el.style.opacity = fade.toFixed(3);
    el.style.color = colorStr;
  }
}

