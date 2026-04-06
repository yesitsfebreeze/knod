// ============================================================
// CAMERA (quaternion orbit)
// ============================================================

import { quatNormalize, quatFromAxisAngle, quatMul, quatToAxes } from './math.js';

export let camDist = 45.0;
export let camDistTarget = 45.0; // smooth zoom target
// Initialize quaternion from the original euler angles (pitch=0.3, yaw=0)
// Rotation: first yaw around Y, then pitch around X
export let camQuat = quatNormalize(quatFromAxisAngle(1, 0, 0, 0.3));

export let mouseX = 0;
export let mouseY = 0;

export function setCamDistTarget(v) { camDistTarget = v; }

let dragging = false;
let lastMX = 0, lastMY = 0;

export function initCamera(canvas) {
  canvas.addEventListener('mousedown', e => {
    dragging = true;
    lastMX = e.clientX;
    lastMY = e.clientY;
  });
  window.addEventListener('mouseup', () => { dragging = false; });
  window.addEventListener('mousemove', e => {
    if (dragging) {
      const dx = -(e.clientX - lastMX) * 0.005;
      const dy = (e.clientY - lastMY) * 0.005;

      // Rotate around world Y for horizontal drag (yaw)
      const qYaw = quatFromAxisAngle(0, 1, 0, dx);
      // Rotate around camera-local X for vertical drag (pitch)
      const { right } = quatToAxes(camQuat);
      const qPitch = quatFromAxisAngle(right[0], right[1], right[2], dy);

      camQuat = quatNormalize(quatMul(qPitch, quatMul(qYaw, camQuat)));
      lastMX = e.clientX;
      lastMY = e.clientY;
    }
    mouseX = e.clientX;
    mouseY = e.clientY;
  });

  canvas.addEventListener('wheel', e => {
    camDistTarget *= 1 + e.deltaY * 0.001;
    camDistTarget = Math.max(5, Math.min(80, camDistTarget));
    e.preventDefault();
  }, { passive: false });
}

export function updateCamDist(dt) {
  camDist += (camDistTarget - camDist) * Math.min(dt * 8, 0.9);
}
