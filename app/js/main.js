// ============================================================
// MAIN — Entry point, render loop, fade-in
// ============================================================

import { initWebGL, gl, resize, initShaders } from './webgl.js';
import { specialists } from './data.js';
import { totalNodes, displaced, restPositions, sphereInfo, bridgeStart } from './geometry.js';
import { mat4Perspective, mat4LookAt, mat4Mul, quatRotateVec3 } from './math.js';
import { updatePhysics, buildStemLines, opacitySettings, lineData, lineBuf, initLineBuf, LINE_STRIDE_EXPORT, dotSizeMul } from './physics.js';
import { camDist, camQuat, mouseX, mouseY, updateCamDist, initCamera } from './camera.js';
import { buildEdgeLines, edgeLineData, edgeLineBuf, initEdgeBuf } from './edges.js';
import { STRIDE, totalPoints, pointData, pointBuf, initPointBuf, glowData, glowBuf, initGlowBuf, glowTagCount, glowNodeIndices, baseNodeColors, BASE_STRIDE, rebakeBaseColors } from './points.js';
import { hoveredNodeIndex, hoverAnim, hoverActiveAnim, updateTooltip, updateHoverAnim } from './hover.js';
import { updateLabels } from './labels.js';
import { initUI } from './ui.js';
import { palette, getOldColor, getRecentColor, buildThemeList, applyTheme, loadSettings, onPaletteChange } from './themes.js';
import { initSettings, DOF_RANGE } from './settings.js';
import { updateLOD, tagFade, thoughtFade } from './lod.js';
import { PROXY_STRIDE, proxyData, proxyBuf, initProxyBuf, updateProxyBuf, updateProxyLabels } from './proxy.js';

async function main() {
  // --- WebGL init ---
  const { canvas } = initWebGL();
  resize(canvas);
  window.addEventListener('resize', () => resize(canvas));

  // --- Load shaders & create programs ---
  const progs = await initShaders();

  // --- Init buffers ---
  initLineBuf();
  initEdgeBuf();
  initPointBuf();
  initGlowBuf();
  initProxyBuf();

  // --- Camera ---
  initCamera(canvas);

  // --- UI ---
  initUI();

  // --- Settings ---
  initSettings();
  buildThemeList();
  if (!loadSettings()) {
    applyTheme('knod');
  }

  // Register palette change hook — rebakes static per-node colors/sizes
  onPaletteChange(() => rebakeBaseColors(palette, getOldColor(), getRecentColor()));
  // Initial bake (palette is now set)
  rebakeBaseColors(palette, getOldColor(), getRecentColor());

  // --- Render state ---
  let lastTime = 0;

  const BPF = 4; // bytes per float
  const STRIDE_BYTES = STRIDE * BPF;
  const LINE_STRIDE = LINE_STRIDE_EXPORT;
  const LINE_STRIDE_BYTES = LINE_STRIDE * BPF;

  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  gl.enable(gl.DEPTH_TEST);
  gl.depthFunc(gl.LEQUAL);

  function render(time) {
    requestAnimationFrame(render);

    const t = time * 0.001;
    const dt = lastTime > 0 ? Math.min(t - lastTime, 0.05) : 0.016;
    lastTime = t;

    // Smooth zoom easing
    updateCamDist(dt);

    // Update physics after camera rotation is set
    updatePhysics(dt, camQuat);

    // Camera eye = quaternion applied to (0, 0, camDist)
    const eyePos = quatRotateVec3(camQuat, [0, 0, camDist]);
    const eyeX = eyePos[0], eyeY = eyePos[1], eyeZ = eyePos[2];
    // Up vector from quaternion
    const upVec = quatRotateVec3(camQuat, [0, 1, 0]);

    // Update per-sphere LOD blend factors
    updateLOD(eyeX, eyeY, eyeZ, dt);

    const aspect = canvas.width / canvas.height;
    const proj = mat4Perspective(Math.PI / 4, aspect, 0.1, 150);
    const view = mat4LookAt([eyeX, eyeY, eyeZ], [0, 0, 0], upVec);
    const mvp = mat4Mul(proj, view);

    // Background from palette base00
    const bg = palette.base00 || [0.039, 0.039, 0.059];
    gl.clearColor(bg[0], bg[1], bg[2], 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    // --- Update hover animation ---
    updateHoverAnim(dt);

    // --- Update point positions to match displaced tips, and apply colors ---
    for (let i = 0; i < totalNodes; i++) {
      const off = i * STRIDE;
      const boff = i * BASE_STRIDE;
      pointData[off]     = displaced[i].x;
      pointData[off + 1] = displaced[i].y;
      pointData[off + 2] = displaced[i].z;

      const h = hoverAnim[i];
      const node = restPositions[i];

      // Use pre-baked base color & size (only recomputed on palette change)
      const baseR    = baseNodeColors[boff];
      const baseG    = baseNodeColors[boff + 1];
      const baseB    = baseNodeColors[boff + 2];
      const baseSize = baseNodeColors[boff + 3];

      pointData[off + 3] = baseR;
      pointData[off + 4] = baseG;
      pointData[off + 5] = baseB;
      // Alpha: hovered nodes go opaque (>1.0 signals shader to bypass dim/dof),
      // non-hovered nodes dim to 50% when something else is hovered
      const dimFactor = 1.0 - (1.0 - h) * hoverActiveAnim * 0.5;
      // LOD fade: tags fade out at LOD_FAR_DIST, thoughts fade out at LOD_MID_DIST
      // Bridge nodes (sphereIdx = -1) are at galaxy level — always visible
      const si = node.sphereIdx;
      const lodFade = (si < 0) ? 1.0 : node.isTag ? tagFade[si] : thoughtFade[si];
      pointData[off + 6] = (dimFactor + h) * lodFade;
      // Size: apply dot size multiplier — at 0 the dots vanish entirely
      pointData[off + 7] = baseSize * (1.0 + h * 0.75) * dotSizeMul;
    }

    // --- DOF_RANGE needs to be read fresh each frame (it's mutable from settings) ---
    // We import the module-level DOF_RANGE but it's a let, so we re-read via the module
    // Actually DOF_RANGE is exported as a let from settings.js, which means the import
    // binding stays live in ES modules. So we can use it directly.

    // --- Dot depth pre-pass: stamp spherical depth into depth buffer (no color) ---
    gl.colorMask(false, false, false, false);
    gl.depthMask(true);
    gl.enable(gl.DEPTH_TEST);

    gl.useProgram(progs.depthProg);
    gl.uniformMatrix4fv(progs.uMVP_depth, false, mvp);
    gl.uniform1f(progs.uScale_depth, canvas.height * 0.15);
    gl.uniform3f(progs.uEye_depth, eyeX, eyeY, eyeZ);
    gl.uniform1f(progs.uDofRange_depth, DOF_RANGE);
    gl.uniform1f(progs.uNear_depth, 0.1);
    gl.uniform1f(progs.uFar_depth, 150.0);

    gl.bindBuffer(gl.ARRAY_BUFFER, pointBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, pointData);
    gl.enableVertexAttribArray(progs.aPos_depth);
    gl.enableVertexAttribArray(progs.aColor_depth);
    gl.enableVertexAttribArray(progs.aSize_depth);
    gl.vertexAttribPointer(progs.aPos_depth, 3, gl.FLOAT, false, STRIDE_BYTES, 0);
    gl.vertexAttribPointer(progs.aColor_depth, 4, gl.FLOAT, false, STRIDE_BYTES, 3 * BPF);
    gl.vertexAttribPointer(progs.aSize_depth, 1, gl.FLOAT, false, STRIDE_BYTES, 7 * BPF);
    gl.drawArrays(gl.POINTS, 0, totalPoints);
    gl.disableVertexAttribArray(progs.aPos_depth);
    gl.disableVertexAttribArray(progs.aColor_depth);
    gl.disableVertexAttribArray(progs.aSize_depth);

    gl.colorMask(true, true, true, true);

    // --- Build and draw tag→thought stem lines ---
    const totalLineVerts = buildStemLines(eyeX, eyeY, eyeZ, hoverAnim, hoverActiveAnim, thoughtFade);

    gl.useProgram(progs.lineProg);
    gl.uniformMatrix4fv(progs.uMVP_lines, false, mvp);
    gl.uniform3f(progs.uEye_lines, eyeX, eyeY, eyeZ);
    gl.uniform1f(progs.uDofRange_lines, DOF_RANGE);
    gl.depthMask(false); // don't let lines write depth — dots own the depth buffer

    gl.bindBuffer(gl.ARRAY_BUFFER, lineBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, lineData);
    gl.enableVertexAttribArray(progs.aPos_lines);
    gl.enableVertexAttribArray(progs.aColor_lines);
    gl.vertexAttribPointer(progs.aPos_lines, 3, gl.FLOAT, false, LINE_STRIDE_BYTES, 0);
    gl.vertexAttribPointer(progs.aColor_lines, 4, gl.FLOAT, false, LINE_STRIDE_BYTES, 3 * BPF);
    gl.drawArrays(gl.LINES, 0, totalLineVerts);
    gl.disableVertexAttribArray(progs.aPos_lines);
    gl.disableVertexAttribArray(progs.aColor_lines);

    // --- Build and draw inter-thought edges (straight lines) ---
    const totalEdgeVerts = buildEdgeLines(eyeX, eyeY, eyeZ, hoverAnim, hoverActiveAnim, thoughtFade);
    if (totalEdgeVerts > 0) {
      gl.bindBuffer(gl.ARRAY_BUFFER, edgeLineBuf);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, edgeLineData);
      gl.enableVertexAttribArray(progs.aPos_lines);
      gl.enableVertexAttribArray(progs.aColor_lines);
      gl.vertexAttribPointer(progs.aPos_lines, 3, gl.FLOAT, false, LINE_STRIDE_BYTES, 0);
      gl.vertexAttribPointer(progs.aColor_lines, 4, gl.FLOAT, false, LINE_STRIDE_BYTES, 3 * BPF);
      gl.drawArrays(gl.LINES, 0, totalEdgeVerts);
      gl.disableVertexAttribArray(progs.aPos_lines);
      gl.disableVertexAttribArray(progs.aColor_lines);
    }

    // --- Draw dots (color pass) — additive blend (no depth sort needed) ---
    gl.useProgram(progs.pointProg);
    gl.uniformMatrix4fv(progs.uMVP_points, false, mvp);
    gl.uniform1f(progs.uScale_points, canvas.height * 0.15);
    gl.uniform3f(progs.uEye_points, eyeX, eyeY, eyeZ);
    gl.uniform1f(progs.uDofRange_points, DOF_RANGE);

    gl.blendFunc(gl.SRC_ALPHA, gl.ONE); // additive — no sort needed
    gl.bindBuffer(gl.ARRAY_BUFFER, pointBuf);
    gl.enableVertexAttribArray(progs.aPos_points);
    gl.enableVertexAttribArray(progs.aColor_points);
    gl.enableVertexAttribArray(progs.aSize_points);
    gl.vertexAttribPointer(progs.aPos_points, 3, gl.FLOAT, false, STRIDE_BYTES, 0);
    gl.vertexAttribPointer(progs.aColor_points, 4, gl.FLOAT, false, STRIDE_BYTES, 3 * BPF);
    gl.vertexAttribPointer(progs.aSize_points, 1, gl.FLOAT, false, STRIDE_BYTES, 7 * BPF);

    gl.disable(gl.DEPTH_TEST);
    gl.drawArrays(gl.POINTS, 0, totalPoints);

    gl.disableVertexAttribArray(progs.aPos_points);
    gl.disableVertexAttribArray(progs.aColor_points);
    gl.disableVertexAttribArray(progs.aSize_points);

    // --- Draw glow halos on tag dots (additive blending) ---
    const glowOp = opacitySettings.glow;
    // Build glow data using pre-built tag index list (avoids full 5560-node scan)
    for (let gi = 0; gi < glowTagCount; gi++) {
      const i = glowNodeIndices[gi];
      const off = gi * STRIDE;
      const pOff = i * STRIDE;
      glowData[off]     = pointData[pOff];
      glowData[off + 1] = pointData[pOff + 1];
      glowData[off + 2] = pointData[pOff + 2];
      glowData[off + 3] = pointData[pOff + 3]; // color r
      glowData[off + 4] = pointData[pOff + 4]; // color g
      glowData[off + 5] = pointData[pOff + 5]; // color b
      glowData[off + 6] = pointData[pOff + 6] * glowOp; // color a with glow opacity
      glowData[off + 7] = pointData[pOff + 7] * 2.5; // 2.5x size for glow halo
    }

    gl.blendFunc(gl.SRC_ALPHA, gl.ONE); // additive blending for glow
    gl.useProgram(progs.glowProg);
    gl.uniformMatrix4fv(progs.uMVP_glow, false, mvp);
    gl.uniform1f(progs.uScale_glow, canvas.height * 0.15);
    gl.uniform3f(progs.uEye_glow, eyeX, eyeY, eyeZ);
    gl.uniform1f(progs.uDofRange_glow, DOF_RANGE);

    gl.bindBuffer(gl.ARRAY_BUFFER, glowBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, glowData);
    gl.enableVertexAttribArray(progs.aPos_glow);
    gl.enableVertexAttribArray(progs.aColor_glow);
    gl.enableVertexAttribArray(progs.aSize_glow);
    gl.vertexAttribPointer(progs.aPos_glow, 3, gl.FLOAT, false, STRIDE_BYTES, 0);
    gl.vertexAttribPointer(progs.aColor_glow, 4, gl.FLOAT, false, STRIDE_BYTES, 3 * BPF);
    gl.vertexAttribPointer(progs.aSize_glow, 1, gl.FLOAT, false, STRIDE_BYTES, 7 * BPF);
    gl.drawArrays(gl.POINTS, 0, glowTagCount);
    gl.disableVertexAttribArray(progs.aPos_glow);
    gl.disableVertexAttribArray(progs.aColor_glow);
    gl.disableVertexAttribArray(progs.aSize_glow);

    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA); // restore normal blending
    gl.depthMask(true);
    gl.enable(gl.DEPTH_TEST);

    // --- Draw LOD proxy dots (one per specialist, visible when sphere is far) ---
    updateProxyBuf(palette);
    const PROXY_STRIDE_BYTES = PROXY_STRIDE * BPF;
    gl.useProgram(progs.pointProg);
    gl.uniformMatrix4fv(progs.uMVP_points, false, mvp);
    gl.uniform1f(progs.uScale_points, canvas.height * 0.15);
    gl.uniform3f(progs.uEye_points, eyeX, eyeY, eyeZ);
    gl.uniform1f(progs.uDofRange_points, DOF_RANGE);

    gl.bindBuffer(gl.ARRAY_BUFFER, proxyBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, proxyData);
    gl.enableVertexAttribArray(progs.aPos_points);
    gl.enableVertexAttribArray(progs.aColor_points);
    gl.enableVertexAttribArray(progs.aSize_points);
    gl.vertexAttribPointer(progs.aPos_points, 3, gl.FLOAT, false, PROXY_STRIDE_BYTES, 0);
    gl.vertexAttribPointer(progs.aColor_points, 4, gl.FLOAT, false, PROXY_STRIDE_BYTES, 3 * BPF);
    gl.vertexAttribPointer(progs.aSize_points, 1, gl.FLOAT, false, PROXY_STRIDE_BYTES, 7 * BPF);
    gl.disable(gl.DEPTH_TEST);
    gl.drawArrays(gl.POINTS, 0, specialists.length);
    gl.disableVertexAttribArray(progs.aPos_points);
    gl.disableVertexAttribArray(progs.aColor_points);
    gl.disableVertexAttribArray(progs.aSize_points);
    gl.enable(gl.DEPTH_TEST);

    // --- Labels follow displaced tag positions (suppressed at far LOD by tagFade) ---
    updateLabels(mvp, eyeX, eyeY, eyeZ, tagFade);
    updateProxyLabels(mvp, palette);
    updateTooltip(mvp, eyeX, eyeY, eyeZ);
  }

  // --- Fade in on load ---
  setTimeout(() => {
    const overlay = document.getElementById('fade-overlay');
    if (overlay) overlay.classList.add('done');
  }, 100);
  // Remove overlay from DOM after transition
  setTimeout(() => {
    const overlay = document.getElementById('fade-overlay');
    if (overlay) overlay.remove();
  }, 2200);

  requestAnimationFrame(render);
}

main().catch(err => {
  console.error('Failed to initialize:', err);
  document.body.innerHTML = `<p style="padding:40px;color:#f88">Init failed: ${err.message}</p>`;
});
