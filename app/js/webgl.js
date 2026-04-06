// ============================================================
// WEBGL INITIALIZATION & SHADER MANAGEMENT
// ============================================================

export let gl;
export let extFragDepth;

export function initWebGL() {
  const canvas = document.getElementById('c');
  gl = canvas.getContext('webgl', { antialias: true, alpha: false });

  if (!gl) {
    document.body.innerHTML = '<p style="padding:40px">WebGL not available</p>';
    throw new Error('No WebGL');
  }

  // Enable fragment depth writing for spherical depth pre-pass
  extFragDepth = gl.getExtension('EXT_frag_depth');
  if (!extFragDepth) console.warn('EXT_frag_depth not supported — node depth will be flat');

  return { canvas, gl };
}

export function resize(canvas) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = window.innerWidth * dpr;
  canvas.height = window.innerHeight * dpr;
  canvas.style.width = window.innerWidth + 'px';
  canvas.style.height = window.innerHeight + 'px';
  gl.viewport(0, 0, canvas.width, canvas.height);
}

// --- Shader loading ---
export async function loadShaderSource(url) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Failed to load shader: ${url}`);
  return resp.text();
}

export function compileShader(src, type) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    console.error(gl.getShaderInfoLog(s));
    return null;
  }
  return s;
}

export function linkProgram(vsSrc, fsSrc) {
  const p = gl.createProgram();
  gl.attachShader(p, compileShader(vsSrc, gl.VERTEX_SHADER));
  gl.attachShader(p, compileShader(fsSrc, gl.FRAGMENT_SHADER));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    console.error(gl.getProgramInfoLog(p));
    return null;
  }
  return p;
}

// Load all shaders and create programs
export async function initShaders() {
  const [pointsVert, pointsFrag, glowFrag, depthFrag, linesVert, linesFrag] = await Promise.all([
    loadShaderSource('shaders/points.vert'),
    loadShaderSource('shaders/points.frag'),
    loadShaderSource('shaders/glow.frag'),
    loadShaderSource('shaders/depth.frag'),
    loadShaderSource('shaders/lines.vert'),
    loadShaderSource('shaders/lines.frag'),
  ]);

  const pointProg = linkProgram(pointsVert, pointsFrag);
  const glowProg = linkProgram(pointsVert, glowFrag);
  const depthProg = linkProgram(pointsVert, depthFrag);
  const lineProg = linkProgram(linesVert, linesFrag);

  return {
    pointProg,
    glowProg,
    depthProg,
    lineProg,
    // Point program uniforms/attribs
    uMVP_points: gl.getUniformLocation(pointProg, 'uMVP'),
    uScale_points: gl.getUniformLocation(pointProg, 'uScale'),
    uEye_points: gl.getUniformLocation(pointProg, 'uEye'),
    uDofRange_points: gl.getUniformLocation(pointProg, 'uDofRange'),
    aPos_points: gl.getAttribLocation(pointProg, 'aPos'),
    aColor_points: gl.getAttribLocation(pointProg, 'aColor'),
    aSize_points: gl.getAttribLocation(pointProg, 'aSize'),
    // Depth program uniforms/attribs
    uMVP_depth: gl.getUniformLocation(depthProg, 'uMVP'),
    uScale_depth: gl.getUniformLocation(depthProg, 'uScale'),
    uEye_depth: gl.getUniformLocation(depthProg, 'uEye'),
    uDofRange_depth: gl.getUniformLocation(depthProg, 'uDofRange'),
    uNear_depth: gl.getUniformLocation(depthProg, 'uNear'),
    uFar_depth: gl.getUniformLocation(depthProg, 'uFar'),
    aPos_depth: gl.getAttribLocation(depthProg, 'aPos'),
    aColor_depth: gl.getAttribLocation(depthProg, 'aColor'),
    aSize_depth: gl.getAttribLocation(depthProg, 'aSize'),
    // Glow program uniforms/attribs
    uMVP_glow: gl.getUniformLocation(glowProg, 'uMVP'),
    uScale_glow: gl.getUniformLocation(glowProg, 'uScale'),
    uEye_glow: gl.getUniformLocation(glowProg, 'uEye'),
    uDofRange_glow: gl.getUniformLocation(glowProg, 'uDofRange'),
    aPos_glow: gl.getAttribLocation(glowProg, 'aPos'),
    aColor_glow: gl.getAttribLocation(glowProg, 'aColor'),
    aSize_glow: gl.getAttribLocation(glowProg, 'aSize'),
    // Line program uniforms/attribs
    uMVP_lines: gl.getUniformLocation(lineProg, 'uMVP'),
    uEye_lines: gl.getUniformLocation(lineProg, 'uEye'),
    uDofRange_lines: gl.getUniformLocation(lineProg, 'uDofRange'),
    aPos_lines: gl.getAttribLocation(lineProg, 'aPos'),
    aColor_lines: gl.getAttribLocation(lineProg, 'aColor'),
  };
}
