// knod — WebGPU 3D polygon cloud visualization
// Tags form vertices of a 3D convex polyhedron (offset on Y so even
// a triangle becomes volumetric). Thoughts cluster in sectors between
// tag vertices. Inertial orbit: Apple-style sphere rotation.

// ---- palette (linear RGB) ----
const PALETTE = [
  [0.482, 0.549, 1.0],
  [1.0, 0.420, 0.616],
  [0.306, 0.804, 0.769],
  [1.0, 0.722, 0.424],
  [0.741, 0.576, 0.976],
  [0.314, 0.980, 0.482],
  [1.0, 0.475, 0.776],
  [0.545, 0.914, 0.992],
  [0.945, 0.980, 0.549],
  [0.384, 0.447, 0.643],
];
const TAG_CLR = [1.0, 0.851, 0.239];
const CTR_CLR = [0.655, 0.545, 0.984];
const POLY_R = 1.8;
const FOV = 65 * Math.PI / 180;
const ZOOM_IN = 0.84;
const ZOOM_OUT = 1.12;
const FRAME_PAD = 1.00;
const FRAME_MARGIN_PX = 48;
const DEFAULT_VIEW = 0.90;
const FOCUS_LERP = 0.18;

// ---- mat4 helpers ----
function m4() { const m = new Float32Array(16); m[0] = m[5] = m[10] = m[15] = 1; return m; }
function m4Mul(a, b) {
  // column-major: C[col*4+row] = sum_k A[k*4+row]*B[col*4+k]
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let s = 0; for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k]; o[c * 4 + r] = s;
  } return o;
}
function m4RotX(a) { const m = m4(), c = Math.cos(a), s = Math.sin(a); m[5] = c; m[6] = s; m[9] = -s; m[10] = c; return m; }
function m4RotY(a) { const m = m4(), c = Math.cos(a), s = Math.sin(a); m[0] = c; m[2] = -s; m[8] = s; m[10] = c; return m; }
function m4Persp(fov, asp, n, f) {
  // WebGPU clip z ∈ [0,1]
  const t = 1 / Math.tan(fov / 2), m = new Float32Array(16);
  m[0] = t / asp; m[5] = t; m[10] = f / (n - f); m[11] = -1; m[14] = n * f / (n - f); return m;
}
function m4Trans(x, y, z) { const m = m4(); m[12] = x; m[13] = y; m[14] = z; return m; }

// ---- state ----
const cvs = document.getElementById("canvas");
let W, H, dpr;
let allNodes = [], tagNodes = [], thoughtNodes = [], edges = [];
let nodeIdx = {}, nbrs = {};
let storeClr = {}, clrI = 0;

// orbit
let yaw = 0.3, pitch = -0.5, dist = 3;
let vYaw = 0.002, vPitch = 0;
const DAMP = 0.97, SENS = 0.005;
let drag = false, lmx = 0, lmy = 0;
let graphCenter = [0, 0, 0], orbitCenter = [0, 0, 0], orbitTarget = [0, 0, 0], graphRadius = POLY_R, minDist = 0.7, baseMinDist = 0.7, maxDist = 6, resetDist = 3;

// pick
let hoveredNode = null, hoveredEdge = null, selected = null, projected = [], edgeProjected = [];

// overlay canvas for labels/tooltips
const ov = document.createElement("canvas");
ov.style.cssText = "position:absolute;inset:0;z-index:1;pointer-events:none;";
document.body.appendChild(ov);
const oc = ov.getContext("2d");

function sClr(s) { if (!storeClr[s]) storeClr[s] = PALETTE[clrI++ % PALETTE.length]; return storeClr[s]; }
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
function lerp(a, b, t) { return a + (b - a) * t; }
function mixColor(a, b, t) {
  return [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
}
function rgba(color, alpha = 1) {
  return `rgba(${Math.round(color[0] * 255)}, ${Math.round(color[1] * 255)}, ${Math.round(color[2] * 255)}, ${alpha})`;
}
function hash01(str) {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0) / 4294967295;
}
function softScore(value, maxValue) {
  if (maxValue <= 0) return 0;
  return Math.sqrt(clamp(value / maxValue, 0, 1));
}
function edgeStrength(e) {
  const traversalBoost = clamp(Math.log1p(e.traversals || 0) / Math.log(12), 0, 1);
  return clamp((e.w || 0) * 0.55 + (e.success || 0) * 0.25 + traversalBoost * 0.20, 0, 1);
}
function truncateText(text, max = 120) {
  const value = (text || "").trim();
  return value.length > max ? value.slice(0, max - 1) + "\u2026" : value;
}
function getFocusState() {
  if (!selected) return null;

  const scores = new Map([[selected.key, 1]]);
  let min = Infinity, max = -Infinity;

  for (const e of edges) {
    if (e.s !== selected && e.t !== selected) continue;
    const other = e.s === selected ? e.t : e.s;
    const weight = clamp(e.w || 0, 0, 1);
    const score = clamp((other.importance || 0) * 0.65 + weight * 0.35, 0, 1);
    scores.set(other.key, score);
    min = Math.min(min, score);
    max = Math.max(max, score);
  }

  if (min === Infinity) { min = 0; max = 1; }
  return { selectedKey: selected.key, scores, min, max };
}
function normalizeFocusScore(score, focus) {
  if (!focus) return 0;
  const range = Math.max(focus.max - focus.min, 0.001);
  return clamp((score - focus.min) / range, 0, 1);
}
function getNodeVisual(n, focus) {
  const baseAlpha = n.kind === "tag" ? 0.85 : 0.55;
  if (!focus) return { color: n.color, alpha: baseAlpha, scale: 1 };
  if (n.key === focus.selectedKey) {
    return { color: mixColor(n.color, TAG_CLR, 0.55), alpha: 1.0, scale: 1.35 };
  }

  const score = focus.scores.get(n.key);
  if (score === undefined) {
    return { color: mixColor(n.color, [0.18, 0.2, 0.28], 0.78), alpha: n.kind === "tag" ? 0.14 : 0.07, scale: 0.92 };
  }

  const t = normalizeFocusScore(score, focus);
  const accent = mixColor([0.384, 0.447, 0.643], TAG_CLR, t);
  return {
    color: mixColor(n.color, accent, 0.35 + 0.4 * t),
    alpha: 0.30 + 0.55 * t,
    scale: 1 + 0.18 * t,
  };
}
function getEdgeVisual(e, focus) {
  const strength = edgeStrength(e);
  if (!focus) {
    const alpha = e.knn ? 0.12 + strength * 0.24 : 0.12 + strength * 0.34;
    const color = e.knn
      ? mixColor([0.306, 0.804, 0.769], [0.545, 0.914, 0.992], 0.28 + (e.success || 0) * 0.2)
      : mixColor([0.482, 0.549, 1.0], TAG_CLR, 0.10 + (e.success || 0) * 0.45);
    return { color, alpha };
  }

  const connected = e.s.key === focus.selectedKey || e.t.key === focus.selectedKey;
  if (!connected) return { color: [0.16, 0.18, 0.24], alpha: 0.03 };

  const other = e.s.key === focus.selectedKey ? e.t : e.s;
  const score = focus.scores.get(other.key) ?? 0;
  const t = normalizeFocusScore(score, focus);
  return {
    color: mixColor([0.306, 0.804, 0.769], TAG_CLR, clamp(0.15 + 0.55 * t + (e.success || 0) * 0.2, 0, 1)),
    alpha: 0.18 + 0.50 * t + strength * 0.20 + (e.knn ? 0.02 : 0.05),
  };
}
function getEdgeMarkerVisual(e, focus) {
  const strength = edgeStrength(e);
  if (!focus) {
    return {
      radius: 1.8 + strength * 2.2 + (e.knn ? 0 : 0.25),
      alpha: 0.16 + strength * 0.28,
      lineWidth: 0.9 + strength * 0.45,
      color: e.knn
        ? mixColor([0.306, 0.804, 0.769], [0.545, 0.914, 0.992], 0.35)
        : mixColor([0.482, 0.549, 1.0], TAG_CLR, 0.18 + (e.success || 0) * 0.4),
    };
  }

  const connected = e.s.key === focus.selectedKey || e.t.key === focus.selectedKey;
  if (!connected) return { radius: 1.5 + strength * 1.2, alpha: 0.06, lineWidth: 0.8, color: [0.18, 0.2, 0.28] };

  const other = e.s.key === focus.selectedKey ? e.t : e.s;
  const t = normalizeFocusScore(focus.scores.get(other.key) ?? 0, focus);
  return {
    radius: 2.1 + strength * 2.4 + t * 0.9,
    alpha: 0.24 + t * 0.34,
    lineWidth: 1.0 + strength * 0.5 + t * 0.35,
    color: mixColor([0.306, 0.804, 0.769], TAG_CLR, 0.30 + t * 0.45),
  };
}
function isSpecialistNode(raw) {
  const source = (raw?.source || "").toLowerCase();
  const label = (raw?.label || "").toLowerCase();
  return source.startsWith("specialist:") || label.includes("[specialist:");
}
function setOrbitTarget(pos, snap = false) {
  orbitTarget = [pos[0], pos[1], pos[2]];
  if (snap) orbitCenter = [pos[0], pos[1], pos[2]];
}
function syncOrbitTarget(snap = false) {
  const pos = selected?.pos || graphCenter;
  setOrbitTarget(pos, snap);
}
function rotateIntoCamera(point, center, yawAngle=yaw, pitchAngle=pitch) {
  const px=point[0]-center[0], py=point[1]-center[1], pz=point[2]-center[2];

  const cy=Math.cos(-yawAngle), sy=Math.sin(-yawAngle);
  const yawX=px*cy + pz*sy;
  const yawY=py;
  const yawZ=-px*sy + pz*cy;

  const cx=Math.cos(-pitchAngle), sx=Math.sin(-pitchAngle);
  return [
    yawX,
    yawY*cx - yawZ*sx,
    yawY*sx + yawZ*cx,
  ];
}
function getNodeExtent(n) {
  return Math.max(n.r*2.0*(n.sizeMul||0.1), n.kind==="tag" ? 0.10 : 0.04) * FRAME_PAD;
}
function getProjectedBounds(center, distance, yawAngle=yaw, pitchAngle=pitch) {
  const aspect=Math.max(W/Math.max(H, 1), 0.1);
  const tanHalf=Math.tan(FOV/2);
  const margin=Math.min(FRAME_MARGIN_PX, Math.min(W, H)*0.18);
  const bounds={minX:Infinity, minY:Infinity, maxX:-Infinity, maxY:-Infinity, maxExtent:0.6, visible:true, margin};

  for (const n of allNodes) {
    const rr=getNodeExtent(n);
    const [rx, ry, rz]=rotateIntoCamera(n.pos, center, yawAngle, pitchAngle);
    const w=distance-rz;
    if (w <= 0.001) {
      bounds.visible=false;
      return bounds;
    }
    const ndcX=rx/(w*tanHalf*aspect);
    const ndcY=ry/(w*tanHalf);
    const sx=(ndcX*0.5+0.5)*W;
    const sy=(1-(ndcY*0.5+0.5))*H;
    const radiusPx=Math.max((rr/(w*tanHalf*aspect))*W*0.5, (rr/(w*tanHalf))*H*0.5);

    bounds.minX=Math.min(bounds.minX, sx-radiusPx);
    bounds.maxX=Math.max(bounds.maxX, sx+radiusPx);
    bounds.minY=Math.min(bounds.minY, sy-radiusPx);
    bounds.maxY=Math.max(bounds.maxY, sy+radiusPx);
    bounds.maxExtent=Math.max(bounds.maxExtent, Math.abs(rx)+rr, Math.abs(ry)+rr, Math.abs(rz)+rr);
  }

  return bounds;
}
function fitsViewport(center, distance, yawAngle=yaw, pitchAngle=pitch) {
  const bounds=getProjectedBounds(center, distance, yawAngle, pitchAngle);
  if (!bounds.visible) return {fits:false, bounds};
  const fits=
    bounds.minX >= bounds.margin &&
    bounds.maxX <= W-bounds.margin &&
    bounds.minY >= bounds.margin &&
    bounds.maxY <= H-bounds.margin;
  return {fits, bounds};
}
function fitDistanceForBounds(center, yawAngle=yaw, pitchAngle=pitch) {
  if (!allNodes.length) return 0.8;

  let low=0.05;
  let high=1.0;
  let probe=fitsViewport(center, high, yawAngle, pitchAngle);
  let expandCount=0;
  while (!probe.fits && expandCount < 24) {
    low=high;
    high*=1.6;
    probe=fitsViewport(center, high, yawAngle, pitchAngle);
    expandCount+=1;
  }

  let lastBounds=probe.bounds;
  for (let i=0;i<28;i++) {
    const mid=(low+high)*0.5;
    const test=fitsViewport(center, mid, yawAngle, pitchAngle);
    lastBounds=test.bounds;
    if (test.fits) high=mid;
    else low=mid;
  }

  graphRadius=Math.max(0.6, lastBounds?.maxExtent || 0.6);
  return high;
}
function updateCameraBounds() {
  if (!allNodes.length) {
    graphCenter = [0, 0, 0];
    graphRadius = POLY_R;
    minDist = 0.7;
    baseMinDist = 0.7;
    maxDist = 6;
    resetDist = 3;
    dist = clamp(dist, minDist, maxDist);
    syncOrbitTarget(true);
    return;
  }

  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  let maxNodeR = 0;

  for (const n of allNodes) {
    const rr = Math.max(n.r * 2.0 * (n.sizeMul || 0.1), n.kind === "tag" ? 0.10 : 0.04);
    maxNodeR = Math.max(maxNodeR, rr);
    minX = Math.min(minX, n.pos[0] - rr);
    minY = Math.min(minY, n.pos[1] - rr);
    minZ = Math.min(minZ, n.pos[2] - rr);
    maxX = Math.max(maxX, n.pos[0] + rr);
    maxY = Math.max(maxY, n.pos[1] + rr);
    maxZ = Math.max(maxZ, n.pos[2] + rr);
  }

  graphCenter = [(minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2];
  const fitDist = fitDistanceForBounds(graphCenter, yaw, pitch);
  baseMinDist = Math.max(maxNodeR * 2.0, fitDist * 0.08);
  minDist = baseMinDist;
  maxDist = Math.max(minDist * 1.4, fitDist);
  resetDist = clamp(fitDist * DEFAULT_VIEW, minDist, maxDist);
  dist = resetDist;
  syncOrbitTarget(true);
}

function applyImportanceLayout(tagPos, angleOf) {
  if (!thoughtNodes.length) return;

  const storeBounds = {};
  for (const n of thoughtNodes) {
    const st = n.store || "global";
    const vec = n.embedPos || [0, 0, 0];
    const bound = (storeBounds[st] ||= [1e-6, 1e-6, 1e-6]);
    bound[0] = Math.max(bound[0], Math.abs(vec[0] || 0));
    bound[1] = Math.max(bound[1], Math.abs(vec[1] || 0));
    bound[2] = Math.max(bound[2], Math.abs(vec[2] || 0));
  }

  const thoughtEdges = edges
    .filter(e => !e.parent && e.s.kind === "thought" && e.t.kind === "thought")
    .sort((a, b) => `${a.s.key}:${a.t.key}`.localeCompare(`${b.s.key}:${b.t.key}`));

  for (const n of thoughtNodes) {
    const isGlobal = n.store === "global";
    const raw = n.embedPos || [0, 0, 0];
    const bounds = storeBounds[n.store || "global"] || [1, 1, 1];
    const vx = clamp((raw[0] || 0) / bounds[0], -1.2, 1.2);
    const vy = clamp((raw[1] || 0) / bounds[1], -1.2, 1.2);
    const vz = clamp((raw[2] || 0) / bounds[2], -1.2, 1.2);

    const tagAnchor = isGlobal ? [0, 0, 0] : (tagPos[n.store] || [0, 0, 0]);
    const baseAngle = isGlobal ? 0 : (angleOf[n.store] ?? 0);
    const radialDir = isGlobal
      ? [0, 0, 1]
      : [Math.cos(baseAngle), 0, Math.sin(baseAngle)];
    const tangentDir = [-radialDir[2], 0, radialDir[0]];
    const upDir = [0, 1, 0];

    const anchorBlend = isGlobal ? 0.0 : 0.56;
    const clusterCenter = [tagAnchor[0] * anchorBlend, tagAnchor[1] * anchorBlend * 0.78, tagAnchor[2] * anchorBlend];
    const spread = POLY_R * (isGlobal ? 0.20 : 0.24) * (0.82 + (1 - n.importance) * 0.35);
    const depthSpread = POLY_R * (isGlobal ? 0.10 : 0.12);

    n.anchor = clusterCenter;
    n.pos = [
      clusterCenter[0] + tangentDir[0] * vx * spread + radialDir[0] * vz * depthSpread,
      clusterCenter[1] + vy * spread * 0.82,
      clusterCenter[2] + tangentDir[2] * vx * spread + radialDir[2] * vz * depthSpread,
    ];
  }

  for (let iter = 0; iter < 14; iter++) {
    for (const e of thoughtEdges) {
      const strength = edgeStrength(e);
      const dx = e.t.pos[0] - e.s.pos[0], dy = e.t.pos[1] - e.s.pos[1], dz = e.t.pos[2] - e.s.pos[2];
      const len = Math.sqrt(dx * dx + dy * dy + dz * dz) || 0.001;
      const ideal = POLY_R * (0.09 + (e.s.store === e.t.store ? 0.05 : 0.11) + (1 - strength) * 0.04);
      const pull = (len - ideal) * (0.004 + strength * 0.010);
      const ux = dx / len, uy = dy / len, uz = dz / len;
      e.s.pos[0] += ux * pull * 0.5; e.s.pos[1] += uy * pull * 0.5; e.s.pos[2] += uz * pull * 0.5;
      e.t.pos[0] -= ux * pull * 0.5; e.t.pos[1] -= uy * pull * 0.5; e.t.pos[2] -= uz * pull * 0.5;
    }

    for (let i = 0; i < thoughtNodes.length; i++) {
      const a = thoughtNodes[i];
      for (let j = i + 1; j < thoughtNodes.length; j++) {
        const b = thoughtNodes[j];
        const dx = b.pos[0] - a.pos[0], dy = b.pos[1] - a.pos[1], dz = b.pos[2] - a.pos[2];
        const distSq = dx * dx + dy * dy + dz * dz + 0.003;
        const sameStore = a.store === b.store;
        const repel = (sameStore ? 0.0028 : 0.0010) / distSq;
        a.pos[0] -= dx * repel; a.pos[1] -= dy * repel; a.pos[2] -= dz * repel;
        b.pos[0] += dx * repel; b.pos[1] += dy * repel; b.pos[2] += dz * repel;
      }
    }

    for (const n of thoughtNodes) {
      const settle = 0.018;
      n.pos[0] += (n.anchor[0] - n.pos[0]) * settle;
      n.pos[1] += (n.anchor[1] - n.pos[1]) * settle;
      n.pos[2] += (n.anchor[2] - n.pos[2]) * settle;
    }
  }
}
function refreshNodeDirections() {
  for (const n of allNodes) {
    let dx = 0, dy = 0, dz = 0;
    for (const e of edges) {
      let other = null;
      if (e.s === n) other = e.t;
      else if (e.t === n) other = e.s;
      else continue;
      const w = 0.15 + edgeStrength(e) * 0.85;
      dx += (other.pos[0] - n.pos[0]) * w;
      dy += (other.pos[1] - n.pos[1]) * w;
      dz += (other.pos[2] - n.pos[2]) * w;
    }
    const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (len > 0.001) n.dir = [dx / len, dy / len, dz / len];
    else {
      const pl = Math.hypot(n.pos[0], n.pos[1], n.pos[2]);
      n.dir = pl > 0.001 ? [n.pos[0] / pl, n.pos[1] / pl, n.pos[2] / pl] : [0, 1, 0];
    }
  }
}

// ---- 3D layout ----
function layout(data) {
  allNodes = []; tagNodes = []; thoughtNodes = []; edges = [];
  nodeIdx = {}; nbrs = {}; storeClr = {}; clrI = 0;

  const specs = [...data.nodes.filter(n => n.type === "strand")]
    .sort((a, b) => String(a.label || a.key).localeCompare(String(b.label || b.key)));
  const rest = [...data.nodes.filter(n => n.type !== "strand")]
    .sort((a, b) => String(a.key).localeCompare(String(b.key)));
  const N = specs.length, angleOf = {}, tagPos = {};
  const now = Date.now() / 1000;

  specs.forEach((s, i) => {
    const a = (2 * Math.PI * i) / Math.max(N, 1);
    const yOff = (i % 2 === 0 ? 0.5 : -0.5) * POLY_R * 0.4;
    angleOf[s.label] = a;
    tagPos[s.label] = [POLY_R * Math.cos(a), yOff, POLY_R * Math.sin(a)];
    const col = sClr(s.label);
    const nd = {
      key: s.key, pos: [...tagPos[s.label]], r: 0.12, color: col,
      label: s.label, kind: "tag", shape: "tetra", store: s.store, source: s.source || "",
      access: s.access_count || 0, created: s.created_at || 0, lastAccess: s.last_accessed || 0
    };
    allNodes.push(nd); tagNodes.push(nd); nodeIdx[nd.key] = nd;
  });

  const sw = N > 0 ? (2 * Math.PI) / N : 2 * Math.PI;
  const bk = {};
  rest.forEach(t => {
    const st = t.store || "global";
    const k = angleOf[st] !== undefined ? st : "__global__";
    (bk[k] ||= []).push(t);
  });

  for (const [store, list] of Object.entries(bk)) {
    list.sort((a, b) => String(a.key).localeCompare(String(b.key)));
    const isG = store === "__global__";
    const base = isG ? 0 : angleOf[store];
    const col = sClr(isG ? "global" : store);

    for (const t of list) {
      const ac = Math.min(t.access_count || 0, 20) / 20;
      const theta = isG
        ? hash01(`${t.key}:theta`) * Math.PI * 2
        : base + (hash01(`${t.key}:theta`) - 0.5) * sw * 0.72;
      const radial = POLY_R * (isG ? 0.18 + hash01(`${t.key}:rad`) * 0.18 : 0.24 + (1 - ac) * 0.14 + hash01(`${t.key}:rad`) * 0.18);
      const yJ = (hash01(`${t.key}:y`) - 0.5) * (isG ? 0.38 : 0.28);
      const nd = {
        key: t.key, pos: [radial * Math.cos(theta), yJ, radial * Math.sin(theta)],
        r: Math.max(0.02, Math.min(0.055, 0.02 + ac * 0.035)),
        color: col, label: t.label, kind: "thought", shape: isSpecialistNode(t) ? "cube" : "tetra", store: isG ? "global" : store,
        source: t.source || "", access: t.access_count || 0, created: t.created_at || 0, lastAccess: t.last_accessed || 0,
        embedPos: Array.isArray(t.embed_pos) ? t.embed_pos : null
      };
      allNodes.push(nd); thoughtNodes.push(nd); nodeIdx[nd.key] = nd;
    }
  }

  for (const e of data.edges) {
    const s = nodeIdx[e.source], t = nodeIdx[e.target];
    if (s && t) edges.push({
      s, t,
      w: e.weight,
      reason: e.reasoning || "",
      success: e.success_rate || 0,
      traversals: e.traversal_count || 0,
      created: e.created_at || 0,
      parent: e.source.startsWith("_spec:") || e.target.startsWith("_spec:"),
      knn: false,
    });
  }
  if (data.knn_edges) for (const e of data.knn_edges) {
    const s = nodeIdx[e.source], t = nodeIdx[e.target];
    if (s && t) edges.push({
      s, t,
      w: e.weight,
      reason: e.reasoning || "",
      success: e.success_rate || 0,
      traversals: e.traversal_count || 0,
      created: e.created_at || 0,
      parent: false,
      knn: true,
    });
  }

  let maxAccess = 1, maxLinks = 1, maxStrength = 1, maxTraversal = 1;
  for (const e of edges) {
    (nbrs[e.s.key] ||= new Set()).add(e.t.key);
    (nbrs[e.t.key] ||= new Set()).add(e.s.key);
  }

  for (const n of allNodes) {
    let strength = 0, successSum = 0, traversalSum = 0, seen = 0;
    for (const e of edges) {
      if (e.s !== n && e.t !== n) continue;
      strength += 0.15 + edgeStrength(e) * 0.85;
      successSum += e.success || 0;
      traversalSum += e.traversals || 0;
      seen += 1;
    }
    n.linkCount = nbrs[n.key]?.size || 0;
    n.linkStrength = strength;
    n.avgSuccess = seen ? successSum / seen : 0;
    n.traversalLoad = traversalSum;
    const stamp = Math.max(n.lastAccess || 0, n.created || 0);
    n.recency = stamp > 0 ? 1 / (1 + Math.max(0, now - stamp) / (3600 * 24 * 14)) : 0;
    maxAccess = Math.max(maxAccess, n.access || 0);
    maxLinks = Math.max(maxLinks, n.linkCount);
    maxStrength = Math.max(maxStrength, strength);
    maxTraversal = Math.max(maxTraversal, traversalSum);
  }

  for (const n of allNodes) {
    const accessScore = softScore(n.access || 0, maxAccess);
    const linkScore = softScore(n.linkCount || 0, maxLinks);
    const strengthScore = softScore(n.linkStrength || 0, maxStrength);
    const traversalScore = softScore(n.traversalLoad || 0, maxTraversal);
    const successScore = clamp(n.avgSuccess || 0, 0, 1);
    const recencyScore = clamp(n.recency || 0, 0, 1);
    const importance = n.kind === "tag"
      ? clamp(0.18 + 0.24 * linkScore + 0.30 * strengthScore + 0.12 * successScore + 0.10 * traversalScore + 0.06 * recencyScore, 0, 1)
      : clamp(0.28 * accessScore + 0.18 * linkScore + 0.24 * strengthScore + 0.12 * successScore + 0.10 * traversalScore + 0.08 * recencyScore, 0, 1);
    n.importance = importance;
    n.sizeMul = 0.10 * (0.55 + 1.05 * importance);
  }

  applyImportanceLayout(tagPos, angleOf);
  refreshNodeDirections();
  updateCameraBounds();
}

// ---- WebGPU init ----
async function initGPU() {
  if (!navigator.gpu) throw new Error("WebGPU not supported");
  const ad = await navigator.gpu.requestAdapter();
  if (!ad) throw new Error("No GPU adapter found");
  const dev = await ad.requestDevice();
  const fmt = navigator.gpu.getPreferredCanvasFormat();
  const gc = cvs.getContext("webgpu");
  gc.configure({ device: dev, format: fmt, alphaMode: "premultiplied" });
  return { dev, fmt, gc };
}

// ---- WGSL shaders ----

const LNS = `
struct U { mvp:mat4x4f, time:f32, focal:f32, res:vec2f, depthMin:f32, depthMax:f32 }
@group(0) @binding(0) var<uniform> u:U;
struct VI { @location(0) pos:vec3f, @location(1) col:vec4f }
struct VO {
  @builtin(position) pos:vec4f,
  @location(0) col:vec4f,
  @location(1) dof:f32,
  @location(2) normDepth:f32,
  @location(3) nearBoost:f32,
}
@vertex fn vs(v:VI) -> VO {
  var o:VO;
  let clip = u.mvp*vec4f(v.pos,1.0);
  o.pos = clip;
  o.col = v.col;

  let depthRange = max(u.depthMax - u.depthMin, 0.001);
  let rawDepth = clamp((clip.w - u.depthMin) / depthRange, 0.0, 1.0);
  let farFromFocus = max(clip.w - (u.focal * 0.72), 0.0);

  o.normDepth = smoothstep(0.02, 1.0, pow(rawDepth, 1.22));
  o.dof = smoothstep(0.0, 1.0, clamp(farFromFocus / max(u.focal * 1.15, 0.001), 0.0, 1.0));
  o.nearBoost = 1.0 - smoothstep(0.0, 0.38, rawDepth);
  return o;
}
@fragment fn fs(f:VO) -> @location(0) vec4f {
  let fogClr = vec3f(0.035, 0.04, 0.085);
  let fogAmt = clamp(f.normDepth * 0.82 + f.dof * 0.28, 0.0, 0.96);
  let nearLit = f.col.rgb * (0.76 + 0.24 * f.nearBoost);
  let c = mix(nearLit, fogClr, fogAmt);
  let alpha = f.col.a * mix(1.0, 0.08, pow(fogAmt, 1.08));
  return vec4f(c, alpha);
}
`;

// ---- GPU buffers ----
function buildBufs(dev) {
  const focus = getFocusState();

  // --- Wireframe shapes per node, locked to a global upright orientation ---
  const sq2 = Math.sqrt(2), sq6 = Math.sqrt(6);
  // Elongated tetrahedron: tip points upward on +Y and never auto-rotates.
  const TV = [[0, 1.5, 0], [0, -0.5, 2 * sq2 / 3], [-sq6 / 3, -0.5, -sq2 / 3], [sq6 / 3, -0.5, -sq2 / 3]];
  const TE = [[0, 1], [0, 2], [0, 3], [1, 2], [2, 3], [3, 1]];
  const CV = [
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
  ];
  const CE = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
  ];
  const UPRIGHT_MAT = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

  function xformPt(p, mat, s, off) {
    return [
      (mat[0][0] * p[0] + mat[1][0] * p[1] + mat[2][0] * p[2]) * s + off[0],
      (mat[0][1] * p[0] + mat[1][1] * p[1] + mat[2][1] * p[2]) * s + off[1],
      (mat[0][2] * p[0] + mat[1][2] * p[1] + mat[2][2] * p[2]) * s + off[2],
    ];
  }

  const FPV = 7;  // pos(3) + col(3) + alpha(1)
  const pd = [];
  for (const n of allNodes) {
    const mat = UPRIGHT_MAT;
    const vis = getNodeVisual(n, focus);
    const isCube = n.shape === "cube";
    const scaleBase = Math.max(n.kind === "tag" ? 0.01 : 0.006, n.r * 2.0 * (n.sizeMul || 0.1)) * vis.scale;
    const verts = (isCube ? CV : TV).map(v => xformPt(v, mat, isCube ? scaleBase * 0.72 : scaleBase, n.pos));
    const edgesForShape = isCube ? CE : TE;
    for (const [ei, ej] of edgesForShape) {
      const va = verts[ei], vb = verts[ej];
      pd.push(
        va[0], va[1], va[2], vis.color[0], vis.color[1], vis.color[2], vis.alpha,
        vb[0], vb[1], vb[2], vis.color[0], vis.color[1], vis.color[2], vis.alpha,
      );
    }
  }
  const pda = new Float32Array(pd);
  const pb = dev.createBuffer({ size: Math.max(pda.byteLength, 16), usage: GPUBufferUsage.VERTEX, mappedAtCreation: true });
  new Float32Array(pb.getMappedRange(0, pda.byteLength || 16)).set(pda); pb.unmap();

  // Edges (skip parent)
  const ve = edges.filter(e => !e.parent);
  const ld = new Float32Array(ve.length * 2 * 7);
  for (let i = 0; i < ve.length; i++) {
    const e = ve[i], vis = getEdgeVisual(e, focus);
    const c = [vis.color[0], vis.color[1], vis.color[2], vis.alpha];
    const o = i * 14;
    ld[o] = e.s.pos[0]; ld[o + 1] = e.s.pos[1]; ld[o + 2] = e.s.pos[2];
    ld[o + 3] = c[0]; ld[o + 4] = c[1]; ld[o + 5] = c[2]; ld[o + 6] = c[3];
    ld[o + 7] = e.t.pos[0]; ld[o + 8] = e.t.pos[1]; ld[o + 9] = e.t.pos[2];
    ld[o + 10] = c[0]; ld[o + 11] = c[1]; ld[o + 12] = c[2]; ld[o + 13] = c[3];
  }
  const lb = dev.createBuffer({ size: Math.max(ld.byteLength, 16), usage: GPUBufferUsage.VERTEX, mappedAtCreation: true });
  new Float32Array(lb.getMappedRange(0, ld.byteLength || 16)).set(ld); lb.unmap();

  // Polyhedron wireframe + center spokes
  const pv = [];
  const ringAlpha = focus ? 0.10 : 0.35;
  const spokeAlpha = focus ? 0.05 : 0.15;
  for (let i = 0; i < tagNodes.length; i++) {
    const a = tagNodes[i], b = tagNodes[(i + 1) % tagNodes.length];
    pv.push(a.pos[0], a.pos[1], a.pos[2], 0.482, 0.549, 1.0, ringAlpha,
      b.pos[0], b.pos[1], b.pos[2], 0.482, 0.549, 1.0, ringAlpha);
    pv.push(0, 0, 0, 0.39, 0.45, 0.78, spokeAlpha,
      a.pos[0], a.pos[1], a.pos[2], 0.39, 0.45, 0.78, spokeAlpha);
  }
  const pvd = new Float32Array(pv);
  const pvb = dev.createBuffer({ size: Math.max(pvd.byteLength, 16), usage: GPUBufferUsage.VERTEX, mappedAtCreation: true });
  new Float32Array(pvb.getMappedRange(0, pvd.byteLength || 16)).set(pvd); pvb.unmap();

  return { pb, pc: pd.length / FPV, lb, lc: ve.length * 2, pvb, pvc: pv.length / 7 };
}

let dev, fmt, gc, ptP, lnP, ub, bg, bufs;
function refreshSceneBuffers() {
  if (!dev) return;
  if (bufs) {
    bufs.pb?.destroy?.();
    bufs.lb?.destroy?.();
    bufs.pvb?.destroy?.();
  }
  bufs = buildBufs(dev);
}

// ---- pipelines ----
function makePipes(dev, fmt) {
  const bgl = dev.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } }
    ]
  });
  const pl = dev.createPipelineLayout({ bindGroupLayouts: [bgl] });
  const blend = {
    color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
    alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
  };

  const ptP = dev.createRenderPipeline({
    layout: pl,
    vertex: {
      module: dev.createShaderModule({ code: LNS }), entryPoint: "vs",
      buffers: [{
        arrayStride: 28, attributes: [
          { shaderLocation: 0, offset: 0, format: "float32x3" },
          { shaderLocation: 1, offset: 12, format: "float32x4" },
        ]
      }]
    },
    fragment: {
      module: dev.createShaderModule({ code: LNS }), entryPoint: "fs",
      targets: [{ format: fmt, blend }]
    },
    primitive: { topology: "line-list" },
    depthStencil: { depthWriteEnabled: false, depthCompare: "less", format: "depth24plus" },
  });

  const lnP = dev.createRenderPipeline({
    layout: pl,
    vertex: {
      module: dev.createShaderModule({ code: LNS }), entryPoint: "vs",
      buffers: [{
        arrayStride: 28, attributes: [
          { shaderLocation: 0, offset: 0, format: "float32x3" },
          { shaderLocation: 1, offset: 12, format: "float32x4" },
        ]
      }]
    },
    fragment: {
      module: dev.createShaderModule({ code: LNS }), entryPoint: "fs",
      targets: [{ format: fmt, blend }]
    },
    primitive: { topology: "line-list" },
    depthStencil: { depthWriteEnabled: false, depthCompare: "less", format: "depth24plus" },
  });

  const ub = dev.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const bg = dev.createBindGroup({ layout: bgl, entries: [{ binding: 0, resource: { buffer: ub } }] });

  return { ptP, lnP, ub, bg };
}

// ---- project for picking ----
let depthMin = 1, depthMax = 10;
function projectPoint(mvp, pos) {
  const [x, y, z] = pos;
  const cx = mvp[0] * x + mvp[4] * y + mvp[8] * z + mvp[12];
  const cy = mvp[1] * x + mvp[5] * y + mvp[9] * z + mvp[13];
  const cw = mvp[3] * x + mvp[7] * y + mvp[11] * z + mvp[15];
  if (cw < 0.01) return null;
  return { x: (cx / cw * 0.5 + 0.5) * W, y: (1 - (cy / cw * 0.5 + 0.5)) * H, w: cw };
}
function projectAll(mvp) {
  projected = []; edgeProjected = [];
  let dMin = Infinity, dMax = -Infinity;
  for (const n of allNodes) {
    const [x, y, z] = n.pos;
    const cx = mvp[0] * x + mvp[4] * y + mvp[8] * z + mvp[12];
    const cy = mvp[1] * x + mvp[5] * y + mvp[9] * z + mvp[13];
    const cw = mvp[3] * x + mvp[7] * y + mvp[11] * z + mvp[15];
    if (cw < 0.01) { projected.push({ n, sx: -9999, sy: -9999, w: Infinity }); continue; }
    if (cw < dMin) dMin = cw;
    if (cw > dMax) dMax = cw;
    projected.push({ n, sx: (cx / cw * 0.5 + 0.5) * W, sy: (1 - (cy / cw * 0.5 + 0.5)) * H, w: cw });
  }
  if (dMin !== Infinity) { depthMin = dMin; depthMax = dMax; }

  const focus = getFocusState();
  const byKey = new Map(projected.map(p => [p.n.key, p]));
  for (const e of edges) {
    if (e.parent) continue;
    const sp = byKey.get(e.s.key), tp = byKey.get(e.t.key);
    if (!sp || !tp || !Number.isFinite(sp.sx) || !Number.isFinite(tp.sx) || sp.w < 0.01 || tp.w < 0.01) continue;
    const mx = (sp.sx + tp.sx) * 0.5, my = (sp.sy + tp.sy) * 0.5;
    if (mx < -24 || mx > W + 24 || my < -24 || my > H + 24) continue;
    edgeProjected.push({ e, sx: mx, sy: my, r: getEdgeMarkerVisual(e, focus).radius, depth: (sp.w + tp.w) * 0.5 });
  }
  edgeProjected.sort((a, b) => a.depth - b.depth);
}

// ---- 2D overlay ----
function drawTooltip(lines, x, y) {
  const filtered = lines.filter(Boolean);
  if (!filtered.length) return;
  const padX = 10, padY = 7, lineH = 15;
  oc.font = "500 12px Inter,system-ui,sans-serif";
  const boxW = Math.max(...filtered.map(line => oc.measureText(line).width), 0) + padX * 2;
  const boxH = filtered.length * lineH + padY * 2;
  const bx = clamp(x, 12, W - boxW - 12);
  const by = clamp(y - boxH * 0.5, 12, H - boxH - 12);
  oc.beginPath();
  oc.roundRect(bx, by, boxW, boxH, 8);
  oc.fillStyle = "rgba(10,10,20,0.88)";
  oc.fill();
  oc.strokeStyle = "rgba(120,140,255,0.18)";
  oc.lineWidth = 0.75;
  oc.stroke();
  oc.textAlign = "left";
  filtered.forEach((line, i) => {
    oc.fillStyle = i === 0 ? "rgba(235,240,255,0.96)" : "rgba(210,215,245,0.88)";
    oc.fillText(line, bx + padX, by + padY + 12 + i * lineH);
  });
}
function drawOv(time) {
  oc.setTransform(dpr, 0, 0, dpr, 0, 0);
  oc.clearRect(0, 0, W, H);
  const focus = getFocusState();

  const pulse = 1 + 0.15 * Math.sin(time * 0.002);
  const cProj = projectPoint(mvpCache, orbitCenter) || { x: W / 2, y: H / 2 };
  oc.save();
  oc.beginPath(); oc.arc(cProj.x, cProj.y, 4 * pulse, 0, Math.PI * 2);
  oc.fillStyle = `rgba(167,139,250,${0.5 + 0.3 * Math.sin(time * 0.003)})`;
  oc.shadowColor = "rgba(167,139,250,0.6)"; oc.shadowBlur = 12;
  oc.fill(); oc.shadowBlur = 0; oc.restore();

  for (const marker of edgeProjected) {
    const vis = getEdgeMarkerVisual(marker.e, focus);
    const hot = hoveredEdge === marker.e;
    const radius = vis.radius * (hot ? 1.16 : 1);
    oc.save();
    oc.beginPath(); oc.arc(marker.sx, marker.sy, radius, 0, Math.PI * 2);
    oc.strokeStyle = rgba(vis.color, hot ? Math.min(0.9, vis.alpha + 0.18) : vis.alpha);
    oc.lineWidth = (vis.lineWidth || 1) * (hot ? 1.2 : 1);
    oc.shadowColor = rgba(vis.color, 0.18 + (hot ? 0.16 : 0));
    oc.shadowBlur = hot ? 8 : 4;
    oc.stroke();
    oc.beginPath(); oc.arc(marker.sx, marker.sy, Math.max(0.9, radius * 0.20), 0, Math.PI * 2);
    oc.fillStyle = rgba(vis.color, hot ? 0.65 : 0.45);
    oc.fill();
    oc.restore();
  }

  oc.font = "500 11px Inter,system-ui,sans-serif";
  const labelCandidates = projected
    .filter(p => p.n.kind === "thought" && p.w < Infinity && (p.n.importance || 0) >= 0.55)
    .sort((a, b) => (b.n.importance || 0) - (a.n.importance || 0))
    .slice(0, 10);
  for (const p of labelCandidates) {
    if (hoveredNode === p.n || selected === p.n) continue;
    const extra = focus && focus.scores.has(p.n.key) ? 0.14 : 0;
    const alpha = Math.min(0.72, 0.16 + (p.n.importance || 0) * 0.35 + extra);
    oc.fillStyle = `rgba(210,215,245,${alpha})`;
    oc.fillText(truncateText(p.n.label, 34), p.sx + 8, p.sy - 8);
  }

  if (hoveredNode) {
    const hp = projected.find(p => p.n === hoveredNode);
    if (hp) drawTooltip([truncateText(hoveredNode.label, 64)], hp.sx + 16, hp.sy);
  } else if (hoveredEdge) {
    const marker = edgeProjected.find(p => p.e === hoveredEdge);
    if (marker) {
      const pair = `${truncateText(hoveredEdge.s.label || hoveredEdge.s.key, 18)} ↔ ${truncateText(hoveredEdge.t.label || hoveredEdge.t.key, 18)}`;
      const stats = `w ${hoveredEdge.w.toFixed(2)} • success ${(hoveredEdge.success || 0).toFixed(2)}${hoveredEdge.traversals ? ` • traversals ${hoveredEdge.traversals}` : ""}`;
      drawTooltip([pair, truncateText(hoveredEdge.reason || "linked by graph evidence", 92), stats], marker.sx + 16, marker.sy);
    }
  }
}

function pickNodeAt(mx, my) {
  let best = null, bd = Infinity;
  for (const p of projected) {
    const dx = mx - p.sx, dy = my - p.sy, d = Math.sqrt(dx * dx + dy * dy);
    const hr = p.n.kind === "tag" ? 16 : 8;
    if (d < hr && d < bd) { best = p.n; bd = d; }
  }
  return best;
}
function pickEdgeAt(mx, my) {
  let best = null, bd = Infinity;
  for (const marker of edgeProjected) {
    const dx = mx - marker.sx, dy = my - marker.sy, d = Math.sqrt(dx * dx + dy * dy);
    const hr = Math.max(6, marker.r + 3);
    if (d < hr && d < bd) { best = marker.e; bd = d; }
  }
  return best;
}

// ---- input ----
function setupInput() {
  cvs.addEventListener("wheel", e => {
    e.preventDefault();
    dist = clamp(dist * (e.deltaY > 0 ? ZOOM_OUT : ZOOM_IN), minDist, maxDist);
  }, { passive: false });

  cvs.addEventListener("pointerdown", e => {
    drag = true; lmx = e.clientX; lmy = e.clientY;
    vYaw = 0; vPitch = 0;
    cvs.setPointerCapture(e.pointerId);
    cvs.style.cursor = "grabbing";
  });

  cvs.addEventListener("pointermove", e => {
    if (!drag) {
      hoveredEdge = pickEdgeAt(e.offsetX, e.offsetY);
      hoveredNode = hoveredEdge ? null : pickNodeAt(e.offsetX, e.offsetY);
      cvs.style.cursor = (hoveredNode || hoveredEdge) ? "pointer" : "grab";
      return;
    }
    const dx = e.clientX - lmx, dy = e.clientY - lmy;
    vYaw = -dx * SENS; vPitch = -dy * SENS;
    yaw += vYaw; pitch += vPitch;
    pitch = Math.max(-Math.PI / 2 + 0.05, Math.min(Math.PI / 2 - 0.05, pitch));
    lmx = e.clientX; lmy = e.clientY;
  });

  window.addEventListener("pointerup", () => { drag = false; cvs.style.cursor = "grab"; });

  window.addEventListener("keydown", e => {
    if (e.key === "Escape") selectNode(null);
  });

  cvs.addEventListener("click", e => {
    const hit = pickNodeAt(e.offsetX, e.offsetY);
    if (hit) { selectNode(hit); return; }
    const edgeHit = pickEdgeAt(e.offsetX, e.offsetY);
    if (edgeHit) {
      const target = (edgeHit.s.importance || 0) >= (edgeHit.t.importance || 0) ? edgeHit.s : edgeHit.t;
      selectNode(target, { align: true });
    }
  });

  cvs.addEventListener("dblclick", () => {
    hoveredNode = null; hoveredEdge = null;
    selectNode(null);
    yaw = 0.3;
    pitch = -0.5;
    dist = resetDist;
    vYaw = 0.002;
    vPitch = 0;
    syncOrbitTarget(true);
  });
}

// ---- detail ----
function selectNode(n, { align = false } = {}) {
  selected = n;
  refreshSceneBuffers();
  if (!n) {
    syncOrbitTarget();
    document.getElementById("detail").style.display = "none";
    return;
  }

  if (align) {
    const dx = n.pos[0] - graphCenter[0];
    const dy = n.pos[1] - graphCenter[1];
    const dz = n.pos[2] - graphCenter[2];
    yaw = Math.atan2(dx, dz);
    pitch = clamp(-Math.atan2(dy, Math.max(Math.hypot(dx, dz), 0.001)), -Math.PI / 2 + 0.05, Math.PI / 2 - 0.05);
  }

  vYaw = 0;
  vPitch = 0;
  setOrbitTarget(n.pos);
  showDetail(n);
}

function showDetail(n) {
  document.getElementById("d-title").textContent =
    n.kind === "tag" ? n.label : "thought #" + n.key;
  document.getElementById("d-meta").innerHTML = n.kind === "tag"
    ? `strand &middot; importance: ${(n.importance || 0).toFixed(2)} &middot; links: ${n.linkCount || 0}`
    : `store: ${n.store} &middot; source: ${n.source || "\u2014"} &middot; access: ${n.access} &middot; importance: ${(n.importance || 0).toFixed(2)}`;
  document.getElementById("d-text").textContent = n.label;
  const el = document.getElementById("d-edge-list"); el.innerHTML = "";
  const related = edges
    .filter(e => e.s === n || e.t === n)
    .sort((a, b) => edgeStrength(b) - edgeStrength(a));
  for (const e of related) {
    const o = e.s === n ? e.t : e.s;
    const d = document.createElement("div"); d.className = "edge-item";
    const bits = [`w ${e.w.toFixed(2)}`];
    if ((e.success || 0) > 0) bits.push(`success ${e.success.toFixed(2)}`);
    if (e.traversals) bits.push(`${e.traversals} traversals`);
    d.innerHTML = `<div><span class="ew">${bits.join(" · ")}</span> → #${o.key} ${truncateText(o.label || "", 60)}${e.knn ? ' <span class="edge-knn">knn</span>' : ''}</div><div style="opacity:.8;margin-top:4px">${truncateText(e.reason || "linked by graph evidence", 120)}</div>`;
    d.onclick = () => selectNode(o, { align: true });
    el.appendChild(d);
  }
  document.getElementById("detail").style.display = "block";
}

// ---- search ----
function setupSearch() {
  const inp = document.getElementById("search"), res = document.getElementById("search-results");
  inp.addEventListener("input", () => {
    const q = inp.value.toLowerCase().trim(); res.innerHTML = "";
    if (q.length < 2) return;
    let c = 0;
    for (const n of allNodes) {
      if (c >= 15) break;
      if (n.label && n.label.toLowerCase().includes(q)) {
        const d = document.createElement("div"); d.className = "sr-item";
        d.textContent = "#" + n.key + " " + n.label.slice(0, 60);
        d.onclick = () => { selectNode(n, { align: true }); inp.value = ""; res.innerHTML = ""; };
        res.appendChild(d); c++;
      }
    }
  });
}

// ---- resize ----
function resize() {
  dpr = window.devicePixelRatio || 1;
  W = window.innerWidth; H = window.innerHeight;
  cvs.width = W * dpr; cvs.height = H * dpr;
  cvs.style.width = W + "px"; cvs.style.height = H + "px";
  ov.width = W * dpr; ov.height = H * dpr;
  ov.style.width = W + "px"; ov.style.height = H + "px";
  updateCameraBounds();
}

// ---- boot ----
resize();
window.addEventListener("resize", () => { resize(); depthTex = null; });

const data = await(await fetch("/graph/full")).json();
document.getElementById("loading").remove();
layout(data);
document.getElementById("s-nodes").textContent = thoughtNodes.length;
document.getElementById("s-edges").textContent = edges.length;

let gpu;
try { gpu = await initGPU(); } catch (e) {
  document.body.innerHTML = '<div style="color:#ff6b6b;padding:40px;font-family:Inter,sans-serif"><h2>WebGPU not available</h2><p>' + e.message + '</p><p>Try Chrome 113+ or Edge 113+.</p></div>';
  throw e;
}

({ dev, fmt, gc } = gpu);
({ ptP, lnP, ub, bg } = makePipes(dev, fmt));
bufs = buildBufs(dev);
let depthTex = null;
let mvpCache = new Float32Array(16);

function getDT() {
  if (!depthTex || depthTex.width !== cvs.width || depthTex.height !== cvs.height) {
    if (depthTex) depthTex.destroy();
    depthTex = dev.createTexture({ size: [cvs.width, cvs.height], format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT });
  }
  return depthTex;
}

function frame(time) {
  // inertia
  if (!drag) {
    yaw += vYaw; pitch += vPitch;
    vYaw *= DAMP; vPitch *= DAMP;
    pitch = Math.max(-Math.PI / 2 + 0.05, Math.min(Math.PI / 2 - 0.05, pitch));
  }

  for (let i = 0; i < 3; i++) orbitCenter[i] += (orbitTarget[i] - orbitCenter[i]) * FOCUS_LERP;

  // MVP — orbit camera: rotate world then pull back
  const asp = W / H;
  const far = Math.max(40, dist + graphRadius * 6);
  const proj = m4Persp(FOV, asp, 0.05, far);
  const view = m4Mul(
    m4Mul(
      m4Mul(m4Trans(0, 0, -dist), m4RotX(-pitch)),
      m4RotY(-yaw)
    ),
    m4Trans(-orbitCenter[0], -orbitCenter[1], -orbitCenter[2])
  );

  const fitDist = fitDistanceForBounds(orbitCenter, yaw, pitch);
  minDist = Math.max(baseMinDist, fitDist * 0.08);
  maxDist = Math.max(minDist * 1.4, fitDist);
  dist = clamp(dist, minDist, maxDist);
  const mvp = m4Mul(proj, view);
  mvpCache = mvp;

  // upload (respect WGSL alignment: vec2f at offset 72 = float index 18)
  const ud = new Float32Array(24);
  ud.set(mvp, 0); ud[16] = time; ud[17] = dist; ud[18] = cvs.width; ud[19] = cvs.height;
  ud[20] = depthMin; ud[21] = depthMax;
  dev.queue.writeBuffer(ub, 0, ud);

  projectAll(mvp);

  const tex = gc.getCurrentTexture();
  const dt = getDT();
  const enc = dev.createCommandEncoder();
  const pass = enc.beginRenderPass({
    colorAttachments: [{ view: tex.createView(), clearValue: { r: 0.027, g: 0.027, b: 0.051, a: 1 }, loadOp: "clear", storeOp: "store" }],
    depthStencilAttachment: { view: dt.createView(), depthClearValue: 1, depthLoadOp: "clear", depthStoreOp: "store" },
  });

  // polyhedron wireframe
  if (bufs.pvc > 0) {
    pass.setPipeline(lnP); pass.setBindGroup(0, bg);
    pass.setVertexBuffer(0, bufs.pvb); pass.draw(bufs.pvc);
  }
  // edges
  if (bufs.lc > 0) {
    pass.setPipeline(lnP); pass.setBindGroup(0, bg);
    pass.setVertexBuffer(0, bufs.lb); pass.draw(bufs.lc);
  }
  // points
  pass.setPipeline(ptP); pass.setBindGroup(0, bg);
  pass.setVertexBuffer(0, bufs.pb); pass.draw(bufs.pc);

  pass.end();
  dev.queue.submit([enc.finish()]);

  drawOv(time);
  requestAnimationFrame(frame);
}

setupInput();
setupSearch();
requestAnimationFrame(frame);
