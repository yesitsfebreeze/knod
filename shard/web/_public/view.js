// shard — WebGPU 3D polygon cloud visualization
// Tags form vertices of a 3D convex polyhedron (offset on Y so even
// a triangle becomes volumetric). Thoughts cluster in sectors between
// tag vertices. Inertial orbit: Apple-style sphere rotation.

// ---- Base16 themes ----
const UI_FONT_FAMILY = '"IBM Plex Mono", "JetBrains Mono", ui-monospace, monospace';
const ANSI_THEME_ORDER = ["base00", "base08", "base0B", "base0A", "base0D", "base0E", "base0C", "base05", "base03", "base08", "base0B", "base0A", "base0D", "base0E", "base0C", "base07"];
const BASE16_THEME_SPECS = [
	{
		id: "default-dark",
		name: "Default Dark",
		author: "Chris Kempson",
		base: {
			base00: "181818", base01: "282828", base02: "383838", base03: "585858",
			base04: "b8b8b8", base05: "d8d8d8", base06: "e8e8e8", base07: "f8f8f8",
			base08: "ab4642", base09: "dc9656", base0A: "f7ca88", base0B: "a1b56c",
			base0C: "86c1b9", base0D: "7cafc2", base0E: "ba8baf", base0F: "a16946",
		},
		cursor: "7cafc2",
	},
	{
		id: "solarized-dark",
		name: "Solarized Dark",
		author: "Ethan Schoonover",
		base: {
			base00: "002b36", base01: "073642", base02: "586e75", base03: "657b83",
			base04: "839496", base05: "93a1a1", base06: "eee8d5", base07: "fdf6e3",
			base08: "dc322f", base09: "cb4b16", base0A: "b58900", base0B: "859900",
			base0C: "2aa198", base0D: "268bd2", base0E: "6c71c4", base0F: "d33682",
		},
		cursor: "268bd2",
	},
	{
		id: "gruvbox-dark-hard",
		name: "Gruvbox Dark Hard",
		author: "Dawid Kurek, morhetz",
		base: {
			base00: "1d2021", base01: "3c3836", base02: "504945", base03: "665c54",
			base04: "bdae93", base05: "d5c4a1", base06: "ebdbb2", base07: "fbf1c7",
			base08: "fb4934", base09: "fe8019", base0A: "fabd2f", base0B: "b8bb26",
			base0C: "8ec07c", base0D: "83a598", base0E: "d3869b", base0F: "d65d0e",
		},
		cursor: "83a598",
	},
	{
		id: "nord",
		name: "Nord",
		author: "arcticicestudio",
		base: {
			base00: "2E3440", base01: "3B4252", base02: "434C5E", base03: "4C566A",
			base04: "D8DEE9", base05: "E5E9F0", base06: "ECEFF4", base07: "8FBCBB",
			base08: "BF616A", base09: "D08770", base0A: "EBCB8B", base0B: "A3BE8C",
			base0C: "88C0D0", base0D: "81A1C1", base0E: "B48EAD", base0F: "5E81AC",
		},
		cursor: "88C0D0",
	},
	{
		id: "tomorrow-night",
		name: "Tomorrow Night",
		author: "Chris Kempson",
		base: {
			base00: "1d1f21", base01: "282a2e", base02: "373b41", base03: "969896",
			base04: "b4b7b4", base05: "c5c8c6", base06: "e0e0e0", base07: "ffffff",
			base08: "cc6666", base09: "de935f", base0A: "f0c674", base0B: "b5bd68",
			base0C: "8abeb7", base0D: "81a2be", base0E: "b294bb", base0F: "a3685a",
		},
		cursor: "81a2be",
	},
	{
		id: "dracula",
		name: "Dracula",
		author: "Mike Barkmin",
		base: {
			base00: "282936", base01: "3a3c4e", base02: "4d4f68", base03: "626483",
			base04: "62d6e8", base05: "e9e9f4", base06: "f1f2f8", base07: "f7f7fb",
			base08: "ea51b2", base09: "b45bcf", base0A: "00f769", base0B: "ebff87",
			base0C: "a1efe4", base0D: "62d6e8", base0E: "b45bcf", base0F: "00f769",
		},
		cursor: "62d6e8",
	},
	{
		id: "monokai",
		name: "Monokai",
		author: "Wimer Hazenberg",
		base: {
			base00: "272822", base01: "383830", base02: "49483e", base03: "75715e",
			base04: "a59f85", base05: "f8f8f2", base06: "f5f4f1", base07: "f9f8f5",
			base08: "f92672", base09: "fd971f", base0A: "f4bf75", base0B: "a6e22e",
			base0C: "a1efe4", base0D: "66d9ef", base0E: "ae81ff", base0F: "cc6633",
		},
		cursor: "f8f8f2",
	},
	{
		id: "onedark",
		name: "One Dark",
		author: "Lalit Magant",
		base: {
			base00: "282c34", base01: "353b45", base02: "3e4451", base03: "545862",
			base04: "565c64", base05: "abb2bf", base06: "b6bdca", base07: "c8ccd4",
			base08: "e06c75", base09: "d19a66", base0A: "e5c07b", base0B: "98c379",
			base0C: "56b6c2", base0D: "61afef", base0E: "c678dd", base0F: "be5046",
		},
		cursor: "61afef",
	},
	{
		id: "tokyo-night-dark",
		name: "Tokyo Night",
		author: "Michaël Ball",
		base: {
			base00: "1a1b26", base01: "16161e", base02: "2f3549", base03: "444b6a",
			base04: "787c99", base05: "a9b1d6", base06: "cbccd1", base07: "d5d6db",
			base08: "f7768e", base09: "ff9e64", base0A: "e0af68", base0B: "9ece6a",
			base0C: "7dcfff", base0D: "7aa2f7", base0E: "bb9af7", base0F: "d18616",
		},
		cursor: "7aa2f7",
	},
	{
		id: "catppuccin-mocha",
		name: "Catppuccin Mocha",
		author: "Catppuccin",
		base: {
			base00: "1e1e2e", base01: "181825", base02: "313244", base03: "45475a",
			base04: "585b70", base05: "cdd6f4", base06: "f5e0dc", base07: "b4befe",
			base08: "f38ba8", base09: "fab387", base0A: "f9e2af", base0B: "a6e3a1",
			base0C: "94e2d5", base0D: "89b4fa", base0E: "cba6f7", base0F: "f2cdcd",
		},
		cursor: "89b4fa",
	},
	{
		id: "everforest-dark",
		name: "Everforest Dark",
		author: "Sainnhe Park",
		base: {
			base00: "2d353b", base01: "343f44", base02: "3d484d", base03: "475258",
			base04: "859289", base05: "d3c6aa", base06: "e6e2cc", base07: "fdf6e3",
			base08: "e67e80", base09: "e69875", base0A: "dbbc7f", base0B: "a7c080",
			base0C: "83c092", base0D: "7fbbb3", base0E: "d699b6", base0F: "d65d0e",
		},
		cursor: "7fbbb3",
	},
	{
		id: "rose-pine",
		name: "Rosé Pine",
		author: "Rosé Pine",
		base: {
			base00: "191724", base01: "1f1d2e", base02: "26233a", base03: "6e6a86",
			base04: "908caa", base05: "e0def4", base06: "e0def4", base07: "524f67",
			base08: "eb6f92", base09: "f6c177", base0A: "ebbcba", base0B: "31748f",
			base0C: "9ccfd8", base0D: "c4a7e7", base0E: "f6c177", base0F: "524f67",
		},
		cursor: "c4a7e7",
	},
	{
		id: "kanagawa",
		name: "Kanagawa",
		author: "rebelot",
		base: {
			base00: "1f1f28", base01: "2a2a37", base02: "223249", base03: "727169",
			base04: "c8c093", base05: "dcd7ba", base06: "938aa9", base07: "363646",
			base08: "c34043", base09: "ffa066", base0A: "dca561", base0B: "76946a",
			base0C: "7fb4ca", base0D: "7e9cd8", base0E: "957fb8", base0F: "d27e99",
		},
		cursor: "7e9cd8",
	},
	{
		id: "ayu-dark",
		name: "Ayu Dark",
		author: "Khue Nguyen",
		base: {
			base00: "0f1419", base01: "131721", base02: "272d38", base03: "3e4b59",
			base04: "bfbdb6", base05: "e6e1cf", base06: "e6e1cf", base07: "f3f4f5",
			base08: "f07178", base09: "ff8f40", base0A: "ffb454", base0B: "b8cc52",
			base0C: "95e6cb", base0D: "59c2ff", base0E: "d2a6ff", base0F: "e6b673",
		},
		cursor: "59c2ff",
	},
	{
		id: "palenight",
		name: "Material Palenight",
		author: "Nate Peterson",
		base: {
			base00: "292d3e", base01: "444267", base02: "32374d", base03: "676e95",
			base04: "8796b0", base05: "959dcb", base06: "959dcb", base07: "ffffff",
			base08: "f07178", base09: "f78c6c", base0A: "ffcb6b", base0B: "c3e88d",
			base0C: "89ddff", base0D: "82aaff", base0E: "c792ea", base0F: "ff5370",
		},
		cursor: "82aaff",
	},
	{
		id: "synthwave-84",
		name: "Synthwave '84",
		author: "Robb Owen",
		base: {
			base00: "2b213a", base01: "34294f", base02: "495495", base03: "848bbd",
			base04: "e4e4e4", base05: "e0d8f0", base06: "f2f0ff", base07: "ffffff",
			base08: "f97e72", base09: "ff8b39", base0A: "fede5d", base0B: "72f1b8",
			base0C: "03edf9", base0D: "36f9f6", base0E: "ff7edb", base0F: "fe4450",
		},
		cursor: "36f9f6",
	},
];
const DEFAULT_THEME_ID = "default-dark";
function mixRgb(a, b, t) {
	return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}
function hex_to_color(hex) {
	const value = String(hex || "").replace(/^#/, "");
	const normalized = value.length === 3 ? value.split("").map(ch => ch + ch).join("") : value;
	return [0, 2, 4].map(index => parseInt(normalized.slice(index, index + 2), 16) / 255);
}
function color_distance(a, b) {
	return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}
function modulateConsoleColor(color, background, foreground) {
	let next = mixRgb(color, foreground, 0.12);
	if (color_distance(next, background) < 0.28) next = mixRgb(next, foreground, 0.28);
	if (color_distance(next, background) < 0.36) next = mixRgb(next, foreground, 0.18);
	return next;
}
function compile_theme(spec) {
	const background = hex_to_color(spec.base.base00);
	const foreground = hex_to_color(spec.base.base05);
	const cursor = hex_to_color(spec.cursor || spec.base.base0D);
	const ansi = ANSI_THEME_ORDER.map(key => hex_to_color(spec.base[key]));
	return {
		id: spec.id,
		name: spec.name,
		author: spec.author,
		background,
		foreground,
		cursor,
		ansi,
		palette: ansi.map(color => modulateConsoleColor(color, background, foreground)),
		backgroundHex: `#${spec.base.base00}`,
		foregroundHex: `#${spec.base.base05}`,
		cursorHex: `#${spec.cursor || spec.base.base0D}`,
		ansiHex: ANSI_THEME_ORDER.map(key => `#${spec.base[key]}`),
		accents: {
			red: hex_to_color(spec.base.base08),
			orange: hex_to_color(spec.base.base09),
			yellow: hex_to_color(spec.base.base0A),
			green: hex_to_color(spec.base.base0B),
			cyan: hex_to_color(spec.base.base0C),
			blue: hex_to_color(spec.base.base0D),
			magenta: hex_to_color(spec.base.base0E),
			brown: hex_to_color(spec.base.base0F),
			muted: hex_to_color(spec.base.base03),
		},
	};
}
const THEMES = BASE16_THEME_SPECS.map(compile_theme);
const POLY_R = 1.8;
const FOV = 65 * Math.PI / 180;
const ZOOM_IN = 0.84;
const ZOOM_OUT = 1.12;
const FRAME_PAD = 1.00;
const FRAME_MARGIN_PX = 48;
const DEFAULT_VIEW = 0.90;
const FOCUS_LERP = 0.18;
const INTERACTIVE_ALPHA_THRESHOLD = 0.10;
const CLICK_MOVE_THRESHOLD = 6;
const SELECTED_FOG_SCALE = 0.45;
const FOCUS_MAX_HOPS = 3;
const FOCUS_DEPTH_FADE_STEP = 0.18;
const FOCUS_VISIBILITY_START = 0.22;
const FOCUS_VISIBILITY_END = 0.92;
const FOCUS_VISIBILITY_POWER = 1.8;
const DOF_CAPSULE_NEAR = 0.16;
const DOF_CAPSULE_FAR = 1.20;
const DOF_CAPSULE_RADIUS = 0.34;
const DOF_CAPSULE_SOFTNESS = 0.68;
const DOF_BLUR_EXPANSION = 0.26;
const DOF_LABEL_SHARPNESS_POWER = 0.8;
const MAX_DIMENSIONS = 50;
const DIMENSION_SLIDER_SCALE = 10000;
const SPIRAL_HEIGHT_DEFAULT = 4.2;
const SPIRAL_HEIGHT_MAX = 10;
const SPIRAL_SPREAD_DEFAULT = 0;
const EDGE_OPACITY_DEFAULT = 0.55;
const EDGE_MARKER_MIN_ALPHA = 0.18;
const EDGE_MARKER_MIN_FOCUS_ALPHA = 0.26;
const EDGE_MARKER_HOVER_BOOST = 0.18;
const SETTINGS_STORAGE_KEY = "shard.viz.settings.v1";
const EDGE_MARKER_T = 0.62;
const SHARD_RING_RADIUS = POLY_R * 1.36;
const SHARD_INNER_PULL = 0.72;
const THOUGHT_SPIRAL_RADIUS = POLY_R * 0.28;
const GLOBAL_SPIRAL_RADIUS = POLY_R * 0.22;
const SPIRAL_TURN_BASE = 1.15;
const SPIRAL_TURN_GROWTH = 1 / 12;
const DIMENSION_STOPWORDS = new Set([
	"about", "after", "also", "been", "being", "between", "both", "can", "could", "data", "does", "each",
	"from", "have", "into", "its", "just", "knowledge", "make", "many", "more", "most", "much", "node",
	"nodes", "over", "same", "should", "some", "than", "that", "their", "them", "then", "there", "these",
	"they", "this", "thought", "thoughts", "through", "under", "using", "used", "very", "what", "when",
	"where", "which", "while", "with", "would", "your", "graph", "graphs", "system", "systems", "because",
	"into", "onto", "across", "such", "those", "were", "will", "shall", "than", "query", "queries"
]);

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
let all_nodes = [], tag_nodes = [], thought_nodes = [], edges = [];
let node_idx = {}, nbrs = {}, edgeAdj = {};
let store_clr = {}, clrI = 0;
let layout_state = { tagPos: {}, angleOf: {}, sw: Math.PI * 2 };
let rawEdgeData = { edges: [], knn_edges: [] };
let knownNodeKeys = new Set(); // tracks all node keys loaded so far for BFS expand
let dimension_tags = [], dimensionControlEls = new Map();
let dimension_weights = new Map(), dimension_vectors = new Map();
let dimension_selection = { strength: 0, boosted: new Set(), suppressed: new Set(), deltas: new Map(), boosts: new Map(), fades: new Map() };
let spiral_height_scale = SPIRAL_HEIGHT_DEFAULT;
let spiral_spread_scale = SPIRAL_SPREAD_DEFAULT;
let edge_opacity_scale = EDGE_OPACITY_DEFAULT;
let recencyRecentKey = "cyan";
let recencyOldKey = "magenta";
const RECENCY_COLOR_OPTIONS = ["red", "orange", "yellow", "green", "cyan", "blue", "magenta", "brown"];
let currentTheme = THEMES.find(theme => theme.id === DEFAULT_THEME_ID) || THEMES[0];
let askState = { open: false, lastFocused: null, debounceId: null, requestSeq: 0, controller: null, serverSessionId: null };

// orbit
let yaw = 0.3, pitch = -0.5, dist = 3;
let vYaw = 0.002, vPitch = 0;
const DAMP = 0.97, SENS = 0.005;
let drag = false, lmx = 0, lmy = 0;
let pointerGesture = null;
let graph_center = [0, 0, 0], orbitCenter = [0, 0, 0], orbitTarget = [0, 0, 0], graph_radius = POLY_R, min_dist = 0.7, base_min_dist = 0.7, maxDist = 6, reset_dist = 3;
let frameHandle = 0;

// pick
let hoveredNode = null, hoveredEdge = null, selected = null, projected = [], edgeProjected = [];
let mvpCache = new Float32Array(16);

// overlay canvas for labels/tooltips
const ov = document.createElement("canvas");
ov.style.cssText = "position:absolute;inset:0;z-index:1;pointer-events:none;";
document.body.appendChild(ov);
const oc = ov.getContext("2d");

function sClr(s) { if (!store_clr[s]) store_clr[s] = currentTheme.palette[clrI++ % currentTheme.palette.length]; return store_clr[s]; }
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
function lerp(a, b, t) { return a + (b - a) * t; }
function smoothstep(a, b, v) {
	const t = invLerp(a, b, v);
	return t * t * (3 - 2 * t);
}
function invLerp(a, b, v) {
	if (Math.abs(b - a) < 1e-6) return 0.5;
	return clamp((v - a) / (b - a), 0, 1);
}
function mixColor(a, b, t) {
	return [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
}
function recencyColor(recency) {
	return mixColor(currentTheme.accents[recencyOldKey], currentTheme.accents[recencyRecentKey], clamp(recency, 0, 1));
}
function rgba(color, alpha = 1) {
	return `rgba(${Math.round(color[0] * 255)}, ${Math.round(color[1] * 255)}, ${Math.round(color[2] * 255)}, ${alpha})`;
}
function rgbTriplet(color) {
	return `${Math.round(color[0] * 255)},${Math.round(color[1] * 255)},${Math.round(color[2] * 255)}`;
}
function colorHex(color) {
	return `#${color.map(channel => Math.round(clamp(channel, 0, 1) * 255).toString(16).padStart(2, "0")).join("")}`;
}
function getThemeById(themeId) {
	return THEMES.find(theme => theme.id === themeId) || THEMES[0];
}
function getThemeShadowColor() {
	return mixColor(currentTheme.background, currentTheme.foreground, 0.14);
}
function applyThemeToDocument(theme) {
	const root = document.documentElement;
	const surface = mixColor(theme.background, theme.foreground, 0.08);
	const surfaceStrong = mixColor(theme.background, theme.foreground, 0.14);
	const border = mixColor(theme.background, theme.foreground, 0.28);
	const muted = mixColor(theme.background, theme.foreground, 0.46);
	const mutedStrong = mixColor(theme.background, theme.foreground, 0.72);
	const shadow = mixColor(theme.background, [0, 0, 0], 0.38);
	const grid = mixColor(theme.background, theme.foreground, 0.22);
	root.style.setProperty("--theme-bg", colorHex(theme.background));
	root.style.setProperty("--theme-bg-rgb", rgbTriplet(theme.background));
	root.style.setProperty("--theme-fg", colorHex(theme.foreground));
	root.style.setProperty("--theme-fg-rgb", rgbTriplet(theme.foreground));
	root.style.setProperty("--theme-cursor", colorHex(theme.cursor));
	root.style.setProperty("--theme-cursor-rgb", rgbTriplet(theme.cursor));
	root.style.setProperty("--theme-surface-rgb", rgbTriplet(surface));
	root.style.setProperty("--theme-surface-strong-rgb", rgbTriplet(surfaceStrong));
	root.style.setProperty("--theme-border-rgb", rgbTriplet(border));
	root.style.setProperty("--theme-muted-rgb", rgbTriplet(muted));
	root.style.setProperty("--theme-muted-strong-rgb", rgbTriplet(mutedStrong));
	root.style.setProperty("--theme-shadow-rgb", rgbTriplet(shadow));
	root.style.setProperty("--theme-grid-rgb", rgbTriplet(grid));
	root.style.setProperty("--theme-red-rgb", rgbTriplet(theme.accents.red));
	root.style.setProperty("--theme-green-rgb", rgbTriplet(theme.accents.green));
	root.style.setProperty("--theme-yellow-rgb", rgbTriplet(theme.accents.yellow));
	root.style.setProperty("--theme-blue-rgb", rgbTriplet(theme.accents.blue));
	root.style.setProperty("--theme-magenta-rgb", rgbTriplet(theme.accents.magenta));
	root.dataset.theme = theme.id;
}
function recolorNodesForTheme() {
	store_clr = {};
	clrI = 0;
	for (const node of all_nodes) {
		node.color = recencyColor(node.recency || 0);
	}
}
function syncThemeControls() {
	const select = document.getElementById("theme-select");
	if (select) select.value = currentTheme.id;
	const meta = document.getElementById("theme-meta");
	if (meta) meta.textContent = `${currentTheme.name} · ${currentTheme.author}`;
	if (window._refreshRecencySwatches) window._refreshRecencySwatches();
}
function applyTheme(themeId, { persist = true, recolorNodes = true, refresh = true } = {}) {
	currentTheme = getThemeById(themeId);
	applyThemeToDocument(currentTheme);
	syncThemeControls();
	if (recolorNodes && all_nodes.length) recolorNodesForTheme();
	if (refresh) {
		if (bufs) refresh_scene_buffers();
		if (selected) showDetail(selected);
		requestRender();
	}
	if (persist) write_stored_viz_settings();
}
function vec3Length(v) {
	return Math.hypot(v[0], v[1], v[2]);
}
function normalize3(v) {
	const len = vec3Length(v);
	if (len < 1e-6) return [0, 0, 1];
	return [v[0] / len, v[1] / len, v[2] / len];
}
function cross3(a, b) {
	return [
		a[1] * b[2] - a[2] * b[1],
		a[2] * b[0] - a[0] * b[2],
		a[0] * b[1] - a[1] * b[0],
	];
}
function getEdgeMarkerCenter(edge) {
	return [
		lerp(edge.s.pos[0], edge.t.pos[0], EDGE_MARKER_T),
		lerp(edge.s.pos[1], edge.t.pos[1], EDGE_MARKER_T),
		lerp(edge.s.pos[2], edge.t.pos[2], EDGE_MARKER_T),
	];
}
function getEdgeMarkerBasis(edge) {
	const forward = normalize3([
		edge.t.pos[0] - edge.s.pos[0],
		edge.t.pos[1] - edge.s.pos[1],
		edge.t.pos[2] - edge.s.pos[2],
	]);
	const guide = Math.abs(forward[1]) > 0.92 ? [1, 0, 0] : [0, 1, 0];
	let right = cross3(guide, forward);
	if (vec3Length(right) < 1e-6) right = [1, 0, 0];
	right = normalize3(right);
	const up = normalize3(cross3(forward, right));
	return { right, up, forward };
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
function escapeHtml(value) {
	return String(value || "")
		.replace(/&/g, "&amp;")
		.replace(/</g, "&lt;")
		.replace(/>/g, "&gt;")
		.replace(/\"/g, "&quot;")
		.replace(/'/g, "&#39;");
}
function formatDimensionLabel(tag) {
	return String(tag || "").replace(/[_-]+/g, " ");
}
function tokenizeDimensionText(text) {
	return String(text || "")
		.toLowerCase()
		.replace(/[_:/\\|]+/g, " ")
		.match(/[a-z0-9]{3,}/g)?.filter(token => !DIMENSION_STOPWORDS.has(token) && !/^\d+$/.test(token)) || [];
}
function fibonacciSpherePoint(index, total) {
	if (total <= 1) return [0, 0, 1];
	const t = index / Math.max(total - 1, 1);
	const y = 1 - t * 2;
	const radius = Math.sqrt(Math.max(0, 1 - y * y));
	const theta = Math.PI * (3 - Math.sqrt(5)) * index;
	return [Math.cos(theta) * radius, y * 0.85, Math.sin(theta) * radius];
}
function collectNodeTagSignals(n) {
	const signals = new Map();
	const add = (text, weight) => {
		for (const token of tokenizeDimensionText(text)) {
			signals.set(token, (signals.get(token) || 0) + weight);
		}
	};
	add(n.label, 1.0);
	if (n.source) add(n.source, 2.2);
	if (n.store && n.store !== "global") add(n.store, 1.8);
	if (n.kind === "tag") add(n.label, 2.5);
	return signals;
}
function getDimensionWeight(tagKey) {
	return dimension_weights.get(tagKey) ?? getBaselineDimensionWeight();
}
function getBaselineDimensionWeight() {
	return dimension_tags.length ? 1 / dimension_tags.length : 1;
}
function getDimensionRatio(weight) {
	const baseline = getBaselineDimensionWeight();
	return baseline > 0 ? weight / baseline : 1;
}
function sliderPositionToWeight(position) {
	const baseline = getBaselineDimensionWeight();
	const t = clamp(position, 0, 1);
	if (t <= 0.5) return baseline * (t / 0.5);
	return baseline + ((t - 0.5) / 0.5) * (1 - baseline);
}
function weightToSliderPosition(weight) {
	const baseline = getBaselineDimensionWeight();
	const w = clamp(weight, 0, 1);
	if (baseline <= 0 || baseline >= 1) return 0.5;
	if (w <= baseline) return 0.5 * (w / baseline);
	return 0.5 + 0.5 * ((w - baseline) / (1 - baseline));
}
function getDimensionSliderPosition(tagKey) {
	if (!tagKey || !dimension_weights.has(tagKey)) return 0.5;
	return clamp(weightToSliderPosition(getDimensionWeight(tagKey)), 0, 1);
}
function applySliderOpacity(baseAlpha, sliderPosition) {
	const position = clamp(sliderPosition, 0, 1);
	if (position <= 0.5) return baseAlpha * (position / 0.5);
	return baseAlpha;
}
function get_spiral_height_span() {
	return clamp(spiral_height_scale, 0, SPIRAL_HEIGHT_MAX);
}
function isSpiralInverted() {
	return spiral_spread_scale < 0;
}
function getSpiralHeightTwistScale() {
	const blend = getSpiralSphereBlend();
	return isSpiralInverted() ? -blend : blend;
}
function getSpiralSphereBlend() {
	return clamp(Math.abs(spiral_spread_scale), 0, 1);
}
function getSpiralHeightLabel() {
	return `${get_spiral_height_span().toFixed(1)} y-span`;
}
function getSpiralSpreadScale() {
	return clamp(spiral_spread_scale, -1, 1);
}
function getSpiralSpreadLabel() {
	const blend = getSpiralSphereBlend();
	if (blend < 0.01) return "flat — no spiral or Fibonacci";
	if (isSpiralInverted()) {
		return `${Math.round(blend * 100)}% inverse spread · −${Math.round(blend * 180)}° twist`;
	}
	return `${Math.round((1 - blend) * 100)}% spiral · ${Math.round(blend * 100)}% Fibonacci · ${Math.round(blend * 180)}° twist`;
}
function getEdgeOpacityScale() {
	return clamp(edge_opacity_scale, 0, 1);
}
function getEdgeOpacityLabel() {
	return `${getEdgeOpacityScale().toFixed(2)} alpha`;
}
function readStoredVizSettings() {
	try {
		const raw = window.localStorage.getItem(SETTINGS_STORAGE_KEY);
		if (!raw) return null;
		const parsed = JSON.parse(raw);
		return parsed && typeof parsed === "object" ? parsed : null;
	} catch {
		return null;
	}
}
function write_stored_viz_settings() {
	try {
		window.localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify({
			themeId: currentTheme.id,
			spiral_height_scale,
			spiral_spread_scale,
			edge_opacity_scale,
			recencyRecentKey,
			recencyOldKey,
		}));
	} catch {
		// Ignore storage failures so the viz still works in restricted contexts.
	}
}
function loadStoredVizSettings() {
	const stored = readStoredVizSettings();
	if (!stored) return;

	if (typeof stored.themeId === "string") {
		currentTheme = getThemeById(stored.themeId);
	}

	if (Number.isFinite(stored.spiral_height_scale)) {
		spiral_height_scale = clamp(stored.spiral_height_scale, 0, SPIRAL_HEIGHT_MAX);
	}
	if (Number.isFinite(stored.spiral_spread_scale)) {
		spiral_spread_scale = clamp(stored.spiral_spread_scale, -1, 1);
	}
	if (Number.isFinite(stored.edge_opacity_scale)) {
		edge_opacity_scale = clamp(stored.edge_opacity_scale, 0, 1);
	}
	if (typeof stored.recencyRecentKey === "string" && RECENCY_COLOR_OPTIONS.includes(stored.recencyRecentKey)) {
		recencyRecentKey = stored.recencyRecentKey;
	}
	if (typeof stored.recencyOldKey === "string" && RECENCY_COLOR_OPTIONS.includes(stored.recencyOldKey)) {
		recencyOldKey = stored.recencyOldKey;
	}
}
function applyEdgeOpacity(alpha, { ignoreSlider = false } = {}) {
	if (ignoreSlider) return clamp(alpha, 0, 1);
	return clamp(alpha * getEdgeOpacityScale(), 0, 1);
}
function applyEdgeMarkerOpacity(alpha, { ignoreSlider = false, minAlpha = EDGE_MARKER_MIN_ALPHA } = {}) {
	const scaled = applyEdgeOpacity(alpha, { ignoreSlider });
	if (scaled <= 0.001) return 0;
	return clamp(Math.max(scaled, minAlpha), 0, 1);
}
function scalePositionFromCenter(pos) {
	return pos;
}
function rotatePointAroundY(pos, angle) {
	const c = Math.cos(angle);
	const s = Math.sin(angle);
	return [
		pos[0] * c - pos[2] * s,
		pos[1],
		pos[0] * s + pos[2] * c,
	];
}
function getHeightTwistAngle(y, minY, maxY) {
	const twistScale = getSpiralHeightTwistScale();
	if (Math.abs(twistScale) < 1e-6) return 0;
	const yT = invLerp(minY, maxY, y);
	return (yT - 0.5) * Math.PI * twistScale;
}
function twistPositionByHeight(pos, minY, maxY) {
	return rotatePointAroundY(pos, getHeightTwistAngle(pos[1], minY, maxY));
}
function getSpiralTurns(count) {
	return SPIRAL_TURN_BASE + Math.min(2.2, Math.max(0, count - 1) * SPIRAL_TURN_GROWTH);
}
function get_spiral_sphere_scale(rank_t) {
	const clampedRank = clamp(rank_t, 0, 1);
	const y = 1 - clampedRank * 2;
	const surfaceRadius = Math.sqrt(Math.max(0, 1 - y * y));
	const blend = getSpiralSphereBlend();
	if (isSpiralInverted()) {
		const invertedRadius = 2 - surfaceRadius;
		return lerp(1, invertedRadius, blend);
	}
	return lerp(1, surfaceRadius, blend);
}
function sort_nodes_for_spiral(nodes) {
	return [...nodes].sort((a, b) => {
		if ((a.importance || 0) !== (b.importance || 0)) return (a.importance || 0) - (b.importance || 0);
		return String(a.label || a.key).localeCompare(String(b.label || b.key));
	});
}
function getEdgeSliderPosition(edge) {
	return ((edge.s.dimension_opacity ?? 0.5) + (edge.t.dimension_opacity ?? 0.5)) * 0.5;
}
function describeDimensionState() {
	if (!dimension_tags.length) return "no strong graph tags detected yet";
	const baseline = getBaselineDimensionWeight();
	const boosted = dimension_tags
		.map(tag => ({ tag, weight: getDimensionWeight(tag.key) }))
		.filter(entry => entry.weight > baseline * 1.15)
		.sort((a, b) => b.weight - a.weight)
		.slice(0, 3);
	const suppressed = dimension_tags
		.map(tag => ({ tag, weight: getDimensionWeight(tag.key) }))
		.filter(entry => entry.weight < baseline * 0.85)
		.sort((a, b) => a.weight - b.weight)
		.slice(0, 3);
	if (!boosted.length && !suppressed.length) return `balanced across ${dimension_tags.length} ranked tags`;
	if (boosted.length && suppressed.length) {
		return `biasing toward ${boosted.map(entry => `${formatDimensionLabel(entry.tag.label)} ${(entry.weight * 100).toFixed(1)}%`).join(" · ")} · hiding ${suppressed.map(entry => `${formatDimensionLabel(entry.tag.label)} ${(entry.weight * 100).toFixed(1)}%`).join(" · ")}`;
	}
	if (suppressed.length) {
		return `hiding ${suppressed.map(entry => `${formatDimensionLabel(entry.tag.label)} ${(entry.weight * 100).toFixed(1)}%`).join(" · ")}`;
	}
	return `biasing toward ${boosted.map(entry => `${formatDimensionLabel(entry.tag.label)} ${(entry.weight * 100).toFixed(1)}%`).join(" · ")}`;
}
function recomputeDimensionSelection() {
	const baseline = getBaselineDimensionWeight();
	const deltas = new Map();
	const boosts = new Map();
	const fades = new Map();
	let positive = 0;
	let negative = 0;

	for (const tag of dimension_tags) {
		const weight = getDimensionWeight(tag.key);
		const delta = weight - baseline;
		const boost = delta > 0 ? delta / Math.max(1 - baseline, 0.001) : 0;
		const fade = delta < 0 ? 1 - (weight / Math.max(baseline, 0.001)) : 0;
		deltas.set(tag.key, delta);
		boosts.set(tag.key, boost);
		fades.set(tag.key, fade);
		if (delta > 0) positive += delta;
		else negative += -delta;
	}

	const active = Math.max(positive, negative);
	dimension_selection = {
		strength: clamp(active * 2.4, 0, 1),
		boosted: new Set([...deltas.entries()].filter(([, delta]) => delta > baseline * 0.08).map(([key]) => key)),
		suppressed: new Set([...fades.entries()].filter(([, fade]) => fade > 0.08).map(([key]) => key)),
		deltas,
		boosts,
		fades,
	};
}
function getFocusState() {
	if (!selected) return null;

	const selectedColor = mixColor(selected.color, currentTheme.cursor, 0.55);
	const scores = new Map([[selected.key, 1]]);
	const distances = new Map([[selected.key, 0]]);
	const edgeScores = new Map();
	const queue = [selected];
	let cursor = 0;

	while (cursor < queue.length) {
		const node = queue[cursor++];
		const depth = distances.get(node.key) ?? 0;
		if (depth >= FOCUS_MAX_HOPS) continue;

		for (const e of edgeAdj[node.key] || []) {
			const other = e.s === node ? e.t : e.s;
			const nextDepth = depth + 1;
			if (nextDepth > FOCUS_MAX_HOPS) continue;
			if (distances.has(other.key)) continue;
			distances.set(other.key, nextDepth);
			queue.push(other);
		}
	}
	for (const [key, depth] of distances.entries()) {
		const visibility = depth === 0 ? 1.0 : clamp(1 - (depth - 1) / 3, 0, 1);
		scores.set(key, visibility);
	}

	for (const e of edges) {
		const sourceDepth = distances.get(e.s.key);
		const targetDepth = distances.get(e.t.key);
		if (sourceDepth === undefined || targetDepth === undefined) continue;
		const edgeDepth = Math.max(sourceDepth, targetDepth);
		const visibility = edgeDepth === 0 ? 1.0 : clamp(1 - (edgeDepth - 1) / 3, 0, 1);
		edgeScores.set(e, visibility);
	}

	return { selectedKey: selected.key, selectedColor, scores, distances, edgeScores, maxDepth: FOCUS_MAX_HOPS, min: 0, max: 1 };
}
function normalizeFocusScore(score, focus) {
	if (!focus) return 0;
	const range = Math.max(focus.max - focus.min, 0.001);
	return clamp((score - focus.min) / range, 0, 1);
}
function getDepthOfFieldState(cx, cy, cw) {
	if (!Number.isFinite(cw) || cw <= 0.0001) {
		return { blur: 1, sharpness: 0, scale: 1 + DOF_BLUR_EXPANSION };
	}

	const ndcX = cx / cw;
	const ndcY = cy / cw;
	const radial = Math.hypot(ndcX * 0.86, ndcY);
	const focusStart = Math.max(depthMin + (depthMax - depthMin) * 0.04, dist * DOF_CAPSULE_NEAR);
	const focusEnd = Math.max(focusStart + 0.001, Math.min(depthMax, dist * DOF_CAPSULE_FAR));
	const depthSpan = Math.max(focusEnd - focusStart, dist * 0.36, 0.001);
	const longitudinal = Math.max((cw - focusStart) / depthSpan - 1, 0);
	const capsule = Math.hypot(radial / DOF_CAPSULE_RADIUS, longitudinal / DOF_CAPSULE_SOFTNESS);
	const blur = smoothstep(0.74, 1.34, capsule);
	const sharpness = 1 - blur;
	return {
		blur,
		sharpness,
		scale: 1 + blur * DOF_BLUR_EXPANSION,
	};
}
function getSelectionFogScale() {
	return selected ? SELECTED_FOG_SCALE : 1;
}
function getFocusProminence(score, focus) {
	const absolute = clamp(score || 0, 0, 1);
	const relative = normalizeFocusScore(absolute, focus);
	return Math.max(absolute, relative * FOCUS_VISIBILITY_END);
}
function getFocusVisibility(score, focus) {
	const prominence = getFocusProminence(score, focus);
	const t = invLerp(FOCUS_VISIBILITY_START, FOCUS_VISIBILITY_END, prominence);
	if (t <= 0) return 0;
	return Math.pow(t, FOCUS_VISIBILITY_POWER);
}
function getNodeVisual(n, focus) {
	const baseAlpha = n.kind === "tag" ? 0.92 : 0.72;
	if (!focus) return { color: n.color, alpha: baseAlpha, scale: 1 };
	if (n.key === focus.selectedKey) {
		return { color: focus.selectedColor, alpha: 1.0, scale: 1.35 };
	}

	const score = focus.scores.get(n.key);
	if (score === undefined) {
		return { color: mixColor(n.color, getThemeShadowColor(), 0.72), alpha: n.kind === "tag" ? 0.08 : 0.035, scale: 0.92 };
	}

	const visibility = clamp(score, 0, 1);
	const prominence = clamp(visibility * 0.78 + (n.importance || 0) * 0.22, 0, 1);
	const accent = mixColor(currentTheme.accents.blue, currentTheme.cursor, clamp(0.12 + prominence * 0.88, 0, 1));
	return {
		color: mixColor(n.color, accent, 0.18 + 0.58 * prominence),
		alpha: visibility,
		scale: 0.96 + 0.28 * visibility,
	};
}
function getUnfocusedNodeVisual(n) {
	const baseAlpha = n.kind === "tag" ? 0.92 : 0.72;
	const sliderPosition = n.dimension_opacity ?? 0.5;
	return { color: n.color, alpha: applySliderOpacity(baseAlpha, sliderPosition), scale: 1 };
}
function getEdgeVisual(e, focus) {
	const strength = edgeStrength(e);
	const color = recencyColor(e.recency || 0);
	if (!focus) {
		const sliderPosition = getEdgeSliderPosition(e);
		const alpha = applyEdgeOpacity(applySliderOpacity(e.knn ? 0.18 + strength * 0.30 : 0.18 + strength * 0.40, sliderPosition));
		return { color, alpha };
	}

	const score = focus.edgeScores.get(e);
	if (score === undefined) return { color: mixColor(currentTheme.background, currentTheme.foreground, 0.1), alpha: 0 };

	const visibility = clamp(score, 0, 1);
	return {
		color,
		alpha: applyEdgeOpacity(0.34 + visibility * (e.knn ? 0.51 : 0.66), { ignoreSlider: true }),
	};
}
function getEdgeLineVisual(e, focus) {
	const vis = getEdgeVisual(e, focus);

	if (focus) {
		return vis.alpha > 0.001 ? vis : { color: vis.color, alpha: 0 };
	}

	const alpha = hoveredEdge === e
		? Math.max(vis.alpha, applyEdgeOpacity(0.32 + edgeStrength(e) * 0.24))
		: vis.alpha;

	return { color: vis.color, alpha };
}
function getEdgeMarkerVisual(e, focus) {
	const strength = edgeStrength(e);
	const color = recencyColor(e.recency || 0);
	if (!focus) {
		const sliderPosition = getEdgeSliderPosition(e);
		const baseAlpha = applySliderOpacity(0.22 + strength * 0.34, sliderPosition);
		return {
			radius: 4.8 + strength * 4.4 + (e.knn ? 0 : 0.6),
			scale: 0.028 + strength * 0.02 + (e.knn ? 0 : 0.003),
			alpha: applyEdgeMarkerOpacity(baseAlpha, {
				ignoreSlider: true,
				minAlpha: Math.min(0.42, EDGE_MARKER_MIN_ALPHA + strength * 0.18),
			}),
			lineWidth: 0.9 + strength * 0.45,
			color,
		};
	}

	const score = focus.edgeScores.get(e);
	if (score === undefined) return { radius: 3.8 + strength * 2.4, scale: 0.022 + strength * 0.01, alpha: 0, lineWidth: 0.8, color: getThemeShadowColor() };

	const visibility = clamp(score, 0, 1);
	return {
		radius: 4.4 + strength * 3.8 + visibility * 2.4,
		scale: 0.026 + strength * 0.018 + visibility * 0.014,
		alpha: applyEdgeMarkerOpacity(0.42 + visibility * 0.58, {
			ignoreSlider: true,
			minAlpha: EDGE_MARKER_MIN_FOCUS_ALPHA,
		}),
		lineWidth: 0.9 + strength * 0.4 + visibility * 0.55,
		color,
	};
}
function isNodeInteractive(n, focus, alpha) {
	if (alpha < INTERACTIVE_ALPHA_THRESHOLD) return false;
	if (!focus) return true;
	return n.key === focus.selectedKey || focus.scores.has(n.key);
}
function isEdgeInteractive(e, focus, alpha) {
	if (alpha < INTERACTIVE_ALPHA_THRESHOLD) return false;
	if (!focus) return true;
	return focus.edgeScores.has(e);
}
function isSpecialistNode(raw) {
	const source = (raw?.source || "").toLowerCase();
	const label = (raw?.label || "").toLowerCase();
	return source.startsWith("specialist:") || label.includes("[specialist:");
}
function getNodeTypeLabel(node) {
	return node?.kind === "tag" ? "shard" : "thought";
}
function getNodeIconGlyph(node) {
	return node?.kind === "tag" ? "●" : "■";
}
function getNodeShape(node) {
	const shape = node?.shape;
	if (shape === "circle" || shape === "cube" || shape === "triangle") return shape;
	return node?.kind === "tag" ? "circle" : "cube";
}
function getNodeShapeScale(shape) {
	switch (shape) {
		case "circle": return 1.04;
		case "triangle": return 0.96;
		default: return 1.0;
	}
}
function get_node_size_multiplier(importance) {
	return 0.55 + 1.05 * clamp(importance || 0, 0, 1);
}
function get_node_world_radius(n) {
	const sizeMul = n.sizeMul || get_node_size_multiplier(n.importance || 0);
	const radius = n.r * sizeMul * 1.9 * getNodeShapeScale(getNodeShape(n));
	return Math.max(radius, n.kind === "tag" ? 0.08 : 0.03);
}
function getNodePrimitiveRadius(n, visualScale = 1) {
	const sizeMul = n.sizeMul || get_node_size_multiplier(n.importance || 0);
	const baseRadius = n.kind === "tag" ? 7.2 : 4.1;
	const radius = baseRadius * sizeMul * getNodeShapeScale(getNodeShape(n));
	const floor = n.kind === "tag" ? 4.5 : 2.4;
	return Math.max(radius, floor) * visualScale;
}
function getNodeIconMarkup(node) {
	const kind = node?.kind === "tag" ? "shard" : "thought";
	return `<span class="list-icon list-icon-${kind}" aria-hidden="true">${getNodeIconGlyph(node)}</span>`;
}
function getEdgeIconMarkup() {
	return '<span class="list-icon list-icon-edge" aria-hidden="true">▲</span>';
}
function setOrbitTarget(pos, snap = false) {
	orbitTarget = [pos[0], pos[1], pos[2]];
	if (snap) orbitCenter = [pos[0], pos[1], pos[2]];
}
function requestRender() {
	if (frameHandle) return;
	frameHandle = requestAnimationFrame(frame);
}
function updateHoverState(x, y) {
	const nextHoveredEdge = pickEdgeAt(x, y);
	const nextHoveredNode = nextHoveredEdge ? null : picshardeAt(x, y);
	const hoverChanged = hoveredEdge !== nextHoveredEdge || hoveredNode !== nextHoveredNode;
	hoveredEdge = nextHoveredEdge;
	hoveredNode = nextHoveredNode;
	cvs.style.cursor = (hoveredNode || hoveredEdge) ? "pointer" : "grab";
	if (hoverChanged) requestRender();
}
function isPointerGestureClick(gesture, event) {
	if (!gesture || gesture.pointerId !== event.pointerId) return false;
	const dx = event.clientX - gesture.startX;
	const dy = event.clientY - gesture.startY;
	return dx * dx + dy * dy <= CLICK_MOVE_THRESHOLD * CLICK_MOVE_THRESHOLD;
}
function hasCameraMotion() {
	if (drag) return true;
	if (Math.abs(vYaw) > 0.0001 || Math.abs(vPitch) > 0.0001) return true;
	return (
		Math.abs(orbitCenter[0] - orbitTarget[0]) > 0.0005 ||
		Math.abs(orbitCenter[1] - orbitTarget[1]) > 0.0005 ||
		Math.abs(orbitCenter[2] - orbitTarget[2]) > 0.0005
	);
}
function syncOrbitTarget(snap = false) {
	const pos = selected?.pos || graph_center;
	setOrbitTarget(pos, snap);
}
function rotateIntoCamera(point, center, yawAngle = yaw, pitchAngle = pitch) {
	const px = point[0] - center[0], py = point[1] - center[1], pz = point[2] - center[2];

	const cy = Math.cos(-yawAngle), sy = Math.sin(-yawAngle);
	const yawX = px * cy + pz * sy;
	const yawY = py;
	const yawZ = -px * sy + pz * cy;

	const cx = Math.cos(-pitchAngle), sx = Math.sin(-pitchAngle);
	return [
		yawX,
		yawY * cx - yawZ * sx,
		yawY * sx + yawZ * cx,
	];
}
function get_node_extent(n) {
	return get_node_world_radius(n) * FRAME_PAD;
}
function getProjectedBounds(center, distance, yawAngle = yaw, pitchAngle = pitch) {
	const aspect = Math.max(W / Math.max(H, 1), 0.1);
	const tanHalf = Math.tan(FOV / 2);
	const margin = Math.min(FRAME_MARGIN_PX, Math.min(W, H) * 0.18);
	const bounds = { minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity, maxExtent: 0.6, visible: true, margin };

	for (const n of all_nodes) {
		const rr = get_node_extent(n);
		const [rx, ry, rz] = rotateIntoCamera(n.pos, center, yawAngle, pitchAngle);
		const w = distance - rz;
		if (w <= 0.001) {
			bounds.visible = false;
			return bounds;
		}
		const ndcX = rx / (w * tanHalf * aspect);
		const ndcY = ry / (w * tanHalf);
		const sx = (ndcX * 0.5 + 0.5) * W;
		const sy = (1 - (ndcY * 0.5 + 0.5)) * H;
		const radiusPx = Math.max((rr / (w * tanHalf * aspect)) * W * 0.5, (rr / (w * tanHalf)) * H * 0.5);

		bounds.minX = Math.min(bounds.minX, sx - radiusPx);
		bounds.maxX = Math.max(bounds.maxX, sx + radiusPx);
		bounds.minY = Math.min(bounds.minY, sy - radiusPx);
		bounds.maxY = Math.max(bounds.maxY, sy + radiusPx);
		bounds.maxExtent = Math.max(bounds.maxExtent, Math.abs(rx) + rr, Math.abs(ry) + rr, Math.abs(rz) + rr);
	}

	return bounds;
}
function fitsViewport(center, distance, yawAngle = yaw, pitchAngle = pitch) {
	const bounds = getProjectedBounds(center, distance, yawAngle, pitchAngle);
	if (!bounds.visible) return { fits: false, bounds };
	const fits =
		bounds.minX >= bounds.margin &&
		bounds.maxX <= W - bounds.margin &&
		bounds.minY >= bounds.margin &&
		bounds.maxY <= H - bounds.margin;
	return { fits, bounds };
}
function fitDistanceForBounds(center, yawAngle = yaw, pitchAngle = pitch) {
	if (!all_nodes.length) return 0.8;

	let low = 0.05;
	let high = 1.0;
	let probe = fitsViewport(center, high, yawAngle, pitchAngle);
	let expandCount = 0;
	while (!probe.fits && expandCount < 24) {
		low = high;
		high *= 1.6;
		probe = fitsViewport(center, high, yawAngle, pitchAngle);
		expandCount += 1;
	}

	let lastBounds = probe.bounds;
	for (let i = 0; i < 28; i++) {
		const mid = (low + high) * 0.5;
		const test = fitsViewport(center, mid, yawAngle, pitchAngle);
		lastBounds = test.bounds;
		if (test.fits) high = mid;
		else low = mid;
	}

	graph_radius = Math.max(0.6, lastBounds?.maxExtent || 0.6);
	return high;
}
function updateCameraBounds(resetView = true) {
	if (!all_nodes.length) {
		graph_center = [0, 0, 0];
		graph_radius = POLY_R;
		min_dist = 0.7;
		base_min_dist = 0.7;
		maxDist = 6;
		reset_dist = 3;
		dist = resetView ? reset_dist : clamp(dist, min_dist, maxDist);
		if (resetView) syncOrbitTarget(true);
		return;
	}

	let minX = Infinity, minY = Infinity, minZ = Infinity;
	let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
	let maxNodeR = 0;

	for (const n of all_nodes) {
		const rr = get_node_world_radius(n);
		maxNodeR = Math.max(maxNodeR, rr);
		minX = Math.min(minX, n.pos[0] - rr);
		minY = Math.min(minY, n.pos[1] - rr);
		minZ = Math.min(minZ, n.pos[2] - rr);
		maxX = Math.max(maxX, n.pos[0] + rr);
		maxY = Math.max(maxY, n.pos[1] + rr);
		maxZ = Math.max(maxZ, n.pos[2] + rr);
	}

	graph_center = [(minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2];
	const fitDist = fitDistanceForBounds(graph_center, yaw, pitch);
	base_min_dist = Math.max(maxNodeR * 2.0, fitDist * 0.08);
	min_dist = base_min_dist;
	maxDist = Math.max(min_dist * 1.4, fitDist);
	reset_dist = clamp(fitDist * DEFAULT_VIEW, min_dist, maxDist);
	dist = resetView ? reset_dist : clamp(dist, min_dist, maxDist);
	if (resetView) syncOrbitTarget(true);
}

function build_dimension_model() {
	const popularity = new Map();

	for (const n of all_nodes) {
		const signals = collectNodeTagSignals(n);
		const peak = Math.max(...signals.values(), 0, 1);
		const tagAffinity = {};
		let tagSignalMass = 0;

		for (const [key, value] of signals) {
			const affinity = clamp(value / peak, 0, 1.6);
			tagAffinity[key] = affinity;
			tagSignalMass += affinity;

			if (n.kind !== "thought") continue;
			const stat = popularity.get(key) || { key, label: key, popularity: 0, nodes: 0 };
			stat.popularity += affinity * (0.35 + (n.baseImportance || 0) * 0.65);
			stat.nodes += 1;
			popularity.set(key, stat);
		}

		n.tagAffinity = tagAffinity;
		n.tagSignalMass = Math.max(tagSignalMass, 1);
	}

	const thoughts_by_store = new Map();
	for (const n of thought_nodes) {
		const key = n.store || "global";
		if (!thoughts_by_store.has(key)) thoughts_by_store.set(key, []);
		thoughts_by_store.get(key).push(n);
	}

	for (const node of tag_nodes) {
		const bucket = thoughts_by_store.get(node.label) || thoughts_by_store.get(node.store || "") || [];
		if (!bucket.length) continue;

		const aggregate = new Map();
		let weightTotal = 0;

		for (const thought of bucket) {
			const weight = 0.35 + (thought.baseImportance || 0) * 0.65;
			weightTotal += weight;
			for (const [key, affinity] of Object.entries(thought.tagAffinity || {})) {
				aggregate.set(key, (aggregate.get(key) || 0) + affinity * weight);
			}
		}

		if (weightTotal <= 0) continue;

		const tagAffinity = {};
		let tagSignalMass = 0;
		for (const [key, value] of aggregate.entries()) {
			const affinity = value / weightTotal;
			tagAffinity[key] = affinity;
			tagSignalMass += affinity;
		}

		node.tagAffinity = tagAffinity;
		node.tagSignalMass = Math.max(tagSignalMass, 1);
	}

	let ranked = [...popularity.values()].filter(tag => tag.nodes >= 2);
	if (ranked.length < 12) ranked = [...popularity.values()];

	ranked.sort((a, b) => {
		if (b.popularity !== a.popularity) return b.popularity - a.popularity;
		return a.label.localeCompare(b.label);
	});

	dimension_tags = ranked.slice(0, MAX_DIMENSIONS).map((tag, index) => ({
		...tag,
		index,
		label: formatDimensionLabel(tag.label),
	}));

	const activeKeys = new Set(dimension_tags.map(tag => tag.key));
	dimension_weights = new Map([...dimension_weights.entries()].filter(([key]) => activeKeys.has(key)));
	dimension_vectors = new Map();
	const baseline = getBaselineDimensionWeight();
	dimension_tags.forEach((tag, index) => {
		dimension_vectors.set(tag.key, fibonacciSpherePoint(index, dimension_tags.length));
		if (!dimension_weights.has(tag.key)) dimension_weights.set(tag.key, baseline);
	});

	tag_nodes.forEach(node => {
		const match = dimension_tags.find(tag => tag.label === node.label || tag.key === String(node.label || "").toLowerCase());
		node.tagKey = match?.key || "";
	});
}

function update_dimension_driven_state() {
	recomputeDimensionSelection();
	for (const n of all_nodes) {
		const ranked = [];
		let weightedTotal = 0;
		let affinityTotal = 0;
		let opacityWeightedSum = 0;
		let hasOpacityMatch = false;

		for (const tag of dimension_tags) {
			const affinity = n.tagAffinity?.[tag.key] || 0;
			if (affinity <= 0) continue;

			const weight = getDimensionWeight(tag.key);
			const ratio = getDimensionRatio(weight);
			const sliderPosition = getDimensionSliderPosition(tag.key);
			const weighted = affinity * ratio;
			ranked.push({ key: tag.key, label: tag.label, score: weighted, affinity });
			weightedTotal += weighted;
			affinityTotal += affinity;
			opacityWeightedSum += sliderPosition * affinity;
			hasOpacityMatch = true;
		}

		ranked.sort((a, b) => b.score - a.score);
		n.topDimensions = ranked.slice(0, 3);
		n.dimension_opacity = hasOpacityMatch && affinityTotal > 0
			? clamp(opacityWeightedSum / affinityTotal, 0, 1)
			: 0.5;

		const adjustedImportance = n.baseImportance || 0;
		n.importance = adjustedImportance;
		n.sizeMul = get_node_size_multiplier(adjustedImportance);
	}

	const thoughts_by_store = new Map();
	for (const node of thought_nodes) {
		const key = node.store || "global";
		if (!thoughts_by_store.has(key)) thoughts_by_store.set(key, []);
		thoughts_by_store.get(key).push(node);
	}

	for (const node of tag_nodes) {
		const bucket = thoughts_by_store.get(node.label) || thoughts_by_store.get(node.store || "") || [];
		if (!bucket.length) continue;

		let strongest = 0;
		let total = 0;
		for (const thought of bucket) {
			const opacity = thought.dimension_opacity ?? 0.5;
			strongest = Math.max(strongest, opacity);
			total += opacity;
		}

		const average = total / bucket.length;
		node.dimension_opacity = clamp(strongest * 0.7 + average * 0.3, 0, 1);
	}
}

function apply_importance_layout(tagPos, angleOf) {
	const ordered_tags = sort_nodes_for_spiral(tag_nodes);
	const tagSpan = get_spiral_height_span() * 0.58;
	const shard_min_y = -tagSpan * 0.5;
	const shard_max_y = tagSpan * 0.5;
	let min_shard_y = Infinity;
	let max_shard_y = -Infinity;

	for (let i = 0; i < ordered_tags.length; i++) {
		const n = ordered_tags[i];
		const rank_t = ordered_tags.length <= 1 ? 0.5 : i / (ordered_tags.length - 1);
		const angle_t = ordered_tags.length <= 1 ? 0.5 : i / ordered_tags.length;
		const angle = -Math.PI * 0.5 + angle_t * Math.PI * 2;
		const sphere_scale = get_spiral_sphere_scale(rank_t);
		const shard_radius = SHARD_RING_RADIUS * sphere_scale;
		const basePos = [
			shard_radius * Math.cos(angle),
			(rank_t - 0.5) * tagSpan,
			shard_radius * Math.sin(angle),
		];
		n.pos = scalePositionFromCenter(twistPositionByHeight(basePos, shard_min_y, shard_max_y));
		tagPos[n.label] = basePos;
		angleOf[n.label] = angle;
		min_shard_y = Math.min(min_shard_y, basePos[1]);
		max_shard_y = Math.max(max_shard_y, basePos[1]);
	}

	if (!Number.isFinite(min_shard_y) || !Number.isFinite(max_shard_y)) {
		min_shard_y = -tagSpan * 0.5;
		max_shard_y = tagSpan * 0.5;
	}

	if (!thought_nodes.length) return;

	const store_buckets = {};
	const store_bounds = {};
	const pending_thought_positions = [];
	for (const n of thought_nodes) {
		const key = n.store || "global";
		(store_buckets[key] ||= []).push(n);
		const vec = n.embedPos || [0, 0, 0];
		const bound = (store_bounds[key] ||= [1e-6, 1e-6, 1e-6]);
		bound[0] = Math.max(bound[0], Math.abs(vec[0] || 0));
		bound[1] = Math.max(bound[1], Math.abs(vec[1] || 0));
		bound[2] = Math.max(bound[2], Math.abs(vec[2] || 0));
	}

	const spiralSpan = get_spiral_height_span();
	for (const [storeKey, bucket] of Object.entries(store_buckets)) {
		const isGlobal = storeKey === "global";
		const ordered = sort_nodes_for_spiral(bucket);
		const anchorAngle = isGlobal ? -Math.PI * 0.5 : (angleOf[storeKey] ?? 0);
		const tagAnchor = isGlobal ? [0, 0, 0] : (tagPos[storeKey] || [0, 0, 0]);
		const center = isGlobal
			? [0, 0, 0]
			: [tagAnchor[0] * SHARD_INNER_PULL, tagAnchor[1] * 0.82, tagAnchor[2] * SHARD_INNER_PULL];
		const radialDir = [Math.cos(anchorAngle), 0, Math.sin(anchorAngle)];
		const tangentDir = [-radialDir[2], 0, radialDir[0]];
		const radius = isGlobal ? GLOBAL_SPIRAL_RADIUS : THOUGHT_SPIRAL_RADIUS;
		const bounds = store_bounds[storeKey] || [1, 1, 1];

		for (let i = 0; i < ordered.length; i++) {
			const n = ordered[i];
			const rank_t = ordered.length <= 1 ? 0.5 : i / (ordered.length - 1);
			const sphere_scale = get_spiral_sphere_scale(rank_t);
			const raw = n.embedPos || [0, 0, 0];
			const vx = clamp((raw[0] || 0) / bounds[0], -1.15, 1.15);
			const vz = clamp((raw[2] || 0) / bounds[2], -1.15, 1.15);
			const lateral = vx * radius * sphere_scale;
			const depth = vz * radius * (isGlobal ? 0.42 : 0.55) * sphere_scale;
			const taperedCenter = [center[0] * sphere_scale, center[1], center[2] * sphere_scale];

			const basePos = [
				taperedCenter[0] + tangentDir[0] * lateral + radialDir[0] * depth,
				center[1] + (rank_t - 0.5) * spiralSpan,
				taperedCenter[2] + tangentDir[2] * lateral + radialDir[2] * depth,
			];

			pending_thought_positions.push({ node: n, basePos });
		}
	}

	let rawThoughtMinY = Infinity;
	let rawThoughtMaxY = -Infinity;
	for (const entry of pending_thought_positions) {
		rawThoughtMinY = Math.min(rawThoughtMinY, entry.basePos[1]);
		rawThoughtMaxY = Math.max(rawThoughtMaxY, entry.basePos[1]);
	}

	const shard_mid_y = (min_shard_y + max_shard_y) * 0.5;
	const thoughtMidY = Number.isFinite(rawThoughtMinY) && Number.isFinite(rawThoughtMaxY)
		? (rawThoughtMinY + rawThoughtMaxY) * 0.5
		: shard_mid_y;

	for (const entry of pending_thought_positions) {
		const normalizedY = rawThoughtMaxY - rawThoughtMinY > 1e-6
			? lerp(min_shard_y, max_shard_y, invLerp(rawThoughtMinY, rawThoughtMaxY, entry.basePos[1]))
			: shard_mid_y;
		entry.basePos[1] = normalizedY;
		entry.node.pos = scalePositionFromCenter(twistPositionByHeight(entry.basePos, min_shard_y, max_shard_y));
	}
}

function sync_spiral_controls() {
	const heightInput = document.getElementById("spiral-height-slider");
	if (heightInput) heightInput.value = get_spiral_height_span().toFixed(1);
	const heightMeta = document.getElementById("spiral-height-meta");
	if (heightMeta) heightMeta.textContent = getSpiralHeightLabel();

	const spreadInput = document.getElementById("spiral-spread-slider");
	if (spreadInput) spreadInput.value = getSpiralSpreadScale().toFixed(2);
	const spreadMeta = document.getElementById("spiral-spread-meta");
	if (spreadMeta) spreadMeta.textContent = getSpiralSpreadLabel();

	const edge_input = document.getElementById("edge-opacity-slider");
	if (edge_input) edge_input.value = getEdgeOpacityScale().toFixed(2);
	const edgeMeta = document.getElementById("edge-opacity-meta");
	if (edgeMeta) edgeMeta.textContent = getEdgeOpacityLabel();
}

function syncDimensionControls() {
	const summaryEl = document.getElementById("dimension-summary");
	if (summaryEl) summaryEl.textContent = describeDimensionState();
	const countEl = document.getElementById("s-dimensions");
	if (countEl) countEl.textContent = dimension_tags.length;

	for (const tag of dimension_tags) {
		const controls = dimensionControlEls.get(tag.key);
		if (!controls) continue;
		const weight = getDimensionWeight(tag.key);
		controls.input.value = String(Math.round(weightToSliderPosition(weight) * DIMENSION_SLIDER_SCALE));
		controls.meta.textContent = `#${tag.index + 1} · ${tag.nodes} nodes · ${(weight * 100).toFixed(2)}%`;
		controls.row.classList.toggle("active", weight > getBaselineDimensionWeight() * 1.15);
	}
}

function relayout_scene({ resetView = false, recomputeMesh = false } = {}) {
	update_dimension_driven_state();
	if (recomputeMesh) {
		apply_importance_layout(layout_state.tagPos, layout_state.angleOf);
		updateCameraBounds(resetView);
		if (selected) setOrbitTarget(selected.pos, resetView);
		else setOrbitTarget(graph_center, resetView);
	}
	refresh_scene_buffers();
	if (selected) showDetail(selected);
	sync_spiral_controls();
	syncDimensionControls();
}

function rebalanceDimension(tagKey, nextWeight) {
	if (!dimension_tags.length || !dimension_weights.has(tagKey)) return;

	const clampedWeight = clamp(nextWeight, 0, 1);
	const oldWeight = getDimensionWeight(tagKey);
	const others = dimension_tags.filter(tag => tag.key !== tagKey);
	const oldOtherTotal = others.reduce((sum, tag) => sum + getDimensionWeight(tag.key), 0);
	const remaining = Math.max(0, 1 - clampedWeight);

	dimension_weights.set(tagKey, clampedWeight);
	if (others.length) {
		if (oldOtherTotal <= 1e-6) {
			const shared = remaining / others.length;
			for (const tag of others) dimension_weights.set(tag.key, shared);
		} else {
			const scale = remaining / oldOtherTotal;
			for (const tag of others) dimension_weights.set(tag.key, getDimensionWeight(tag.key) * scale);
		}
	}

	if (Math.abs(oldWeight - clampedWeight) > 1e-6) relayout_scene({ resetView: false, recomputeMesh: false });
}

function resetDimensionWeights() {
	if (!dimension_tags.length) return;
	const baseline = getBaselineDimensionWeight();
	for (const tag of dimension_tags) dimension_weights.set(tag.key, baseline);
	relayout_scene({ resetView: false, recomputeMesh: false });
}

function renderDimensionControls() {
	const root = document.getElementById("dimension-controls");
	if (!root) return;

	root.innerHTML = "";
	dimensionControlEls = new Map();

	if (!dimension_tags.length) {
		const empty = document.createElement("div");
		empty.className = "dimension-empty";
		empty.textContent = "No stable graph tags yet. Add more varied thoughts and sources to unlock dimension balancing.";
		root.appendChild(empty);
		syncDimensionControls();
		return;
	}

	for (const tag of dimension_tags) {
		const row = document.createElement("div");
		row.className = "dimension-row";

		const head = document.createElement("div");
		head.className = "dimension-row-head";

		const name = document.createElement("div");
		name.className = "dimension-name";
		name.textContent = tag.label;

		const meta = document.createElement("div");
		meta.className = "dimension-meta";

		const input = document.createElement("input");
		input.type = "range";
		input.min = "0";
		input.max = String(DIMENSION_SLIDER_SCALE);
		input.step = "1";
		input.value = String(Math.round(weightToSliderPosition(getDimensionWeight(tag.key)) * DIMENSION_SLIDER_SCALE));
		input.addEventListener("input", () => rebalanceDimension(tag.key, sliderPositionToWeight(Number(input.value) / DIMENSION_SLIDER_SCALE)));

		head.appendChild(name);
		head.appendChild(meta);
		row.appendChild(head);
		row.appendChild(input);
		root.appendChild(row);
		dimensionControlEls.set(tag.key, { row, input, meta });
	}

	const resetButton = document.getElementById("dimension-reset");
	if (resetButton) resetButton.onclick = resetDimensionWeights;
	syncDimensionControls();
}

function setupSpiralControls() {
	const heightInput = document.getElementById("spiral-height-slider");
	if (heightInput) {
		heightInput.value = get_spiral_height_span().toFixed(1);
		heightInput.addEventListener("input", () => {
			spiral_height_scale = clamp(Number(heightInput.value), 0, SPIRAL_HEIGHT_MAX);
			write_stored_viz_settings();
			relayout_scene({ resetView: false, recomputeMesh: true });
		});
	}

	const spreadInput = document.getElementById("spiral-spread-slider");
	if (spreadInput) {
		spreadInput.value = getSpiralSpreadScale().toFixed(2);
		spreadInput.addEventListener("input", () => {
			spiral_spread_scale = clamp(Number(spreadInput.value), -1, 1);
			write_stored_viz_settings();
			relayout_scene({ resetView: false, recomputeMesh: true });
		});
	}

	const edge_input = document.getElementById("edge-opacity-slider");
	if (edge_input) {
		edge_input.value = getEdgeOpacityScale().toFixed(2);
		edge_input.addEventListener("input", () => {
			edge_opacity_scale = clamp(Number(edge_input.value), 0, 1);
			write_stored_viz_settings();
			sync_spiral_controls();
			refresh_scene_buffers();
		});
	}

	if (!heightInput && !spreadInput && !radiusInput && !edge_input) return;
	sync_spiral_controls();
}

function setupThemeControls() {
	const select = document.getElementById("theme-select");
	if (!select) return;
	select.innerHTML = "";
	for (const theme of THEMES) {
		const option = document.createElement("option");
		option.value = theme.id;
		option.textContent = theme.name;
		select.appendChild(option);
	}
	select.addEventListener("input", () => applyTheme(select.value));
	syncThemeControls();
}

function setupRecencyColorControls() {
	const picker = document.getElementById("recency-picker");
	if (!picker) return;
	let openTarget = null; // "recent" | "old" | null

	function render() {
		picker.innerHTML = "";

		// Main row: two swatches with labels
		const row = document.createElement("div");
		row.className = "recency-main-row";

		for (const which of ["recent", "old"]) {
			const key = which === "recent" ? recencyRecentKey : recencyOldKey;
			const accent = currentTheme.accents[key];
			const wrap = document.createElement("div");
			wrap.className = "recency-main-wrap" + (openTarget === which ? " is-open" : "");

			const swatch = document.createElement("div");
			swatch.className = "recency-main-swatch";
			swatch.style.background = rgba(accent);
			swatch.title = `${which}: ${key}`;
			swatch.addEventListener("click", () => {
				openTarget = openTarget === which ? null : which;
				render();
			});

			const label = document.createElement("span");
			label.className = "recency-main-label";
			label.textContent = which;

			wrap.appendChild(swatch);
			wrap.appendChild(label);
			row.appendChild(wrap);
		}
		picker.appendChild(row);

		// Expandable options row
		if (openTarget) {
			const options = document.createElement("div");
			options.className = "recency-options";
			const currentKey = openTarget === "recent" ? recencyRecentKey : recencyOldKey;
			for (const key of RECENCY_COLOR_OPTIONS) {
				const opt = document.createElement("div");
				opt.className = "recency-option-swatch" + (key === currentKey ? " is-active" : "");
				opt.style.background = rgba(currentTheme.accents[key]);
				opt.title = key;
				opt.addEventListener("click", () => {
					if (openTarget === "recent") recencyRecentKey = key;
					else recencyOldKey = key;
					openTarget = null;
					applyRecencyChange();
					render();
				});
				options.appendChild(opt);
			}
			picker.appendChild(options);
		}
	}

	function applyRecencyChange() {
		recolorNodesForTheme();
		write_stored_viz_settings();
		refresh_scene_buffers();
	}

	render();
	window._refreshRecencySwatches = render;
}

// ---- 3D layout ----
function layout(data) {
	if (!data?.nodes) return;
	all_nodes = []; tag_nodes = []; thought_nodes = []; edges = [];
	node_idx = {}; nbrs = {}; edgeAdj = {}; store_clr = {}; clrI = 0;

	const shards = [...data.nodes.filter(n => n.type === "shard")]
		.sort((a, b) => String(a.label || a.key).localeCompare(String(b.label || b.key)));
	const rest = [...data.nodes.filter(n => n.type !== "shard")]
		.sort((a, b) => String(a.key).localeCompare(String(b.key)));
	const N = shards.length, angleOf = {}, tagPos = {};
	rawEdgeData = { edges: data.edges || [], knn_edges: data.knn_edges || [] };
	const now = Date.now() / 1000;

	shards.forEach((s, i) => {
		const a = (2 * Math.PI * i) / Math.max(N, 1);
		angleOf[s.label] = a;
		tagPos[s.label] = [SHARD_RING_RADIUS * Math.cos(a), 0, SHARD_RING_RADIUS * Math.sin(a)];
		const col = sClr(s.label);
		const nd = {
			key: s.key, pos: [...tagPos[s.label]], r: 0.12, color: col, paletteKey: s.label,
			label: s.label, kind: "tag", shape: "circle", store: s.store, source: s.source || "",
			access: s.access_count || 0, created: s.created_at || 0, lastAccess: s.last_accessed || 0
		};
		all_nodes.push(nd); tag_nodes.push(nd); node_idx[nd.key] = nd;
	});

	const sw = N > 0 ? (2 * Math.PI) / N : 2 * Math.PI;
	layout_state = { tagPos, angleOf, sw };
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
				color: col, paletteKey: isG ? "global" : store, label: t.label, kind: "thought", shape: "cube", store: isG ? "global" : store,
				source: t.source || "", access: t.access_count || 0, created: t.created_at || 0, lastAccess: t.last_accessed || 0,
				embedPos: Array.isArray(t.embed_pos) ? t.embed_pos : null
			};
			all_nodes.push(nd); thought_nodes.push(nd); node_idx[nd.key] = nd;
		}
	}

	for (const e of data.edges) {
		const s = node_idx[e.source], t = node_idx[e.target];
		if (s && t) edges.push({
			s, t,
			w: e.weight,
			reason: e.reasoning || "",
			success: e.success_rate || 0,
			traversals: e.traversal_count || 0,
			created: e.created_at || 0,
			parent: e.source.startsWith("_") || e.target.startsWith("_"),
			knn: false,
		});
	}
	if (data.knn_edges) for (const e of data.knn_edges) {
		const s = node_idx[e.source], t = node_idx[e.target];
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
		(edgeAdj[e.s.key] ||= []).push(e);
		(edgeAdj[e.t.key] ||= []).push(e);
		const eStamp = e.created || 0;
		e.recency = eStamp > 0 ? 1 / (1 + Math.max(0, now - eStamp) / (3600 * 24 * 14)) : 0;
	}

	for (const n of all_nodes) {
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
		n.color = recencyColor(n.recency);
		maxAccess = Math.max(maxAccess, n.access || 0);
		maxLinks = Math.max(maxLinks, n.linkCount);
		maxStrength = Math.max(maxStrength, strength);
		maxTraversal = Math.max(maxTraversal, traversalSum);
	}

	for (const n of all_nodes) {
		const accessScore = softScore(n.access || 0, maxAccess);
		const linkScore = softScore(n.linkCount || 0, maxLinks);
		const strengthScore = softScore(n.linkStrength || 0, maxStrength);
		const traversalScore = softScore(n.traversalLoad || 0, maxTraversal);
		const successScore = clamp(n.avgSuccess || 0, 0, 1);
		const recencyScore = clamp(n.recency || 0, 0, 1);
		const importance = n.kind === "tag"
			? clamp(0.18 + 0.24 * linkScore + 0.30 * strengthScore + 0.12 * successScore + 0.10 * traversalScore + 0.06 * recencyScore, 0, 1)
			: clamp(0.28 * accessScore + 0.18 * linkScore + 0.24 * strengthScore + 0.12 * successScore + 0.10 * traversalScore + 0.08 * recencyScore, 0, 1);
		n.baseImportance = importance;
		n.importance = importance;
		n.sizeMul = get_node_size_multiplier(importance);
	}

	build_dimension_model();
	renderDimensionControls();
	relayout_scene({ resetView: true, recomputeMesh: true });
}

let _appendRelayoutTimer = null;
function appendThoughts(batch) {
	if (!batch.length) return;
	const { angleOf, sw } = layout_state;
	const now = Date.now() / 1000;
	for (const t of batch) {
		if (node_idx[t.key]) continue;
		const st = t.store || "global";
		const isG = angleOf[st] === undefined;
		const base = isG ? 0 : angleOf[st];
		const col = sClr(isG ? "global" : st);
		const ac = Math.min(t.access_count || 0, 20) / 20;
		const theta = isG
			? hash01(`${t.key}:theta`) * Math.PI * 2
			: base + (hash01(`${t.key}:theta`) - 0.5) * sw * 0.72;
		const radial = POLY_R * (isG ? 0.18 + hash01(`${t.key}:rad`) * 0.18 : 0.24 + (1 - ac) * 0.14 + hash01(`${t.key}:rad`) * 0.18);
		const yJ = (hash01(`${t.key}:y`) - 0.5) * (isG ? 0.38 : 0.28);
		const stamp = Math.max(t.last_accessed || 0, t.created_at || 0);
		const nd = {
			key: t.key, pos: [radial * Math.cos(theta), yJ, radial * Math.sin(theta)],
			r: Math.max(0.02, Math.min(0.055, 0.02 + ac * 0.035)),
			color: col, paletteKey: isG ? "global" : st, label: t.label, kind: "thought", shape: "cube",
			store: isG ? "global" : st, source: t.source || "", access: t.access_count || 0,
			created: t.created_at || 0, lastAccess: t.last_accessed || 0, embedPos: Array.isArray(t.embed_pos) ? t.embed_pos : null,
			linkCount: 0, linkStrength: 0, avgSuccess: 0, traversalLoad: 0, importance: 0.3,
			recency: stamp > 0 ? 1 / (1 + Math.max(0, now - stamp) / (3600 * 24 * 14)) : 0,
		};
		nd.color = recencyColor(nd.recency);
		all_nodes.push(nd); thought_nodes.push(nd); node_idx[nd.key] = nd;
	}
	resolveEdges();
	recolorNodesForTheme();
	refresh_scene_buffers();
	// Debounce the full spiral relayout so parallel BFS chains don't thrash it
	clearTimeout(_appendRelayoutTimer);
	_appendRelayoutTimer = setTimeout(() => relayout_scene({ recomputeMesh: true }), 80);
}

function resolveEdges() {
	const edgeSet = new Set(edges.map(e => `${e.s.key}|${e.t.key}`));
	for (const e of [...rawEdgeData.edges, ...(rawEdgeData.knn_edges || [])]) {
		const s = node_idx[e.source], t = node_idx[e.target];
		if (!s || !t) continue;
		const k = `${s.key}|${t.key}`;
		if (edgeSet.has(k)) continue;
		edgeSet.add(k);
		const edgeObj = {
			s, t, w: e.weight, reason: e.reasoning || "", success: e.success_rate || 0,
			traversals: e.traversal_count || 0, created: e.created_at || 0,
			parent: e.source.startsWith("_") || e.target.startsWith("_"),
			knn: e.type === "knn",
		};
		edges.push(edgeObj);
		(nbrs[s.key] ||= new Set()).add(t.key);
		(nbrs[t.key] ||= new Set()).add(s.key);
		(edgeAdj[s.key] ||= []).push(edgeObj);
		(edgeAdj[t.key] ||= []).push(edgeObj);
	}
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

const EDGE_SHADER = `
struct U { mvp:mat4x4f, focal:f32, depthMin:f32, depthMax:f32, pad:f32, res:vec2f, focusFog:f32, pad2:f32 }
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
  let clip = u.mvp * vec4f(v.pos, 1.0);
  o.pos = clip;
  o.col = v.col;

  let depthRange = max(u.depthMax - u.depthMin, 0.001);
  let rawDepth = clamp((clip.w - u.depthMin) / depthRange, 0.0, 1.0);
	let ndc = clip.xy / max(clip.w, 0.0001);
	let radial = length(vec2f(ndc.x * 0.86, ndc.y));
	let focusStart = max(u.depthMin + depthRange * 0.04, u.focal * 0.16);
	let focusEnd = max(focusStart + 0.001, min(u.depthMax, u.focal * 1.20));
	let depthSpan = max(focusEnd - focusStart, u.focal * 0.36);
	let longitudinal = max((clip.w - focusStart) / depthSpan - 1.0, 0.0);
	let capsule = length(vec2f(radial / 0.34, longitudinal / 0.68));

  o.normDepth = smoothstep(0.04, 1.0, pow(rawDepth, 1.0));
	o.dof = smoothstep(0.82, 1.52, capsule);
  o.nearBoost = 1.0 - smoothstep(0.0, 0.38, rawDepth);
  return o;
}
@fragment fn fs(f:VO) -> @location(0) vec4f {
	let fogAmt = clamp(f.normDepth * 0.30 + f.dof * 0.38, 0.0, 0.78);
  let nearLit = f.col.rgb * (0.86 + 0.14 * f.nearBoost);
  let alpha = f.col.a * mix(1.0, 0.22, pow(fogAmt, 1.08));
	return vec4f(nearLit, alpha);
}
`;

const SPRITE_SHADER = `
struct U { mvp:mat4x4f, focal:f32, depthMin:f32, depthMax:f32, pad:f32, res:vec2f, focusFog:f32, pad2:f32 }
@group(0) @binding(0) var<uniform> u:U;
struct GV { @location(0) corner:vec2f }
struct IV { @location(1) center:vec3f, @location(2) radius:f32, @location(3) col:vec4f }
struct VO {
	@builtin(position) pos:vec4f,
	@location(0) local:vec2f,
	@location(1) col:vec4f,
	@location(2) dof:f32,
	@location(3) normDepth:f32,
	@location(4) nearBoost:f32,
}
@vertex fn vs(v:GV, inst:IV) -> VO {
	var o:VO;
	let clipCenter = u.mvp * vec4f(inst.center, 1.0);
	let depthRange = max(u.depthMax - u.depthMin, 0.001);
	let rawDepth = clamp((clipCenter.w - u.depthMin) / depthRange, 0.0, 1.0);
	let ndc = clipCenter.xy / max(clipCenter.w, 0.0001);
	let radial = length(vec2f(ndc.x * 0.86, ndc.y));
	let focusStart = max(u.depthMin + depthRange * 0.04, u.focal * 0.16);
	let focusEnd = max(focusStart + 0.001, min(u.depthMax, u.focal * 1.20));
	let depthSpan = max(focusEnd - focusStart, u.focal * 0.36);
	let longitudinal = max((clipCenter.w - focusStart) / depthSpan - 1.0, 0.0);
	let capsule = length(vec2f(radial / 0.34, longitudinal / 0.68));
	let dof = smoothstep(0.82, 1.52, capsule);
	let blurScale = mix(1.0, 1.26, dof);
	let offset = vec2f(
		v.corner.x * inst.radius * blurScale * 2.0 / max(u.res.x, 1.0) * clipCenter.w,
		v.corner.y * inst.radius * blurScale * 2.0 / max(u.res.y, 1.0) * clipCenter.w
	);
	o.pos = clipCenter + vec4f(offset, 0.0, 0.0);
	o.local = v.corner;
	o.col = inst.col;

	o.normDepth = smoothstep(0.04, 1.0, pow(rawDepth, 1.0));
	o.dof = dof;
	o.nearBoost = 1.0 - smoothstep(0.0, 0.38, rawDepth);
	return o;
}
fn triA() -> vec2f { return vec2f(0.0, 0.94); }
fn triB() -> vec2f { return vec2f(-0.82, -0.48); }
fn triC() -> vec2f { return vec2f(0.82, -0.48); }
fn edgeCross(a:vec2f, b:vec2f, p:vec2f) -> f32 {
	return (p.x - a.x) * (b.y - a.y) - (p.y - a.y) * (b.x - a.x);
}
fn triangleInside(p:vec2f) -> bool {
	let e0 = edgeCross(triA(), triB(), p);
	let e1 = edgeCross(triB(), triC(), p);
	let e2 = edgeCross(triC(), triA(), p);
	return (e0 >= 0.0 && e1 >= 0.0 && e2 >= 0.0) || (e0 <= 0.0 && e1 <= 0.0 && e2 <= 0.0);
}
fn segmentDistance(a:vec2f, b:vec2f, p:vec2f) -> f32 {
	let ab = b - a;
	let t = clamp(dot(p - a, ab) / max(dot(ab, ab), 0.000001), 0.0, 1.0);
	return length((a + ab * t) - p);
}
fn triangleEdgeDistance(p:vec2f) -> f32 {
	let d0 = segmentDistance(triA(), triB(), p);
	let d1 = segmentDistance(triB(), triC(), p);
	let d2 = segmentDistance(triC(), triA(), p);
	return min(d0, min(d1, d2));
}
fn foggedColor(f:VO, alphaMul:f32) -> vec4f {
	let fogAmt = clamp((f.normDepth * 0.28 + f.dof * 0.44) * u.focusFog, 0.0, 0.76);
	let litClr = f.col.rgb * (0.87 + 0.13 * f.nearBoost);
	let alpha = f.col.a * alphaMul * mix(1.0, 0.26, pow(fogAmt, 1.08));
	return vec4f(litClr, alpha);
}
fn interiorFillColor() -> vec4f {
	return vec4f(0.045, 0.045, 0.072, 1.0);
}
@fragment fn fsMaskCircle(f:VO) -> @location(0) vec4f {
	if (dot(f.local, f.local) > 1.0) { discard; }
	return vec4f(0.0);
}
@fragment fn fsFillCircle(f:VO) -> @location(0) vec4f {
	if (dot(f.local, f.local) > 1.0) { discard; }
	return interiorFillColor();
}
@fragment fn fsCircle(f:VO) -> @location(0) vec4f {
	let r = length(f.local);
	if (r > 1.0) { discard; }
	let d = abs(r - 0.88);
	let px = fwidth(r);
	let ring = 1.0 - smoothstep(0.0, px * 1.5, d);
	if (ring <= 0.001) { discard; }
	return foggedColor(f, ring);
}
@fragment fn fsMaskBox(f:VO) -> @location(0) vec4f {
	if (max(abs(f.local.x), abs(f.local.y)) > 1.0) { discard; }
	return vec4f(0.0);
}
@fragment fn fsFillBox(f:VO) -> @location(0) vec4f {
	if (max(abs(f.local.x), abs(f.local.y)) > 1.0) { discard; }
	return interiorFillColor();
}
@fragment fn fsBox(f:VO) -> @location(0) vec4f {
	let m = max(abs(f.local.x), abs(f.local.y));
	if (m > 1.0) { discard; }
	let px = fwidth(m);
	if (m < 1.0 - px * 1.5) { discard; }
	return foggedColor(f, 1.0);
}
@fragment fn fsMaskTriangle(f:VO) -> @location(0) vec4f {
	if (!triangleInside(f.local)) { discard; }
	return vec4f(0.0);
}
@fragment fn fsFillTriangle(f:VO) -> @location(0) vec4f {
	if (!triangleInside(f.local)) { discard; }
	return interiorFillColor();
}
@fragment fn fsTriangle(f:VO) -> @location(0) vec4f {
	if (!triangleInside(f.local)) { discard; }
	let d = triangleEdgeDistance(f.local);
	let px = fwidth(d);
	let rim = 1.0 - smoothstep(0.0, px * 2.0, d);
	if (rim <= 0.001) { discard; }
	return foggedColor(f, rim);
}
`;

const MARKER_SHADER = `
struct U { mvp:mat4x4f, focal:f32, depthMin:f32, depthMax:f32, pad:f32, res:vec2f, focusFog:f32, pad2:f32 }
@group(0) @binding(0) var<uniform> u:U;
struct GV { @location(0) local:vec2f }
struct IV { @location(1) source:vec3f, @location(2) dest:vec3f, @location(3) radius:f32, @location(4) col:vec4f }
struct VO {
  @builtin(position) pos:vec4f,
	@location(0) col:vec4f,
	@location(1) dof:f32,
	@location(2) normDepth:f32,
	@location(3) nearBoost:f32,
}
@vertex fn vs(v:GV, inst:IV) -> VO {
  var o:VO;
	let center = mix(inst.source, inst.dest, 0.62);
	let clipSource = u.mvp * vec4f(inst.source, 1.0);
	let clipTarget = u.mvp * vec4f(inst.dest, 1.0);
	let clipCenter = u.mvp * vec4f(center, 1.0);
	let depthRange = max(u.depthMax - u.depthMin, 0.001);
	let rawDepth = clamp((clipCenter.w - u.depthMin) / depthRange, 0.0, 1.0);
	let ndc = clipCenter.xy / max(clipCenter.w, 0.0001);
	let radial = length(vec2f(ndc.x * 0.86, ndc.y));
	let focusStart = max(u.depthMin + depthRange * 0.04, u.focal * 0.16);
	let focusEnd = max(focusStart + 0.001, min(u.depthMax, u.focal * 1.20));
	let depthSpan = max(focusEnd - focusStart, u.focal * 0.36);
	let longitudinal = max((clipCenter.w - focusStart) / depthSpan - 1.0, 0.0);
	let capsule = length(vec2f(radial / 0.28, longitudinal / 0.56));
	let dof = smoothstep(0.72, 1.42, capsule);
	let blurScale = mix(1.0, 1.18, dof);
	let ndcDelta = clipTarget.xy / max(clipTarget.w, 0.0001) - clipSource.xy / max(clipSource.w, 0.0001);
	let dirScreen = normalize(vec2(ndcDelta.x * u.res.x, ndcDelta.y * u.res.y));
	let safeDir = select(vec2(1.0, 0.0), dirScreen, dot(dirScreen, dirScreen) > 0.000001);
	let normalScreen = vec2(-safeDir.y, safeDir.x);
	let offsetScreen = safeDir * v.local.x * inst.radius * blurScale + normalScreen * v.local.y * inst.radius * blurScale;
	let offsetClip = vec2(
		offsetScreen.x * 2.0 / max(u.res.x, 1.0) * clipCenter.w,
		offsetScreen.y * 2.0 / max(u.res.y, 1.0) * clipCenter.w
	);
	o.pos = clipCenter + vec4f(offsetClip, 0.0, 0.0);
  o.col = inst.col;

	o.normDepth = smoothstep(0.04, 1.0, pow(rawDepth, 1.0));
	o.dof = dof;
	o.nearBoost = 1.0 - smoothstep(0.0, 0.38, rawDepth);
  return o;
}
@fragment fn fsFill(f:VO) -> @location(0) vec4f {
	return vec4f(0.032, 0.035, 0.055, 1.0);
}
@fragment fn fs(f:VO) -> @location(0) vec4f {
	let fogAmt = clamp((f.normDepth * 0.28 + f.dof * 0.44) * u.focusFog, 0.0, 0.76);
	let litClr = f.col.rgb * (0.88 + 0.12 * f.nearBoost);
	let alpha = max(f.col.a * mix(1.0, 0.38, pow(fogAmt, 1.08)), f.col.a * 0.52);
	return vec4f(litClr, alpha);
}
`;

function createFloatBuffer(dev, data) {
	const buffer = dev.createBuffer({ size: Math.max(data.byteLength, 16), usage: GPUBufferUsage.VERTEX, mappedAtCreation: true });
	new Float32Array(buffer.getMappedRange(0, data.byteLength || 16)).set(data);
	buffer.unmap();
	return buffer;
}

function pushLine(out, a, b) {
	out.push(a[0], a[1], a[2], b[0], b[1], b[2]);
}

function buildStaticGeometry(dev) {
	const quadVerts = new Float32Array([
		-1, -1,
		1, -1,
		1, 1,
		-1, -1,
		1, 1,
		-1, 1,
	]);
	const triangleVerts = new Float32Array([
		1.0, 0.0,
		-0.5, -0.8660254,
		-0.5, -0.8660254,
		-0.5, 0.8660254,
		-0.5, 0.8660254,
		1.0, 0.0,
	]);
	const triangleFillVerts = new Float32Array([
		1.0, 0.0,
		-0.5, -0.8660254,
		-0.5, 0.8660254,
	]);

	return {
		quad: { vb: createFloatBuffer(dev, quadVerts), count: quadVerts.length / 2 },
		triangle: { vb: createFloatBuffer(dev, triangleVerts), count: triangleVerts.length / 2 },
		triangleFill: { vb: createFloatBuffer(dev, triangleFillVerts), count: triangleFillVerts.length / 2 },
	};
}

function pushInstance(out, pos, scaleOrRadius, color, alpha) {
	out.push(pos[0], pos[1], pos[2], scaleOrRadius, color[0], color[1], color[2], alpha);
}
function pushMarkerInstance(out, source, target, radius, color, alpha) {
	out.push(
		source[0], source[1], source[2],
		target[0], target[1], target[2],
		radius,
		color[0], color[1], color[2], alpha,
	);
}

function buildBufs(dev) {
	const focus = getFocusState();
	const circleInstances = [];
	const cubeInstances = [];
	const triangleInstances = [];

	for (const n of all_nodes) {
		const vis = focus ? getNodeVisual(n, focus) : getUnfocusedNodeVisual(n);
		const scaleBase = getNodePrimitiveRadius(n, vis.scale);
		const shape = getNodeShape(n);
		const target = shape === "circle"
			? circleInstances
			: (shape === "triangle" || shape === "tetra")
				? triangleInstances
				: cubeInstances;
		pushInstance(target, n.pos, scaleBase, vis.color, vis.alpha);
	}

	const circleData = new Float32Array(circleInstances);
	const cubeData = new Float32Array(cubeInstances);
	const triangleData = new Float32Array(triangleInstances);
	const circleBuffer = circleData.length ? createFloatBuffer(dev, circleData) : null;
	const cubeBuffer = cubeData.length ? createFloatBuffer(dev, cubeData) : null;
	const triangleBuffer = triangleData.length ? createFloatBuffer(dev, triangleData) : null;

	const lineEdges = edges;
	const markerEdges = edges.filter(e => !e.parent);
	const ld = new Float32Array(lineEdges.length * 2 * 7);
	const markerInstances = [];
	for (let i = 0; i < lineEdges.length; i++) {
		const e = lineEdges[i];
		const lineVis = getEdgeLineVisual(e, focus);
		const c = [lineVis.color[0], lineVis.color[1], lineVis.color[2], lineVis.alpha];
		const o = i * 14;
		ld[o] = e.s.pos[0]; ld[o + 1] = e.s.pos[1]; ld[o + 2] = e.s.pos[2];
		ld[o + 3] = c[0]; ld[o + 4] = c[1]; ld[o + 5] = c[2]; ld[o + 6] = c[3];
		ld[o + 7] = e.t.pos[0]; ld[o + 8] = e.t.pos[1]; ld[o + 9] = e.t.pos[2];
		ld[o + 10] = c[0]; ld[o + 11] = c[1]; ld[o + 12] = c[2]; ld[o + 13] = c[3];
	}

	for (const e of markerEdges) {
		const markerVis = getEdgeMarkerVisual(e, focus);
		if (markerVis.alpha > 0.01) {
			pushMarkerInstance(markerInstances, e.s.pos, e.t.pos, markerVis.radius, markerVis.color, markerVis.alpha);
		}
	}

	const lb = createFloatBuffer(dev, ld);
	const markerData = new Float32Array(markerInstances);
	const markerBuffer = markerData.length ? createFloatBuffer(dev, markerData) : null;

	return {
		circleBuffer,
		circleCount: circleInstances.length / 8,
		cubeBuffer,
		cubeCount: cubeInstances.length / 8,
		triangleBuffer,
		triangleCount: triangleInstances.length / 8,
		lb,
		lc: lineEdges.length * 2,
		markerBuffer,
		markerCount: markerInstances.length / 11,
	};
}

let dev, fmt, gc, geom, circleMaskP, boxMaskP, triangleMaskP, circleFillP, boxFillP, triangleFillP, circleP, boxP, triangleP, edgeP, markerFillP, markerP, ub, bg, bufs;
function refresh_scene_buffers() {
	if (!dev) return;
	if (bufs) {
		bufs.circleBuffer?.destroy?.();
		bufs.cubeBuffer?.destroy?.();
		bufs.triangleBuffer?.destroy?.();
		bufs.lb?.destroy?.();
		bufs.markerBuffer?.destroy?.();
	}
	bufs = buildBufs(dev);
	requestRender();
}

function drawSpriteBatch(pass, pipeline, instanceBuffer, instanceCount) {
	if (!pipeline || !instanceBuffer || instanceCount <= 0) return;
	pass.setPipeline(pipeline);
	pass.setBindGroup(0, bg);
	pass.setVertexBuffer(0, geom.quad.vb);
	pass.setVertexBuffer(1, instanceBuffer);
	pass.draw(geom.quad.count, instanceCount);
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

	const edgeP = dev.createRenderPipeline({
		layout: pl,
		vertex: {
			module: dev.createShaderModule({ code: EDGE_SHADER }), entryPoint: "vs",
			buffers: [{
				arrayStride: 28, attributes: [
					{ shaderLocation: 0, offset: 0, format: "float32x3" },
					{ shaderLocation: 1, offset: 12, format: "float32x4" },
				]
			}]
		},
		fragment: {
			module: dev.createShaderModule({ code: EDGE_SHADER }), entryPoint: "fs",
			targets: [{ format: fmt, blend }]
		},
		primitive: { topology: "line-list" },
		depthStencil: { depthWriteEnabled: false, depthCompare: "less", format: "depth24plus" },
	});

	const spriteBuffers = [
		{
			arrayStride: 8,
			attributes: [
				{ shaderLocation: 0, offset: 0, format: "float32x2" },
			]
		},
		{
			arrayStride: 32,
			stepMode: "instance",
			attributes: [
				{ shaderLocation: 1, offset: 0, format: "float32x3" },
				{ shaderLocation: 2, offset: 12, format: "float32" },
				{ shaderLocation: 3, offset: 16, format: "float32x4" },
			]
		}
	];
	const maskTargets = [{ format: fmt, writeMask: 0 }];
	const colorTargets = [{ format: fmt, blend }];
	const fillTargets = [{ format: fmt }];
	function makeSpritePipeline(entryPoint, depthWriteEnabled, depthCompare, targets) {
		return dev.createRenderPipeline({
			layout: pl,
			vertex: {
				module: dev.createShaderModule({ code: SPRITE_SHADER }), entryPoint: "vs",
				buffers: spriteBuffers,
			},
			fragment: {
				module: dev.createShaderModule({ code: SPRITE_SHADER }), entryPoint,
				targets,
			},
			primitive: { topology: "triangle-list" },
			depthStencil: { depthWriteEnabled, depthCompare, format: "depth24plus" },
		});
	}

	const circleMaskP = makeSpritePipeline("fsMaskCircle", true, "less", maskTargets);
	const boxMaskP = makeSpritePipeline("fsMaskBox", true, "less", maskTargets);
	const triangleMaskP = makeSpritePipeline("fsMaskTriangle", true, "less", maskTargets);
	const circleFillP = makeSpritePipeline("fsFillCircle", false, "less-equal", fillTargets);
	const boxFillP = makeSpritePipeline("fsFillBox", false, "less-equal", fillTargets);
	const triangleFillP = makeSpritePipeline("fsFillTriangle", false, "less-equal", fillTargets);
	const circleP = makeSpritePipeline("fsCircle", true, "less", colorTargets);
	const boxP = makeSpritePipeline("fsBox", true, "less", colorTargets);
	const triangleP = makeSpritePipeline("fsTriangle", true, "less", colorTargets);

	const markerFillP = dev.createRenderPipeline({
		layout: pl,
		vertex: {
			module: dev.createShaderModule({ code: MARKER_SHADER }), entryPoint: "vs",
			buffers: [
				{
					arrayStride: 8,
					attributes: [
						{ shaderLocation: 0, offset: 0, format: "float32x2" },
					]
				},
				{
					arrayStride: 44,
					stepMode: "instance",
					attributes: [
						{ shaderLocation: 1, offset: 0, format: "float32x3" },
						{ shaderLocation: 2, offset: 12, format: "float32x3" },
						{ shaderLocation: 3, offset: 24, format: "float32" },
						{ shaderLocation: 4, offset: 28, format: "float32x4" },
					]
				}
			]
		},
		fragment: {
			module: dev.createShaderModule({ code: MARKER_SHADER }), entryPoint: "fsFill",
			targets: [{ format: fmt }]
		},
		primitive: { topology: "triangle-list" },
		depthStencil: { depthWriteEnabled: true, depthCompare: "less", format: "depth24plus" },
	});

	const markerP = dev.createRenderPipeline({
		layout: pl,
		vertex: {
			module: dev.createShaderModule({ code: MARKER_SHADER }), entryPoint: "vs",
			buffers: [
				{
					arrayStride: 8,
					attributes: [
						{ shaderLocation: 0, offset: 0, format: "float32x2" },
					]
				},
				{
					arrayStride: 44,
					stepMode: "instance",
					attributes: [
						{ shaderLocation: 1, offset: 0, format: "float32x3" },
						{ shaderLocation: 2, offset: 12, format: "float32x3" },
						{ shaderLocation: 3, offset: 24, format: "float32" },
						{ shaderLocation: 4, offset: 28, format: "float32x4" },
					]
				}
			]
		},
		fragment: {
			module: dev.createShaderModule({ code: MARKER_SHADER }), entryPoint: "fs",
			targets: [{ format: fmt, blend }]
		},
		primitive: { topology: "line-list" },
		depthStencil: { depthWriteEnabled: false, depthCompare: "less-equal", format: "depth24plus" },
	});

	const ub = dev.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
	const bg = dev.createBindGroup({ layout: bgl, entries: [{ binding: 0, resource: { buffer: ub } }] });

	return { circleMaskP, boxMaskP, triangleMaskP, circleFillP, boxFillP, triangleFillP, circleP, boxP, triangleP, edgeP, markerFillP, markerP, ub, bg };
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
	const focus = getFocusState();
	const rawProjected = [];
	for (const n of all_nodes) {
		const [x, y, z] = n.pos;
		const cx = mvp[0] * x + mvp[4] * y + mvp[8] * z + mvp[12];
		const cy = mvp[1] * x + mvp[5] * y + mvp[9] * z + mvp[13];
		const cw = mvp[3] * x + mvp[7] * y + mvp[11] * z + mvp[15];
		const vis = focus ? getNodeVisual(n, focus) : getUnfocusedNodeVisual(n);
		const interactive = isNodeInteractive(n, focus, vis.alpha);
		if (cw < 0.01) {
			rawProjected.push({ n, sx: -9999, sy: -9999, w: Infinity, radius: 0, alpha: vis.alpha, interactive, blur: 1, sharpness: 0 });
			continue;
		}
		if (cw < dMin) dMin = cw;
		if (cw > dMax) dMax = cw;
		rawProjected.push({
			n,
			sx: (cx / cw * 0.5 + 0.5) * W,
			sy: (1 - (cy / cw * 0.5 + 0.5)) * H,
			w: cw,
			cx,
			cy,
			radius: getNodePrimitiveRadius(n, vis.scale),
			alpha: vis.alpha,
			interactive,
		});
	}
	if (dMin !== Infinity) { depthMin = dMin; depthMax = dMax; }
	projected = rawProjected.map(p => {
		if (p.w === Infinity) return p;
		const dof = getDepthOfFieldState(p.cx, p.cy, p.w);
		return {
			...p,
			radius: p.radius * dof.scale,
			blur: dof.blur,
			sharpness: dof.sharpness,
		};
	});

	const byKey = new Map(projected.map(p => [p.n.key, p]));
	for (const e of edges) {
		if (e.parent) continue;
		const sp = byKey.get(e.s.key), tp = byKey.get(e.t.key);
		if (!sp || !tp || !Number.isFinite(sp.sx) || !Number.isFinite(tp.sx) || sp.w < 0.01 || tp.w < 0.01) continue;
		const markerVis = getEdgeMarkerVisual(e, focus);
		const markerAlpha = markerVis.alpha;
		if (markerAlpha <= 0.01) continue;
		const interactive = isEdgeInteractive(e, focus, markerAlpha);
		const markerPos = getEdgeMarkerCenter(e);
		const mp = projectPoint(mvp, markerPos);
		if (!mp) continue;
		const markerClipX = mvp[0] * markerPos[0] + mvp[4] * markerPos[1] + mvp[8] * markerPos[2] + mvp[12];
		const markerClipY = mvp[1] * markerPos[0] + mvp[5] * markerPos[1] + mvp[9] * markerPos[2] + mvp[13];
		const markerDof = getDepthOfFieldState(markerClipX, markerClipY, mp.w);
		const mx = mp.x, my = mp.y;
		if (mx < -24 || mx > W + 24 || my < -24 || my > H + 24) continue;
		edgeProjected.push({
			e,
			sx: mx,
			sy: my,
			r: markerVis.radius * lerp(1, 1.2, markerDof.blur),
			alpha: markerAlpha,
			interactive,
			angle: Math.atan2(tp.sy - sp.sy, tp.sx - sp.sx),
			depth: mp.w,
			blur: markerDof.blur,
		});
	}
	edgeProjected.sort((a, b) => a.depth - b.depth);
}

// ---- 2D overlay ----
function drawTooltip(lines, x, y) {
	const filtered = lines.filter(Boolean);
	if (!filtered.length) return;
	const padX = 10, padY = 7, lineH = 15;
	oc.font = `500 12px ${UI_FONT_FAMILY}`;
	const boxW = Math.max(...filtered.map(line => oc.measureText(line).width), 0) + padX * 2;
	const boxH = filtered.length * lineH + padY * 2;
	const bx = clamp(x, 12, W - boxW - 12);
	const by = clamp(y - boxH * 0.5, 12, H - boxH - 12);
	oc.beginPath();
	oc.roundRect(bx, by, boxW, boxH, 8);
	oc.fillStyle = rgba(currentTheme.background, 0.9);
	oc.fill();
	oc.strokeStyle = rgba(currentTheme.cursor, 0.24);
	oc.lineWidth = 0.75;
	oc.stroke();
	oc.textAlign = "left";
	filtered.forEach((line, i) => {
		oc.fillStyle = i === 0 ? rgba(currentTheme.foreground, 0.98) : rgba(mixColor(currentTheme.foreground, currentTheme.cursor, 0.18), 0.88);
		oc.fillText(line, bx + padX, by + padY + 12 + i * lineH);
	});
}
function drawDirectionalMarkerOutline(x, y, radius, angle, color, alpha, lineWidth = 1) {
	const head = radius + 2;
	const backX = -radius * 0.8;
	const halfHeight = radius * 0.92;
	const points = [
		[head, 0],
		[backX, -halfHeight],
		[backX, halfHeight],
	];
	oc.save();
	oc.translate(x, y);
	oc.rotate(angle);
	oc.beginPath();
	oc.moveTo(points[0][0], points[0][1]);
	for (let i = 1; i < points.length; i++) oc.lineTo(points[i][0], points[i][1]);
	oc.closePath();
	oc.strokeStyle = `rgba(6,8,16,${Math.min(0.94, Math.max(0.34, alpha * 0.9))})`;
	oc.lineWidth = lineWidth + 2.2;
	oc.shadowColor = "rgba(0,0,0,0)";
	oc.shadowBlur = 0;
	oc.stroke();
	oc.strokeStyle = rgba(color, Math.min(0.98, Math.max(alpha, EDGE_MARKER_MIN_ALPHA) + 0.16));
	oc.lineWidth = lineWidth;
	oc.shadowColor = rgba(color, 0.24 * Math.max(alpha, 0.35));
	oc.shadowBlur = 10;
	oc.stroke();
	oc.restore();
}
function drawOv() {
	oc.setTransform(dpr, 0, 0, dpr, 0, 0);
	oc.clearRect(0, 0, W, H);
	const focus = getFocusState();

	if (hoveredEdge) {
		const marker = edgeProjected.find(entry => entry.e === hoveredEdge);
		if (marker) {
			const vis = getEdgeMarkerVisual(marker.e, focus);
			const alpha = Math.min(vis.alpha, marker.alpha ?? vis.alpha);
			const radius = vis.radius * 1.22;
			drawDirectionalMarkerOutline(marker.sx, marker.sy, radius, marker.angle || 0, vis.color, Math.min(1, alpha + EDGE_MARKER_HOVER_BOOST), (vis.lineWidth || 1) * 1.45);
		}
	}

	oc.font = `500 11px ${UI_FONT_FAMILY}`;
	const labelCandidates = projected
		.filter(p => p.n.kind === "thought" && p.w < Infinity && (p.n.importance || 0) >= 0.45 && (p.sharpness || 0) > 0.03)
		.sort((a, b) => (b.n.importance || 0) - (a.n.importance || 0))
		.slice(0, 10);
	for (const p of labelCandidates) {
		if (hoveredNode === p.n || selected === p.n) continue;
		const extra = focus && focus.scores.has(p.n.key) ? 0.14 : 0;
		const sharpness = Math.pow(p.sharpness || 0, DOF_LABEL_SHARPNESS_POWER);
		const alpha = Math.min(0.85, (0.24 + (p.n.importance || 0) * 0.40 + extra) * sharpness);
		if (alpha <= 0.015) continue;
		oc.shadowBlur = 14 * (p.blur || 0);
		oc.shadowColor = `rgba(140,160,255,${0.14 * (p.blur || 0)})`;
		oc.fillStyle = rgba(currentTheme.foreground, alpha);
		oc.fillText(truncateText(p.n.label, 34), p.sx + 8, p.sy - 8);
	}
	oc.shadowBlur = 0;
	oc.shadowColor = "rgba(0,0,0,0)";

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

function picshardeAt(mx, my) {
	let best = null, bd = Infinity;
	for (const p of projected) {
		if (!p.interactive) continue;
		const dx = mx - p.sx, dy = my - p.sy, d = Math.sqrt(dx * dx + dy * dy);
		const hr = Math.max(p.n.kind === "tag" ? 10 : 6, (p.radius || 0) + 2);
		if (d < hr && d < bd) { best = p.n; bd = d; }
	}
	return best;
}
function pickEdgeAt(mx, my) {
	let best = null, bd = Infinity;
	for (const marker of edgeProjected) {
		if (!marker.interactive) continue;
		const dx = mx - marker.sx, dy = my - marker.sy, d = Math.sqrt(dx * dx + dy * dy);
		const hr = Math.max(6, marker.r + 3);
		if (d < hr && d < bd) { best = marker.e; bd = d; }
	}
	return best;
}

function isTextEditingTarget(target) {
	if (!target) return false;
	const tag = target.tagName;
	return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable;
}

function rank_thoughtMatches(query) {
	const q = query.toLowerCase().trim();
	if (q.length < 2) return [];

	return thought_nodes
		.map(node => {
			const label = String(node.label || "").toLowerCase();
			const source = String(node.source || "").toLowerCase();
			let matchScore = 0;
			if (label.startsWith(q)) matchScore += 5;
			else if (label.includes(q)) matchScore += 3;
			if (source.includes(q)) matchScore += 1.5;
			const importance = node.importance || 0;
			const access = Math.min(node.access || 0, 12) * 0.03;
			return { node, matchScore, importance, access };
		})
		.filter(entry => entry.matchScore > 0)
		.sort((a, b) => {
			if (b.importance !== a.importance) return b.importance - a.importance;
			if (b.matchScore !== a.matchScore) return b.matchScore - a.matchScore;
			return b.access - a.access;
		})
		.map(entry => entry.node);
}

function setupAskUI() {
	const overlay = document.getElementById("ask-overlay");
	const dialog = document.getElementById("ask-dialog");
	const stack = document.getElementById("ask-stack");
	const form = document.getElementById("ask-form");
	const input = document.getElementById("ask-input");
	const status = document.getElementById("ask-status");
	const answer = document.getElementById("ask-answer");
	const answerText = document.getElementById("ask-answer-text");
	const thoughts = document.getElementById("ask-thoughts");
	const thoughtsList = document.getElementById("ask-thoughts-list");

	const sendBtn = document.getElementById("ask-send");

	const setLoading = (v) => {
		input.disabled = v;
		if (sendBtn) {
			sendBtn.classList.toggle("loading", v);
			sendBtn.disabled = v;
		}
		if (!v) requestAnimationFrame(() => input.focus());
	};

	const cancelPending = () => {
		if (askState.debounceId) {
			clearTimeout(askState.debounceId);
			askState.debounceId = null;
		}
		if (askState.controller) {
			askState.controller.abort();
			askState.controller = null;
		}
	};

	const resetResults = () => {
		answer.hidden = true;
		thoughts.hidden = true;
		answerText.textContent = "";
		thoughtsList.innerHTML = "";
	};

	const setStatus = (message) => {
		status.textContent = message;
	};

	const closeAsk = () => {
		if (!askState.open) return;
		cancelPending();
		askState.open = false;
		overlay.classList.remove("open");
		overlay.setAttribute("aria-hidden", "true");
		overlay.hidden = true;
		document.body.classList.remove("ask-open");
		const previous = askState.lastFocused;
		askState.lastFocused = null;
		if (previous && typeof previous.focus === "function") previous.focus();
		else cvs.focus?.();
		requestRender();
	};

	const openAsk = (prefill = "") => {
		if (askState.open) {
			if (prefill) input.value = prefill;
			input.focus();
			input.select();
			return;
		}
		askState.open = true;
		askState.lastFocused = document.activeElement instanceof HTMLElement ? document.activeElement : null;
		overlay.hidden = false;
		overlay.classList.add("open");
		overlay.setAttribute("aria-hidden", "false");
		document.body.classList.add("ask-open");
		if (prefill) input.value = prefill;
		if (!prefill && !input.value.trim()) setStatus("Type to search thoughts and query the model. Esc closes.");
		requestRender();
		requestAnimationFrame(() => {
			input.focus();
			input.select();
		});
	};

	const renderThoughtMatches = (query) => {
		thoughtsList.innerHTML = "";
		const matches = rank_thoughtMatches(query);
		thoughts.hidden = !matches.length;
		for (const node of matches) {
			const card = document.createElement("div");
			card.className = "ask-source-item is-clickable";
			card.innerHTML = `
				<div class="ask-source-title">${getNodeIconMarkup(node)}${escapeHtml(`${getNodeTypeLabel(node)} #${node.key}`)}</div>
				<div class="ask-source-meta">${escapeHtml(`${node.store || "global"} · ${node.source || "graph"} · importance ${(node.importance || 0).toFixed(2)}`)}</div>
        <div class="ask-source-text">${escapeHtml(truncateText(node.label || "", 180))}</div>
      `;
			card.addEventListener("click", () => {
				closeAsk();
				selectNode(node, { align: true });
			});
			thoughtsList.appendChild(card);
		}
	};

	const renderAnswer = (answerValue) => {
		const hasAnswer = !!String(answerValue || "").trim();
		answer.hidden = !hasAnswer;
		answerText.textContent = answerValue || "";
	};

	const requestAnswer = async (query) => {
		cancelPending();

		const trimmed = query.trim();
		if (trimmed.length < 2) {
			answer.hidden = true;
			setStatus("Type to search thoughts and query the model. Esc closes.");
			return;
		}

		const seq = ++askState.requestSeq;
		const controller = new AbortController();
		askState.controller = controller;
		setLoading(true);
		setStatus("Thinking…");
		try {
			const response = await fetch("/agent", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ message: trimmed, session_id: askState.serverSessionId }),
				signal: controller.signal,
			});
			if (!response.ok) throw new Error(`Request failed (${response.status})`);
			const payload = await response.json();
			if (seq !== askState.requestSeq) return;
			if (payload.session_id) askState.serverSessionId = payload.session_id;
			renderAnswer(payload?.content || payload?.answer || "No answer returned.");
			setStatus("Enter to send · Esc to close");
		} catch (error) {
			if (controller.signal.aborted) return;
			if (seq !== askState.requestSeq) return;
			renderAnswer("The model could not answer that right now.");
			setStatus(error instanceof Error ? error.message : "Query failed.");
		} finally {
			if (askState.controller === controller) askState.controller = null;
			setLoading(false);
		}
	};

	form.addEventListener("submit", event => {
		event.preventDefault();
		requestAnswer(input.value);
	});

	input.addEventListener("input", () => {
		if (!input.value.trim()) resetResults();
		renderThoughtMatches(input.value);
	});

	overlay.addEventListener("click", event => {
		if (event.target === overlay) closeAsk();
	});

	window.addEventListener("keydown", event => {
		if (askState.open) {
			if (event.key === "Escape") {
				event.preventDefault();
				closeAsk();
				return;
			}
			if (event.key === "Enter" && document.activeElement === input) {
				event.preventDefault();
				requestAnswer(input.value);
			}
			return;
		}
		if (event.key !== " " || event.ctrlKey || event.metaKey || event.altKey) return;
		if (isTextEditingTarget(event.target)) return;
		event.preventDefault();
		openAsk();
	});

	dialog.addEventListener("keydown", event => {
		if (event.key === "Escape") {
			event.preventDefault();
			closeAsk();
		}
	});

	stack.addEventListener("pointerdown", event => {
		event.stopPropagation();
	});
}

// ---- input ----
function setupInput() {
	cvs.addEventListener("wheel", e => {
		e.preventDefault();
		dist = clamp(dist * (e.deltaY > 0 ? ZOOM_OUT : ZOOM_IN), min_dist, maxDist);
		requestRender();
	}, { passive: false });

	cvs.addEventListener("pointerdown", e => {
		drag = true; lmx = e.clientX; lmy = e.clientY;
		pointerGesture = { pointerId: e.pointerId, startX: e.clientX, startY: e.clientY };
		vYaw = 0; vPitch = 0;
		cvs.setPointerCapture(e.pointerId);
		cvs.style.cursor = "grabbing";
		requestRender();
	});

	cvs.addEventListener("pointermove", e => {
		if (!drag) {
			updateHoverState(e.offsetX, e.offsetY);
			return;
		}
		const dx = e.clientX - lmx, dy = e.clientY - lmy;
		vYaw = -dx * SENS; vPitch = -dy * SENS;
		yaw += vYaw; pitch += vPitch;
		pitch = Math.max(-Math.PI / 2 + 0.05, Math.min(Math.PI / 2 - 0.05, pitch));
		lmx = e.clientX; lmy = e.clientY;
		requestRender();
	});

	cvs.addEventListener("pointerup", e => {
		if (!drag) return;
		const shouldSelect = isPointerGestureClick(pointerGesture, e);
		drag = false;
		pointerGesture = null;
		cvs.releasePointerCapture?.(e.pointerId);
		updateHoverState(e.offsetX, e.offsetY);
		if (shouldSelect) {
			const hit = picshardeAt(e.offsetX, e.offsetY);
			if (hit) {
				selectNode(hit);
			} else {
				const edgeHit = pickEdgeAt(e.offsetX, e.offsetY);
				if (edgeHit) {
					const target = (edgeHit.s.importance || 0) >= (edgeHit.t.importance || 0) ? edgeHit.s : edgeHit.t;
					selectNode(target, { align: true });
				} else {
					selectNode(null);
				}
			}
		}
		requestRender();
	});

	cvs.addEventListener("pointercancel", e => {
		if (pointerGesture?.pointerId !== e.pointerId) return;
		drag = false;
		pointerGesture = null;
		cvs.style.cursor = "grab";
		requestRender();
	});

	window.addEventListener("keydown", e => {
		if (askState.open) return;
		if (e.key === "Escape") selectNode(null);
	});

	cvs.addEventListener("dblclick", () => {
		hoveredNode = null; hoveredEdge = null;
		pointerGesture = null;
		selectNode(null);
		yaw = 0.3;
		pitch = -0.5;
		dist = reset_dist;
		vYaw = 0.002;
		vPitch = 0;
		syncOrbitTarget(true);
		requestRender();
	});
}

// ---- detail ----
function selectNode(n, { align = false } = {}) {
	selected = n;
	refresh_scene_buffers();
	if (!n) {
		syncOrbitTarget();
		document.getElementById("detail").style.display = "none";
		return;
	}

	if (align) {
		const dx = n.pos[0] - graph_center[0];
		const dy = n.pos[1] - graph_center[1];
		const dz = n.pos[2] - graph_center[2];
		yaw = Math.atan2(dx, dz);
		pitch = clamp(-Math.atan2(dy, Math.max(Math.hypot(dx, dz), 0.001)), -Math.PI / 2 + 0.05, Math.PI / 2 - 0.05);
	}

	vYaw = 0;
	vPitch = 0;
	setOrbitTarget(n.pos);
	showDetail(n);
}

function showDetail(n) {
	document.getElementById("d-title").innerHTML =
		`${getNodeIconMarkup(n)}${escapeHtml(n.kind === "tag" ? n.label : `${getNodeTypeLabel(n)} #${n.key}`)}`;
	const dimText = n.topDimensions?.length
		? ` &middot; tags: ${n.topDimensions.map(dim => `${dim.label} ${dim.score.toFixed(2)}`).join(", ")}`
		: "";
	document.getElementById("d-meta").innerHTML = n.kind === "tag"
		? `shard &middot; importance: ${(n.importance || 0).toFixed(2)} &middot; links: ${n.linkCount || 0}${dimText}`
		: `store: ${n.store} &middot; source: ${n.source || "\u2014"} &middot; access: ${n.access} &middot; importance: ${(n.importance || 0).toFixed(2)}${dimText}`;
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
		d.innerHTML = `<div class="edge-item-main">${getEdgeIconMarkup()}<span class="ew">${bits.join(" · ")}</span> → <span class="edge-target">${getNodeIconMarkup(o)}#${o.key} ${escapeHtml(truncateText(o.label || "", 60))}</span>${e.knn ? ' <span class="edge-knn">knn</span>' : ''}</div><div class="edge-item-reason">${escapeHtml(truncateText(e.reason || "linked by graph evidence", 120))}</div>`;
		d.onclick = () => selectNode(o, { align: true });
		el.appendChild(d);
	}
	document.getElementById("detail").style.display = "block";
}

function setupPanelSettings() {
	const panel = document.getElementById("panel");
	const toggle = document.getElementById("panel-settings-toggle");
	if (!panel || !toggle) return;

	const syncExpanded = () => {
		toggle.setAttribute("aria-expanded", panel.classList.contains("is-open") ? "true" : "false");
	};

	toggle.addEventListener("click", event => {
		event.preventDefault();
		event.stopPropagation();
		panel.classList.toggle("is-open");
		syncExpanded();
	});

	panel.addEventListener("keydown", event => {
		if (event.key !== "Escape") return;
		panel.classList.remove("is-open");
		syncExpanded();
		toggle.blur();
	});

	document.addEventListener("pointerdown", event => {
		if (!panel.classList.contains("is-open")) return;
		if (panel.contains(event.target)) return;
		panel.classList.remove("is-open");
		syncExpanded();
	});

	syncExpanded();
}

// ---- resize ----
function resize() {
	dpr = window.devicePixelRatio || 1;
	W = window.innerWidth; H = window.innerHeight;
	cvs.width = W * dpr; cvs.height = H * dpr;
	cvs.style.width = W + "px"; cvs.style.height = H + "px";
	ov.width = W * dpr; ov.height = H * dpr;
	ov.style.width = W + "px"; ov.style.height = H + "px";
	updateCameraBounds(true);
	requestRender();
}

// ---- boot ----
let depthTex = null;
resize();
window.addEventListener("resize", () => { resize(); depthTex = null; });

loadStoredVizSettings();
applyTheme(currentTheme.id, { persist: false, recolorNodes: false, refresh: false });

let gpu;
try { gpu = await initGPU(); } catch (e) {
	document.body.innerHTML = `<div style="color:${colorHex(currentTheme.accents.red)};background:${currentTheme.backgroundHex};padding:40px;font-family:${UI_FONT_FAMILY}"><h2 style="margin-bottom:12px">WebGPU not available</h2><p style="color:${currentTheme.foregroundHex};margin-bottom:8px">${escapeHtml(e.message)}</p><p style="color:${currentTheme.foregroundHex}">Try Chrome 113+ or Edge 113+.</p></div>`;
	throw e;
}
({ dev, fmt, gc } = gpu);
geom = buildStaticGeometry(dev);
({ circleMaskP, boxMaskP, triangleMaskP, circleFillP, boxFillP, triangleFillP, circleP, boxP, triangleP, edgeP, markerFillP, markerP, ub, bg } = makePipes(dev, fmt));
bufs = buildBufs(dev);

// Seed load: a few random thoughts + their edges, then BFS-expand the whole graph
let metaData = null;
try {
	const res = await fetch("/graph/seed?n=12");
	if (res.ok) metaData = await res.json();
} catch (e) {
	document.getElementById("loading").textContent = `Load failed: ${e.message}`;
	throw e;
}
if (!metaData) {
	try {
		metaData = await (await fetch("/graph/full")).json();
	} catch (e) {
		document.getElementById("loading").textContent = `Load failed: ${e.message}`;
		throw e;
	}
}

layout(metaData);
document.getElementById("loading").remove();
document.getElementById("s-nodes").textContent = thought_nodes.length;
document.getElementById("s-edges").textContent = edges.length;
setupSpiralControls();
setupThemeControls();
setupRecencyColorControls();
setupPanelSettings();

// Seed the known set
for (const n of all_nodes) knownNodeKeys.add(n.key);

// One BFS chain per seed node — all run in parallel, share knownNodeKeys to avoid duplicates
async function bfsChain(startKey) {
	let frontier = [startKey];
	while (frontier.length > 0) {
		let data;
		try {
			data = await (await fetch("/graph/expand", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ keys: frontier, known: [...knownNodeKeys] }),
			})).json();
		} catch (e) {
			console.error("BFS chain error:", e);
			break;
		}
		rawEdgeData.edges.push(...(data.edges || []));
		// Claim new nodes immediately before other chains see them
		const newNodes = (data.nodes || []).filter(n => !knownNodeKeys.has(n.key));
		newNodes.forEach(n => knownNodeKeys.add(n.key));
		if (newNodes.length) {
			appendThoughts(newNodes);
			frontier = newNodes.map(n => n.key);
		} else {
			resolveEdges();
			frontier = [];
		}
		document.getElementById("s-nodes").textContent = thought_nodes.length;
		document.getElementById("s-edges").textContent = edges.length;
		await new Promise(r => setTimeout(r, 0)); // yield to renderer
	}
}

(async function bfsLoad() {
	// Fire one chain per seed in parallel — they race across the graph simultaneously
	await Promise.all(metaData.nodes.map(n => bfsChain(n.key)));

	// Final sweep: pick up any thoughts unreachable by edges (isolated nodes)
	const total = metaData?.total_thoughts ?? 0;
	if (thought_nodes.length < total) {
		const BATCH = 200;
		const offsets = [];
		for (let off = 0; off < total; off += BATCH) offsets.push(off);
		await Promise.all(offsets.map(off =>
			fetch(`/graph/thoughts?offset=${off}&limit=${BATCH}`)
				.then(r => r.json())
				.then(batch => {
					if (batch.nodes?.length) {
						appendThoughts(batch.nodes);
						batch.nodes.forEach(n => knownNodeKeys.add(n.key));
						document.getElementById("s-nodes").textContent = thought_nodes.length;
						document.getElementById("s-edges").textContent = edges.length;
					}
				})
				.catch(e => console.error("Thought sweep error:", e))
		));
	}

	// Load KNN edges — cross-shard similarity links that BFS can't discover
	fetch("/graph/knn_edges")
		.then(r => r.json())
		.then(data => {
			if (data.knn_edges?.length) {
				rawEdgeData.knn_edges.push(...data.knn_edges);
				resolveEdges();
				refresh_scene_buffers();
				document.getElementById("s-edges").textContent = edges.length;
			}
		})
		.catch(e => console.error("KNN edges error:", e));
})();

// ---- live poll for updates ----
let lastThoughtCount = metaData?.total_thoughts ?? thought_nodes.length;
let lastEdgeCount = metaData?.edges?.length ?? edges.length;
let pollInterval = null;

async function pollForUpdates() {
	try {
		const diff = await (await fetch("/diff")).json();
		let changed = false;

		if (diff.added_count > 0) {
			const newData = await (await fetch(`/graph/thoughts?offset=${lastThoughtCount}&limit=${diff.added_count}`)).json();
			if (diff.added_edges?.length) rawEdgeData.edges.push(...diff.added_edges);
			if (newData.nodes?.length) appendThoughts(newData.nodes);
			lastThoughtCount = diff.thought_count;
			changed = true;
		} else if (diff.initial) {
			lastThoughtCount = diff.thought_count;
		}

		if (diff.edge_count !== undefined && diff.edge_count !== lastEdgeCount) {
			const expandData = await (await fetch("/graph/expand", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ keys: [...knownNodeKeys], known: [...knownNodeKeys] }),
			})).json();
			rawEdgeData.edges.push(...(expandData.edges || []));
			resolveEdges();
			lastEdgeCount = diff.edge_count;
			changed = true;
		}

		if (changed) {
			document.getElementById("s-nodes").textContent = thought_nodes.length;
			document.getElementById("s-edges").textContent = edges.length;
			refresh_scene_buffers();
		}
	} catch (e) {
		console.error("Poll error:", e);
	}
}

function startPolling() {
	pollForUpdates();
	pollInterval = setInterval(pollForUpdates, 3000);
}
startPolling();

function getDT() {
	if (!depthTex || depthTex.width !== cvs.width || depthTex.height !== cvs.height) {
		if (depthTex) depthTex.destroy();
		depthTex = dev.createTexture({ size: [cvs.width, cvs.height], format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT });
	}
	return depthTex;
}

function frame(time) {
	frameHandle = 0;
	if (!dev || !gc || !ub || !bufs || !geom) return;

	// inertia
	if (!drag) {
		yaw += vYaw; pitch += vPitch;
		vYaw *= DAMP; vPitch *= DAMP;
		if (Math.abs(vYaw) < 0.0001) vYaw = 0;
		if (Math.abs(vPitch) < 0.0001) vPitch = 0;
		pitch = Math.max(-Math.PI / 2 + 0.05, Math.min(Math.PI / 2 - 0.05, pitch));
	}

	for (let i = 0; i < 3; i++) {
		const delta = orbitTarget[i] - orbitCenter[i];
		orbitCenter[i] += delta * FOCUS_LERP;
		if (Math.abs(delta) < 0.0005) orbitCenter[i] = orbitTarget[i];
	}

	// MVP — orbit camera: rotate world then pull back
	const asp = W / H;
	const far = Math.max(40, dist + graph_radius * 6);
	const proj = m4Persp(FOV, asp, 0.05, far);
	const view = m4Mul(
		m4Mul(
			m4Mul(m4Trans(0, 0, -dist), m4RotX(-pitch)),
			m4RotY(-yaw)
		),
		m4Trans(-orbitCenter[0], -orbitCenter[1], -orbitCenter[2])
	);

	dist = clamp(dist, min_dist, maxDist);
	const mvp = m4Mul(proj, view);
	mvpCache = mvp;
	projectAll(mvp);

	const ud = new Float32Array(24);
	ud.set(mvp, 0);
	ud[16] = dist;
	ud[17] = depthMin;
	ud[18] = depthMax;
	ud[20] = cvs.width;
	ud[21] = cvs.height;
	ud[22] = getSelectionFogScale();
	dev.queue.writeBuffer(ub, 0, ud);

	const tex = gc.getCurrentTexture();
	const dt = getDT();
	const enc = dev.createCommandEncoder();
	const pass = enc.beginRenderPass({
		colorAttachments: [{ view: tex.createView(), clearValue: { r: currentTheme.background[0], g: currentTheme.background[1], b: currentTheme.background[2], a: 1 }, loadOp: "clear", storeOp: "store" }],
		depthStencilAttachment: { view: dt.createView(), depthClearValue: 1, depthLoadOp: "clear", depthStoreOp: "store" },
	});

	drawSpriteBatch(pass, circleP, bufs.circleBuffer, bufs.circleCount);
	drawSpriteBatch(pass, boxP, bufs.cubeBuffer, bufs.cubeCount);
	drawSpriteBatch(pass, triangleP, bufs.triangleBuffer, bufs.triangleCount);
	if (bufs.markerCount > 0) {
		pass.setPipeline(markerP); pass.setBindGroup(0, bg);
		pass.setVertexBuffer(0, geom.triangle.vb);
		pass.setVertexBuffer(1, bufs.markerBuffer);
		pass.draw(geom.triangle.count, bufs.markerCount);
	}
	if (bufs.lc > 0) {
		pass.setPipeline(edgeP); pass.setBindGroup(0, bg);
		pass.setVertexBuffer(0, bufs.lb); pass.draw(bufs.lc);
	}

	pass.end();
	dev.queue.submit([enc.finish()]);

	drawOv();
	if (hasCameraMotion()) requestRender();
}

setupInput();
setupAskUI();
requestRender();
