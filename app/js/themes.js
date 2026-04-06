// ============================================================
// SETTINGS PANEL — Base16 palette, physics, opacity controls
// ============================================================

// Note: No import from labels.js — we query label elements from the DOM
// to avoid circular dependency (themes -> labels -> physics -> themes)

// --- Base16 theme definitions ---
// Each has 16 colors: base00..base0F following the Base16 spec
export const BASE16_THEMES = {
  'default-dark': {
    name: 'default dark',
    base00: '181818', base01: '282828', base02: '383838', base03: '585858',
    base04: 'b8b8b8', base05: 'd8d8d8', base06: 'e8e8e8', base07: 'f8f8f8',
    base08: 'ab4642', base09: 'dc9656', base0A: 'f7ca88', base0B: 'a1b56c',
    base0C: '86c1b9', base0D: '7cafc2', base0E: 'ba8baf', base0F: 'a16946',
  },
  'ocean': {
    name: 'ocean',
    base00: '2b303b', base01: '343d46', base02: '4f5b66', base03: '65737e',
    base04: 'a7adba', base05: 'c0c5ce', base06: 'dfe1e8', base07: 'eff1f5',
    base08: 'bf616a', base09: 'd08770', base0A: 'ebcb8b', base0B: 'a3be8c',
    base0C: '96b5b4', base0D: '8fa1b3', base0E: 'b48ead', base0F: 'ab7967',
  },
  'monokai': {
    name: 'monokai',
    base00: '272822', base01: '383830', base02: '49483e', base03: '75715e',
    base04: 'a59f85', base05: 'f8f8f2', base06: 'f5f4f1', base07: 'f9f8f5',
    base08: 'f92672', base09: 'fd971f', base0A: 'f4bf75', base0B: 'a6e22e',
    base0C: 'a1efe4', base0D: '66d9ef', base0E: 'ae81ff', base0F: 'cc6633',
  },
  'solarized-dark': {
    name: 'solarized',
    base00: '002b36', base01: '073642', base02: '586e75', base03: '657b83',
    base04: '839496', base05: '93a1a1', base06: 'eee8d5', base07: 'fdf6e3',
    base08: 'dc322f', base09: 'cb4b16', base0A: 'b58900', base0B: '859900',
    base0C: '2aa198', base0D: '268bd2', base0E: '6c71c4', base0F: 'd33682',
  },
  'gruvbox-dark': {
    name: 'gruvbox',
    base00: '282828', base01: '3c3836', base02: '504945', base03: '665c54',
    base04: 'bdae93', base05: 'd5c4a1', base06: 'ebdbb2', base07: 'fbf1c7',
    base08: 'fb4934', base09: 'fe8019', base0A: 'fabd2f', base0B: 'b8bb26',
    base0C: '8ec07c', base0D: '83a598', base0E: 'd3869b', base0F: 'd65d0e',
  },
  'nord': {
    name: 'nord',
    base00: '2e3440', base01: '3b4252', base02: '434c5e', base03: '4c566a',
    base04: 'd8dee9', base05: 'e5e9f0', base06: 'eceff4', base07: '8fbcbb',
    base08: 'bf616a', base09: 'd08770', base0A: 'ebcb8b', base0B: 'a3be8c',
    base0C: '88c0d0', base0D: '81a1c1', base0E: 'b48ead', base0F: '5e81ac',
  },
  'dracula': {
    name: 'dracula',
    base00: '282936', base01: '3a3c4e', base02: '4d4f68', base03: '626483',
    base04: '62d6e8', base05: 'e9e9f4', base06: 'f1f2f8', base07: 'f7f7fb',
    base08: 'ea51b2', base09: 'b45bcf', base0A: 'ebff87', base0B: '00f769',
    base0C: 'a1efe4', base0D: '62d6e8', base0E: 'b45bcf', base0F: '00f769',
  },
  'tomorrow-night': {
    name: 'tomorrow',
    base00: '1d1f21', base01: '282a2e', base02: '373b41', base03: '969896',
    base04: 'b4b7b4', base05: 'c5c8c6', base06: 'e0e0e0', base07: 'ffffff',
    base08: 'cc6666', base09: 'de935f', base0A: 'f0c674', base0B: 'b5bd68',
    base0C: '8abeb7', base0D: '81a2be', base0E: 'b294bb', base0F: 'a3685a',
  },
  'zenburn': {
    name: 'zenburn',
    base00: '383838', base01: '404040', base02: '606060', base03: '6f6f6f',
    base04: '808080', base05: 'dcdccc', base06: 'c0c0c0', base07: 'ffffff',
    base08: 'dca3a3', base09: 'dfaf8f', base0A: 'e0cf9f', base0B: '5f7f5f',
    base0C: '93e0e3', base0D: '7cb8bb', base0E: 'dc8cc3', base0F: '000000',
  },
  'knod': {
    name: 'knod',
    base00: '0a0a0f', base01: '12121a', base02: '1e1e2a', base03: '585868',
    base04: '8888a0', base05: 'c8c8d0', base06: 'e0e0e8', base07: 'f0f0f4',
    base08: 'dd6644', base09: 'c8a040', base0A: 'f0c674', base0B: '448888',
    base0C: '86c1b9', base0D: '8cb4ff', base0E: 'ba8baf', base0F: '7a9ab8',
  },
  'catppuccin-mocha': {
    name: 'catppuccin',
    base00: '1e1e2e', base01: '181825', base02: '313244', base03: '45475a',
    base04: '585b70', base05: 'cdd6f4', base06: 'f5e0dc', base07: 'b4befe',
    base08: 'f38ba8', base09: 'fab387', base0A: 'f9e2af', base0B: 'a6e3a1',
    base0C: '94e2d5', base0D: '89b4fa', base0E: 'cba6f7', base0F: 'f2cdcd',
  },
  'rose-pine': {
    name: 'rose pine',
    base00: '191724', base01: '1f1d2e', base02: '26233a', base03: '6e6a86',
    base04: '908caa', base05: 'e0def4', base06: 'e0def4', base07: '524f67',
    base08: 'eb6f92', base09: 'ebbcba', base0A: 'f6c177', base0B: '31748f',
    base0C: '9ccfd8', base0D: 'c4a7e7', base0E: 'c4a7e7', base0F: '56526e',
  },
  'tokyo-night': {
    name: 'tokyo night',
    base00: '1a1b26', base01: '16161e', base02: '2f3549', base03: '444b6a',
    base04: '787c99', base05: 'a9b1d6', base06: 'cbccd1', base07: 'd5d6db',
    base08: 'f7768e', base09: 'ff9e64', base0A: 'e0af68', base0B: '9ece6a',
    base0C: '2ac3de', base0D: '7aa2f7', base0E: 'bb9af7', base0F: 'db4b4b',
  },
  'one-dark': {
    name: 'one dark',
    base00: '282c34', base01: '353b45', base02: '3e4451', base03: '545862',
    base04: '565c64', base05: 'abb2bf', base06: 'b6bdca', base07: 'c8ccd4',
    base08: 'e06c75', base09: 'd19a66', base0A: 'e5c07b', base0B: '98c379',
    base0C: '56b6c2', base0D: '61afef', base0E: 'c678dd', base0F: 'be5046',
  },
  'kanagawa': {
    name: 'kanagawa',
    base00: '1f1f28', base01: '16161d', base02: '223249', base03: '54546d',
    base04: '727169', base05: 'dcd7ba', base06: 'c8c093', base07: '717c7c',
    base08: 'c34043', base09: 'ffa066', base0A: 'c0a36e', base0B: '76946a',
    base0C: '6a9589', base0D: '7e9cd8', base0E: '957fb8', base0F: 'd27e99',
  },
  'everforest': {
    name: 'everforest',
    base00: '2d353b', base01: '343f44', base02: '3d484d', base03: '475258',
    base04: '859289', base05: 'd3c6aa', base06: 'e4dfc5', base07: 'fdf6e3',
    base08: 'e67e80', base09: 'e69875', base0A: 'dbbc7f', base0B: 'a7c080',
    base0C: '83c092', base0D: '7fbbb3', base0E: 'd699b6', base0F: 'e66f4e',
  },
  'ayu-dark': {
    name: 'ayu',
    base00: '0a0e14', base01: '1f2430', base02: '232834', base03: '707a8c',
    base04: '8a9199', base05: 'b3b1ad', base06: 'c7c7c7', base07: 'd9d7ce',
    base08: 'f07178', base09: 'ff8f40', base0A: 'ffb454', base0B: 'c2d94c',
    base0C: '95e6cb', base0D: '59c2ff', base0E: 'd2a6ff', base0F: 'e6b673',
  },
  'synthwave-84': {
    name: 'synthwave',
    base00: '262335', base01: '34294f', base02: '413a5e', base03: '6c6783',
    base04: '9b97b0', base05: 'e0dfe1', base06: 'f0eff1', base07: 'fefeff',
    base08: 'f97e72', base09: 'ff8b39', base0A: 'fede5d', base0B: '72f1b8',
    base0C: '36f9f6', base0D: '69bbff', base0E: 'ff7edb', base0F: 'fe4450',
  },
  'palenight': {
    name: 'palenight',
    base00: '292d3e', base01: '313548', base02: '444267', base03: '676e95',
    base04: '8796b0', base05: '959dcb', base06: 'bfc7d5', base07: 'ffffff',
    base08: 'f07178', base09: 'f78c6c', base0A: 'ffcb6b', base0B: 'c3e88d',
    base0C: '89ddff', base0D: '82aaff', base0E: 'c792ea', base0F: 'ff5370',
  },
  'github-dark': {
    name: 'github dark',
    base00: '0d1117', base01: '161b22', base02: '21262d', base03: '30363d',
    base04: '484f58', base05: 'c9d1d9', base06: 'ecf2f8', base07: 'f0f6fc',
    base08: 'ff7b72', base09: 'ffa657', base0A: 'f0c674', base0B: '7ee787',
    base0C: 'a5d6ff', base0D: '79c0ff', base0E: 'd2a8ff', base0F: 'ffa198',
  },
  'vesper': {
    name: 'vesper',
    base00: '101010', base01: '1a1a1a', base02: '2a2a2a', base03: '3a3a3a',
    base04: '6a6a6a', base05: 'b0b0b0', base06: 'd4d4d4', base07: 'ffffff',
    base08: 'f5a191', base09: 'ffb86c', base0A: 'ffc799', base0B: 'a1c181',
    base0C: '8cc7dc', base0D: 'daa1f0', base0E: 'c8a1ff', base0F: 'e0876a',
  },
};

// Parse hex to [r,g,b] floats (0..1)
export function hexToRgb(hex) {
  const n = parseInt(hex, 16);
  return [(n >> 16 & 0xff) / 255, (n >> 8 & 0xff) / 255, (n & 0xff) / 255];
}

// Current active palette (resolved to float arrays)
export let activeThemeKey = 'knod';
export let palette = {}; // base00..base0F as [r,g,b]

// Color mapping: which palette slots are used for old/recent/stems
// Only offer base08..base0F (the accent colors) plus base04/base05 for stems
export const COLOR_SLOTS = ['base04','base05','base06','base07','base08','base09','base0A','base0B','base0C','base0D','base0E','base0F'];
export let colorMapOld = 'base0B';
export let colorMapRecent = 'base08';
export let colorMapStem = 'base05';

export function setColorMapOld(s) { colorMapOld = s; }
export function setColorMapRecent(s) { colorMapRecent = s; }
export function setColorMapStem(s) { colorMapStem = s; }

export function getOldColor() { return palette[colorMapOld] || [0.27, 0.53, 0.53]; }
export function getRecentColor() { return palette[colorMapRecent] || [0.86, 0.40, 0.27]; }
export function getStemColor() { return palette[colorMapStem] || [0.78, 0.78, 0.82]; }

// --- Palette change callbacks ---
const _paletteChangeCallbacks = [];
export function onPaletteChange(fn) { _paletteChangeCallbacks.push(fn); }
function firePaletteChange() { for (const fn of _paletteChangeCallbacks) fn(); }

export function applyTheme(key, skipSave) {
  const theme = BASE16_THEMES[key];
  if (!theme) return;
  activeThemeKey = key;
  palette = {};
  for (const k of Object.keys(theme)) {
    if (k.startsWith('base')) palette[k] = hexToRgb(theme[k]);
  }

  // Apply to DOM elements
  const b00 = theme.base00, b01 = theme.base01, b02 = theme.base02, b03 = theme.base03;
  const b04 = theme.base04, b05 = theme.base05, b06 = theme.base06, b07 = theme.base07;
  const b08 = theme.base08, b0B = theme.base0B, b0C = theme.base0C, b0D = theme.base0D;
  const b0E = theme.base0E, b0F = theme.base0F, b09 = theme.base09, b0A = theme.base0A;

  document.body.style.background = '#' + b00;
  document.body.style.color = '#' + b05;

  // Info panel
  const infoH1 = document.querySelector('#info h1');
  if (infoH1) infoH1.style.color = '#' + b06;

  // Tooltip
  const tip = document.getElementById('tooltip');
  tip.style.background = `rgba(${parseInt(b00.slice(0,2),16)}, ${parseInt(b00.slice(2,4),16)}, ${parseInt(b00.slice(4,6),16)}, 0.92)`;
  tip.style.borderColor = `rgba(${parseInt(b0D.slice(0,2),16)}, ${parseInt(b0D.slice(2,4),16)}, ${parseInt(b0D.slice(4,6),16)}, 0.3)`;
  const tipLabel = tip.querySelector('.label');
  if (tipLabel) tipLabel.style.color = '#' + b0D;
  const tipDetail = tip.querySelector('.detail');
  if (tipDetail) tipDetail.style.color = '#' + b04;
  const tipSource = tip.querySelector('.source');
  if (tipSource) tipSource.style.color = '#' + b0F;
  const tipConn = tip.querySelector('.connections');
  if (tipConn) tipConn.style.color = '#' + b09;

  // Legend dots — follow color map selections
  updateLegendColors(theme);

  // Settings panel styling
  const panel = document.getElementById('settings-panel');
  panel.style.background = `rgba(${parseInt(b00.slice(0,2),16)}, ${parseInt(b00.slice(2,4),16)}, ${parseInt(b00.slice(4,6),16)}, 0.92)`;

  // Tag labels follow palette
  const labelContainer = document.getElementById('label-container');
  if (labelContainer) {
    for (const el of labelContainer.children) {
      el.style.color = `rgba(${parseInt(b0D.slice(0,2),16)}, ${parseInt(b0D.slice(2,4),16)}, ${parseInt(b0D.slice(4,6),16)}, 0.9)`;
      el.style.background = `rgba(${parseInt(b00.slice(0,2),16)}, ${parseInt(b01.slice(2,4),16)}, ${parseInt(b01.slice(4,6),16)}, 0.55)`;
      el.style.borderColor = `rgba(${parseInt(b0D.slice(0,2),16)}, ${parseInt(b0D.slice(2,4),16)}, ${parseInt(b0D.slice(4,6),16)}, 0.12)`;
    }
  }

  // Swatch display
  renderPaletteGrid(theme);
  renderAllColorMaps();
  updateLegendColors(theme);

  // Highlight active theme button
  document.querySelectorAll('.theme-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.key === key);
  });

  if (!skipSave) saveSettings();
  firePaletteChange();
}

// --- Build theme selector buttons ---
export function buildThemeList() {
  const container = document.getElementById('theme-list');
  container.innerHTML = '';
  for (const key of Object.keys(BASE16_THEMES)) {
    const btn = document.createElement('button');
    btn.className = 'theme-btn';
    btn.dataset.key = key;
    btn.textContent = key;
    btn.addEventListener('click', () => applyTheme(key));
    container.appendChild(btn);
  }
}

// --- Palette grid: 16 swatches showing current palette ---
function renderPaletteGrid(theme) {
  const container = document.getElementById('palette-grid');
  container.innerHTML = '';
  const keys = Object.keys(theme).filter(k => k.startsWith('base'));
  for (const k of keys) {
    const swatch = document.createElement('div');
    swatch.className = 'palette-swatch';
    swatch.style.background = '#' + theme[k];
    swatch.textContent = k.replace('base', '');
    swatch.title = k + ': #' + theme[k];
    container.appendChild(swatch);
  }
}

// --- Color map row: clickable swatches for picking a palette slot ---
function renderColorMapRow(containerId, currentSlot, onChange) {
  const container = document.getElementById(containerId);
  if (!container) return;
  container.innerHTML = '';
  const theme = BASE16_THEMES[activeThemeKey];
  if (!theme) return;
  for (const slot of COLOR_SLOTS) {
    const swatch = document.createElement('div');
    swatch.className = 'cmap-swatch' + (slot === currentSlot ? ' selected' : '');
    swatch.style.background = '#' + theme[slot];
    swatch.title = slot;
    swatch.addEventListener('click', () => {
      onChange(slot);
      renderAllColorMaps();
      updateLegendColors(theme);
      saveSettings();
      firePaletteChange();
    });
    container.appendChild(swatch);
  }
}

export function renderAllColorMaps() {
  renderColorMapRow('cmap-old', colorMapOld, s => { colorMapOld = s; });
  renderColorMapRow('cmap-recent', colorMapRecent, s => { colorMapRecent = s; });
  renderColorMapRow('cmap-stem', colorMapStem, s => { colorMapStem = s; });
}

export function updateLegendColors(theme) {
  if (!theme) theme = BASE16_THEMES[activeThemeKey];
  if (!theme) return;
  const legendDots = document.querySelectorAll('#legend .dot');
  if (legendDots[0]) legendDots[0].style.background = '#' + theme[colorMapOld];
  if (legendDots[1]) legendDots[1].style.background = '#' + theme[colorMapRecent];
  if (legendDots[2]) legendDots[2].style.background = '#' + theme[colorMapStem];
}

// --- Settings persistence ---
const SETTINGS_KEY = 'knod-viz-settings';
const ALL_SLIDER_IDS = [
  'op-edges', 'op-recent-edges', 'op-stems',
  'op-glow', 'op-dots', 'op-labels',
  'ph-stiffness', 'ph-reactivity',
  'cam-zoom', 'cam-dof',
];

export function saveSettings() {
  const data = {
    theme: activeThemeKey,
    sliders: {},
    colorMaps: { old: colorMapOld, recent: colorMapRecent, stem: colorMapStem }
  };
  for (const id of ALL_SLIDER_IDS) {
    const el = document.getElementById(id);
    if (el) data.sliders[id] = parseFloat(el.value);
  }
  try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(data)); } catch (e) {}
}

export function loadSettings() {
  let data;
  try { data = JSON.parse(localStorage.getItem(SETTINGS_KEY)); } catch (e) {}
  if (!data) return false;

  // Apply theme first
  if (data.theme && BASE16_THEMES[data.theme]) {
    applyTheme(data.theme, true); // true = skip save (we're loading)
  }

  // Restore color map selections
  if (data.colorMaps) {
    if (data.colorMaps.old && COLOR_SLOTS.includes(data.colorMaps.old)) colorMapOld = data.colorMaps.old;
    if (data.colorMaps.recent && COLOR_SLOTS.includes(data.colorMaps.recent)) colorMapRecent = data.colorMaps.recent;
    if (data.colorMaps.stem && COLOR_SLOTS.includes(data.colorMaps.stem)) colorMapStem = data.colorMaps.stem;
    renderAllColorMaps();
    updateLegendColors();
  }

  // Restore slider positions and fire their handlers
  if (data.sliders) {
    for (const id of ALL_SLIDER_IDS) {
      if (data.sliders[id] !== undefined) {
        const el = document.getElementById(id);
        if (el) {
          el.value = data.sliders[id];
          el.dispatchEvent(new Event('input'));
        }
      }
    }
  }
  return true;
}
