// ============================================================
// SETTINGS — Slider bindings, panel toggle, section collapse
// ============================================================

import { opacitySettings, setStiffness, setDamping, setInertiaStrength, setDotSizeMul,
         STIFFNESS_TAG, STIFFNESS_THOUGHT } from './physics.js';
import { setCamDistTarget } from './camera.js';
import { saveSettings } from './themes.js';

export let DOF_RANGE = 20.0;

export function bindSlider(id, valId, fmt, onChange) {
  const slider = document.getElementById(id);
  const valEl = document.getElementById(valId);
  const handler = () => {
    const v = parseFloat(slider.value);
    const mapped = onChange(v);
    if (valEl) valEl.textContent = fmt(mapped);
    saveSettings();
  };
  slider.addEventListener('input', handler);
  return slider;
}

export function initSettings() {
  // --- Panel toggle ---
  const panelToggle = document.getElementById('panel-toggle');
  const settingsPanel = document.getElementById('settings-panel');

  panelToggle.addEventListener('click', () => {
    const open = settingsPanel.classList.toggle('open');
    panelToggle.classList.toggle('open', open);
    // Move legend out of the way when panel opens
    const legend = document.getElementById('legend');
    legend.style.transition = 'right 0.35s cubic-bezier(0.4, 0, 0.2, 1)';
    legend.style.right = open ? '316px' : '16px';
  });

  // Prevent panel interactions from rotating the globe
  settingsPanel.addEventListener('mousedown', e => e.stopPropagation());
  settingsPanel.addEventListener('wheel', e => e.stopPropagation(), { passive: true });

  // --- Section collapse toggles ---
  document.querySelectorAll('.panel-heading').forEach(heading => {
    heading.addEventListener('click', () => {
      heading.classList.toggle('collapsed');
      const section = heading.nextElementSibling;
      if (section) section.classList.toggle('collapsed');
    });
  });

  // --- Opacity sliders ---
  bindSlider('op-edges', 'op-edges-val', v => v.toFixed(2), v => {
    opacitySettings.edges = v / 100;
    return opacitySettings.edges;
  });
  bindSlider('op-recent-edges', 'op-recent-edges-val', v => v.toFixed(2), v => {
    opacitySettings.recencyBias = v / 100;
    return opacitySettings.recencyBias;
  });
  bindSlider('op-stems', 'op-stems-val', v => v.toFixed(2), v => {
    opacitySettings.stems = v / 100;
    return opacitySettings.stems;
  });
  bindSlider('op-glow', 'op-glow-val', v => v.toFixed(2), v => {
    opacitySettings.glow = v / 100;
    return opacitySettings.glow;
  });
  bindSlider('op-dots', 'op-dots-val', v => v.toFixed(2), v => {
    setDotSizeMul(v / 100);
    return v / 100;
  });
  bindSlider('op-labels', 'op-labels-val', v => v.toFixed(2), v => {
    opacitySettings.labels = v / 100;
    return opacitySettings.labels;
  });

  // --- Physics sliders ---
  const STIFFNESS_TOTAL = 7.0;
  bindSlider('ph-stiffness', 'ph-stiffness-val',
    () => `tag ${STIFFNESS_TAG.toFixed(1)} \u00b7 node ${STIFFNESS_THOUGHT.toFixed(1)}`,
    v => {
      const bias = v / 100; // 0 = all tag, 1 = all thought
      setStiffness(STIFFNESS_TOTAL * (1 - bias), STIFFNESS_TOTAL * bias);
      return bias;
    }
  );
  bindSlider('ph-reactivity', 'ph-reactivity-val', v => v.toFixed(2), v => {
    const t = v / 100; // 0 = stiff, 1 = very reactive
    setInertiaStrength(t * 5.0); // 0..5
    setDamping(0.80 + t * 0.18);  // 0.80..0.98
    return t;
  });

  // --- Camera sliders ---
  bindSlider('cam-zoom', 'cam-zoom-val', v => v.toFixed(1), v => {
    const target = 5 + (v / 100) * 75; // 5..80
    setCamDistTarget(target);
    return target;
  });
  bindSlider('cam-dof', 'cam-dof-val', v => v.toFixed(2), v => {
    DOF_RANGE = 2 + (v / 100) * 48; // 2..50
    return DOF_RANGE;
  });
}
