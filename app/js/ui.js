// ============================================================
// UI — Store stats, initial DOM updates
// ============================================================

import { STORE, specialists, GLOBAL_TAGS } from './data.js';
import { totalNodes, sphereInfo, bridgeNodes } from './geometry.js';

export function initUI() {
  document.getElementById('store-name').textContent = STORE.name;
  document.getElementById('store-purpose').textContent = STORE.purpose;

  // Count total tags across all spheres
  let totalTags = 0;
  for (let si = 0; si < specialists.length; si++) {
    totalTags += sphereInfo[si].tagCount;
  }
  totalTags += bridgeNodes.length;

  document.getElementById('store-stats').textContent =
    `${STORE.thought_count} thoughts \u00b7 ${STORE.edge_count} edges \u00b7 ${totalTags} tags \u00b7 ${specialists.length} specialists`;
}
