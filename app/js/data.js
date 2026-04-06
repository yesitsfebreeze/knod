// ============================================================
// DATA — MULTI-SPHERE GALAXY (25 specialists)
// ============================================================

// Simple seedable PRNG
function mulberry32(a) {
  return function() {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

export const STORE = {
  name: "knod galaxy",
  purpose: "25 specialist knowledge spheres connected by shared concepts",
  thought_count: 0,
  edge_count: 0,
  specialist_count: 25,
};

// -------------------------------------------------------
// GLOBAL TAG POOL — 60 unique tags. Each specialist picks 20.
// Overlapping picks create shared/bridge tags between spheres.
// -------------------------------------------------------
export const GLOBAL_TAGS = [
  "memory-mgmt","concurrency","networking","file-systems","compilers",
  "operating-systems","data-structures","algorithms","cpu-arch","type-systems",
  "garbage-collection","virtual-memory","scheduling","cache-coherence","linkers",
  "syscalls","IPC","SIMD","lock-free","page-tables",
  "machine-learning","neural-networks","transformers","embeddings","optimization",
  "backpropagation","reinforcement-learning","computer-vision","NLP","generative-models",
  "databases","SQL","indexing","transactions","replication",
  "distributed-sys","consensus","fault-tolerance","load-balancing","sharding",
  "cryptography","hashing","encryption","signatures","zero-knowledge",
  "web-dev","HTTP","REST","GraphQL","WebSockets",
  "security","authentication","authorization","sandboxing","fuzzing",
  "graphics","shaders","ray-tracing","rasterization","GPU-compute",
];

// 25 specialists — each has a name, purpose, and 20 indices into GLOBAL_TAGS
export const SPECIALIST_DEFS = [
  { name:"systems-prog", purpose:"OS internals, memory, compilers", tags:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19] },
  { name:"machine-learning", purpose:"ML models, training, inference", tags:[20,21,22,23,24,25,26,27,28,29,6,7,8,17,24,30,31,32,33,34] },
  { name:"databases", purpose:"Storage engines, queries, indexing", tags:[30,31,32,33,34,6,7,3,0,11,35,36,37,38,39,12,1,16,14,9] },
  { name:"distributed-sys", purpose:"Consensus, replication, fault tolerance", tags:[35,36,37,38,39,1,2,12,30,34,33,16,6,7,8,40,41,42,43,44] },
  { name:"cryptography", purpose:"Encryption, hashing, zero-knowledge", tags:[40,41,42,43,44,6,7,8,0,9,1,17,24,45,46,47,48,49,50,51] },
  { name:"web-dev", purpose:"HTTP, APIs, real-time communication", tags:[45,46,47,48,49,2,1,6,7,30,38,50,51,52,53,54,55,56,57,3] },
  { name:"security", purpose:"Auth, sandboxing, vulnerability research", tags:[50,51,52,53,54,40,41,42,0,5,15,2,45,46,1,6,7,11,16,19] },
  { name:"graphics", purpose:"Rendering, shaders, GPU compute", tags:[55,56,57,58,59,8,17,6,7,0,1,24,9,4,11,20,21,22,23,25] },
  { name:"compiler-design", purpose:"Parsing, optimization, code gen", tags:[4,9,14,0,6,7,8,13,17,18,1,5,15,11,12,55,59,20,24,30] },
  { name:"networking-deep", purpose:"Protocol design, packet processing", tags:[2,16,1,38,39,45,49,0,6,7,8,12,35,36,37,5,15,17,50,52] },
  { name:"concurrency-models", purpose:"Lock-free, actors, CSP, coroutines", tags:[1,18,13,12,0,6,7,8,16,17,10,11,5,15,19,35,36,37,38,39] },
  { name:"functional-prog", purpose:"Type theory, category theory, purity", tags:[9,6,7,24,4,0,1,10,11,20,21,22,28,29,25,40,43,44,14,30] },
  { name:"embedded-sys", purpose:"Microcontrollers, RTOS, bare metal", tags:[0,5,8,12,13,15,17,1,3,6,7,14,16,18,19,11,4,55,58,59] },
  { name:"cloud-native", purpose:"Containers, orchestration, serverless", tags:[35,36,37,38,39,45,46,47,2,1,30,34,50,51,12,6,7,16,52,53] },
  { name:"data-science", purpose:"Statistics, visualization, pipelines", tags:[20,21,23,24,25,29,30,31,32,6,7,28,33,34,38,39,9,55,56,57] },
  { name:"game-dev", purpose:"Physics engines, ECS, real-time rendering", tags:[55,56,57,58,59,6,7,8,0,1,12,13,17,20,24,4,9,18,11,3] },
  { name:"devops", purpose:"CI/CD, monitoring, infrastructure as code", tags:[35,36,38,39,45,46,50,51,52,53,2,30,34,12,1,6,7,16,37,54] },
  { name:"quantum-comp", purpose:"Qubits, gates, quantum algorithms", tags:[40,43,44,6,7,8,24,25,20,21,9,0,1,17,22,23,29,41,42,55] },
  { name:"robotics", purpose:"Control systems, perception, planning", tags:[20,26,27,8,6,7,0,1,12,17,24,25,55,58,59,5,13,15,35,36] },
  { name:"NLP-deep", purpose:"Language models, parsing, generation", tags:[28,29,22,23,20,21,24,25,6,7,9,30,31,32,33,34,4,14,40,43] },
  { name:"blockchain", purpose:"Consensus, smart contracts, DeFi", tags:[40,41,42,43,44,35,36,37,1,6,7,30,33,34,38,39,2,16,45,49] },
  { name:"audio-eng", purpose:"DSP, synthesis, codecs", tags:[17,8,6,7,0,1,55,59,4,9,24,25,20,21,14,11,3,18,12,13] },
  { name:"bioinformatics", purpose:"Genomics, protein folding, sequences", tags:[6,7,20,21,22,23,24,25,30,31,32,33,34,9,0,1,28,29,40,43] },
  { name:"formal-verify", purpose:"Model checking, theorem proving", tags:[9,4,6,7,0,1,40,43,44,14,18,19,24,25,20,21,10,11,5,15] },
  { name:"edge-computing", purpose:"IoT, fog, low-latency inference", tags:[2,35,36,38,39,0,1,5,8,12,13,17,20,24,45,46,50,55,58,59] },
];

export const SOURCES = [
  "textbook","research-papers","documentation","stack-overflow",
  "man-pages","blog-posts","conference-talks",
];

export const EDGE_REASONINGS = [
  "Both relate to memory hierarchy performance",
  "Shared concern with resource lifecycle management",
  "Both address concurrency safety mechanisms",
  "Connected through implementation details",
  "Both optimize for cache locality patterns",
  "Shared dependency on atomic operations",
  "Both address data structure efficiency",
  "Connected through parallel execution models",
  "Both relate to distributed coordination",
  "Shared concern with fault tolerance",
  "Both optimize for reduced latency",
  "Connected through mathematical foundations",
];

export const GENERIC_THOUGHTS = [
  "First principles reasoning cuts through accumulated complexity",
  "Abstraction boundaries define what information crosses module interfaces",
  "Performance depends on the critical path through the data dependency graph",
  "Correctness proofs give stronger guarantees than testing alone",
  "Cache locality dominates performance for data-intensive workloads",
  "Incremental computation avoids redundant work by tracking dependencies",
  "Composition is more flexible than inheritance for code reuse",
  "Idempotent operations simplify retry logic in unreliable environments",
  "Immutable data structures eliminate whole classes of concurrency bugs",
  "Property-based testing explores the input space more thoroughly than examples",
  "Profiling should guide optimization — avoid premature optimization",
  "The read path and write path often have very different performance needs",
  "Layered abstractions enable independent evolution of components",
  "Trade-offs between throughput and latency shape system architecture",
  "Backpressure prevents fast producers from overwhelming slow consumers",
];

// -------------------------------------------------------
// Generate all specialists: tags, thoughts, edges
// -------------------------------------------------------
export const specialists = [];
export const globalTagSet = new Map(); // label -> { owners: Set<specialistIdx> }
export const allThoughts = [];
export const allEdges = [];
let globalThoughtId = 1;

// Register all tags first to find overlaps
for (let si = 0; si < SPECIALIST_DEFS.length; si++) {
  const def = SPECIALIST_DEFS[si];
  // Deduplicate tag indices within this specialist
  const uniqueTags = [...new Set(def.tags)];
  def.tags = uniqueTags.slice(0, 20);
  for (const tagIdx of def.tags) {
    const label = GLOBAL_TAGS[tagIdx];
    if (!globalTagSet.has(label)) {
      globalTagSet.set(label, { owners: new Set() });
    }
    globalTagSet.get(label).owners.add(si);
  }
}

// Build per-specialist data
for (let si = 0; si < SPECIALIST_DEFS.length; si++) {
  const def = SPECIALIST_DEFS[si];
  const rng = mulberry32(42 + si * 1337);
  const now = Date.now();
  const DAY = 86400000;

  const localTags = def.tags.map(gi => ({
    dim_index: gi * 25 + Math.floor(gi * 1.7),
    label: GLOBAL_TAGS[gi],
    globalIndex: gi,
  }));

  // Generate 200 thoughts per specialist
  const thoughts = [];
  for (let i = 0; i < 200; i++) {
    const affinities = localTags.map((tag, ti) => {
      const primary = Math.floor(rng() * localTags.length);
      const secondary = Math.floor(rng() * localTags.length);
      const base = rng() * 0.15;
      const boost = (ti === primary) ? 0.6 + rng() * 0.3 :
                    (ti === secondary) ? 0.3 + rng() * 0.2 :
                    (Math.abs(ti - primary) <= 2) ? 0.15 + rng() * 0.15 : base;
      return boost;
    });

    const age = rng();
    const created_at = now - Math.floor(age * 90 * DAY);
    const last_accessed = created_at + Math.floor(rng() * (now - created_at));
    const text = GENERIC_THOUGHTS[Math.floor(rng() * GENERIC_THOUGHTS.length)];

    thoughts.push({
      id: globalThoughtId++,
      text,
      affinities,
      access_count: Math.floor(rng() * 50),
      created_at,
      last_accessed,
      age,
      source_id: SOURCES[Math.floor(rng() * SOURCES.length)],
      specialistIdx: si,
    });
  }

  // Generate intra-specialist edges
  const edges = [];
  const edgeRng = mulberry32(67890 + si * 999);
  for (let i = 0; i < thoughts.length; i++) {
    const numE = 1 + Math.floor(edgeRng() * 3);
    for (let e = 0; e < numE; e++) {
      let target;
      if (edgeRng() < 0.6) {
        target = (i + 1 + Math.floor(edgeRng() * 15)) % thoughts.length;
      } else {
        target = Math.floor(edgeRng() * thoughts.length);
      }
      if (target === i) continue;
      const a = thoughts[i].affinities, b = thoughts[target].affinities;
      let dot = 0, mA = 0, mB = 0;
      for (let d = 0; d < a.length; d++) { dot += a[d]*b[d]; mA += a[d]*a[d]; mB += b[d]*b[d]; }
      const cosine = dot / (Math.sqrt(mA) * Math.sqrt(mB) + 1e-8);
      const weight = 0.1 + cosine * 0.8;
      if (weight < 0.3) continue;
      edges.push({
        source_id: thoughts[i].id, target_id: thoughts[target].id, weight,
        reasoning: EDGE_REASONINGS[Math.floor(edgeRng() * EDGE_REASONINGS.length)],
        created_at: Math.min(thoughts[i].created_at, thoughts[target].created_at) + Math.floor(edgeRng() * DAY * 5),
      });
    }
  }
  // Deduplicate edges
  const seen = new Map();
  for (const edge of edges) {
    const key = Math.min(edge.source_id, edge.target_id) + ',' + Math.max(edge.source_id, edge.target_id);
    if (!seen.has(key) || seen.get(key).weight < edge.weight) seen.set(key, edge);
  }
  const dedupEdges = Array.from(seen.values());

  specialists.push({ name: def.name, purpose: def.purpose, tags: localTags, thoughts, edges: dedupEdges });
  allThoughts.push(...thoughts);
  allEdges.push(...dedupEdges);
}

// Normalize edge ages globally
{
  let minT = Infinity, maxT = -Infinity;
  for (const e of allEdges) { if (e.created_at < minT) minT = e.created_at; if (e.created_at > maxT) maxT = e.created_at; }
  const range = maxT - minT || 1;
  for (const e of allEdges) { e.age = 1.0 - (e.created_at - minT) / range; }
}

// Build adjacency for quick lookup
export const adjacency = new Map();
allEdges.forEach((edge, ei) => {
  if (!adjacency.has(edge.source_id)) adjacency.set(edge.source_id, []);
  if (!adjacency.has(edge.target_id)) adjacency.set(edge.target_id, []);
  adjacency.get(edge.source_id).push({ target: edge.target_id, edgeIdx: ei });
  adjacency.get(edge.target_id).push({ target: edge.source_id, edgeIdx: ei });
});

STORE.thought_count = allThoughts.length;
STORE.edge_count = allEdges.length;

// -------------------------------------------------------
// SHARED TAGS — tags owned by 2+ specialists become bridge nodes
// -------------------------------------------------------
export const sharedTags = [];
for (const [label, info] of globalTagSet) {
  if (info.owners.size >= 2) {
    sharedTags.push({ label, owners: Array.from(info.owners) });
  }
}
