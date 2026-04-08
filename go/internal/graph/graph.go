// Package graph loads and queries a single .shard graph file.
// One graph = one specialist. This package owns nothing else.
package graph

// Thought is one atomic statement stored in the graph.
type Thought struct {
	ID        uint64
	Text      string
	Embedding []float32 // 1536-dim from external embedding API
	Source    string
	CreatedAt int64
}

// Edge is a directed, weighted link between two thoughts.
type Edge struct {
	SourceID  uint64
	TargetID  uint64
	Weight    float32
	Reasoning string
	Embedding []float32 // embedding of the reasoning text
}

// Signal is a floating edge — an open question with no target yet.
type Signal struct {
	FromID uint64
	Query  string
}

// Graph holds the in-memory state of one specialist's knowledge.
type Graph struct {
	Name    string
	Purpose string
	Path    string

	thoughts map[uint64]*Thought
	edges    []*Edge
	signals  []*Signal
	adj      map[uint64][]*Edge // adjacency list
}

func New(name, purpose, path string) *Graph {
	return &Graph{
		Name:     name,
		Purpose:  purpose,
		Path:     path,
		thoughts: make(map[uint64]*Thought),
		adj:      make(map[uint64][]*Edge),
	}
}

func (g *Graph) ThoughtCount() int { return len(g.thoughts) }
func (g *Graph) EdgeCount() int    { return len(g.edges) }
func (g *Graph) Signals() []*Signal { return g.signals }

func (g *Graph) AddThought(t *Thought) {
	g.thoughts[t.ID] = t
}

func (g *Graph) AddEdge(e *Edge) {
	g.edges = append(g.edges, e)
	g.adj[e.SourceID] = append(g.adj[e.SourceID], e)
}

func (g *Graph) AddSignal(s *Signal) {
	g.signals = append(g.signals, s)
}

// Neighbors returns all thoughts reachable from id within maxDepth hops.
func (g *Graph) Neighbors(id uint64, maxDepth int) []*Thought {
	visited := map[uint64]bool{id: true}
	frontier := []uint64{id}
	var result []*Thought

	for depth := 0; depth < maxDepth && len(frontier) > 0; depth++ {
		var next []uint64
		for _, fid := range frontier {
			for _, e := range g.adj[fid] {
				if !visited[e.TargetID] {
					visited[e.TargetID] = true
					next = append(next, e.TargetID)
					if t, ok := g.thoughts[e.TargetID]; ok {
						result = append(result, t)
					}
				}
			}
		}
		frontier = next
	}
	return result
}

// TopK returns the k thoughts with highest cosine similarity to query.
func (g *Graph) TopK(query []float32, k int) []*Thought {
	type scored struct {
		t    *Thought
		score float32
	}
	var candidates []scored
	for _, t := range g.thoughts {
		if len(t.Embedding) != len(query) {
			continue
		}
		candidates = append(candidates, scored{t, cosine(query, t.Embedding)})
	}
	// partial sort: bubble top-k to front
	for i := 0; i < k && i < len(candidates); i++ {
		for j := i + 1; j < len(candidates); j++ {
			if candidates[j].score > candidates[i].score {
				candidates[i], candidates[j] = candidates[j], candidates[i]
			}
		}
	}
	end := k
	if end > len(candidates) {
		end = len(candidates)
	}
	out := make([]*Thought, end)
	for i := range out {
		out[i] = candidates[i].t
	}
	return out
}

func cosine(a, b []float32) float32 {
	var dot, na, nb float32
	for i := range a {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (sqrt32(na) * sqrt32(nb))
}

func sqrt32(x float32) float32 {
	if x <= 0 {
		return 0
	}
	// Newton's method — avoids importing math for one function
	z := x
	for i := 0; i < 10; i++ {
		z -= (z*z - x) / (2 * z)
	}
	return z
}
