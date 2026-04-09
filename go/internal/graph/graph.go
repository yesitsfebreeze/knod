// Package graph loads and queries a single specialist's knowledge graph.
// One graph = one specialist. This package owns nothing else.
package graph

import (
	"crypto/sha256"
	"encoding/gob"
	"fmt"
	"os"
)

// Thought is one atomic statement stored in the graph.
type Thought struct {
	ID        uint64
	IP        string    // sha256(text:source)[:16] — content address
	PID       int       // process that committed this thought
	Score     float32   // quality / confidence (default 1.0)
	Text      string
	Embedding []float32 // 1536-dim from external embedding API
	Source    string
	CreatedAt int64
}

// Edge is a directed, weighted link between two thoughts.
type Edge struct {
	SourceID  uint64
	TargetID  uint64
	EdgeIP    string    // sha256(source_id + target_id + reasoning)[:16]
	PID       int       // process that created this edge
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

func (g *Graph) ThoughtCount() int  { return len(g.thoughts) }
func (g *Graph) EdgeCount() int     { return len(g.edges) }
func (g *Graph) Signals() []*Signal { return g.signals }

func (g *Graph) AddThought(t *Thought) {
	if t.Score == 0 {
		t.Score = 1.0
	}
	g.thoughts[t.ID] = t
}

func (g *Graph) AddEdge(e *Edge) {
	g.edges = append(g.edges, e)
	g.adj[e.SourceID] = append(g.adj[e.SourceID], e)
}

func (g *Graph) AddSignal(s *Signal) {
	g.signals = append(g.signals, s)
}

// AllThoughts returns all thoughts as a flat slice (for tagcloud input).
func (g *Graph) AllThoughts() []*Thought {
	out := make([]*Thought, 0, len(g.thoughts))
	for _, t := range g.thoughts {
		out = append(out, t)
	}
	return out
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
		t     *Thought
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

// ThoughtIP computes the stable content address for a thought.
func ThoughtIP(text, source string) string {
	h := sha256.Sum256([]byte(text + ":" + source))
	return fmt.Sprintf("%x", h[:8])
}

// EdgeIP computes the stable content address for an edge.
func EdgeIP(sourceID, targetID uint64, reasoning string) string {
	h := sha256.Sum256([]byte(fmt.Sprintf("%d:%d:%s", sourceID, targetID, reasoning)))
	return fmt.Sprintf("%x", h[:8])
}

// --- Persistence ---

// graphFile is the serialized form used by gob encoding.
type graphFile struct {
	Name     string
	Purpose  string
	Thoughts []*Thought
	Edges    []*Edge
	Signals  []*Signal
}

// Save writes the graph to path using gob encoding.
// Uses .gshard extension by convention (Go-native format, decoupled from Python .shard).
func (g *Graph) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("graph save: %w", err)
	}
	defer f.Close()

	thoughts := make([]*Thought, 0, len(g.thoughts))
	for _, t := range g.thoughts {
		thoughts = append(thoughts, t)
	}

	gf := graphFile{
		Name:     g.Name,
		Purpose:  g.Purpose,
		Thoughts: thoughts,
		Edges:    g.edges,
		Signals:  g.signals,
	}
	if err := gob.NewEncoder(f).Encode(&gf); err != nil {
		return fmt.Errorf("graph save encode: %w", err)
	}
	return nil
}

// Load reads a graph from a .gshard file.
// Returns a new empty graph if the file does not exist.
func Load(name, purpose, path string) (*Graph, error) {
	f, err := os.Open(path)
	if os.IsNotExist(err) {
		return New(name, purpose, path), nil
	}
	if err != nil {
		return nil, fmt.Errorf("graph load: %w", err)
	}
	defer f.Close()

	var gf graphFile
	if err := gob.NewDecoder(f).Decode(&gf); err != nil {
		return nil, fmt.Errorf("graph load decode: %w", err)
	}

	g := New(gf.Name, gf.Purpose, path)
	for _, t := range gf.Thoughts {
		g.AddThought(t)
	}
	for _, e := range gf.Edges {
		g.AddEdge(e)
	}
	for _, s := range gf.Signals {
		g.AddSignal(s)
	}
	return g, nil
}

// --- Math helpers ---

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
	z := x
	for i := 0; i < 10; i++ {
		z -= (z*z - x) / (2 * z)
	}
	return z
}
