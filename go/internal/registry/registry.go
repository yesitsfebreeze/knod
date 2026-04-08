// Package registry maintains the tag index and specialist roster.
// It ingests gossip packets and answers routing queries.
package registry

import (
	"shard/internal/gossip"
	"sort"
	"sync"
)

// Entry is one specialist's record in the registry.
type Entry struct {
	AgentID    string
	GraphPath  string
	AgentScore float32
	Tags       map[string]float32 // tag → tag_score
}

// Registry is the parent's view of all known specialists.
type Registry struct {
	mu      sync.RWMutex
	agents  map[string]*Entry        // agent_id → entry
	tagIdx  map[string][]*tagRecord  // tag → sorted list of (agent_id, combined_score)
}

type tagRecord struct {
	agentID string
	score   float32 // tag_score × agent_score
}

func New() *Registry {
	return &Registry{
		agents: make(map[string]*Entry),
		tagIdx: make(map[string][]*tagRecord),
	}
}

// Register adds a shard by name. Path is always ~/.shards/<name>.shard.
func (r *Registry) Register(name string, dbPath string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.agents[name]; !ok {
		r.agents[name] = &Entry{
			AgentID:    name,
			GraphPath:  dbPath,
			AgentScore: 1.0,
			Tags:       make(map[string]float32),
		}
	}
}

// IngestPacket processes an incoming gossip broadcast.
func (r *Registry) IngestPacket(p gossip.Packet) {
	r.mu.Lock()
	defer r.mu.Unlock()

	entry, ok := r.agents[p.AgentID]
	if !ok {
		entry = &Entry{
			AgentID:    p.AgentID,
			AgentScore: p.AgentScore,
			Tags:       make(map[string]float32),
		}
		r.agents[p.AgentID] = entry
	}
	entry.Tags[p.Tag] = p.TagScore
	r.rebuildTagIndex(p.Tag)
}

// RecordFeedback adjusts an agent's score based on consumer rating (0–1).
func (r *Registry) RecordFeedback(agentID string, rating float32) {
	r.mu.Lock()
	defer r.mu.Unlock()
	entry, ok := r.agents[agentID]
	if !ok {
		return
	}
	// Exponential moving average
	entry.AgentScore = 0.9*entry.AgentScore + 0.1*rating
	// Rebuild index for all this agent's tags
	for tag := range entry.Tags {
		r.rebuildTagIndex(tag)
	}
}

// Resolve finds the best agent for a signal query by tag matching.
// Returns the top-n agent IDs ranked by combined score.
func (r *Registry) Resolve(tags []string, n int) []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	scores := make(map[string]float32)
	for _, tag := range tags {
		for _, rec := range r.tagIdx[tag] {
			scores[rec.agentID] += rec.score
		}
	}

	type kv struct {
		id    string
		score float32
	}
	ranked := make([]kv, 0, len(scores))
	for id, s := range scores {
		ranked = append(ranked, kv{id, s})
	}
	sort.Slice(ranked, func(i, j int) bool { return ranked[i].score > ranked[j].score })

	out := make([]string, 0, n)
	for i := 0; i < n && i < len(ranked); i++ {
		out = append(out, ranked[i].id)
	}
	return out
}

// GraphPath returns the graph file path for a known agent.
func (r *Registry) GraphPath(agentID string) (string, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	e, ok := r.agents[agentID]
	if !ok {
		return "", false
	}
	return e.GraphPath, true
}

// AllAgents returns all known agent IDs.
func (r *Registry) AllAgents() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	ids := make([]string, 0, len(r.agents))
	for id := range r.agents {
		ids = append(ids, id)
	}
	return ids
}

func (r *Registry) rebuildTagIndex(tag string) {
	var records []*tagRecord
	for _, entry := range r.agents {
		if ts, ok := entry.Tags[tag]; ok {
			records = append(records, &tagRecord{
				agentID: entry.AgentID,
				score:   ts * entry.AgentScore,
			})
		}
	}
	sort.Slice(records, func(i, j int) bool { return records[i].score > records[j].score })
	r.tagIdx[tag] = records
}
