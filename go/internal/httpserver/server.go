// Package httpserver exposes the agent's capabilities over HTTP.
//
// Endpoints:
//
//	POST /query   — broadcast tag query, collect scored fragments from network
//	GET  /peers   — live peers known via gossip
//	GET  /tags    — this agent's current tag cloud
//	POST /rate    — submit feedback rating for an agent
//	GET  /health  — this agent's basic stats
package httpserver

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"shard/internal/gossip"
	"shard/internal/graph"
	"shard/internal/registry"
)

// Deps holds all dependencies the HTTP server needs.
type Deps struct {
	AgentID     string
	Graph       *graph.Graph
	Broadcaster *gossip.Broadcaster
	Registry    *registry.Registry
	GetScore    func() float32
	RecordRate  func(agentID string, rating float32)
	QueryTimeout time.Duration
	QueryTopK   int
}

// Server is the HTTP handler.
type Server struct {
	d   Deps
	mux *http.ServeMux
}

func New(d Deps) *Server {
	s := &Server{d: d, mux: http.NewServeMux()}
	s.mux.HandleFunc("/query", s.handleQuery)
	s.mux.HandleFunc("/peers", s.handlePeers)
	s.mux.HandleFunc("/tags", s.handleTags)
	s.mux.HandleFunc("/rate", s.handleRate)
	s.mux.HandleFunc("/health", s.handleHealth)
	return s
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.mux.ServeHTTP(w, r)
}

// POST /query
// Body: {"tag": "machine-learning", "timeout": 2.0}
// Response: {"fragments": [{ip, text, score, agent_id, tag_score, agent_score, composite}]}
func (s *Server) handleQuery(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	var body struct {
		Tag     string  `json:"tag"`
		Timeout float64 `json:"timeout"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		http.Error(w, "bad JSON", http.StatusBadRequest)
		return
	}
	timeout := s.d.QueryTimeout
	if body.Timeout > 0 {
		timeout = time.Duration(body.Timeout * float64(time.Second))
	}
	frags, err := s.d.Broadcaster.QueryNetwork(body.Tag, timeout, s.d.QueryTopK*5)
	if err != nil {
		http.Error(w, fmt.Sprintf("query error: %v", err), http.StatusInternalServerError)
		return
	}
	writeJSON(w, map[string]any{"fragments": frags})
}

// GET /peers
// Response: [{agent_id, host, http_port, pid, tags, agent_score, last_seen}]
func (s *Server) handlePeers(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "GET only", http.StatusMethodNotAllowed)
		return
	}
	peers := s.d.Registry.LivePeers(s.d.AgentID)
	type peerView struct {
		AgentID    string             `json:"agent_id"`
		Host       string             `json:"host"`
		HTTPPort   int                `json:"http_port"`
		PID        int                `json:"pid"`
		Tags       map[string]float32 `json:"tags"`
		AgentScore float32            `json:"agent_score"`
		LastSeen   string             `json:"last_seen"`
	}
	out := make([]peerView, len(peers))
	for i, p := range peers {
		out[i] = peerView{
			AgentID:    p.AgentID,
			Host:       p.Host,
			HTTPPort:   p.HTTPPort,
			PID:        p.PID,
			Tags:       p.Tags,
			AgentScore: p.AgentScore,
			LastSeen:   p.LastSeen.Format(time.RFC3339),
		}
	}
	writeJSON(w, out)
}

// GET /tags
// Response: [{tag, score}]
func (s *Server) handleTags(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "GET only", http.StatusMethodNotAllowed)
		return
	}
	pkts := s.d.Broadcaster.CurrentTags()
	type tagView struct {
		Tag   string  `json:"tag"`
		Score float32 `json:"score"`
	}
	out := make([]tagView, len(pkts))
	for i, p := range pkts {
		out[i] = tagView{Tag: p.Tag, Score: p.TagScore}
	}
	writeJSON(w, out)
}

// POST /rate
// Body: {"agent_id": "abc8f3d2", "rating": 0.9}
func (s *Server) handleRate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	var body struct {
		AgentID string  `json:"agent_id"`
		Rating  float32 `json:"rating"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		http.Error(w, "bad JSON", http.StatusBadRequest)
		return
	}
	if body.Rating < 0 || body.Rating > 1 {
		http.Error(w, "rating must be 0.0–1.0", http.StatusBadRequest)
		return
	}
	s.d.RecordRate(body.AgentID, body.Rating)
	writeJSON(w, map[string]string{"status": "ok"})
}

// GET /health
// Response: {agent_id, n_thoughts, n_edges, agent_score, tags}
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "GET only", http.StatusMethodNotAllowed)
		return
	}
	pkts := s.d.Broadcaster.CurrentTags()
	tags := make([]string, len(pkts))
	for i, p := range pkts {
		tags[i] = p.Tag
	}
	writeJSON(w, map[string]any{
		"agent_id":   s.d.AgentID,
		"n_thoughts": s.d.Graph.ThoughtCount(),
		"n_edges":    s.d.Graph.EdgeCount(),
		"agent_score": s.d.GetScore(),
		"tags":       tags,
	})
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(v)
}
