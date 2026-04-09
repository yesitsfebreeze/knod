// Package child runs the binary in single-specialist mode.
// It is spawned by the parent as: shard --child=<graph-path>
// It loads one graph, answers one query (from stdin), writes results to stdout, exits.
package child

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"shard/internal/graph"
)

// Request is sent to the child via stdin.
type Request struct {
	Query     string    `json:"query"`
	Embedding []float32 `json:"embedding"`
	TopK      int       `json:"top_k"`
}

// Response is written by the child to stdout.
type Response struct {
	AgentID string   `json:"agent_id"`
	Results []Result `json:"results"`
}

// Result is one matching thought.
type Result struct {
	ThoughtID uint64  `json:"thought_id"`
	IP        string  `json:"ip"`
	Text      string  `json:"text"`
	Score     float32 `json:"score"`
}

// Run is the child's main loop: read request → query graph → write response → exit.
func Run(graphPath string) {
	g, err := graph.Load("", "", graphPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "child: load graph: %v\n", err)
		os.Exit(1)
	}

	var req Request
	if err := json.NewDecoder(os.Stdin).Decode(&req); err != nil {
		fmt.Fprintf(os.Stderr, "child: bad request: %v\n", err)
		os.Exit(1)
	}

	thoughts := g.TopK(req.Embedding, req.TopK)
	results := make([]Result, len(thoughts))
	for i, t := range thoughts {
		results[i] = Result{
			ThoughtID: t.ID,
			IP:        t.IP,
			Text:      t.Text,
			Score:     t.Score,
		}
	}

	resp := Response{AgentID: graphPath, Results: results}
	json.NewEncoder(os.Stdout).Encode(resp)
}

// Spawn launches the binary as a child for the given shard path.
func Spawn(selfPath, shardPath string, req Request) (*Response, error) {
	data, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	cmd := exec.Command(selfPath, "--child="+shardPath)
	cmd.Stdin = bytes.NewReader(data)

	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("child %s: %w", shardPath, err)
	}

	var resp Response
	if err := json.Unmarshal(out, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// SpawnAll launches one child per shard path in parallel and merges results.
func SpawnAll(selfPath string, shardPaths []string, req Request) ([]Result, error) {
	type outcome struct {
		resp *Response
		err  error
	}

	ch := make(chan outcome, len(shardPaths))
	for _, p := range shardPaths {
		go func(sp string) {
			resp, err := Spawn(selfPath, sp, req)
			ch <- outcome{resp, err}
		}(p)
	}

	var all []Result
	for range shardPaths {
		o := <-ch
		if o.err != nil {
			fmt.Fprintf(os.Stderr, "child error: %v\n", o.err)
			continue
		}
		all = append(all, o.resp.Results...)
	}
	return all, nil
}
