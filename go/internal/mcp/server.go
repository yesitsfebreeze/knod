// Package mcp implements a minimal MCP stdio server (JSON-RPC 2.0 over stdin/stdout).
//
// Framing: LSP-style Content-Length headers.
//
//	Content-Length: 123\r\n
//	\r\n
//	{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{...}}
//
// Tools exposed:
//
//	query_network(tag)             — broadcast query, return scored fragments
//	list_peers()                   — live peers with tags and scores
//	get_tags()                     — this agent's tag cloud
//	rate_agent(agent_id, rating)   — submit EMA rating
//	ask(query)                     — local graph cosine search
package mcp

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"time"

	"shard/internal/gossip"
	"shard/internal/graph"
	"shard/internal/registry"
)

// Deps holds all dependencies the MCP server needs.
type Deps struct {
	AgentID      string
	Graph        *graph.Graph
	Broadcaster  *gossip.Broadcaster
	Registry     *registry.Registry
	GetScore     func() float32
	QueryTimeout time.Duration
	QueryTopK    int
}

// Server is the MCP stdio server.
type Server struct {
	d      Deps
	in     *bufio.Reader
	out    io.Writer
}

func New(d Deps) *Server {
	return &Server{d: d, in: bufio.NewReader(os.Stdin), out: os.Stdout}
}

// Serve reads JSON-RPC requests from stdin and writes responses to stdout.
// Runs until stdin is closed.
func (s *Server) Serve() {
	for {
		msg, err := s.readMessage()
		if err != nil {
			if err != io.EOF {
				fmt.Fprintf(os.Stderr, "mcp read: %v\n", err)
			}
			return
		}
		resp := s.dispatch(msg)
		if resp != nil {
			s.writeMessage(resp)
		}
	}
}

// rpcMessage is a generic JSON-RPC 2.0 message.
type rpcMessage struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      any             `json:"id,omitempty"`
	Method  string          `json:"method,omitempty"`
	Params  json.RawMessage `json:"params,omitempty"`
	Result  any             `json:"result,omitempty"`
	Error   *rpcError       `json:"error,omitempty"`
}

type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

func (s *Server) readMessage() (*rpcMessage, error) {
	// Read headers until blank line.
	contentLen := 0
	for {
		line, err := s.in.ReadString('\n')
		if err != nil {
			return nil, err
		}
		line = strings.TrimRight(line, "\r\n")
		if line == "" {
			break
		}
		if strings.HasPrefix(line, "Content-Length:") {
			v := strings.TrimSpace(strings.TrimPrefix(line, "Content-Length:"))
			contentLen, _ = strconv.Atoi(v)
		}
	}
	if contentLen == 0 {
		return nil, fmt.Errorf("no Content-Length")
	}
	body := make([]byte, contentLen)
	if _, err := io.ReadFull(s.in, body); err != nil {
		return nil, err
	}
	var msg rpcMessage
	if err := json.Unmarshal(body, &msg); err != nil {
		return nil, err
	}
	return &msg, nil
}

func (s *Server) writeMessage(msg *rpcMessage) {
	msg.JSONRPC = "2.0"
	data, err := json.Marshal(msg)
	if err != nil {
		return
	}
	fmt.Fprintf(s.out, "Content-Length: %d\r\n\r\n", len(data))
	s.out.Write(data)
}

func (s *Server) ok(id any, result any) *rpcMessage {
	return &rpcMessage{ID: id, Result: result}
}

func (s *Server) errMsg(id any, code int, msg string) *rpcMessage {
	return &rpcMessage{ID: id, Error: &rpcError{Code: code, Message: msg}}
}

func (s *Server) dispatch(msg *rpcMessage) *rpcMessage {
	switch msg.Method {
	case "initialize":
		return s.ok(msg.ID, map[string]any{
			"protocolVersion": "2024-11-05",
			"capabilities":    map[string]any{"tools": map[string]any{}},
			"serverInfo":      map[string]string{"name": "shard", "version": "0.1.0"},
		})

	case "initialized":
		return nil // notification, no response

	case "tools/list":
		return s.ok(msg.ID, map[string]any{"tools": toolDefinitions()})

	case "tools/call":
		return s.handleToolCall(msg)

	default:
		return s.errMsg(msg.ID, -32601, "method not found: "+msg.Method)
	}
}

func (s *Server) handleToolCall(msg *rpcMessage) *rpcMessage {
	var p struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}
	if err := json.Unmarshal(msg.Params, &p); err != nil {
		return s.errMsg(msg.ID, -32600, "invalid params")
	}

	var result string
	var callErr error

	switch p.Name {
	case "query_network":
		var args struct {
			Tag string `json:"tag"`
		}
		json.Unmarshal(p.Arguments, &args)
		frags, err := s.d.Broadcaster.QueryNetwork(args.Tag, s.d.QueryTimeout, s.d.QueryTopK*5)
		if err != nil {
			callErr = err
			break
		}
		data, _ := json.MarshalIndent(frags, "", "  ")
		result = string(data)

	case "list_peers":
		peers := s.d.Registry.LivePeers(s.d.AgentID)
		data, _ := json.MarshalIndent(peers, "", "  ")
		result = string(data)

	case "get_tags":
		pkts := s.d.Broadcaster.CurrentTags()
		type tagView struct {
			Tag   string  `json:"tag"`
			Score float32 `json:"score"`
		}
		out := make([]tagView, len(pkts))
		for i, p := range pkts {
			out[i] = tagView{Tag: p.Tag, Score: p.TagScore}
		}
		data, _ := json.MarshalIndent(out, "", "  ")
		result = string(data)

	case "rate_agent":
		var args struct {
			AgentID string  `json:"agent_id"`
			Rating  float32 `json:"rating"`
		}
		json.Unmarshal(p.Arguments, &args)
		s.d.Registry.RecordFeedback(args.AgentID, args.Rating)
		result = `{"status":"ok"}`

	case "ask":
		var args struct {
			Query string `json:"query"`
		}
		json.Unmarshal(p.Arguments, &args)
		// Local cosine search — no embedding (word overlap approximation).
		thoughts := s.d.Graph.AllThoughts()
		qLower := strings.ToLower(args.Query)
		type hit struct {
			IP    string  `json:"ip"`
			Text  string  `json:"text"`
			Score float32 `json:"score"`
		}
		var hits []hit
		for _, t := range thoughts {
			score := float32(0)
			tl := strings.ToLower(t.Text)
			for _, w := range strings.Fields(qLower) {
				if len(w) > 3 && strings.Contains(tl, w) {
					score += 1.0
				}
			}
			if score > 0 {
				hits = append(hits, hit{IP: t.IP, Text: t.Text, Score: score})
			}
		}
		// Sort descending.
		for i := 0; i < len(hits); i++ {
			for j := i + 1; j < len(hits); j++ {
				if hits[j].Score > hits[i].Score {
					hits[i], hits[j] = hits[j], hits[i]
				}
			}
		}
		if len(hits) > s.d.QueryTopK {
			hits = hits[:s.d.QueryTopK]
		}
		data, _ := json.MarshalIndent(hits, "", "  ")
		result = string(data)

	default:
		return s.errMsg(msg.ID, -32601, "unknown tool: "+p.Name)
	}

	if callErr != nil {
		return s.errMsg(msg.ID, -32000, callErr.Error())
	}
	return s.ok(msg.ID, map[string]any{
		"content": []map[string]string{
			{"type": "text", "text": result},
		},
	})
}

func toolDefinitions() []map[string]any {
	str := func(desc string) map[string]any {
		return map[string]any{"type": "string", "description": desc}
	}
	num := func(desc string) map[string]any {
		return map[string]any{"type": "number", "description": desc}
	}
	tool := func(name, desc string, props map[string]any, required []string) map[string]any {
		return map[string]any{
			"name":        name,
			"description": desc,
			"inputSchema": map[string]any{
				"type":       "object",
				"properties": props,
				"required":   required,
			},
		}
	}
	return []map[string]any{
		tool("query_network",
			"Broadcast a tag query to the gossip network and collect scored thought fragments from all matching specialist agents.",
			map[string]any{"tag": str("The topic tag to query (e.g. 'machine learning', 'graph neural networks')")},
			[]string{"tag"}),

		tool("list_peers",
			"Return all live specialist agents known via gossip, with their tag clouds and scores.",
			map[string]any{}, nil),

		tool("get_tags",
			"Return this agent's current tag cloud — the topics this specialist knows best.",
			map[string]any{}, nil),

		tool("rate_agent",
			"Submit a quality rating (0.0–1.0) for a specialist agent. Updates its agent_score via EMA.",
			map[string]any{
				"agent_id": str("The agent ID to rate"),
				"rating":   num("Quality rating 0.0 (poor) to 1.0 (excellent)"),
			},
			[]string{"agent_id", "rating"}),

		tool("ask",
			"Search this agent's local knowledge graph for thoughts matching a query.",
			map[string]any{"query": str("The question or topic to search for")},
			[]string{"query"}),
	}
}
