package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"shard/internal/gnn"
	"shard/internal/gossip"
	"shard/internal/graph"
	"shard/internal/mcp"
	"shard/internal/registry"
	"shard/internal/tagcloud"
	httpserver "shard/internal/httpserver"
)

// cfg holds runtime configuration from environment variables.
type cfg struct {
	openAIKey     string
	model         string
	httpPort      int
	gossipPort    int
	gossipInterval time.Duration
	queryTimeout  time.Duration
	queryTopK     int
	gnnPath       string
}

func loadCfg() cfg {
	c := cfg{
		openAIKey:      os.Getenv("OPENAI_API_KEY"),
		model:          envOr("OPENAI_MODEL", "gpt-4o-mini"),
		httpPort:       envInt("SHARD_HTTP_PORT", 8080),
		gossipPort:     envInt("SHARD_GOSSIP_PORT", gossip.DefaultPort),
		gossipInterval: envDur("SHARD_GOSSIP_INTERVAL", 30*time.Second),
		queryTimeout:   envDur("SHARD_QUERY_TIMEOUT", 2*time.Second),
		queryTopK:      envInt("SHARD_QUERY_TOP_K", 5),
		gnnPath:        envOr("SHARD_GNN_PATH", filepath.Join(shardsDir(), "base.gnn")),
	}
	return c
}

// shardsDir is the one place all shard DBs live.
func shardsDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		panic("cannot resolve home directory: " + err.Error())
	}
	return filepath.Join(home, ".shards")
}

// shardPath returns the DB path for a named shard (.gshard Go-native format).
func shardPath(name string) string {
	return filepath.Join(shardsDir(), name+".gshard")
}

// agentIDFor computes the stable agent ID for a shard file path.
func agentIDFor(path string) string {
	abs, _ := filepath.Abs(path)
	return graph.ThoughtIP(abs, "agent") // reuse sha256-based helper
}

func main() {
	// Child mode: spawned by parent, owns exactly one shard DB.
	childPath := flag.String("child", "", "Run as child process for this shard path")
	flag.Parse()

	if *childPath != "" {
		// Import cycle avoided: child package logic inlined here.
		runChildMode(*childPath)
		return
	}

	args := flag.Args()
	if len(args) == 0 {
		fmt.Fprintln(os.Stderr, "commands: serve [--shard-file <path>], new <name>, list, ask <query>")
		os.Exit(1)
	}

	c := loadCfg()

	switch args[0] {
	case "serve":
		fs := flag.NewFlagSet("serve", flag.ExitOnError)
		shardFile := fs.String("shard-file", shardPath("default"), "Path to .gshard file")
		httpPort := fs.Int("http-port", c.httpPort, "HTTP port")
		fs.Parse(args[1:])
		c.httpPort = *httpPort
		runServe(*shardFile, c)

	case "new":
		if len(args) < 2 {
			fmt.Fprintln(os.Stderr, "usage: shard new <name>")
			os.Exit(1)
		}
		runNew(args[1], c)

	case "list":
		runList()

	case "ask":
		if len(args) < 2 {
			fmt.Fprintln(os.Stderr, "usage: shard ask <query>")
			os.Exit(1)
		}
		runAsk(strings.Join(args[1:], " "), c)

	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n", args[0])
		fmt.Fprintln(os.Stderr, "commands: serve, new <name>, list, ask <query>")
		os.Exit(1)
	}
}

// runServe starts a full agent: gossip + tag loop + HTTP + MCP.
func runServe(shardFile string, c cfg) {
	if err := os.MkdirAll(filepath.Dir(shardFile), 0o755); err != nil {
		fatalf("mkdir: %v", err)
	}

	agentID := agentIDFor(shardFile)

	// Load or create graph.
	g, err := graph.Load("default", "", shardFile)
	if err != nil {
		fatalf("load graph: %v", err)
	}

	// Load GNN weights (falls back to random init if file absent).
	model := gnn.LoadWeights(c.gnnPath)
	_ = model // used by httpserver for scoring

	// Registry.
	reg := registry.New()
	reg.Register(agentID, shardFile)

	// Agent score (shared mutable state).
	agentScore := float32(1.0)
	getScore := func() float32 { return agentScore }

	// Gossip listener.
	listener, err := gossip.NewListener(agentID, c.gossipPort)
	if err != nil {
		fatalf("gossip listen: %v", err)
	}
	listener.GetScore = getScore
	listener.OnRecv = func(p gossip.Packet, host string) {
		reg.IngestPacket(p, host)
	}
	listener.OnQuery = func(tag string) (float32, []gossip.Fragment) {
		return scoreAndSearch(g, model, tag, c.queryTopK)
	}

	// Gossip broadcaster.
	broadcaster, err := gossip.NewBroadcaster(agentID, c.httpPort, c.gossipPort)
	if err != nil {
		fatalf("gossip broadcast: %v", err)
	}

	stop := make(chan struct{})

	go listener.Start()
	go gossip.BroadcastLoop(broadcaster, c.gossipInterval, stop)

	// Tag computation loop.
	if c.openAIKey != "" {
		go tagLoop(g, broadcaster, c)
	} else {
		fmt.Fprintln(os.Stderr, "warn: OPENAI_API_KEY not set — tag computation disabled")
	}

	// HTTP server.
	srv := httpserver.New(httpserver.Deps{
		AgentID:     agentID,
		Graph:       g,
		Broadcaster: broadcaster,
		Registry:    reg,
		GetScore:    getScore,
		RecordRate: func(id string, rating float32) {
			if id == agentID {
				agentScore = 0.9*agentScore + 0.1*rating
			}
			reg.RecordFeedback(id, rating)
		},
		QueryTimeout: c.queryTimeout,
		QueryTopK:    c.queryTopK,
	})
	httpSrv := &http.Server{
		Addr:    ":" + strconv.Itoa(c.httpPort),
		Handler: srv,
	}
	go func() {
		fmt.Printf("agent %s listening on :%d (gossip :%d)\n", agentID, c.httpPort, c.gossipPort)
		if err := httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			fmt.Fprintf(os.Stderr, "http: %v\n", err)
		}
	}()

	// MCP server (stdio — only when stdin is not a terminal, i.e. piped).
	mcpSrv := mcp.New(mcp.Deps{
		AgentID:      agentID,
		Graph:        g,
		Broadcaster:  broadcaster,
		Registry:     reg,
		GetScore:     getScore,
		QueryTimeout: c.queryTimeout,
		QueryTopK:    c.queryTopK,
	})
	go mcpSrv.Serve()

	// Wait for SIGINT / SIGTERM.
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	<-sig

	close(stop)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	httpSrv.Shutdown(ctx)
	listener.Close()
	broadcaster.Close()

	if err := g.Save(shardFile); err != nil {
		fmt.Fprintf(os.Stderr, "save graph: %v\n", err)
	}
}

// tagLoop periodically clusters thought vectors and updates the broadcaster's tag cloud.
func tagLoop(g *graph.Graph, b *gossip.Broadcaster, c cfg) {
	tick := time.NewTicker(2 * time.Minute)
	defer tick.Stop()
	// Run once immediately on startup.
	computeAndSetTags(g, b, c)
	for range tick.C {
		computeAndSetTags(g, b, c)
	}
}

func computeAndSetTags(g *graph.Graph, b *gossip.Broadcaster, c cfg) {
	thoughts := g.AllThoughts()
	if len(thoughts) == 0 {
		return
	}
	tc := make([]tagcloud.Thought, len(thoughts))
	for i, t := range thoughts {
		tc[i] = tagcloud.Thought{Text: t.Text, Embedding: t.Embedding}
	}
	tags, err := tagcloud.Extract(tc, c.openAIKey, c.model)
	if err != nil {
		fmt.Fprintf(os.Stderr, "tagcloud: %v\n", err)
		return
	}
	pkts := make([]gossip.Packet, len(tags))
	for i, t := range tags {
		pkts[i] = gossip.Packet{Tag: t.Name, TagScore: t.Score}
	}
	b.UpdateTags(pkts)
}

// scoreAndSearch scores this agent against a query tag and returns top-k fragments.
func scoreAndSearch(g *graph.Graph, model *gnn.MPNN, tag string, topK int) (float32, []gossip.Fragment) {
	thoughts := g.AllThoughts()
	if len(thoughts) == 0 || model == nil {
		return 0, nil
	}

	// Embed the tag name as a pseudo-query by finding thoughts whose text contains
	// the tag words (cheap approximation without a live embedding call).
	tagLower := strings.ToLower(tag)
	var best float32
	var frags []gossip.Fragment

	for _, t := range thoughts {
		score := float32(0)
		tl := strings.ToLower(t.Text)
		// Simple word-overlap score.
		for _, word := range strings.Fields(tagLower) {
			if len(word) > 3 && strings.Contains(tl, word) {
				score += 1.0 / float32(len(strings.Fields(tagLower)))
			}
		}
		if score > 0 {
			if score > best {
				best = score
			}
			frags = append(frags, gossip.Fragment{
				IP:    t.IP,
				Text:  t.Text,
				Score: score * t.Score,
			})
		}
	}

	// Cap at topK.
	if len(frags) > topK {
		frags = frags[:topK]
	}
	return best, frags
}

// runNew creates a new shard file and spawns a child serve process for it.
func runNew(name string, c cfg) {
	path := shardPath(name)
	if err := os.MkdirAll(shardsDir(), 0o755); err != nil {
		fatalf("mkdir: %v", err)
	}
	g := graph.New(name, "", path)
	if err := g.Save(path); err != nil {
		fatalf("create shard: %v", err)
	}
	fmt.Printf("created shard '%s' → %s\n", name, path)
	if err := spawnShard(path); err != nil {
		fmt.Fprintf(os.Stderr, "warn: could not spawn serve process: %v\n", err)
	}
}

// spawnShard launches a detached child process serving the given shard file.
func spawnShard(shardFile string) error {
	exe, err := os.Executable()
	if err != nil {
		return err
	}
	cmd := exec.Command(exe, "serve", "--shard-file", shardFile)
	cmd.Stdout = nil
	cmd.Stderr = nil
	if err := cmd.Start(); err != nil {
		return err
	}
	fmt.Printf("spawned serve pid=%d for %s\n", cmd.Process.Pid, shardFile)
	return nil
}

func runList() {
	dir := shardsDir()
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			fmt.Println("no shards yet")
			return
		}
		fatalf("%v", err)
	}
	for _, e := range entries {
		if filepath.Ext(e.Name()) == ".gshard" {
			name := strings.TrimSuffix(e.Name(), ".gshard")
			fmt.Printf("  %s → %s\n", name, filepath.Join(dir, e.Name()))
		}
	}
}

// runAsk broadcasts a query tag, collects fragments, synthesizes an answer via LLM.
func runAsk(query string, c cfg) {
	if c.openAIKey == "" {
		fatalf("OPENAI_API_KEY not set")
	}

	agentID := fmt.Sprintf("ask-%d", time.Now().UnixNano())
	b, err := gossip.NewBroadcaster(agentID, 0, c.gossipPort)
	if err != nil {
		fatalf("gossip: %v", err)
	}
	defer b.Close()

	frags, err := b.QueryNetwork(query, c.queryTimeout, c.queryTopK*5)
	if err != nil {
		fatalf("query: %v", err)
	}

	if len(frags) == 0 {
		fmt.Println("no results from network")
		return
	}

	answer, err := synthesize(c.openAIKey, c.model, query, frags)
	if err != nil {
		fatalf("synthesize: %v", err)
	}
	fmt.Println(answer)
}

// synthesize calls the OpenAI chat API to answer query given scored fragments as context.
func synthesize(apiKey, model, query string, frags []gossip.ScoredFragment) (string, error) {
	var sb strings.Builder
	sb.WriteString("Answer the following question using only the provided context.\n\n")
	sb.WriteString("Context:\n")
	for i, f := range frags {
		if i >= 10 {
			break
		}
		fmt.Fprintf(&sb, "- [agent:%s score:%.2f] %s\n", f.AgentID, f.Composite, f.Text)
	}
	sb.WriteString("\nQuestion: ")
	sb.WriteString(query)

	body := map[string]any{
		"model": model,
		"messages": []map[string]string{
			{"role": "user", "content": sb.String()},
		},
		"max_tokens": 512,
	}
	data, _ := json.Marshal(body)

	req, err := http.NewRequest("POST", "https://api.openai.com/v1/chat/completions", bytes.NewReader(data))
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var result struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", err
	}
	if len(result.Choices) == 0 {
		return "", fmt.Errorf("no choices in response")
	}
	return result.Choices[0].Message.Content, nil
}

// runChildMode handles the --child=<path> subprocess mode.
func runChildMode(graphPath string) {
	g, err := graph.Load("", "", graphPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "child: load graph: %v\n", err)
		os.Exit(1)
	}

	var req struct {
		Query     string    `json:"query"`
		Embedding []float32 `json:"embedding"`
		TopK      int       `json:"top_k"`
	}
	if err := json.NewDecoder(os.Stdin).Decode(&req); err != nil {
		fmt.Fprintf(os.Stderr, "child: bad request: %v\n", err)
		os.Exit(1)
	}

	thoughts := g.TopK(req.Embedding, req.TopK)
	type result struct {
		IP    string  `json:"ip"`
		Text  string  `json:"text"`
		Score float32 `json:"score"`
	}
	results := make([]result, len(thoughts))
	for i, t := range thoughts {
		results[i] = result{IP: t.IP, Text: t.Text, Score: t.Score}
	}
	json.NewEncoder(os.Stdout).Encode(map[string]any{
		"agent_id": graphPath,
		"results":  results,
	})
}

// --- env helpers ---

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func envInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func envDur(key string, def time.Duration) time.Duration {
	if v := os.Getenv(key); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			return d
		}
	}
	return def
}

func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "shard: "+format+"\n", args...)
	os.Exit(1)
}
