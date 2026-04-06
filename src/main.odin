package knod

import "core:mem"
import "core:os"
import os2 "core:os/os2"
import "core:path/filepath"
import "core:strings"
import "core:time"

import "cli"
import cfg "config"
import "gnn"
import "graph"
import "ingest"
import log "logger"
import "protocol"
import "provider"
import "registry"
import "repl"
import "util"

main :: proc() {
	cmd := cli.parse_and_dispatch()
	if cmd.action == .EXIT {
		return
	}

	if cmd.action == .ASK {
		handle_ask_subprocess(cmd)
		return
	}

	log.init()
	defer log.shutdown()

	c, config_ok := cfg.load()
	if !config_ok {
		path := cfg.write_default()
		if len(path) > 0 {
			log.info("created default config at: %s", path)
			log.info("edit the config file to set your api_key, then run again.")
			delete(path)
		} else {
			log.err("could not determine home directory for config file")
		}
		os.exit(1)
	}
	defer cfg.release(&c)

	if len(c.api_key) == 0 {
		config_path := cfg.config_path()
		log.err("api_key not set in %s", config_path)
		delete(config_path)
		os.exit(1)
	}

	provider.configure(
		provider.Config {
			api_key = c.api_key,
			base_url = c.base_url,
			embedding_model = c.embedding_model,
			chat_model = c.chat_model,
			timeout_ms = c.timeout_ms,
		},
	)
	p := provider.openai_create()
	defer provider.openai_destroy(&p)

	graph_path := c.graph_path
	if !filepath.is_abs(graph_path) {
		graph_path = data_path(c.graph_path)
	}
	log.info("graph path: %s", graph_path)

	// Ensure the parent directory exists for graph persistence.
	{
		dir := filepath.dir(graph_path)
		ensure_data_dir(dir)
		delete(dir)
	}

	graph.configure(
		graph.Config {
			data_path = graph_path,
			max_thoughts = c.max_thoughts,
			max_edges = c.max_edges,
			edge_decay = c.edge_decay,
			similarity_threshold = c.similarity_threshold,
			default_find_k = c.find_k,
			max_context_edges = c.max_context_edges,
		},
	)

	g: graph.Graph
	graph.create(&g)
	defer graph.release(&g)

	is_strand_file := strings.has_suffix(graph_path, util.STRAND_EXTENSION)

	if is_strand_file {
		// .strand container format: load graph data from the SECTION_GRAPH block.
		_, load_ok := graph.load_strand(&g, graph_path)
		if load_ok && graph.thought_count(&g) > 0 {
			log.info(
				"graph loaded: %d thoughts, %d edges",
				graph.thought_count(&g),
				graph.edge_count(&g),
			)
		} else {
			log.info("starting with empty graph")
		}
	} else if graph.load(&g, graph_path) {
		log.info(
			"graph loaded: %d thoughts, %d edges",
			graph.thought_count(&g),
			graph.edge_count(&g),
		)
	} else {
		log.info("starting with empty graph")
	}

	if !graph.stream_open(&g, graph_path) {
		log.warn("could not open graph stream for persistence")
	}

	// Resolve base MPNN checkpoint path
	base_checkpoint_path: string
	if len(c.base_path) > 0 {
		base_checkpoint_path = strings.clone(c.base_path)
	} else {
		base_checkpoint_path = cli.base_model_path()
	}
	defer delete(base_checkpoint_path)

	gnn_model: gnn.MPNN
	gnn.create(&gnn_model)
	defer gnn.release(&gnn_model)

	if gnn.load_checkpoint(&gnn_model, base_checkpoint_path) {
		log.info("base checkpoint loaded from %s", base_checkpoint_path)
	} else {
		log.info("starting with fresh base GNN model")
	}

	// Load strand from .strand file
	strand_model: gnn.StrandMPNN
	strand_loaded := false
	{
		file_data, strand_off, load_ok := graph.load_strand_bytes(graph_path)
		if load_ok && strand_off > 0 {
			off := strand_off
			if gnn.strand_load(&strand_model, file_data, &off) {
				strand_loaded = true
				log.info("strand loaded from %s", graph_path)
			}
		}
		if file_data != nil {delete(file_data)}
	}
	if !strand_loaded {
		gnn.strand_create(&strand_model, gnn_model.hidden_dim)
		log.info("starting with fresh strand model")
		if graph.thought_count(&g) >= 2 {
			log.info("training strand on existing graph...")
			n := graph.thought_count(&g)
			strand_steps := gnn.adaptive_steps(n, gnn.STRAND_TRAIN_STEPS_MAX, gnn.STRAND_TRAIN_STEPS_MIN)
			base_steps := gnn.adaptive_steps(n, gnn.BASE_REFINE_STEPS_MAX, gnn.BASE_REFINE_STEPS_MIN)
			gnn.train_strand(&gnn_model, &strand_model, &g, strand_steps)
			gnn.train_base_refine(&gnn_model, &strand_model, &g, base_steps)
			gnn.save_checkpoint(&gnn_model, base_checkpoint_path)
			strand_bytes := gnn.strand_save_bytes(&strand_model)
			if strand_bytes != nil {
				graph.save_strand(&g, strand_bytes, graph_path)
				delete(strand_bytes)
			}
		}
	}
	defer gnn.strand_release(&strand_model)

	limbo_g: graph.Graph
	limbo_path := registry.limbo_path()
	defer delete(limbo_path)
	limbo_enabled := c.limbo_cluster_min > 0 && !cmd.no_limbo

	if limbo_enabled {
		// Ensure the limbo data directory exists.
		{
			limbo_dir := filepath.dir(limbo_path)
			ensure_data_dir(limbo_dir)
			delete(limbo_dir)
		}

		graph.create(&limbo_g)
		defer graph.release(&limbo_g)

		if graph.load(&limbo_g, limbo_path) {
			log.info("limbo: loaded %d thoughts", graph.thought_count(&limbo_g))
		}
		if !graph.stream_open(&limbo_g, limbo_path) {
			log.warn("limbo: could not open stream, dropped thoughts will not be preserved")
			limbo_enabled = false
		}
	}

	handler := protocol.Handler {
		g = &g,
		p = &p,
		model = &gnn_model,
		strand = &strand_model,
		graph_path = graph_path,
		base_checkpoint_path = base_checkpoint_path,
		limbo = limbo_enabled ? &limbo_g : nil,
		ingest_cfg = ingest.Config {
			max_similar = c.max_similar,
			edge_threshold = c.edge_threshold,
			maturity_threshold = c.maturity_threshold,
			max_tags = c.max_tags,
			min_link_weight = c.min_link_weight,
			limbo_graph = limbo_enabled ? &limbo_g : nil,
			limbo_threshold = util.LIMBO_THRESHOLD,
		},
	}

	if !protocol.handler_start_queue(&handler) {
		log.warn("ingest queue could not start, ingestion will be synchronous")
	}
	defer protocol.handler_stop_queue(&handler)

	tcp, tcp_ok := protocol.tcp_create(c.tcp_port, &handler)
	if !tcp_ok {
		os.exit(1)
	}
	defer protocol.tcp_destroy(&tcp)

	http_ok := false
	http_proto: protocol.HTTP
	if !cmd.no_http {
		http_proto, http_ok = protocol.http_create(c.http_port, &handler)
		if !http_ok {
			log.warn("http: could not start, continuing without HTTP")
		}
	} else {
		log.info("http: disabled (--no-http)")
	}
	defer if http_ok {protocol.http_destroy(&http_proto)}

	config_path := cfg.config_path()
	defer delete(config_path)
	log.info("config: %s", config_path)
	log.info("provider: OpenAI (embed: %s, chat: %s)", c.embedding_model, c.chat_model)
	if len(g.purpose) > 0 {
		log.info("purpose: \"%s\"", g.purpose)
	} else {
		log.info("no purpose set")
	}
	log.info("graph: %d thoughts, %d edges", graph.thought_count(&g), graph.edge_count(&g))
	log.info("gnn base: %d params, step %d", gnn_model.num_parameters, gnn_model.adam_t)
	log.info("gnn strand: %d params, step %d", strand_model.num_parameters, strand_model.adam_t)
	if limbo_enabled {
		log.info(
			"limbo: enabled (%d thoughts, cluster_min=%d)",
			graph.thought_count(&limbo_g),
			c.limbo_cluster_min,
		)
	} else {
		log.info("limbo: disabled")
	}

	repl_state := repl.init(&g, &p, graph_path, &gnn_model, &strand_model, limbo_enabled ? &limbo_g : nil)
	defer repl.destroy(&repl_state)

	LIMBO_SCAN_INTERVAL :: 60 * time.Second
	last_limbo_scan := time.now()

	for {
		if !repl.poll(&repl_state) {
			break
		}

		protocol.tcp_poll(&tcp)

		if limbo_enabled && time.since(last_limbo_scan) >= LIMBO_SCAN_INTERVAL {
			ingest.scan_limbo(&limbo_g, limbo_path, &p, c.limbo_cluster_min, util.LIMBO_THRESHOLD)
			last_limbo_scan = time.now()
		}

		time.sleep(1 * time.Millisecond)
	}
}

data_path :: proc(filename: string) -> string {
	dir := util.exe_dir()
	defer delete(dir)
	return filepath.join({dir, "data", filename})
}

ensure_data_dir :: proc(path: string) {
	if os.exists(path) {
		return
	}
	parent := filepath.dir(path)
	if len(parent) > 0 && parent != path {
		ensure_data_dir(parent)
		delete(parent)
	}
	os.make_directory(path)
}

handle_ask_subprocess :: proc(cmd: cli.Command) {
	defer delete(cmd.query)

	c, config_ok := cfg.load()
	if !config_ok {
		log.err("config not found. run 'knod' first to create default config.")
		return
	}
	defer cfg.release(&c)

	if len(c.api_key) == 0 {
		config_path := cfg.config_path()
		log.err("api_key not set in %s", config_path)
		delete(config_path)
		return
	}

	provider.configure(
		provider.Config {
			api_key = c.api_key,
			base_url = c.base_url,
			embedding_model = c.embedding_model,
			chat_model = c.chat_model,
			timeout_ms = c.timeout_ms,
		},
	)
	p := provider.openai_create()
	defer provider.openai_destroy(&p)

	query_embedding, embed_ok := p.embed_text(&p, cmd.query)
	if !embed_ok {
		log.err("failed to embed query")
		return
	}

	stores := cli.get_store_paths(cmd.knid_scope)
	defer cli.release_store_list(stores)

	if len(stores) == 0 {
		log.info("no stores registered. use 'knod new' or 'knod register' first.")
		return
	}

	embedding_bytes := mem.slice_ptr(cast(^u8)&query_embedding, graph.EMBEDDING_DIM * size_of(f32))

	exe_path := os.args[0]

	SubprocessState :: struct {
		process:    os2.Process,
		stdin_w:    ^os2.File,
		stdout_r:   ^os2.File,
		store_name: string,
		valid:      bool,
	}

	procs := make([]SubprocessState, len(stores))
	defer delete(procs)

	for i in 0 ..< len(stores) {
		graph_arg := strings.concatenate({"--graph=", stores[i].path})
		defer delete(graph_arg)
		query_arg := strings.concatenate({"--query=", cmd.query})
		defer delete(query_arg)

		stdin_r, stdin_w, pipe_err1 := os2.pipe()
		if pipe_err1 != nil {
			log.err("could not create stdin pipe for store '%s'", stores[i].name)
			continue
		}

		stdout_r, stdout_w, pipe_err2 := os2.pipe()
		if pipe_err2 != nil {
			log.err("could not create stdout pipe for store '%s'", stores[i].name)
			os2.close(stdin_r)
			os2.close(stdin_w)
			continue
		}

		desc := os2.Process_Desc {
			command = {exe_path, "--internal-query", graph_arg, query_arg},
			stdin   = stdin_r,
			stdout  = stdout_w,
		}

		process, start_err := os2.process_start(desc)
		if start_err != nil {
			log.err("could not start subprocess for store '%s'", stores[i].name)
			os2.close(stdin_r)
			os2.close(stdin_w)
			os2.close(stdout_r)
			os2.close(stdout_w)
			continue
		}

		os2.close(stdin_r)
		os2.close(stdout_w)

		os2.write(stdin_w, embedding_bytes)
		os2.close(stdin_w)

		procs[i] = SubprocessState {
			process    = process,
			stdin_w    = nil,
			stdout_r   = stdout_r,
			store_name = stores[i].name,
			valid      = true,
		}
	}

	ThoughtResult :: struct {
		score: f32,
		id:    u64,
		text:  string,
	}

	EdgeResult :: struct {
		score:     f32,
		reasoning: string,
	}

	all_thoughts: [dynamic]ThoughtResult
	all_edges: [dynamic]EdgeResult
	defer {
		for &t in all_thoughts {delete(t.text)}
		delete(all_thoughts)
		for &e in all_edges {delete(e.reasoning)}
		delete(all_edges)
	}

	for i in 0 ..< len(procs) {
		if !procs[i].valid {continue}

		stdout_data: [dynamic]u8
		defer delete(stdout_data)

		buf: [4096]u8
		for {
			n, read_err := os2.read(procs[i].stdout_r, buf[:])
			if n > 0 {
				append(&stdout_data, ..buf[:n])
			}
			if read_err != nil || n <= 0 {
				break
			}
		}
		os2.close(procs[i].stdout_r)

		_, _ = os2.process_wait(procs[i].process)
		_ = os2.process_close(procs[i].process)

		output := string(stdout_data[:])
		lines := strings.split(output, "\n")
		defer delete(lines)

		for line in lines {
			if len(line) == 0 {continue}
			trimmed := strings.trim_right(line, "\r")

			if len(trimmed) >= 2 && trimmed[0] == 'T' && trimmed[1] == '\t' {

				parts := strings.split_n(trimmed[2:], "\t", 3)
				defer delete(parts)
				if len(parts) >= 3 {
					score := parse_f32(parts[0])
					id := parse_u64(parts[1])
					text := cli.unescape_line(parts[2])
					append(&all_thoughts, ThoughtResult{score = score, id = id, text = text})
				}
			} else if len(trimmed) >= 2 && trimmed[0] == 'E' && trimmed[1] == '\t' {

				parts := strings.split_n(trimmed[2:], "\t", 2)
				defer delete(parts)
				if len(parts) >= 2 {
					score := parse_f32(parts[0])
					reasoning := cli.unescape_line(parts[1])
					append(&all_edges, EdgeResult{score = score, reasoning = reasoning})
				}
			}
		}
	}

	if len(all_thoughts) == 0 && len(all_edges) == 0 {
		log.rawf("I don't have enough knowledge to answer that.\n")
		return
	}

	seen_text: map[string]int
	defer delete(seen_text)

	deduped: [dynamic]ThoughtResult
	defer {
		for &t in deduped {delete(t.text)}
		delete(deduped)
	}

	for &t in all_thoughts {
		if idx, found := seen_text[t.text]; found {
			if t.score > deduped[idx].score {
				deduped[idx].score = t.score
			}
			delete(t.text)
			t.text = ""
		} else {
			seen_text[t.text] = len(deduped)
			append(
				&deduped,
				ThoughtResult{score = t.score, id = t.id, text = strings.clone(t.text)},
			)
			delete(t.text)
			t.text = ""
		}
	}

	for i in 1 ..< len(deduped) {
		j := i
		for j > 0 && deduped[j].score > deduped[j - 1].score {
			deduped[j], deduped[j - 1] = deduped[j - 1], deduped[j]
			j -= 1
		}
	}

	for i in 1 ..< len(all_edges) {
		j := i
		for j > 0 && all_edges[j].score > all_edges[j - 1].score {
			all_edges[j], all_edges[j - 1] = all_edges[j - 1], all_edges[j]
			j -= 1
		}
	}

	context_parts: [dynamic]string
	defer delete(context_parts)

	n := min(c.find_k, len(deduped))
	for i in 0 ..< n {
		append(&context_parts, deduped[i].text)
	}

	edge_n := min(c.max_context_edges, len(all_edges))
	for i in 0 ..< edge_n {
		append(&context_parts, all_edges[i].reasoning)
	}

	context_text := strings.join(context_parts[:], "\n")
	defer delete(context_text)

	answer, answer_ok := p.generate_answer(&p, cmd.query, context_text)
	if !answer_ok {
		log.err("failed to generate answer")
		return
	}
	defer delete(answer)

	log.rawf("%s\n", answer)
}

parse_f32 :: proc(s: string) -> f32 {
	if len(s) == 0 {return 0}

	neg := false
	i := 0
	if s[0] == '-' {
		neg = true
		i = 1
	} else if s[0] == '+' {
		i = 1
	}

	whole: f64 = 0
	for i < len(s) && s[i] >= '0' && s[i] <= '9' {
		whole = whole * 10 + f64(s[i] - '0')
		i += 1
	}

	frac: f64 = 0
	if i < len(s) && s[i] == '.' {
		i += 1
		scale: f64 = 0.1
		for i < len(s) && s[i] >= '0' && s[i] <= '9' {
			frac += f64(s[i] - '0') * scale
			scale *= 0.1
			i += 1
		}
	}

	result := whole + frac
	if neg {result = -result}
	return f32(result)
}

parse_u64 :: proc(s: string) -> u64 {
	result: u64 = 0
	for c in s {
		if c >= '0' && c <= '9' {
			result = result * 10 + u64(c - '0')
		} else {
			break
		}
	}
	return result
}
