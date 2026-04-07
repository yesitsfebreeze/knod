package main

import "core:fmt"
import "core:os"
import "core:path/filepath"
import "core:strconv"
import "core:strings"
import "core:thread"
import "core:time"

import cfg_pkg    "../config"
import gnn_pkg    "../gnn"
import graph_pkg  "../graph"
import ingest_pkg "../ingest"
import log        "../logger"
import prov       "../provider"
import proto      "../protocols"
import query_pkg  "../query"
import reg_pkg    "../registry"
import util       "../util"


main :: proc() {
	args := os.args
	if len(args) < 2 {
		print_usage()
		os.exit(0)
	}

	cfg, cfg_ok := cfg_pkg.load()
	if !cfg_ok {
		cfg = cfg_pkg.DEFAULT
	}

	// Internal subprocess mode: spawned by the parent for multi-store queries.
	// Must be checked before normal command routing.
	for arg in args[1:] {
		if arg == "--internal-query" {
			// Strip --internal-query from the remaining args.
			rest := make([dynamic]string)
			for a in args[1:] {
				if a != "--internal-query" {
					append(&rest, a)
				}
			}
			do_internal_query(&cfg, rest[:])
			os.exit(0)
		}
	}

	cmd_start := 1
	if args[1] == "-v" || args[1] == "--verbose" {
		cmd_start = 2
	}

	if len(args) <= cmd_start {
		print_usage()
		os.exit(0)
	}

	cmd  := args[cmd_start]
	rest := args[cmd_start + 1:]

	switch cmd {
	case "serve":
		do_serve(&cfg, rest)
	case "ingest":
		do_ingest(&cfg, rest)
	case "ask":
		do_ask(&cfg, rest)
	case "explore":
		do_explore(&cfg, rest)
	case "ingest-corpus":
		do_ingest_corpus(&cfg, rest)
	case "new":
		do_new(&cfg, rest)
	case "register":
		do_register(&cfg, rest)
	case "list":
		do_list(&cfg, rest)
	case "knid":
		do_knid(&cfg, rest)
	case "init":
		do_init()
	case "help", "--help", "-h":
		print_usage()
	case:
		fmt.fprintf(os.stderr, "unknown command: %s\n", cmd)
		print_usage()
		os.exit(1)
	}
}


print_usage :: proc() {
	fmt.println("usage: knod [-v] <command> [args]")
	fmt.println()
	fmt.println("commands:")
	fmt.println("  serve             start HTTP + TCP server")
	fmt.println("  ingest <file>     ingest a text file")
	fmt.println("  ask <query>       ask a question")
	fmt.println("  explore           show graph stats")
	fmt.println("  ingest-corpus     ingest all .txt files in a directory")
	fmt.println("  new               create a new specialist graph interactively")
	fmt.println("  register <path>   register an existing graph file")
	fmt.println("  list              list registered stores")
	fmt.println("  knid              manage knid groupings")
	fmt.println("  init              write default config to ~/.config/knod/config")
	fmt.println()
	fmt.println("flags:")
	fmt.println("  --graph <path>       override graph_path")
	fmt.println("  --port <n>           override http_port")
	fmt.println("  --tcp-port <n>       override tcp_port")
	fmt.println("  --descriptor <name>  use a named descriptor (ingest)")
	fmt.println("  --knid <name>        scope to a knid group (list)")
	fmt.println("  --dir <path>         directory to scan (ingest-corpus, default: corpus)")
}


load_handler :: proc(cfg: ^cfg_pkg.Config) -> (h: proto.Handler, ok: bool) {
	prov.configure(prov.Config{
		api_key         = cfg.api_key,
		base_url        = cfg.base_url,
		embedding_model = cfg.embedding_model,
		chat_model      = cfg.chat_model,
		timeout_ms      = cfg.timeout_ms,
	})

	p := new(prov.Provider)
	p^ = prov.openai_create()

	g := new(graph_pkg.Graph)
	graph_pkg.create(g)
	g.max_thoughts      = cfg.max_thoughts
	g.max_edges         = cfg.max_edges
	g.maturity_threshold = cfg.maturity_threshold

	model := new(gnn_pkg.MPNN)
	gnn_pkg.create(model)

	strand := new(gnn_pkg.StrandMPNN)
	gnn_pkg.strand_create(strand, gnn_pkg.DEFAULT_HIDDEN_DIM)

	base_path: string
	if len(cfg.base_path) > 0 {
		base_path = cfg.base_path
	} else {
		home := util.home_dir()
		if len(home) > 0 {
			base_path = filepath.join({home, ".config", "knod", "knod"})
			delete(home)
		}
	}

	if len(base_path) > 0 {
		gnn_pkg.load_checkpoint(model, base_path)
	}

	if os.exists(cfg.graph_path) {
		_, load_ok := graph_pkg.load_and_replay(g, cfg.graph_path)
		if !load_ok {
			log.warn("failed to load graph from %s", cfg.graph_path)
		} else {
			strand_bytes, sb_ok := graph_pkg.load_strand_bytes(cfg.graph_path)
			if sb_ok && len(strand_bytes) > 0 {
				off := 0
				gnn_pkg.strand_load(strand, strand_bytes, &off)
				delete(strand_bytes)
			}
		}
	}

	stored_base := strings.clone(base_path)
	if base_path != cfg.base_path {
		delete(base_path)
	}

	h = proto.Handler{
		g                    = g,
		p                    = p,
		model                = model,
		strand               = strand,
		graph_path           = strings.clone(cfg.graph_path),
		base_checkpoint_path = stored_base,
		ingest_cfg = ingest_pkg.Config{
			max_similar     = cfg.max_similar,
			edge_threshold  = cfg.edge_threshold,
			min_link_weight = cfg.min_link_weight,
			dedup_threshold = cfg.dedup_threshold,
		},
		query_cfg = query_pkg.Config{
			top_k                = cfg.find_k,
			similarity_threshold = cfg.similarity_threshold,
			max_context_edges    = cfg.max_context_edges,
			confidence_threshold = cfg.confidence_threshold,
		},
		query_routing_threshold = cfg.query_routing_threshold,
		edge_decay              = cfg.edge_decay,
	}
	return h, true
}

release_handler :: proc(h: ^proto.Handler) {
	proto.handler_stop_queue(h)

	if h.strand != nil {
		gnn_pkg.strand_release(h.strand)
		free(h.strand)
		h.strand = nil
	}
	if h.model != nil {
		gnn_pkg.release(h.model)
		free(h.model)
		h.model = nil
	}
	if h.g != nil {
		graph_pkg.release(h.g)
		free(h.g)
		h.g = nil
	}
	if h.p != nil {
		prov.openai_destroy(h.p)
		free(h.p)
		h.p = nil
	}
	delete(h.graph_path)
	delete(h.base_checkpoint_path)
	delete(h.subs)
}


parse_flag :: proc(args: []string, flag: string) -> (value: string, found: bool, remaining: []string) {
	for i in 0 ..< len(args) {
		if args[i] == flag && i + 1 < len(args) {
			merged := make([dynamic]string)
			append(&merged, ..args[:i])
			append(&merged, ..args[i + 2:])
			return args[i + 1], true, merged[:]
		}
	}
	return "", false, args
}

has_flag :: proc(args: []string, flag: string) -> (found: bool, remaining: []string) {
	for i in 0 ..< len(args) {
		if args[i] == flag {
			merged := make([dynamic]string)
			append(&merged, ..args[:i])
			append(&merged, ..args[i + 1:])
			return true, merged[:]
		}
	}
	return false, args
}

parse_int_arg :: proc(s: string) -> (int, bool) {
	return strconv.parse_int(s)
}


do_serve :: proc(cfg: ^cfg_pkg.Config, args: []string) {
	remaining := args

	if graph_val, ok, r := parse_flag(remaining, "--graph"); ok {
		cfg.graph_path = graph_val
		remaining = r
	}
	if port_val, ok, r := parse_flag(remaining, "--port"); ok {
		if v, pok := parse_int_arg(port_val); pok {cfg.http_port = v}
		remaining = r
	}
	if tcp_val, ok, r := parse_flag(remaining, "--tcp-port"); ok {
		if v, pok := parse_int_arg(tcp_val); pok {cfg.tcp_port = v}
		remaining = r
	}
	no_http, r1 := has_flag(remaining, "--no-http")
	remaining = r1
	no_tcp, r2 := has_flag(remaining, "--no-tcp")
	remaining = r2
	_ = remaining

	h, ok := load_handler(cfg)
	if !ok {os.exit(1)}
	defer release_handler(&h)

	proto.handler_start_queue(&h)

	tcp: proto.TCP
	tcp_ok := false
	if !no_tcp {
		tcp, tcp_ok = proto.tcp_create(cfg.tcp_port, &h)
		if !tcp_ok {
			log.warn("tcp server failed to start")
		}
	}

	if !no_http {
		http_srv, http_ok := proto.http_create(cfg.http_port, &h)
		if !http_ok {
			log.warn("http server failed to start")
		}
		if tcp_ok {
			for {
				proto.tcp_poll(&tcp)
				time.sleep(time.Millisecond)
			}
		} else {
			thread.join(http_srv.t)
		}
		proto.http_destroy(&http_srv)
	} else {
		log.info("http: disabled")
		for tcp_ok {
			proto.tcp_poll(&tcp)
			time.sleep(time.Millisecond)
		}
	}

	if tcp_ok {
		proto.tcp_destroy(&tcp)
	}
}


do_ingest :: proc(cfg: ^cfg_pkg.Config, args: []string) {
	remaining := args

	if graph_val, ok, r := parse_flag(remaining, "--graph"); ok {
		cfg.graph_path = graph_val
		remaining = r
	}
	descriptor := ""
	if d, ok, r := parse_flag(remaining, "--descriptor"); ok {
		descriptor = d
		remaining = r
	}

	if len(remaining) == 0 {
		fmt.fprintln(os.stderr, "usage: knod ingest [--graph <path>] [--descriptor <name>] <file>")
		os.exit(1)
	}

	file_path := remaining[0]
	data, read_ok := os.read_entire_file(file_path)
	if !read_ok {
		fmt.fprintf(os.stderr, "error: cannot read file: %s\n", file_path)
		os.exit(1)
	}
	defer delete(data)

	h, ok := load_handler(cfg)
	if !ok {os.exit(1)}
	defer release_handler(&h)

	result := proto.handle_ingest(&h, string(data), descriptor)
	fmt.printf("ingested from %s\n", file_path)
	fmt.printf("graph: %d thoughts, %d edges\n", result.thoughts, result.edges)
}


do_ask :: proc(cfg: ^cfg_pkg.Config, args: []string) {
	remaining := args

	// --graph overrides to single-store mode (bypass registry entirely).
	if graph_val, ok, r := parse_flag(remaining, "--graph"); ok {
		cfg.graph_path = graph_val
		remaining = r

		if len(remaining) == 0 {
			fmt.fprintln(os.stderr, "usage: knod ask [--graph <path>] <query>")
			os.exit(1)
		}

		query_text := strings.join(remaining, " ")
		defer delete(query_text)

		h, ok2 := load_handler(cfg)
		if !ok2 {os.exit(1)}
		defer release_handler(&h)

		answer, answer_ok := proto.handle_ask(&h, query_text)
		if !answer_ok {
			fmt.fprintln(os.stderr, "no answer generated")
			os.exit(1)
		}
		defer delete(answer)

		fmt.println()
		fmt.println(answer)
		return
	}

	// Default: multi-store mode. do_multi_ask handles registry lookup,
	// profile routing, subprocess fan-out, and answer aggregation.
	// If no stores are registered it will auto-add cfg.graph_path as a
	// fallback so a single-graph setup still works.
	do_multi_ask(cfg, remaining)
}


do_explore :: proc(cfg: ^cfg_pkg.Config, args: []string) {
	remaining := args
	if graph_val, ok, r := parse_flag(remaining, "--graph"); ok {
		cfg.graph_path = graph_val
		remaining = r
	}
	_ = remaining

	if !os.exists(cfg.graph_path) {
		fmt.fprintln(os.stderr, "no graph found")
		os.exit(1)
	}

	g: graph_pkg.Graph
	graph_pkg.create(&g)
	defer graph_pkg.release(&g)

	_, load_ok := graph_pkg.load_and_replay(&g, cfg.graph_path)
	if !load_ok {
		fmt.fprintf(os.stderr, "error: failed to load graph from %s\n", cfg.graph_path)
		os.exit(1)
	}

	purpose := g.purpose
	if len(purpose) == 0 {purpose = "(none)"}

	fmt.printf("purpose:  %s\n", purpose)
	fmt.printf("thoughts: %d\n", graph_pkg.thought_count(&g))
	fmt.printf("edges:    %d\n", graph_pkg.edge_count(&g))
	fmt.printf("maturity: %.2f\n", graph_pkg.maturity(&g))
	fmt.printf("graph:    %s\n", cfg.graph_path)

	if len(g.descriptors) > 0 {
		b := strings.builder_make()
		defer strings.builder_destroy(&b)
		first := true
		for k in g.descriptors {
			if !first {strings.write_string(&b, ", ")}
			strings.write_string(&b, k)
			first = false
		}
		fmt.printf("descriptors: %s\n", strings.to_string(b))
	}
}


do_ingest_corpus :: proc(cfg: ^cfg_pkg.Config, args: []string) {
	remaining := args
	corpus_dir := "corpus"

	if dir_val, ok, r := parse_flag(remaining, "--dir"); ok {
		corpus_dir = dir_val
		remaining = r
	}
	if graph_val, ok, r := parse_flag(remaining, "--graph"); ok {
		cfg.graph_path = graph_val
		remaining = r
	}
	_ = remaining

	if !os.exists(corpus_dir) {
		fmt.fprintf(os.stderr, "directory not found: %s\n", corpus_dir)
		os.exit(1)
	}

	h, ok := load_handler(cfg)
	if !ok {os.exit(1)}
	defer release_handler(&h)

	d, open_err := os.open(corpus_dir)
	if open_err != os.ERROR_NONE {
		fmt.fprintf(os.stderr, "error: cannot open directory: %s\n", corpus_dir)
		os.exit(1)
	}

	entries, read_err := os.read_dir(d, -1)
	os.close(d)
	if read_err != os.ERROR_NONE {
		fmt.fprintf(os.stderr, "error: cannot read directory: %s\n", corpus_dir)
		os.exit(1)
	}
	defer os.file_info_slice_delete(entries)

	count := 0
	for entry in entries {
		if entry.is_dir {continue}
		if !strings.has_suffix(entry.name, ".txt") {continue}
		if entry.name == "manifest.txt" {continue}
		count += 1
	}
	if count == 0 {
		fmt.fprintf(os.stderr, "no .txt files in %s\n", corpus_dir)
		os.exit(1)
	}

	i := 0
	for entry in entries {
		if entry.is_dir {continue}
		if !strings.has_suffix(entry.name, ".txt") {continue}
		if entry.name == "manifest.txt" {continue}
		i += 1

		full_path := filepath.join({corpus_dir, entry.name})
		defer delete(full_path)

		fmt.printf("[%d/%d] %s\n", i, count, entry.name)

		data, file_ok := os.read_entire_file(full_path)
		if !file_ok {
			fmt.fprintf(os.stderr, "  error: cannot read %s\n", entry.name)
			continue
		}

		result := proto.handle_ingest(&h, string(data), "")
		delete(data)
		fmt.printf("  -> %d thoughts, %d edges\n", result.thoughts, result.edges)
	}

	fmt.printf("\ndone. graph: %d thoughts, %d edges\n",
		graph_pkg.thought_count(h.g),
		graph_pkg.edge_count(h.g),
	)
}


do_new :: proc(cfg: ^cfg_pkg.Config, args: []string) {
	remaining := args
	knid_name := ""

	if knid_val, ok, r := parse_flag(remaining, "--knid"); ok {
		knid_name = knid_val
		remaining = r
	}
	_ = remaining

	fmt.print("purpose: ")
	purpose_raw := read_line()
	defer delete(purpose_raw)
	purpose := strings.trim_space(purpose_raw)
	if len(purpose) == 0 {
		fmt.fprintln(os.stderr, "error: purpose is required")
		os.exit(1)
	}

	fmt.print("name: ")
	name_raw := read_line()
	defer delete(name_raw)
	name := strings.trim_space(name_raw)
	if len(name) == 0 {
		fmt.fprintln(os.stderr, "error: name is required")
		os.exit(1)
	}

	cwd := os.get_current_directory()
	defer delete(cwd)
	fmt.printf("location [%s]: ", cwd)
	loc_raw := read_line()
	defer delete(loc_raw)
	location := strings.trim_space(loc_raw)
	if len(location) == 0 {location = cwd}

	safe_lower := strings.to_lower(name)
	defer delete(safe_lower)
	safe_name, _ := strings.replace_all(safe_lower, " ", "_")
	defer delete(safe_name)

	base := filepath.join({location, safe_name})
	defer delete(base)

	graph_path := strings.concatenate({base, util.STRAND_EXTENSION})
	defer delete(graph_path)

	g: graph_pkg.Graph
	graph_pkg.create(&g)
	g.max_thoughts = cfg.max_thoughts
	g.max_edges    = cfg.max_edges
	graph_pkg.set_purpose(&g, purpose)
	defer graph_pkg.release(&g)

	strand: gnn_pkg.StrandMPNN
	gnn_pkg.strand_create(&strand, gnn_pkg.DEFAULT_HIDDEN_DIM)
	defer gnn_pkg.strand_release(&strand)

	strand_bytes := gnn_pkg.strand_save_bytes(&strand)
	if strand_bytes == nil {strand_bytes = make([]u8, 0)}
	defer delete(strand_bytes)

	if !graph_pkg.save_with_strand(&g, graph_path, strand_bytes) {
		fmt.fprintf(os.stderr, "error: failed to save graph to %s\n", graph_path)
		os.exit(1)
	}

	r, _ := reg_pkg.load()
	defer reg_pkg.release(&r)

	reg_pkg.add_store(&r, name, graph_path)
	if len(knid_name) > 0 {
		reg_pkg.knid_add_store(&r, knid_name, name)
		fmt.printf("added to knid '%s'\n", knid_name)
	}
	reg_pkg.save(&r)

	fmt.printf("created specialist '%s' at %s\n", name, graph_path)
}


do_register :: proc(cfg: ^cfg_pkg.Config, args: []string) {
	remaining := args
	knid_name := ""

	if knid_val, ok, r := parse_flag(remaining, "--knid"); ok {
		knid_name = knid_val
		remaining = r
	}

	if len(remaining) == 0 {
		fmt.fprintln(os.stderr, "usage: knod register [--knid <name>] <path>")
		os.exit(1)
	}

	path := remaining[0]
	if !os.exists(path) {
		fmt.fprintf(os.stderr, "file not found: %s\n", path)
		os.exit(1)
	}

	g: graph_pkg.Graph
	graph_pkg.create(&g)
	defer graph_pkg.release(&g)

	_, load_ok := graph_pkg.load_and_replay(&g, path)
	if !load_ok {
		fmt.fprintf(os.stderr, "invalid graph file: %s\n", path)
		os.exit(1)
	}

	base := filepath.base(path)
	name := strings.trim_suffix(strings.trim_suffix(base, util.STRAND_EXTENSION), ".graph")

	abs_path, _ := filepath.abs(path)
	defer delete(abs_path)

	r, _ := reg_pkg.load()
	defer reg_pkg.release(&r)

	reg_pkg.add_store(&r, name, abs_path)
	if len(knid_name) > 0 {
		reg_pkg.knid_add_store(&r, knid_name, name)
		fmt.printf("added to knid '%s'\n", knid_name)
	}
	reg_pkg.save(&r)

	purpose := g.purpose
	if len(purpose) == 0 {purpose = "(none)"}
	fmt.printf("registered '%s' (purpose: %s)\n", name, purpose)
}


do_list :: proc(cfg: ^cfg_pkg.Config, args: []string) {
	_ = cfg
	remaining := args
	knid_name := ""

	if knid_val, ok, r := parse_flag(remaining, "--knid"); ok {
		knid_name = knid_val
		remaining = r
	}
	_ = remaining

	r, _ := reg_pkg.load()
	defer reg_pkg.release(&r)

	if len(knid_name) > 0 {
		members := reg_pkg.knid_stores(&r, knid_name)
		defer delete(members)
		if len(members) == 0 {
			fmt.fprintf(os.stderr, "no stores in knid '%s'\n", knid_name)
			return
		}
		fmt.printf("stores in knid '%s':\n", knid_name)
		for s in members {
			fmt.printf("  %s = %s\n", s.name, s.path)
		}
		return
	}

	if len(r.stores) == 0 {
		fmt.println("no registered stores.")
		return
	}
	for &s in r.stores {
		fmt.printf("  %s = %s\n", s.name, s.path)
	}
}


do_knid :: proc(cfg: ^cfg_pkg.Config, args: []string) {
	_ = cfg
	if len(args) == 0 {
		fmt.fprintln(os.stderr, "usage: knod knid {new|add|remove|list} [args]")
		os.exit(1)
	}

	r, _ := reg_pkg.load()
	defer reg_pkg.release(&r)

	subcmd := args[0]
	rest   := args[1:]

	switch subcmd {
	case "new":
		if len(rest) == 0 {
			fmt.fprintln(os.stderr, "usage: knod knid new <name>")
			os.exit(1)
		}
		if !reg_pkg.add_knid(&r, rest[0]) {
			fmt.fprintf(os.stderr, "knid '%s' already exists\n", rest[0])
			os.exit(1)
		}
		reg_pkg.save(&r)
		fmt.printf("created knid '%s'\n", rest[0])

	case "add":
		if len(rest) < 2 {
			fmt.fprintln(os.stderr, "usage: knod knid add <knid> <store>")
			os.exit(1)
		}
		if !reg_pkg.knid_add_store(&r, rest[0], rest[1]) {
			fmt.fprintf(os.stderr, "failed: knid or store not found, or already a member\n")
			os.exit(1)
		}
		reg_pkg.save(&r)
		fmt.printf("added '%s' to knid '%s'\n", rest[1], rest[0])

	case "remove":
		if len(rest) < 2 {
			fmt.fprintln(os.stderr, "usage: knod knid remove <knid> <store>")
			os.exit(1)
		}
		if !reg_pkg.knid_remove_store(&r, rest[0], rest[1]) {
			fmt.fprintf(os.stderr, "'%s' not found in knid '%s'\n", rest[1], rest[0])
			os.exit(1)
		}
		reg_pkg.save(&r)
		fmt.printf("removed '%s' from knid '%s'\n", rest[1], rest[0])

	case "list":
		if len(rest) > 0 {
			members := reg_pkg.knid_stores(&r, rest[0])
			defer delete(members)
			if len(members) == 0 {
				fmt.fprintf(os.stderr, "no stores in knid '%s'\n", rest[0])
				return
			}
			fmt.printf("knid '%s':\n", rest[0])
			for s in members {fmt.printf("  %s\n", s.name)}
		} else {
			if len(r.knids) == 0 {
				fmt.println("no knids defined.")
				return
			}
			for &k in r.knids {
				if len(k.members) > 0 {
					joined := strings.join(k.members[:], ", ")
					fmt.printf("  [%s] %s\n", k.name, joined)
					delete(joined)
				} else {
					fmt.printf("  [%s] (empty)\n", k.name)
				}
			}
		}

	case:
		fmt.fprintf(os.stderr, "unknown knid subcommand: %s\n", subcmd)
		fmt.fprintln(os.stderr, "usage: knod knid {new|add|remove|list} [args]")
		os.exit(1)
	}
}


do_init :: proc() {
	path := cfg_pkg.write_default()
	if len(path) == 0 {
		fmt.fprintln(os.stderr, "error: could not write config")
		os.exit(1)
	}
	fmt.printf("config written to: %s\n", path)
}


read_line :: proc() -> string {
	buf: [dynamic]u8
	tmp: [1]u8
	for {
		n, err := os.read(os.stdin, tmp[:])
		if err != os.ERROR_NONE || n == 0 {break}
		if tmp[0] == '\n' {break}
		if tmp[0] == '\r' {continue}
		append(&buf, tmp[0])
	}
	return string(buf[:])
}
