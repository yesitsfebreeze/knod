package main

// internal_query.odin — subprocess entry point for multi-store parallel queries.
//
// Invoked by the parent as:
//   knod --internal-query --graph=<path> --query=<text>
//
// Loads graph + base GNN + specialist strand, embeds the query via the
// provider, runs retrieve(), and writes results to stdout as newline-
// delimited JSON records, one per scored thought:
//
//   {"text":"...","score":0.85,"source":"..."}
//   {"text":"...","score":0.70,"source":"..."}
//
// Exits 0 on success, 1 on any fatal error.

import "core:fmt"
import "core:os"
import "core:path/filepath"
import "core:strings"

import cfg_pkg   "../config"
import gnn_pkg   "../gnn"
import graph_pkg "../graph"
import log       "../logger"
import prov      "../provider"
import query_pkg "../query"
import util      "../util"

// do_internal_query is the entry point when --internal-query is present.
// args is os.args[1:] with "--internal-query" already consumed.
do_internal_query :: proc(cfg: ^cfg_pkg.Config, args: []string) {
	// Redirect logger to stderr so stdout is clean for JSON output.
	log.redirect_to_stderr()
	graph_path := ""
	query_text := ""

	for i := 0; i < len(args); i += 1 {
		arg := args[i]
		if strings.has_prefix(arg, "--graph=") {
			graph_path = arg[len("--graph="):]
		} else if arg == "--graph" && i+1 < len(args) {
			i += 1
			graph_path = args[i]
		} else if strings.has_prefix(arg, "--query=") {
			query_text = arg[len("--query="):]
		} else if arg == "--query" && i+1 < len(args) {
			i += 1
			query_text = strings.join(args[i:], " ")
			break
		}
	}

	if len(graph_path) == 0 {
		fmt.fprintln(os.stderr, "internal-query: --graph is required")
		os.exit(1)
	}
	if len(query_text) == 0 {
		fmt.fprintln(os.stderr, "internal-query: --query is required")
		os.exit(1)
	}

	// Configure provider.
	prov.configure(prov.Config{
		api_key         = cfg.api_key,
		base_url        = cfg.base_url,
		embedding_model = cfg.embedding_model,
		chat_model      = cfg.chat_model,
		timeout_ms      = cfg.timeout_ms,
	})
	p := new(prov.Provider)
	p^ = prov.openai_create()
	defer {
		prov.openai_destroy(p)
		free(p)
	}

	// Load graph.
	g := new(graph_pkg.Graph)
	graph_pkg.create(g)
	defer {
		graph_pkg.release(g)
		free(g)
	}

	if !os.exists(graph_path) {
		fmt.fprintf(os.stderr, "internal-query: graph not found: %s\n", graph_path)
		os.exit(1)
	}
	_, load_ok := graph_pkg.load_and_replay(g, graph_path)
	if !load_ok {
		fmt.fprintf(os.stderr, "internal-query: failed to load graph: %s\n", graph_path)
		os.exit(1)
	}

	// Load base GNN.
	model := new(gnn_pkg.MPNN)
	gnn_pkg.create(model)
	defer {
		gnn_pkg.release(model)
		free(model)
	}

	base_path: string
	if len(cfg.base_path) > 0 {
		base_path = cfg.base_path
	} else {
		home := util.home_dir()
		if len(home) > 0 {
			base_path = filepath.join({home, ".config", "knod", "knod"})
			defer delete(home)
		}
	}
	if len(base_path) > 0 {
		gnn_pkg.load_checkpoint(model, base_path)
		if base_path != cfg.base_path {
			delete(base_path)
		}
	}

	// Load specialist strand.
	strand := new(gnn_pkg.StrandMPNN)
	gnn_pkg.strand_create(strand, gnn_pkg.DEFAULT_HIDDEN_DIM)
	defer {
		gnn_pkg.strand_release(strand)
		free(strand)
	}
	strand_bytes, sb_ok := graph_pkg.load_strand_bytes(graph_path)
	if sb_ok && len(strand_bytes) > 0 {
		off := 0
		gnn_pkg.strand_load(strand, strand_bytes, &off)
		delete(strand_bytes)
	}

	// Embed the query.
	query_embedding, embed_ok := p.embed_text(p, query_text)
	if !embed_ok {
		fmt.fprintln(os.stderr, "internal-query: failed to embed query")
		os.exit(1)
	}

	qcfg := query_pkg.Config{
		top_k                = cfg.find_k,
		similarity_threshold = cfg.similarity_threshold,
		max_context_edges    = cfg.max_context_edges,
		confidence_threshold = 0, // parent handles confidence gate
	}
	if qcfg.top_k == 0 {qcfg = query_pkg.DEFAULT_CONFIG}

	// Retrieve.
	scored := query_pkg.retrieve(g, &query_embedding, model, strand, qcfg)
	defer delete(scored)

	// Write results as newline-delimited JSON to stdout.
	for st in scored {
		text_escaped   := _json_escape(st.thought.text)
		source_escaped := _json_escape(st.thought.source)
		defer delete(text_escaped)
		defer delete(source_escaped)
		fmt.printf("{{\"text\":%s,\"score\":%.6f,\"source\":%s}}\n",
			text_escaped,
			st.score,
			source_escaped,
		)
	}
}

// _json_escape produces a JSON-quoted string (with surrounding double-quotes).
@(private)
_json_escape :: proc(s: string) -> string {
	b := strings.builder_make()
	strings.write_byte(&b, '"')
	for ch in s {
		switch ch {
		case '"':  strings.write_string(&b, "\\\"")
		case '\\': strings.write_string(&b, "\\\\")
		case '\n': strings.write_string(&b, "\\n")
		case '\r': strings.write_string(&b, "\\r")
		case '\t': strings.write_string(&b, "\\t")
		case:
			if ch < 0x20 {
				fmt.sbprintf(&b, "\\u%04x", int(ch))
			} else {
				strings.write_rune(&b, ch)
			}
		}
	}
	strings.write_byte(&b, '"')
	return strings.to_string(b)
}
