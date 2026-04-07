package main

// multi_ask.odin — parent-side multi-store query fan-out (Plan Phase 5, items 25-26).
//
// When `do_ask` finds registered stores, it calls `do_multi_ask` instead of
// querying a single graph directly.  Each store is queried by spawning a
// subprocess of the same executable in --internal-query mode.  Subprocess
// stdout is parsed as newline-delimited JSON.  Results are merged, ranked,
// and fed to the LLM for a final synthesised answer.

import "core:fmt"
import "core:os"
import os2 "core:os/os2"
import "core:strconv"
import "core:strings"
import "core:thread"

import cfg_pkg   "../config"
import graph_pkg "../graph"
import log       "../logger"
import prov      "../provider"
import reg_pkg   "../registry"


// Sub_Result holds the parsed results from one subprocess.
Sub_Result :: struct {
	store_name: string,
	thoughts:   [dynamic]Sub_Thought,
}

Sub_Thought :: struct {
	text:   string,
	score:  f32,
	source: string,
}

sub_result_release :: proc(r: ^Sub_Result) {
	for &t in r.thoughts {
		delete(t.text)
		delete(t.source)
	}
	delete(r.thoughts)
}

// Spawn_Task is at package level so _spawn_store_query (also package-level) can see it.
Spawn_Task :: struct {
	exe_path:   string,
	graph_path: string,
	query_text: string,
	result:     ^Sub_Result,
}


// do_multi_ask fans out to all registered stores (filtered by knid_name) in
// parallel, collects results, deduplicates, and generates a final answer.
do_multi_ask :: proc(cfg: ^cfg_pkg.Config, args: []string) {
	remaining := args
	knid_name := ""

	if knid_val, ok, r := parse_flag(remaining, "--knid"); ok {
		knid_name = knid_val
		remaining = r
	}

	if len(remaining) == 0 {
		fmt.fprintln(os.stderr, "usage: knod ask [--knid <name>] <query>")
		os.exit(1)
	}

	query_text := strings.join(remaining, " ")
	defer delete(query_text)

	// Gather stores.
	reg, _ := reg_pkg.load()
	defer reg_pkg.release(&reg)

	store_paths := make([dynamic]string)
	defer delete(store_paths)
	store_names := make([dynamic]string)
	defer delete(store_names)

	if len(knid_name) > 0 {
		members := reg_pkg.knid_stores(&reg, knid_name)
		defer delete(members)
		for s in members {
			append(&store_paths, s.path)
			append(&store_names, s.name)
		}
	} else {
		for &s in reg.stores {
			append(&store_paths, s.path)
			append(&store_names, s.name)
		}
	}

	if len(store_paths) == 0 {
		// No registered stores — fall back to cfg.graph_path as a single store.
		if len(cfg.graph_path) > 0 && os.exists(cfg.graph_path) {
			append(&store_paths, cfg.graph_path)
			append(&store_names, "default")
		} else {
			fmt.fprintln(os.stderr, "no registered stores and no default graph found")
			fmt.fprintln(os.stderr, "run 'knod new' to create a specialist, or use 'knod ask --graph <path>'")
			os.exit(1)
		}
	}

	// Configure provider for query embedding + final answer generation.
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

	// Embed query for profile-based routing.
	query_embedding, embed_ok := p.embed_text(p, query_text)
	if !embed_ok {
		fmt.fprintln(os.stderr, "error: failed to embed query")
		os.exit(1)
	}

	// Filter stores by profile similarity (query routing threshold).
	filtered_paths := make([dynamic]string)
	filtered_names := make([dynamic]string)
	defer delete(filtered_paths)
	defer delete(filtered_names)

	routing_threshold := cfg.query_routing_threshold
	for i in 0 ..< len(store_paths) {
		if routing_threshold > 0.0 {
			g: graph_pkg.Graph
			graph_pkg.create(&g)
			_, load_ok := graph_pkg.load_and_replay(&g, store_paths[i])
			if load_ok && g.profile_count > 0 {
				sim := graph_pkg.cosine_similarity(&query_embedding, &g.profile)
				if sim >= routing_threshold {
					append(&filtered_paths, store_paths[i])
					append(&filtered_names, store_names[i])
				} else {
					log.info("[multi_ask] skipping '%s' (profile sim=%.3f < threshold=%.3f)",
						store_names[i], sim, routing_threshold)
				}
				graph_pkg.release(&g)
				continue
			}
			graph_pkg.release(&g)
		}
		// No threshold, or graph has no profile yet — always include.
		append(&filtered_paths, store_paths[i])
		append(&filtered_names, store_names[i])
	}

	if len(filtered_paths) == 0 {
		fmt.fprintln(os.stderr, "no stores matched query (all below routing threshold)")
		os.exit(1)
	}

	// Get path to the current executable.
	exe_path, exe_err := os2.get_executable_path(context.allocator)
	if exe_err != nil {
		fmt.fprintln(os.stderr, "error: could not determine executable path")
		os.exit(1)
	}
	defer delete(exe_path)

	// Spawn one subprocess per store in parallel.
	n := len(filtered_paths)
	results := make([]Sub_Result, n)
	defer {
		for i in 0 ..< n {
			sub_result_release(&results[i])
		}
		delete(results)
	}

	task_datas := make([]Spawn_Task, n)
	tasks      := make([]^thread.Thread, n)
	defer {
		delete(task_datas)
		delete(tasks)
	}

	for i in 0 ..< n {
		results[i].store_name = filtered_names[i]
		results[i].thoughts   = make([dynamic]Sub_Thought)

		task_datas[i] = Spawn_Task{
			exe_path   = exe_path,
			graph_path = filtered_paths[i],
			query_text = query_text,
			result     = &results[i],
		}

		t := thread.create_and_start_with_poly_data(&task_datas[i], _spawn_store_query, context)
		tasks[i] = t
	}

	// Wait for all subprocesses.
	for i in 0 ..< n {
		if tasks[i] != nil {
			thread.join(tasks[i])
			thread.destroy(tasks[i])
		}
	}

	// Aggregate: deduplicate by text, keep highest score.
	// Use a flat list + de-dup by text.
	flat := make([dynamic]Sub_Thought)
	defer delete(flat)

	seen := make(map[string]int) // text -> index in flat
	defer delete(seen)

	for i in 0 ..< n {
		for &st in results[i].thoughts {
			if idx, exists := seen[st.text]; exists {
				if st.score > flat[idx].score {
					flat[idx].score = st.score
				}
			} else {
				new_idx := len(flat)
				seen[st.text] = new_idx
				append(&flat, Sub_Thought{
					text   = strings.clone(st.text),
					score  = st.score,
					source = strings.clone(st.source),
				})
			}
		}
	}
	delete(seen)

	// Free the cloned entries on exit.
	defer {
		for &entry in flat {
			delete(entry.text)
			delete(entry.source)
		}
	}

	// Sort by score descending (insertion sort — small N).
	for i in 1 ..< len(flat) {
		j := i
		for j > 0 && flat[j].score > flat[j-1].score {
			flat[j], flat[j-1] = flat[j-1], flat[j]
			j -= 1
		}
	}

	top_k := cfg.find_k
	if top_k == 0 {top_k = 10}
	n_results := min(top_k, len(flat))

	if n_results == 0 {
		fmt.fprintln(os.stderr, "no relevant thoughts found across stores")
		os.exit(1)
	}

	// Confidence gate: if top score >= threshold, return thoughts directly.
	conf_threshold := cfg.confidence_threshold
	if conf_threshold > 0.0 && flat[0].score >= conf_threshold {
		fmt.println()
		for i in 0 ..< n_results {
			if i > 0 {fmt.println()}
			fmt.println(flat[i].text)
		}
		return
	}

	// Build context string for LLM answer generation.
	ctx_b := strings.builder_make()
	defer strings.builder_destroy(&ctx_b)
	for i in 0 ..< n_results {
		if i > 0 {strings.write_string(&ctx_b, "\n\n")}
		strings.write_string(&ctx_b, flat[i].text)
	}
	context_text := strings.to_string(ctx_b)

	answer, answer_ok := p.generate_answer(p, query_text, context_text)
	if !answer_ok {
		fmt.fprintln(os.stderr, "error: failed to generate answer")
		os.exit(1)
	}
	defer delete(answer)

	fmt.println()
	fmt.println(answer)
}


// _spawn_store_query runs one --internal-query subprocess and parses its output.
@(private)
_spawn_store_query :: proc(task: ^Spawn_Task) {
	graph_arg := strings.concatenate({"--graph=", task.graph_path})
	query_arg := strings.concatenate({"--query=", task.query_text})
	defer delete(graph_arg)
	defer delete(query_arg)

	cmd_slice := []string{task.exe_path, "--internal-query", graph_arg, query_arg}

	desc: os2.Process_Desc
	desc.command = cmd_slice

	state, stdout_bytes, stderr_bytes, err := os2.process_exec(desc, context.allocator)
	defer delete(stdout_bytes)
	defer delete(stderr_bytes)

	if err != nil || !state.success {
		log.warn("[multi_ask] subprocess for '%s' failed (exit=%d)",
			task.graph_path, state.exit_code)
		if len(stderr_bytes) > 0 {
			log.warn("[multi_ask] stderr: %s", string(stderr_bytes))
		}
		return
	}

	// Parse newline-delimited JSON.
	output := string(stdout_bytes)
	lines  := strings.split_lines(output)
	defer delete(lines)

	for line in lines {
		trimmed := strings.trim_space(line)
		if len(trimmed) == 0 {continue}
		st, ok := _parse_sub_thought(trimmed)
		if ok {
			append(&task.result.thoughts, st)
		}
	}
}


// _parse_sub_thought parses one JSON line of the form:
//   {"text":"...","score":0.85,"source":"..."}
// Returns heap-allocated strings in Sub_Thought (caller must free).
@(private)
_parse_sub_thought :: proc(line: string) -> (st: Sub_Thought, ok: bool) {
	s := strings.trim_space(line)
	if len(s) < 2 || s[0] != '{' || s[len(s)-1] != '}' {return {}, false}
	s = s[1:len(s)-1]

	text, source: string
	score_str: string
	has_text, has_score: bool

	for len(s) > 0 {
		s = strings.trim_space(s)
		if len(s) == 0 {break}
		if s[0] != '"' {
			idx := strings.index_any(s, ",}")
			if idx < 0 {break}
			s = s[idx+1:]
			continue
		}
		key, rest, key_ok := _json_read_string(s)
		if !key_ok {break}
		s = strings.trim_space(rest)
		if len(s) == 0 || s[0] != ':' {
			delete(key)
			break
		}
		s = strings.trim_space(s[1:])
		switch key {
		case "text":
			v, r, vok := _json_read_string(s)
			if vok {
				text = v
				has_text = true
				s = r
			}
		case "source":
			v, r, vok := _json_read_string(s)
			if vok {
				source = v
				s = r
			}
		case "score":
			end := strings.index_any(s, ",}")
			if end < 0 {end = len(s)}
			score_str = strings.trim_space(s[:end])
			has_score = true
			s = s[end:]
		case:
			// Skip unknown field value.
			if len(s) > 0 && s[0] == '"' {
				_, r, _ := _json_read_string(s)
				s = r
			} else {
				end := strings.index_any(s, ",}")
				if end < 0 {break}
				s = s[end:]
			}
		}
		delete(key)
		s = strings.trim_space(s)
		if len(s) > 0 && s[0] == ',' {s = s[1:]}
	}

	if !has_text || !has_score {
		if has_text {delete(text)}
		if len(source) > 0 {delete(source)}
		return {}, false
	}

	score_val, sval_ok := strconv.parse_f32(score_str)
	if !sval_ok {
		delete(text)
		if len(source) > 0 {delete(source)}
		return {}, false
	}

	return Sub_Thought{text = text, score = score_val, source = source}, true
}


// _json_read_string reads a JSON-quoted string starting at s[0] == '"'.
// Returns the unescaped heap-allocated string, the remainder of s, and ok.
@(private)
_json_read_string :: proc(s: string) -> (value: string, rest: string, ok: bool) {
	if len(s) == 0 || s[0] != '"' {return "", s, false}
	b := strings.builder_make()
	i := 1
	for i < len(s) {
		ch := s[i]
		if ch == '"' {
			return strings.to_string(b), s[i+1:], true
		}
		if ch == '\\' && i+1 < len(s) {
			i += 1
			switch s[i] {
			case '"':  strings.write_byte(&b, '"')
			case '\\': strings.write_byte(&b, '\\')
			case 'n':  strings.write_byte(&b, '\n')
			case 'r':  strings.write_byte(&b, '\r')
			case 't':  strings.write_byte(&b, '\t')
			case:      strings.write_byte(&b, s[i])
			}
		} else {
			strings.write_byte(&b, ch)
		}
		i += 1
	}
	strings.builder_destroy(&b)
	return "", s, false
}
