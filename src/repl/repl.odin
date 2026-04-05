package repl

import "core:fmt"
import "core:os"
import "core:strconv"
import "core:strings"
import "core:time"

import "../graph"
import "../ingest"
import log "../logger"
import "../provider"

State :: struct {
	line_buf:   [dynamic]u8,
	g:          ^graph.Graph,
	p:          ^provider.Provider,
	graph_path: string,
	limbo:      ^graph.Graph,
	quit:       bool,
}

init :: proc(
	g: ^graph.Graph,
	p: ^provider.Provider,
	graph_path: string,
	limbo_graph: ^graph.Graph = nil,
) -> State {
	print_banner()
	prompt()
	return State {
		line_buf = make([dynamic]u8, 0, 1024),
		g = g,
		p = p,
		graph_path = graph_path,
		limbo = limbo_graph,
		quit = false,
	}
}

destroy :: proc(s: ^State) {
	delete(s.line_buf)
}


poll :: proc(s: ^State) -> bool {
	if s.quit {
		return false
	}

	if !stdin_has_input() {
		return true
	}

	buf: [256]u8
	n, err := os.read(os.stdin, buf[:])
	if err != os.ERROR_NONE || n <= 0 {
		return true
	}

	for i in 0 ..< n {
		ch := buf[i]
		if ch == '\n' || ch == '\r' {
			if ch == '\r' && i + 1 < n && buf[i + 1] == '\n' {
				continue
			}

			line := string(s.line_buf[:])
			trimmed := strings.trim_space(line)
			if len(trimmed) > 0 {
				dispatch(s, trimmed)
			}
			clear(&s.line_buf)

			if s.quit {
				return false
			}
			prompt()
		} else {
			append(&s.line_buf, ch)
		}
	}

	return true
}


@(private)
dispatch :: proc(s: ^State, line: string) {
	if line[0] != '/' {
		fmt.println("(hint: prefix with /ingest to ingest, or /ask to query)")
		return
	}

	space_idx := strings.index(line, " ")
	cmd: string
	args: string
	if space_idx < 0 {
		cmd = line
		args = ""
	} else {
		cmd = line[:space_idx]
		args = strings.trim_space(line[space_idx + 1:])
	}

	switch cmd {
	case "/help", "/h", "/?":
		cmd_help()
	case "/status", "/s":
		cmd_status(s)
	case "/ask", "/a":
		cmd_ask(s, args)
	case "/ingest", "/i":
		cmd_ingest(s, args)
	case "/descriptor", "/d":
		cmd_descriptor(s, args)
	case "/purpose", "/p":
		cmd_purpose(s, args)
	case "/thoughts", "/t":
		cmd_thoughts(s, args)
	case "/edges", "/e":
		cmd_edges(s, args)
	case "/find", "/f":
		cmd_find(s, args)
	case "/save":
		cmd_save(s)
	case "/limbo", "/l":
		cmd_limbo(s)
	case "/quit", "/q":
		cmd_quit(s)
	case:
		fmt.printf("unknown command: %s (type /help for a list)\n", cmd)
	}
}


@(private)
cmd_help :: proc() {
	fmt.println("commands:")
	fmt.println("  /help,       /h  - show this help")
	fmt.println("  /status,     /s  - graph stats and purpose")
	fmt.println("  /ask,        /a  - query the knowledge graph")
	fmt.println("  /ingest,     /i  - ingest text (optional: -d <name>)")
	fmt.println("  /descriptor, /d  - manage descriptors (add/remove/list)")
	fmt.println("  /purpose,    /p  - view or set the node's purpose")
	fmt.println("  /thoughts,   /t  - list thoughts (optional: count)")
	fmt.println("  /edges,      /e  - list edges (optional: thought id)")
	fmt.println("  /find,       /f  - find thoughts by similarity")
	fmt.println("  /save            - force-save the full graph")
	fmt.println("  /limbo,      /l  - show limbo stats")
	fmt.println("  /quit,       /q  - exit knod")
}

@(private)
cmd_status :: proc(s: ^State) {
	tc := graph.thought_count(s.g)
	ec := graph.edge_count(s.g)
	fmt.printf("thoughts: %d\n", tc)
	fmt.printf("edges:    %d\n", ec)
	if len(s.g.purpose) > 0 {
		fmt.printf("purpose:  \"%s\"\n", s.g.purpose)
	} else {
		fmt.println("purpose:  (not set)")
	}
	fmt.printf("graph:    %s\n", s.graph_path)
}

@(private)
cmd_ask :: proc(s: ^State, query: string) {
	if len(query) == 0 {
		fmt.println("usage: /ask <question>")
		return
	}

	fmt.printf("querying: \"%s\" ...\n", query)

	query_embedding, embed_ok := s.p.embed_text(s.p, query)
	if !embed_ok {
		fmt.println("error: failed to embed query")
		return
	}

	results := graph.find_thoughts(s.g, &query_embedding, graph.cfg.default_find_k)
	defer delete(results)

	if len(results) == 0 {
		fmt.println("no relevant thoughts found.")
		return
	}

	context_parts: [dynamic]string
	defer delete(context_parts)

	for result in results {
		thought := graph.get_thought(s.g, result.id)
		if thought != nil {
			append(&context_parts, thought.text)
		}
	}

	context_text := strings.join(context_parts[:], "\n")
	defer delete(context_text)

	answer, answer_ok := s.p.generate_answer(s.p, query, context_text)
	if !answer_ok {
		fmt.println("error: failed to generate answer")
		return
	}
	defer delete(answer)

	fmt.println()
	fmt.println(answer)
}

@(private)
cmd_ingest :: proc(s: ^State, args: string) {
	if len(args) == 0 {
		fmt.println("usage: /ingest [-d <descriptor>] <text>")
		return
	}

	text := args
	descriptor_text := ""

	if strings.has_prefix(args, "-d ") {
		rest := args[3:]
		space := strings.index(rest, " ")
		if space < 0 {
			fmt.println("usage: /ingest -d <descriptor> <text>")
			return
		}
		desc_name := rest[:space]
		text = strings.trim_space(rest[space + 1:])

		d := graph.get_descriptor(s.g, desc_name)
		if d == nil {
			fmt.printf("error: descriptor \"%s\" not found (use /descriptor list)\n", desc_name)
			return
		}
		descriptor_text = d.text
	}

	if len(text) == 0 {
		fmt.println("usage: /ingest [-d <descriptor>] <text>")
		return
	}

	fmt.printf("ingesting %d bytes ...\n", len(text))
	added := ingest.ingest(s.g, s.p, text, ingest.DEFAULT_CONFIG, descriptor_text)
	if added < 0 {
		fmt.println("error: ingestion failed")
	} else {
		fmt.printf(
			"added %d thoughts (total: %d thoughts, %d edges)\n",
			added,
			graph.thought_count(s.g),
			graph.edge_count(s.g),
		)
	}
}

@(private)
cmd_descriptor :: proc(s: ^State, args: string) {
	if len(args) == 0 {
		fmt.println("usage: /descriptor <add|remove|list> [name] [text]")
		return
	}

	space := strings.index(args, " ")
	subcmd: string
	rest: string
	if space < 0 {
		subcmd = args
		rest = ""
	} else {
		subcmd = args[:space]
		rest = strings.trim_space(args[space + 1:])
	}

	switch subcmd {
	case "list":
		if graph.descriptor_count(s.g) == 0 {
			fmt.println("no descriptors.")
			return
		}
		fmt.printf("descriptors (%d):\n", graph.descriptor_count(s.g))
		for _, &d in s.g.descriptors {
			preview := d.text
			if len(preview) > 80 {
				preview = preview[:80]
			}
			fmt.printf("  %-20s  %s\n", d.name, preview)
		}

	case "add":
		if len(rest) == 0 {
			fmt.println("usage: /descriptor add <name> <text>")
			return
		}
		name_end := strings.index(rest, " ")
		if name_end < 0 {
			fmt.println("usage: /descriptor add <name> <text>")
			return
		}
		name := rest[:name_end]
		text := strings.trim_space(rest[name_end + 1:])
		if len(text) == 0 {
			fmt.println("error: descriptor text cannot be empty")
			return
		}
		graph.set_descriptor(s.g, name, text)
		graph.save(s.g, s.graph_path)
		fmt.printf("descriptor \"%s\" saved (%d bytes)\n", name, len(text))

	case "remove":
		if len(rest) == 0 {
			fmt.println("usage: /descriptor remove <name>")
			return
		}
		if graph.remove_descriptor(s.g, rest) {
			graph.save(s.g, s.graph_path)
			fmt.printf("descriptor \"%s\" removed\n", rest)
		} else {
			fmt.printf("descriptor \"%s\" not found\n", rest)
		}

	case:
		fmt.printf("unknown subcommand: %s (use add, remove, list)\n", subcmd)
	}
}

@(private)
cmd_purpose :: proc(s: ^State, text: string) {
	if len(text) == 0 {
		if len(s.g.purpose) > 0 {
			fmt.printf("purpose: \"%s\"\n", s.g.purpose)
		} else {
			fmt.println("purpose not set. usage: /purpose <text>")
		}
		return
	}

	graph.set_purpose(s.g, text)
	graph.save(s.g, s.graph_path)
	fmt.printf("purpose set to: \"%s\"\n", s.g.purpose)
	log.info("[repl] purpose set to: \"%s\"", s.g.purpose)
}

@(private)
cmd_thoughts :: proc(s: ^State, args: string) {
	limit := 10
	if len(args) > 0 {
		if v, ok := strconv.parse_int(args); ok {
			limit = v
		}
	}

	tc := graph.thought_count(s.g)
	if tc == 0 {
		fmt.println("no thoughts in graph.")
		return
	}

	fmt.printf("thoughts (%d total, showing up to %d):\n", tc, limit)

	count := 0
	for id, &t in s.g.thoughts {
		if count >= limit {
			break
		}
		display := t.text
		if len(display) > 120 {
			display = t.text[:120]
		}
		ts := format_unix_time(t.created_at)
		fmt.printf("  [%d] %s  %s\n", id, ts, display)
		count += 1
	}
}

@(private)
cmd_edges :: proc(s: ^State, args: string) {
	ec := graph.edge_count(s.g)
	if ec == 0 {
		fmt.println("no edges in graph.")
		return
	}

	if len(args) > 0 {
		if v, ok := strconv.parse_uint(args); ok {
			id := u64(v)
			thought := graph.get_thought(s.g, id)
			if thought == nil {
				fmt.printf("thought %d not found.\n", id)
				return
			}

			out := graph.outgoing(s.g, id)
			defer delete(out)
			inc := graph.incoming(s.g, id)
			defer delete(inc)

			fmt.printf("thought %d: \"%s\"\n", id, truncate(thought.text, 80))

			if len(out) > 0 {
				fmt.printf("  outgoing (%d):\n", len(out))
				for e in out {
					fmt.printf(
						"    → %d  w=%.2f  %s\n",
						e.target_id,
						e.weight,
						truncate(e.reasoning, 60),
					)
				}
			}
			if len(inc) > 0 {
				fmt.printf("  incoming (%d):\n", len(inc))
				for e in inc {
					fmt.printf(
						"    ← %d  w=%.2f  %s\n",
						e.source_id,
						e.weight,
						truncate(e.reasoning, 60),
					)
				}
			}
			if len(out) == 0 && len(inc) == 0 {
				fmt.println("  no edges.")
			}
			return
		}
	}

	limit := min(ec, 20)
	fmt.printf("edges (%d total, showing %d):\n", ec, limit)
	for i in 0 ..< limit {
		e := s.g.edges[i]
		fmt.printf(
			"  %d → %d  w=%.2f  %s\n",
			e.source_id,
			e.target_id,
			e.weight,
			truncate(e.reasoning, 60),
		)
	}
}

@(private)
cmd_find :: proc(s: ^State, query: string) {
	if len(query) == 0 {
		fmt.println("usage: /find <query>")
		return
	}

	if graph.thought_count(s.g) == 0 {
		fmt.println("graph is empty - nothing to find.")
		return
	}

	fmt.printf("finding: \"%s\" ...\n", query)

	embedding, embed_ok := s.p.embed_text(s.p, query)
	if !embed_ok {
		fmt.println("error: failed to embed query")
		return
	}

	results := graph.find_thoughts(s.g, &embedding, graph.cfg.default_find_k)
	defer delete(results)

	if len(results) == 0 {
		fmt.println("no results.")
		return
	}

	fmt.printf("results (%d):\n", len(results))
	for r in results {
		thought := graph.get_thought(s.g, r.id)
		if thought != nil {
			fmt.printf("  [%d] %.4f  %s\n", r.id, r.score, truncate(thought.text, 80))
		}
	}
}

@(private)
cmd_save :: proc(s: ^State) {
	fmt.printf("saving graph to %s ...\n", s.graph_path)
	if graph.save(s.g, s.graph_path) {
		fmt.printf(
			"saved (%d thoughts, %d edges)\n",
			graph.thought_count(s.g),
			graph.edge_count(s.g),
		)
	} else {
		fmt.println("error: failed to save graph")
	}
}

@(private)
cmd_quit :: proc(s: ^State) {
	fmt.println("shutting down ...")
	s.quit = true
}

@(private)
cmd_limbo :: proc(s: ^State) {
	if s.limbo == nil {
		fmt.println("limbo is disabled (set limbo_cluster_min > 0 in config)")
		return
	}
	tc := graph.thought_count(s.limbo)
	fmt.printf("limbo thoughts: %d\n", tc)
	if tc == 0 {
		fmt.println("(no unconnected thoughts yet)")
		return
	}
	n := min(10, tc)
	fmt.printf("latest %d thoughts in limbo:\n", n)
	i := 0
	for _, &t in s.limbo.thoughts {
		if i >= n {break}
		fmt.printf("  [%d] %s\n", t.id, truncate(t.text, 80))
		i += 1
	}
}

@(private)
prompt :: proc() {
	fmt.print("knod> ")
}

@(private)
print_banner :: proc() {
	fmt.println("knod interactive mode - type /help for commands")
}

@(private)
truncate :: proc(s: string, max_len: int) -> string {
	if len(s) <= max_len {
		return s
	}
	return s[:max_len]
}

@(private)
format_unix_time :: proc(ts: i64) -> string {
	t := time.unix(ts, 0)
	y, mon, d := time.date(t)
	h, m, _ := time.clock(t)
	return fmt.tprintf("%4d-%02d-%02d %02d:%02d", y, mon, d, h, m)
}

when ODIN_OS == .Windows {
	foreign import kernel32 "system:kernel32.lib"

	HANDLE :: rawptr
	DWORD :: u32

	@(default_calling_convention = "stdcall")
	foreign kernel32 {
		GetStdHandle :: proc(nStdHandle: DWORD) -> HANDLE ---
		GetNumberOfConsoleInputEvents :: proc(hConsoleInput: HANDLE, lpcNumberOfEvents: ^DWORD) -> DWORD ---
		GetConsoleMode :: proc(hConsoleHandle: HANDLE, lpMode: ^DWORD) -> DWORD ---
		SetConsoleMode :: proc(hConsoleHandle: HANDLE, dwMode: DWORD) -> DWORD ---
	}

	STD_INPUT_HANDLE :: DWORD(0xFFFFFFF6)

	@(private)
	stdin_has_input :: proc() -> bool {
		handle := GetStdHandle(STD_INPUT_HANDLE)
		count: DWORD
		if GetNumberOfConsoleInputEvents(handle, &count) != 0 {
			return count > 1
		}
		return false
	}
} else {
	foreign import libc "system:c"

	pollfd :: struct {
		fd:      i32,
		events:  i16,
		revents: i16,
	}

	POLLIN :: i16(0x0001)

	@(default_calling_convention = "c")
	foreign libc {
		poll :: proc(fds: ^pollfd, nfds: u64, timeout: i32) -> i32 ---
	}

	@(private)
	stdin_has_input :: proc() -> bool {
		pfd := pollfd {
			fd      = 0,
			events  = POLLIN,
			revents = 0,
		}
		result := poll(&pfd, 1, 0)
		return result > 0 && (pfd.revents & POLLIN) != 0
	}
}
