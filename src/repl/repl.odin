package repl

import "core:os"
import "core:strconv"
import "core:strings"
import "core:time"

import "../gnn"
import "../graph"
import "../ingest"
import log "../logger"
import "../provider"
import "../util"

State :: struct {
	line_buf:   [dynamic]u8,
	g:          ^graph.Graph,
	p:          ^provider.Provider,
	model:      ^gnn.MPNN,
	strand:     ^gnn.StrandMPNN,
	graph_path: string,
	limbo:      ^graph.Graph,
	quit:       bool,
	headless:   bool,
}

init :: proc(
	g: ^graph.Graph,
	p: ^provider.Provider,
	graph_path: string,
	model: ^gnn.MPNN = nil,
	strand: ^gnn.StrandMPNN = nil,
	limbo_graph: ^graph.Graph = nil,
) -> State {
	headless := !is_console_stdin()
	if !headless {
		print_banner()
		prompt()
	} else {
		log.info("headless mode (stdin is not a console)")
	}
	return State {
		line_buf = make([dynamic]u8, 0, 1024),
		g = g,
		p = p,
		model = model,
		strand = strand,
		graph_path = graph_path,
		limbo = limbo_graph,
		quit = false,
		headless = headless,
	}
}

destroy :: proc(s: ^State) {
	delete(s.line_buf)
}


poll :: proc(s: ^State) -> bool {
	if s.quit {
		return false
	}

	// In headless mode, never read stdin — just keep the main loop alive.
	if s.headless {
		return true
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
		log.raw("(hint: prefix with /ingest to ingest, or /ask to query)\n")
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
		log.rawf("unknown command: %s (type /help for a list)\n", cmd)
	}
}


@(private)
cmd_help :: proc() {
	log.raw("commands:\n")
	log.raw("  /help,       /h  - show this help\n")
	log.raw("  /status,     /s  - graph stats and purpose\n")
	log.raw("  /ask,        /a  - query the knowledge graph\n")
	log.raw("  /ingest,     /i  - ingest text (optional: -d <name>)\n")
	log.raw("  /descriptor, /d  - manage descriptors (add/remove/list)\n")
	log.raw("  /purpose,    /p  - view or set the node's purpose\n")
	log.raw("  /thoughts,   /t  - list thoughts (optional: count)\n")
	log.raw("  /edges,      /e  - list edges (optional: thought id)\n")
	log.raw("  /find,       /f  - find thoughts by similarity\n")
	log.raw("  /save            - force-save the full graph\n")
	log.raw("  /limbo,      /l  - show limbo stats\n")
	log.raw("  /quit,       /q  - exit knod\n")
}

@(private)
cmd_status :: proc(s: ^State) {
	tc := graph.thought_count(s.g)
	ec := graph.edge_count(s.g)
	log.rawf("thoughts: %d\n", tc)
	log.rawf("edges:    %d\n", ec)
	if len(s.g.purpose) > 0 {
		log.rawf("purpose:  \"%s\"\n", s.g.purpose)
	} else {
		log.raw("purpose:  (not set)\n")
	}
	log.rawf("graph:    %s\n", s.graph_path)
}

@(private)
cmd_ask :: proc(s: ^State, query: string) {
	if len(query) == 0 {
		log.raw("usage: /ask <question>\n")
		return
	}

	log.rawf("querying: \"%s\" ...\n", query)

	query_embedding, embed_ok := s.p.embed_text(s.p, query)
	if !embed_ok {
		log.err("failed to embed query")
		return
	}

	results := graph.find_thoughts(s.g, &query_embedding, graph.cfg.default_find_k)
	defer delete(results)

	if len(results) == 0 {
		log.raw("no relevant thoughts found.\n")
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
		log.err("failed to generate answer")
		return
	}
	defer delete(answer)

	log.raw("\n")
	log.rawf("%s\n", answer)
}

@(private)
cmd_ingest :: proc(s: ^State, args: string) {
	if len(args) == 0 {
		log.raw("usage: /ingest [-d <descriptor>] <text>\n")
		return
	}

	text := args
	descriptor_text := ""

	if strings.has_prefix(args, "-d ") {
		rest := args[3:]
		space := strings.index(rest, " ")
		if space < 0 {
			log.raw("usage: /ingest -d <descriptor> <text>\n")
			return
		}
		desc_name := rest[:space]
		text = strings.trim_space(rest[space + 1:])

		d := graph.get_descriptor(s.g, desc_name)
		if d == nil {
			log.rawf("error: descriptor \"%s\" not found (use /descriptor list)\n", desc_name)
			return
		}
		descriptor_text = d.text
	}

	if len(text) == 0 {
		log.raw("usage: /ingest [-d <descriptor>] <text>\n")
		return
	}

	log.rawf("ingesting %d bytes ...\n", len(text))
	added := ingest.ingest(s.g, s.p, text, ingest.DEFAULT_CONFIG, descriptor_text)
	if added < 0 {
		log.err("ingestion failed")
	} else {
		log.rawf(
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
		log.raw("usage: /descriptor <add|remove|list> [name] [text]\n")
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
			log.raw("no descriptors.\n")
			return
		}
		log.rawf("descriptors (%d):\n", graph.descriptor_count(s.g))
		for _, &d in s.g.descriptors {
			preview := d.text
			if len(preview) > 80 {
				preview = preview[:80]
			}
			log.rawf("  %-20s  %s\n", d.name, preview)
		}

	case "add":
		if len(rest) == 0 {
			log.raw("usage: /descriptor add <name> <text>\n")
			return
		}
		name_end := strings.index(rest, " ")
		if name_end < 0 {
			log.raw("usage: /descriptor add <name> <text>\n")
			return
		}
		name := rest[:name_end]
		text := strings.trim_space(rest[name_end + 1:])
		if len(text) == 0 {
			log.err("descriptor text cannot be empty")
			return
		}
		graph.set_descriptor(s.g, name, text)
		repl_save_graph(s)
		log.rawf("descriptor \"%s\" saved (%d bytes)\n", name, len(text))

	case "remove":
		if len(rest) == 0 {
			log.raw("usage: /descriptor remove <name>\n")
			return
		}
		if graph.remove_descriptor(s.g, rest) {
			repl_save_graph(s)
			log.rawf("descriptor \"%s\" removed\n", rest)
		} else {
			log.rawf("descriptor \"%s\" not found\n", rest)
		}

	case:
		log.rawf("unknown subcommand: %s (use add, remove, list)\n", subcmd)
	}
}

@(private)
cmd_purpose :: proc(s: ^State, text: string) {
	if len(text) == 0 {
		if len(s.g.purpose) > 0 {
			log.rawf("purpose: \"%s\"\n", s.g.purpose)
		} else {
			log.raw("purpose not set. usage: /purpose <text>\n")
		}
		return
	}

	graph.set_purpose(s.g, text)
	repl_save_graph(s)
	log.rawf("purpose set to: \"%s\"\n", s.g.purpose)
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
		log.raw("no thoughts in graph.\n")
		return
	}

	log.rawf("thoughts (%d total, showing up to %d):\n", tc, limit)

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
		log.rawf("  [%d] %s  %s\n", id, ts, display)
		count += 1
	}
}

@(private)
cmd_edges :: proc(s: ^State, args: string) {
	ec := graph.edge_count(s.g)
	if ec == 0 {
		log.raw("no edges in graph.\n")
		return
	}

	if len(args) > 0 {
		if v, ok := strconv.parse_uint(args); ok {
			id := u64(v)
			thought := graph.get_thought(s.g, id)
			if thought == nil {
				log.rawf("thought %d not found.\n", id)
				return
			}

			out := graph.outgoing(s.g, id)
			defer delete(out)
			inc := graph.incoming(s.g, id)
			defer delete(inc)

			log.rawf("thought %d: \"%s\"\n", id, truncate(thought.text, 80))

			if len(out) > 0 {
				log.rawf("  outgoing (%d):\n", len(out))
				for e in out {
					log.rawf(
						"    → %d  w=%.2f  %s\n",
						e.target_id,
						e.weight,
						truncate(e.reasoning, 60),
					)
				}
			}
			if len(inc) > 0 {
				log.rawf("  incoming (%d):\n", len(inc))
				for e in inc {
					log.rawf(
						"    ← %d  w=%.2f  %s\n",
						e.source_id,
						e.weight,
						truncate(e.reasoning, 60),
					)
				}
			}
			if len(out) == 0 && len(inc) == 0 {
				log.raw("  no edges.\n")
			}
			return
		}
	}

	limit := min(ec, 20)
	log.rawf("edges (%d total, showing %d):\n", ec, limit)
	for i in 0 ..< limit {
		e := s.g.edges[i]
		log.rawf(
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
		log.raw("usage: /find <query>\n")
		return
	}

	if graph.thought_count(s.g) == 0 {
		log.raw("graph is empty - nothing to find.\n")
		return
	}

	log.rawf("finding: \"%s\" ...\n", query)

	embedding, embed_ok := s.p.embed_text(s.p, query)
	if !embed_ok {
		log.err("failed to embed query")
		return
	}

	results := graph.find_thoughts(s.g, &embedding, graph.cfg.default_find_k)
	defer delete(results)

	if len(results) == 0 {
		log.raw("no results.\n")
		return
	}

	log.rawf("results (%d):\n", len(results))
	for r in results {
		thought := graph.get_thought(s.g, r.id)
		if thought != nil {
			log.rawf("  [%d] %.4f  %s\n", r.id, r.score, truncate(thought.text, 80))
		}
	}
}

@(private)
cmd_save :: proc(s: ^State) {
	log.rawf("saving graph to %s ...\n", s.graph_path)
	if repl_save_graph(s) {
		log.rawf(
			"saved (%d thoughts, %d edges)\n",
			graph.thought_count(s.g),
			graph.edge_count(s.g),
		)
	} else {
		log.err("failed to save graph")
	}
}

// repl_save_graph persists the graph in the correct format based on the file extension.
@(private)
repl_save_graph :: proc(s: ^State) -> bool {
	if strings.has_suffix(s.graph_path, util.STRAND_EXTENSION) {
		strand_bytes: []u8
		if s.strand != nil {
			strand_bytes = gnn.strand_save_bytes(s.strand)
		}
		if strand_bytes == nil {
			strand_bytes = make([]u8, 0)
		}
		ok := graph.save_strand(s.g, strand_bytes, s.graph_path)
		delete(strand_bytes)
		return ok
	} else {
		return graph.save(s.g, s.graph_path)
	}
}

@(private)
cmd_quit :: proc(s: ^State) {
	log.raw("shutting down ...\n")
	s.quit = true
}

@(private)
cmd_limbo :: proc(s: ^State) {
	if s.limbo == nil {
		log.raw("limbo is disabled (set limbo_cluster_min > 0 in config)\n")
		return
	}
	tc := graph.thought_count(s.limbo)
	log.rawf("limbo thoughts: %d\n", tc)
	if tc == 0 {
		log.raw("(no unconnected thoughts yet)\n")
		return
	}
	n := min(10, tc)
	log.rawf("latest %d thoughts in limbo:\n", n)
	i := 0
	for _, &t in s.limbo.thoughts {
		if i >= n {break}
		log.rawf("  [%d] %s\n", t.id, truncate(t.text, 80))
		i += 1
	}
}

@(private)
prompt :: proc() {
	log.raw("knod> ")
}

@(private)
print_banner :: proc() {
	log.raw("knod interactive mode - type /help for commands\n")
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

	buf: [16]u8
	buf[0] = u8('0' + (y / 1000) % 10)
	buf[1] = u8('0' + (y / 100) % 10)
	buf[2] = u8('0' + (y / 10) % 10)
	buf[3] = u8('0' + y % 10)
	buf[4] = '-'
	buf[5] = u8('0' + int(mon) / 10)
	buf[6] = u8('0' + int(mon) % 10)
	buf[7] = '-'
	buf[8] = u8('0' + d / 10)
	buf[9] = u8('0' + d % 10)
	buf[10] = ' '
	buf[11] = u8('0' + h / 10)
	buf[12] = u8('0' + h % 10)
	buf[13] = ':'
	buf[14] = u8('0' + m / 10)
	buf[15] = u8('0' + m % 10)
	return strings.clone(string(buf[:]))
}

when ODIN_OS == .Windows {
	foreign import kernel32 "system:kernel32.lib"

	HANDLE :: rawptr
	DWORD :: u32

	@(default_calling_convention = "stdcall")
	foreign kernel32 {
		GetStdHandle :: proc(nStdHandle: DWORD) -> HANDLE ---
		GetNumberOfConsoleInputEvents :: proc(hConsoleInput: HANDLE, lpcNumberOfEvents: ^DWORD) -> DWORD ---
		PeekConsoleInputW :: proc(hConsoleInput: HANDLE, lpBuffer: ^INPUT_RECORD, nLength: DWORD, lpNumberOfEventsRead: ^DWORD) -> DWORD ---
		GetConsoleMode :: proc(hConsoleHandle: HANDLE, lpMode: ^DWORD) -> DWORD ---
		SetConsoleMode :: proc(hConsoleHandle: HANDLE, dwMode: DWORD) -> DWORD ---
	}

	KEY_EVENT :: DWORD(0x0001)

	INPUT_RECORD :: struct {
		event_type: u16,
		_padding:   u16,
		event:      [16]u8, // union, we only check event_type
	}

	STD_INPUT_HANDLE :: DWORD(0xFFFFFFF6)

	@(private)
	stdin_has_input :: proc() -> bool {
		handle := GetStdHandle(STD_INPUT_HANDLE)
		count: DWORD
		if GetNumberOfConsoleInputEvents(handle, &count) == 0 || count == 0 {
			return false
		}
		// Peek at events to check for actual key events (not window/focus/mouse).
		buf: [8]INPUT_RECORD
		peek_count: DWORD
		n := min(DWORD(len(buf)), count)
		if PeekConsoleInputW(handle, &buf[0], n, &peek_count) == 0 {
			return false
		}
		for i in 0 ..< peek_count {
			if buf[i].event_type == u16(KEY_EVENT) {
				return true
			}
		}
		return false
	}

	@(private)
	is_console_stdin :: proc() -> bool {
		handle := GetStdHandle(STD_INPUT_HANDLE)
		mode: DWORD
		return GetConsoleMode(handle, &mode) != 0
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
		isatty :: proc(fd: i32) -> i32 ---
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

	@(private)
	is_console_stdin :: proc() -> bool {
		return isatty(0) != 0
	}
}
