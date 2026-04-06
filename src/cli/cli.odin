package cli

import "core:fmt"
import "core:mem"
import "core:os"
import "core:path/filepath"
import "core:strings"

import "../gnn"
import "../graph"
import "../registry"
import "../util"

// Use graph's canonical constants instead of local redefinitions.
GRAPH_MAGIC :: graph.LOG_MAGIC
GRAPH_VERSION :: graph.LOG_VERSION
KNOD_MAGIC :: graph.KNOD_MAGIC

Action :: enum {
	RUN,
	EXIT,
	ASK,
}

Command :: struct {
	action:     Action,
	knid_scope: string,
	query:      string,
	no_limbo:   bool,
}

parse_and_dispatch :: proc() -> Command {
	cmd := Command {
		action = .RUN,
	}

	args := os.args[1:]
	if len(args) == 0 {
		return cmd
	}

	filtered := make([dynamic]string, 0, len(args))
	defer delete(filtered)

	internal_query := false
	graph_path: string
	query_text: string

	for arg in args {
		if strings.has_prefix(arg, "--knid=") {
			cmd.knid_scope = arg[7:]
		} else if arg == "--internal-query" {
			internal_query = true
		} else if strings.has_prefix(arg, "--graph=") {
			graph_path = arg[8:]
		} else if strings.has_prefix(arg, "--query=") {
			query_text = arg[8:]
		} else if arg == "--no-limbo" || arg == "-nl" {
			cmd.no_limbo = true
		} else {
			append(&filtered, arg)
		}
	}


	if internal_query {
		if len(graph_path) == 0 || len(query_text) == 0 {
			fmt.println("usage: knod --internal-query --graph=<path> --query=<text>")
			cmd.action = .EXIT
			return cmd
		}
		cmd_internal_query(graph_path, query_text)
		cmd.action = .EXIT
		return cmd
	}

	if len(filtered) == 0 {

		return cmd
	}

	subcmd := filtered[0]
	rest := filtered[1:]

	switch subcmd {
	case "new":
		cmd_new(cmd.knid_scope)
		cmd.action = .EXIT
	case "register":
		if len(rest) < 1 {
			fmt.println("usage: knod register <path>")
			cmd.action = .EXIT
		} else {
			cmd_register(rest[0], cmd.knid_scope)
			cmd.action = .EXIT
		}
	case "list":
		cmd_list(cmd.knid_scope)
		cmd.action = .EXIT
	case "knid":
		cmd_knid(rest)
		cmd.action = .EXIT
	case "ask":
		if len(rest) < 1 {
			fmt.println("usage: knod [--knid=<name>] ask <query>")
			cmd.action = .EXIT
		} else {

			cmd.query = strings.join(rest, " ")
			cmd.action = .ASK
		}
	case:
		fmt.printf("unknown command: %s\n", subcmd)
		print_usage()
		cmd.action = .EXIT
	}

	return cmd
}


cmd_new :: proc(knid_scope: string) {

	fmt.print("purpose: ")
	purpose := read_line()
	if len(purpose) == 0 {
		fmt.println("error: purpose cannot be empty")
		return
	}
	defer delete(purpose)


	default_name := derive_name(purpose)
	defer delete(default_name)

	fmt.printf("name [%s]: ", default_name)
	name_input := read_line()
	name: string
	if len(name_input) == 0 {
		name = strings.clone(default_name)
	} else {
		name = name_input
	}


	reg, _ := registry.load()
	defer registry.release(&reg)

	if registry.find_store(&reg, name) != nil {
		fmt.printf("error: store '%s' already exists\n", name)
		if len(name_input) == 0 {

		}
		delete(name)
		return
	}


	default_path := default_store_path(name)
	defer delete(default_path)

	fmt.printf("location [%s]: ", default_path)
	path_input := read_line()
	store_path: string
	if len(path_input) == 0 {
		store_path = strings.clone(default_path)
	} else {
		store_path = path_input
	}


	dir := filepath.dir(store_path)
	defer delete(dir)
	ensure_dir(dir)


	if !create_empty_strand(store_path, purpose) {
		fmt.printf("error: could not create store at %s\n", store_path)
		delete(name)
		delete(store_path)
		return
	}


	abs_path: string
	if filepath.is_abs(store_path) {
		abs_path = strings.clone(store_path)
	} else {
		abs_path = abs_or_clone(store_path)
	}
	delete(store_path)
	defer delete(abs_path)


	registry.add_store(&reg, name, abs_path)


	if len(knid_scope) > 0 {
		if registry.find_knid(&reg, knid_scope) == nil {
			registry.add_knid(&reg, knid_scope)
		}
		registry.knid_add_store(&reg, knid_scope, name)
	}

	if !registry.save(&reg) {
		fmt.println("error: could not save registry")
		delete(name)
		return
	}

	fmt.printf("created store '%s' at %s\n", name, abs_path)
	if len(knid_scope) > 0 {
		fmt.printf("added to knid '%s'\n", knid_scope)
	}
	delete(name)
}


cmd_register :: proc(path: string, knid_scope: string) {
	if !os.exists(path) {
		fmt.printf("error: file not found: %s\n", path)
		return
	}


	purpose, valid := read_store_header(path)
	if !valid {
		fmt.printf("error: %s is not a valid knod store file\n", path)
		return
	}
	defer if len(purpose) > 0 {delete(purpose)}


	base := filepath.stem(path)


	abs_path := abs_or_clone(path)
	defer delete(abs_path)

	reg, _ := registry.load()
	defer registry.release(&reg)

	name := strings.clone(base)

	if registry.find_store(&reg, name) != nil {
		fmt.printf("error: store '%s' already registered\n", name)
		delete(name)
		return
	}

	registry.add_store(&reg, name, abs_path)


	if len(knid_scope) > 0 {
		if registry.find_knid(&reg, knid_scope) == nil {
			registry.add_knid(&reg, knid_scope)
		}
		registry.knid_add_store(&reg, knid_scope, name)
	}

	if !registry.save(&reg) {
		fmt.println("error: could not save registry")
		delete(name)
		return
	}

	if len(purpose) > 0 {
		fmt.printf("registered '%s' (%s)\n  purpose: %s\n", name, abs_path, purpose)
	} else {
		fmt.printf("registered '%s' (%s)\n  no purpose set\n", name, abs_path)
	}
	if len(knid_scope) > 0 {
		fmt.printf("added to knid '%s'\n", knid_scope)
	}
	delete(name)
}


cmd_list :: proc(knid_scope: string) {
	reg, _ := registry.load()
	defer registry.release(&reg)

	if len(knid_scope) > 0 {

		k := registry.find_knid(&reg, knid_scope)
		if k == nil {
			fmt.printf("error: knid '%s' not found\n", knid_scope)
			return
		}
		fmt.printf("knid '%s':\n", knid_scope)
		stores := registry.knid_stores(&reg, knid_scope)
		defer delete(stores)
		if len(stores) == 0 {
			fmt.println("  (empty)")
		} else {
			for s in stores {
				fmt.printf("  %s = %s\n", s.name, s.path)
			}
		}
	} else {

		if len(reg.stores) == 0 {
			fmt.println("no stores registered")
			return
		}
		fmt.println("stores:")
		for &s in reg.stores {
			fmt.printf("  %s = %s\n", s.name, s.path)
		}


		if len(reg.knids) > 0 {
			fmt.println("\nknids:")
			for &k in reg.knids {
				members := strings.join(k.members[:], ", ")
				defer delete(members)
				fmt.printf("  [%s] %s\n", k.name, members)
			}
		}
	}
}


cmd_knid :: proc(args: []string) {
	if len(args) == 0 {
		fmt.println("usage: knod knid <new|add|remove|list> [args...]")
		return
	}

	sub := args[0]
	rest := args[1:]

	switch sub {
	case "new":
		if len(rest) < 1 {
			fmt.println("usage: knod knid new <name>")
			return
		}
		knid_new(rest[0])
	case "add":
		if len(rest) < 2 {
			fmt.println("usage: knod knid add <knid> <store>")
			return
		}
		knid_add(rest[0], rest[1])
	case "remove":
		if len(rest) < 2 {
			fmt.println("usage: knod knid remove <knid> <store>")
			return
		}
		knid_remove(rest[0], rest[1])
	case "list":
		if len(rest) >= 1 {
			knid_list_one(rest[0])
		} else {
			knid_list_all()
		}
	case:
		fmt.printf("unknown knid command: %s\n", sub)
		fmt.println("usage: knod knid <new|add|remove|list> [args...]")
	}
}

knid_new :: proc(name: string) {
	reg, _ := registry.load()
	defer registry.release(&reg)

	if !registry.add_knid(&reg, name) {
		fmt.printf("error: knid '%s' already exists\n", name)
		return
	}

	if !registry.save(&reg) {
		fmt.println("error: could not save registry")
		return
	}

	fmt.printf("created knid '%s'\n", name)
}

knid_add :: proc(knid_name, store_name: string) {
	reg, _ := registry.load()
	defer registry.release(&reg)

	if registry.find_knid(&reg, knid_name) == nil {
		fmt.printf("error: knid '%s' not found\n", knid_name)
		return
	}

	if registry.find_store(&reg, store_name) == nil {
		fmt.printf("error: store '%s' not registered\n", store_name)
		return
	}

	if !registry.knid_add_store(&reg, knid_name, store_name) {
		fmt.printf("error: '%s' is already in knid '%s'\n", store_name, knid_name)
		return
	}

	if !registry.save(&reg) {
		fmt.println("error: could not save registry")
		return
	}

	fmt.printf("added '%s' to knid '%s'\n", store_name, knid_name)
}

knid_remove :: proc(knid_name, store_name: string) {
	reg, _ := registry.load()
	defer registry.release(&reg)

	if !registry.knid_remove_store(&reg, knid_name, store_name) {
		fmt.printf("error: could not remove '%s' from knid '%s'\n", store_name, knid_name)
		return
	}

	if !registry.save(&reg) {
		fmt.println("error: could not save registry")
		return
	}

	fmt.printf("removed '%s' from knid '%s'\n", store_name, knid_name)
}

knid_list_all :: proc() {
	reg, _ := registry.load()
	defer registry.release(&reg)

	if len(reg.knids) == 0 {
		fmt.println("no knids defined")
		return
	}

	for &k in reg.knids {
		members := strings.join(k.members[:], ", ")
		defer delete(members)
		fmt.printf("[%s] %s\n", k.name, members)
	}
}

knid_list_one :: proc(name: string) {
	reg, _ := registry.load()
	defer registry.release(&reg)

	k := registry.find_knid(&reg, name)
	if k == nil {
		fmt.printf("error: knid '%s' not found\n", name)
		return
	}

	stores := registry.knid_stores(&reg, name)
	defer delete(stores)

	if len(stores) == 0 {
		fmt.printf("knid '%s': (empty)\n", name)
	} else {
		fmt.printf("knid '%s':\n", name)
		for s in stores {
			fmt.printf("  %s = %s\n", s.name, s.path)
		}
	}
}


cmd_internal_query :: proc(graph_path, query_text: string) {

	g: graph.Graph
	graph.create(&g)
	defer graph.release(&g)

	is_strand := strings.has_suffix(graph_path, util.STRAND_EXTENSION)
	strand_data: []u8
	strand_offset := 0
	loaded := false

	if is_strand {
		file_data, s_off, s_ok := graph.load_strand_bytes(graph_path)
		if s_ok {
			strand_data = file_data
			strand_offset = s_off
			g2: graph.Graph
			graph.create(&g2)
			_, g_ok := graph.load_strand(&g2, graph_path)
			if g_ok {
				graph.release(&g)
				g = g2
				loaded = true
			} else {
				graph.release(&g2)
			}
		}
	}

	if !loaded {
		if !graph.load(&g, graph_path) {
			fmt.eprintln("error: could not load graph:", graph_path)
			if strand_data != nil {delete(strand_data)}
			return
		}
	}

	if graph.thought_count(&g) == 0 {
		if strand_data != nil {delete(strand_data)}
		return
	}

	embedding_bytes: [graph.EMBEDDING_DIM * size_of(f32)]u8
	total_read := 0
	for total_read < len(embedding_bytes) {
		n, err := os.read(os.stdin, embedding_bytes[total_read:])
		if err != os.ERROR_NONE || n <= 0 {
			break
		}
		total_read += n
	}

	if total_read != len(embedding_bytes) {
		fmt.eprintln("error: expected", len(embedding_bytes), "bytes on stdin, got", total_read)
		if strand_data != nil {delete(strand_data)}
		return
	}

	query_embedding := (^graph.Embedding)(raw_data(embedding_bytes[:]))^
	K := graph.cfg.default_find_k

	base_path := base_model_path()
	defer delete(base_path)

	base_model: gnn.MPNN
	has_gnn := false

	if graph.thought_count(&g) >= 2 && graph.edge_count(&g) > 0 {
		if gnn.load_checkpoint(&base_model, base_path) {
			has_gnn = true
		} else {
			gnn.create(&base_model)
			has_gnn = true
		}
	}
	if has_gnn {defer gnn.release(&base_model)}

	strand_model: gnn.StrandMPNN
	has_strand := false

	if has_gnn && strand_data != nil && strand_offset > 0 {
		off := strand_offset
		if gnn.strand_load(&strand_model, strand_data, &off) {
			has_strand = true
		}
	}
	if strand_data != nil {delete(strand_data)}
	if has_strand {defer gnn.strand_release(&strand_model)}

	if !has_strand && has_gnn {
		gnn.strand_create(&strand_model, base_model.hidden_dim)
		has_strand = true
	}

	seen: map[u64]f32
	defer delete(seen)

	if has_gnn {
		snap := gnn.build_snapshot(&g)
		defer gnn.release_snapshot(&snap)

		gnn_emb: gnn.Embedding
		for i in 0 ..< gnn.EMBEDDING_DIM {gnn_emb[i] = query_embedding[i]}

		strand_ptr: ^gnn.StrandMPNN = nil
		if has_strand {strand_ptr = &strand_model}

		gnn_results := gnn.score_nodes(&base_model, strand_ptr, &snap, &gnn_emb, K)
		defer delete(gnn_results)

		for result in gnn_results {
			existing, found := seen[result.node_id]
			if !found || result.score > existing {
				seen[result.node_id] = result.score
			}
		}
	}

	cosine_results := graph.find_thoughts(&g, &query_embedding, K)
	defer delete(cosine_results)

	for result in cosine_results {
		existing, found := seen[result.id]
		if !found || result.score > existing {
			seen[result.id] = result.score
		}
	}

	edge_results := graph.find_edges(&g, &query_embedding, K)
	defer delete(edge_results)

	for er in edge_results {
		edge := &g.edges[er.edge_index]
		edge_score := er.score * util.EDGE_SCORE_DISCOUNT

		src_existing, src_found := seen[edge.source_id]
		if !src_found || edge_score > src_existing {
			seen[edge.source_id] = edge_score
		}
		dst_existing, dst_found := seen[edge.target_id]
		if !dst_found || edge_score > dst_existing {
			seen[edge.target_id] = edge_score
		}
	}

	ranked := make([dynamic]graph.findResult, 0, len(seen))
	defer delete(ranked)
	for id, score in seen {
		append(&ranked, graph.findResult{id = id, score = score})
	}

	for i in 1 ..< len(ranked) {
		j := i
		for j > 0 && ranked[j].score > ranked[j - 1].score {
			ranked[j], ranked[j - 1] = ranked[j - 1], ranked[j]
			j -= 1
		}
	}

	n := min(K, len(ranked))
	for i in 0 ..< n {
		thought := graph.get_thought(&g, ranked[i].id)
		if thought != nil {
			escaped := escape_newlines(thought.text)
			defer delete(escaped)
			fmt.printf("T\t%.6f\t%d\t%s\n", ranked[i].score, ranked[i].id, escaped)
		}
	}

	edge_context_count := 0
	for er in edge_results {
		if edge_context_count >= graph.cfg.max_context_edges {break}
		edge := &g.edges[er.edge_index]
		if len(edge.reasoning) > 0 {
			escaped := escape_newlines(edge.reasoning)
			defer delete(escaped)
			fmt.printf("E\t%.6f\t%s\n", er.score, escaped)
			edge_context_count += 1
		}
	}
}


get_store_paths :: proc(knid_scope: string) -> []registry.Store {
	reg, _ := registry.load()
	defer registry.release(&reg)

	if len(knid_scope) > 0 {
		stores := registry.knid_stores(&reg, knid_scope)
		if stores == nil {
			fmt.printf("error: knid '%s' not found\n", knid_scope)
			return {}
		}

		result := make([]registry.Store, len(stores))
		for i in 0 ..< len(stores) {
			result[i] = registry.Store {
				name = strings.clone(stores[i].name),
				path = strings.clone(stores[i].path),
			}
		}
		delete(stores)
		return result
	}


	result := make([]registry.Store, len(reg.stores))
	for i in 0 ..< len(reg.stores) {
		result[i] = registry.Store {
			name = strings.clone(reg.stores[i].name),
			path = strings.clone(reg.stores[i].path),
		}
	}
	return result
}


release_store_list :: proc(stores: []registry.Store) {
	for &s in stores {
		delete(s.name)
		delete(s.path)
	}
	delete(stores)
}


@(private)
escape_newlines :: proc(s: string) -> string {
	b := strings.builder_make()
	for c in s {
		switch c {
		case '\n':
			strings.write_string(&b, "\\n")
		case '\r':
			strings.write_string(&b, "\\r")
		case '\t':
			strings.write_string(&b, "\\t")
		case '\\':
			strings.write_string(&b, "\\\\")
		case:
			strings.write_rune(&b, c)
		}
	}
	return strings.to_string(b)
}


unescape_line :: proc(s: string) -> string {
	b := strings.builder_make()
	i := 0
	for i < len(s) {
		if i + 1 < len(s) && s[i] == '\\' {
			switch s[i + 1] {
			case 'n':
				strings.write_rune(&b, '\n')
				i += 2
				continue
			case 'r':
				strings.write_rune(&b, '\r')
				i += 2
				continue
			case 't':
				strings.write_rune(&b, '\t')
				i += 2
				continue
			case '\\':
				strings.write_rune(&b, '\\')
				i += 2
				continue
			}
		}
		strings.write_byte(&b, s[i])
		i += 1
	}
	return strings.to_string(b)
}


gnn_checkpoint_path :: proc(graph_path_arg: string) -> string {
	for i := len(graph_path_arg) - 1; i >= 0; i -= 1 {
		if graph_path_arg[i] == '.' {
			prefix := strings.clone(graph_path_arg[:i])
			defer delete(prefix)
			return strings.concatenate({prefix, ".gnn"})
		}
	}
	return strings.concatenate({graph_path_arg, ".gnn"})
}


print_usage :: proc() {
	fmt.println("usage: knod [options] <command>")
	fmt.println("")
	fmt.println("options:")
	fmt.println("  --knid=<name>              scope to a named collection")
	fmt.println("  --no-limbo, -nl            disable the limbo holding graph")
	fmt.println("")
	fmt.println("commands:")
	fmt.println("  (none)                     start the main process")
	fmt.println("  new                        create a new store")
	fmt.println("  register <path>            register an existing store")
	fmt.println("  list                       list registered stores")
	fmt.println("  ask <query>                query across registered stores")
	fmt.println("")
	fmt.println("  knid new <name>            create a named collection")
	fmt.println("  knid add <knid> <store>    add a store to a knid")
	fmt.println("  knid remove <knid> <store> remove a store from a knid")
	fmt.println("  knid list [knid]           list knids or stores in a knid")
}


read_line :: proc() -> string {
	buf: [1024]u8
	n, err := os.read(os.stdin, buf[:])
	if err != os.ERROR_NONE || n <= 0 {
		return ""
	}

	line := string(buf[:n])
	line = strings.trim_right(line, "\r\n")
	return strings.clone(line)
}


derive_name :: proc(purpose: string) -> string {
	lower := strings.to_lower(purpose)
	defer delete(lower)

	result := make([dynamic]u8, 0, len(lower))
	prev_dash := false

	for c in lower {
		if c >= 'a' && c <= 'z' || c >= '0' && c <= '9' {
			append(&result, u8(c))
			prev_dash = false
		} else if !prev_dash && len(result) > 0 {
			append(&result, '-')
			prev_dash = true
		}
	}


	if len(result) > 0 && result[len(result) - 1] == '-' {
		pop(&result)
	}

	s := strings.clone(string(result[:]))
	delete(result)
	return s
}


default_store_path :: proc(name: string) -> string {
	home := home_dir()
	if len(home) == 0 {
		return strings.clone(name)
	}
	defer delete(home)
	return filepath.join({home, ".config", "knod", "data", fmt.tprintf("%s.strand", name)})
}


create_empty_graph :: proc(path: string, purpose: string) -> bool {
	fd, err := os.open(path, os.O_WRONLY | os.O_CREATE | os.O_TRUNC, 0o644)
	if err != os.ERROR_NONE {
		return false
	}
	defer os.close(fd)

	magic: i32 = GRAPH_MAGIC
	if !write_val(fd, &magic) {return false}

	version: i32 = GRAPH_VERSION
	if !write_val(fd, &version) {return false}

	next_id: u64 = 1
	if !write_val(fd, &next_id) {return false}

	slen := i32(len(purpose))
	if !write_val(fd, &slen) {return false}
	if slen > 0 {
		_, werr := os.write(fd, transmute([]u8)purpose)
		if werr != os.ERROR_NONE {return false}
	}

	profile_count: u64 = 0
	if !write_val(fd, &profile_count) {return false}

	tag_count: u16 = 0
	if !write_val(fd, &tag_count) {return false}

	return true
}

create_empty_strand :: proc(path: string, purpose: string) -> bool {
	fd, err := os.open(path, os.O_WRONLY | os.O_CREATE | os.O_TRUNC, 0o644)
	if err != os.ERROR_NONE {
		return false
	}
	defer os.close(fd)

	knod_magic: i32 = KNOD_MAGIC
	knod_version: i32 = graph.KNOD_VERSION
	if !write_val(fd, &knod_magic) {return false}
	if !write_val(fd, &knod_version) {return false}

	sec_graph := graph.SECTION_GRAPH
	if !write_val(fd, &sec_graph) {return false}
	graph_len_placeholder: i64 = 0
	if !write_val(fd, &graph_len_placeholder) {return false}

	graph_start: i64 = 17

	log_magic: i32 = graph.LOG_MAGIC
	log_version: i32 = graph.LOG_VERSION
	if !write_val(fd, &log_magic) {return false}
	if !write_val(fd, &log_version) {return false}

	next_id: u64 = 1
	if !write_val(fd, &next_id) {return false}

	slen := i32(len(purpose))
	if !write_val(fd, &slen) {return false}
	if slen > 0 {
		_, werr := os.write(fd, transmute([]u8)purpose)
		if werr != os.ERROR_NONE {return false}
	}

	profile_count: u64 = 0
	if !write_val(fd, &profile_count) {return false}

	tag_count: u16 = 0
	if !write_val(fd, &tag_count) {return false}

	desc_count: u16 = 0
	if !write_val(fd, &desc_count) {return false}

	graph_end, seek_err1 := os.seek(fd, 0, os.SEEK_CUR)
	if seek_err1 != os.ERROR_NONE {return false}
	graph_section_len := graph_end - graph_start

	_, seek_err2 := os.seek(fd, 9, os.SEEK_SET)
	if seek_err2 != os.ERROR_NONE {return false}
	if !write_val(fd, &graph_section_len) {return false}
	_, seek_err3 := os.seek(fd, graph_end, os.SEEK_SET)
	if seek_err3 != os.ERROR_NONE {return false}

	sec_strand := graph.SECTION_STRAND
	if !write_val(fd, &sec_strand) {return false}
	strand_len: i64 = 0
	if !write_val(fd, &strand_len) {return false}

	return true
}

read_store_header :: proc(path: string) -> (purpose: string, ok: bool) {
	data, read_ok := os.read_entire_file(path)
	if !read_ok {
		return "", false
	}
	defer delete(data)

	if len(data) < 8 {
		return "", false
	}

	off := 0
	magic := (^i32)(raw_data(data[off:]))^

	if magic == KNOD_MAGIC {
		off += 4
		off += 4

		for off < len(data) {
			if off + 1 > len(data) {break}
			section_tag := data[off]
			off += 1
			if off + 8 > len(data) {break}
			section_len := int((^i64)(raw_data(data[off:]))^)
			off += 8

			if section_tag == graph.SECTION_GRAPH {
				return read_graph_purpose_from_bytes(data[off:off + section_len])
			}
			off += section_len
		}
		return "", true
	}

	if magic == GRAPH_MAGIC {
		return read_graph_header(path)
	}

	return "", false
}


read_graph_header :: proc(path: string) -> (purpose: string, ok: bool) {
	data, read_ok := os.read_entire_file(path)
	if !read_ok {
		return "", false
	}
	defer delete(data)

	if len(data) < 16 {
		return "", false
	}

	off := 0

	magic := (^i32)(raw_data(data[off:]))^
	off += 4
	if magic != GRAPH_MAGIC {
		return "", false
	}

	version := (^i32)(raw_data(data[off:]))^
	off += 4
	if version != GRAPH_VERSION {
		return "", false
	}

	off += 8

	if off + 4 > len(data) {
		return "", true
	}
	slen := int((^i32)(raw_data(data[off:]))^)
	off += 4
	if slen <= 0 || off + slen > len(data) {
		return "", true
	}

	buf := make([]u8, slen)
	copy(buf, data[off:off + slen])
	return string(buf), true
}

@(private)
read_graph_purpose_from_bytes :: proc(data: []u8) -> (purpose: string, ok: bool) {
	if len(data) < 16 {
		return "", false
	}

	off := 0
	magic := (^i32)(raw_data(data[off:]))^
	off += 4
	if magic != graph.LOG_MAGIC {
		return "", false
	}
	off += 4
	off += 8

	if off + 4 > len(data) {
		return "", true
	}
	slen := int((^i32)(raw_data(data[off:]))^)
	off += 4
	if slen <= 0 || off + slen > len(data) {
		return "", true
	}

	buf := make([]u8, slen)
	copy(buf, data[off:off + slen])
	return string(buf), true
}

base_model_path :: proc() -> string {
	home := home_dir()
	if len(home) == 0 {
		return strings.clone("knod")
	}
	defer delete(home)
	return filepath.join({home, ".config", "knod", "knod"})
}

@(private)
write_val :: proc(fd: os.Handle, val: ^$T) -> bool {
	bytes := mem.slice_ptr(cast(^u8)val, size_of(T))
	_, err := os.write(fd, bytes)
	return err == os.ERROR_NONE
}

@(private)
home_dir :: proc() -> string {
	return util.home_dir()
}

@(private)
ensure_dir :: proc(path: string) {
	util.ensure_dir(path)
}

@(private)
abs_or_clone :: proc(path: string) -> string {
	abs, ok := filepath.abs(path)
	if ok {
		return abs
	}
	return strings.clone(path)
}
