package registry

import "core:fmt"
import "core:os"
import "core:path/filepath"
import "core:strings"

import "../util"

Store :: struct {
	name: string,
	path: string,
}

Knid :: struct {
	name:    string,
	members: [dynamic]string,
}

Registry :: struct {
	stores: [dynamic]Store,
	knids:  [dynamic]Knid,
}

release :: proc(r: ^Registry) {
	for &s in r.stores {
		delete(s.name)
		delete(s.path)
	}
	delete(r.stores)
	for &k in r.knids {
		delete(k.name)
		for &m in k.members {
			delete(m)
		}
		delete(k.members)
	}
	delete(r.knids)
}

stores_path :: proc() -> string {
	home := home_dir()
	if len(home) == 0 {
		return ""
	}
	defer delete(home)
	return filepath.join({home, ".config", "knod", "stores"})
}

limbo_path :: proc() -> string {
	home := home_dir()
	if len(home) == 0 {
		return ""
	}
	defer delete(home)
	return filepath.join({home, ".config", "knod", "data", "limbo.graph"})
}

load :: proc() -> (r: Registry, ok: bool) {
	path := stores_path()
	if len(path) == 0 {
		return {}, false
	}
	defer delete(path)

	data, read_ok := os.read_entire_file(path)
	if !read_ok {

		r.stores = make([dynamic]Store)
		r.knids = make([dynamic]Knid)
		return r, false
	}
	defer delete(data)

	return parse(string(data))
}

parse :: proc(content: string) -> (r: Registry, ok: bool) {
	r.stores = make([dynamic]Store)
	r.knids = make([dynamic]Knid)

	lines := strings.split(content, "\n")
	defer delete(lines)

	current_knid: int = -1

	for line in lines {
		trimmed := strings.trim_space(line)

		if len(trimmed) == 0 || trimmed[0] == '#' {
			continue
		}

		if trimmed[0] == '[' {
			end := strings.index(trimmed, "]")
			if end > 1 {
				knid_name := strings.clone(strings.trim_space(trimmed[1:end]))
				append(&r.knids, Knid{name = knid_name, members = make([dynamic]string)})
				current_knid = len(r.knids) - 1
			}
			continue
		}

		if current_knid >= 0 {

			append(&r.knids[current_knid].members, strings.clone(trimmed))
		} else {

			eq_idx := strings.index(trimmed, "=")
			if eq_idx < 0 {
				continue
			}
			name := strings.clone(strings.trim_space(trimmed[:eq_idx]))
			path := strings.clone(strings.trim_space(trimmed[eq_idx + 1:]))
			append(&r.stores, Store{name = name, path = path})
		}
	}

	return r, true
}

save :: proc(r: ^Registry) -> bool {
	path := stores_path()
	if len(path) == 0 {
		return false
	}
	defer delete(path)

	dir := filepath.dir(path)
	defer delete(dir)
	ensure_dir(dir)

	content := serialize(r)
	defer delete(content)

	return os.write_entire_file(path, transmute([]u8)content)
}

serialize :: proc(r: ^Registry) -> string {
	b := strings.builder_make()

	for &s in r.stores {
		fmt.sbprintf(&b, "%s = %s\n", s.name, s.path)
	}

	for &k in r.knids {
		fmt.sbprintf(&b, "\n[%s]\n", k.name)
		for &m in k.members {
			fmt.sbprintf(&b, "%s\n", m)
		}
	}

	return strings.to_string(b)
}

find_store :: proc(r: ^Registry, name: string) -> ^Store {
	for &s in r.stores {
		if s.name == name {
			return &s
		}
	}
	return nil
}

find_knid :: proc(r: ^Registry, name: string) -> ^Knid {
	for &k in r.knids {
		if k.name == name {
			return &k
		}
	}
	return nil
}

add_store :: proc(r: ^Registry, name, path: string) -> bool {
	if find_store(r, name) != nil {
		return false
	}
	append(&r.stores, Store{name = strings.clone(name), path = strings.clone(path)})
	return true
}

add_knid :: proc(r: ^Registry, name: string) -> bool {
	if find_knid(r, name) != nil {
		return false
	}
	append(&r.knids, Knid{name = strings.clone(name), members = make([dynamic]string)})
	return true
}

knid_add_store :: proc(r: ^Registry, knid_name, store_name: string) -> bool {
	k := find_knid(r, knid_name)
	if k == nil {
		return false
	}

	if find_store(r, store_name) == nil {
		return false
	}

	for &m in k.members {
		if m == store_name {
			return false
		}
	}
	append(&k.members, strings.clone(store_name))
	return true
}

knid_remove_store :: proc(r: ^Registry, knid_name, store_name: string) -> bool {
	k := find_knid(r, knid_name)
	if k == nil {
		return false
	}
	for i in 0 ..< len(k.members) {
		if k.members[i] == store_name {
			delete(k.members[i])
			ordered_remove(&k.members, i)
			return true
		}
	}
	return false
}

knid_stores :: proc(r: ^Registry, knid_name: string) -> []^Store {
	k := find_knid(r, knid_name)
	if k == nil {
		return {}
	}
	result := make([dynamic]^Store, 0, len(k.members))
	for &m in k.members {
		s := find_store(r, m)
		if s != nil {
			append(&result, s)
		}
	}
	return result[:]
}

@(private)
home_dir :: proc() -> string {
	return util.home_dir()
}

@(private)
ensure_dir :: proc(path: string) {
	util.ensure_dir(path)
}
