package graph

import "../util"

set_descriptor :: proc(g: ^Graph, name: string, text: string) {
	remove_descriptor(g, name)

	d := Descriptor {
		name = util.clone_string(name),
		text = util.clone_string(text),
	}
	g.descriptors[d.name] = d
}

get_descriptor :: proc(g: ^Graph, name: string) -> ^Descriptor {
	if name in g.descriptors {
		return &g.descriptors[name]
	}
	return nil
}

remove_descriptor :: proc(g: ^Graph, name: string) -> bool {
	if name in g.descriptors {
		d := g.descriptors[name]
		delete(d.name)
		delete(d.text)
		delete_key(&g.descriptors, name)
		return true
	}
	return false
}

descriptor_count :: proc(g: ^Graph) -> int {
	return len(g.descriptors)
}

release_descriptors :: proc(g: ^Graph) {
	for _, &d in g.descriptors {
		delete(d.name)
		delete(d.text)
	}
	delete(g.descriptors)
}
