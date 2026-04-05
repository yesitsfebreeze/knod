package graph

import "../util"

DimEntry :: struct {
	index: u16,
	value: f32,
}

top_dimensions :: proc(g: ^Graph, n: int) -> []DimEntry {
	if n <= 0 || g.profile_count == 0 {
		return {}
	}

	cap_n := min(n, EMBEDDING_DIM)

	candidates := make([dynamic]DimEntry, 0, cap_n)
	defer delete(candidates)

	for i in 0 ..< EMBEDDING_DIM {
		if g.profile[i] > 0.0 {
			append(&candidates, DimEntry{index = u16(i), value = g.profile[i]})
		}
	}

	if len(candidates) == 0 {
		return {}
	}

	for i in 1 ..< len(candidates) {
		j := i
		for j > 0 && candidates[j].value > candidates[j - 1].value {
			candidates[j], candidates[j - 1] = candidates[j - 1], candidates[j]
			j -= 1
		}
	}

	take := min(cap_n, len(candidates))
	result := make([]DimEntry, take)
	for i in 0 ..< take {
		result[i] = candidates[i]
	}
	return result
}

set_tags :: proc(g: ^Graph, tags: []Tag) {
	release_tags(g)
	g.tags = make([dynamic]Tag, 0, len(tags))
	for &tag in tags {
		append(&g.tags, Tag{dim_index = tag.dim_index, label = util.clone_string(tag.label)})
	}
}

get_tags :: proc(g: ^Graph) -> []Tag {
	return g.tags[:]
}

release_tags :: proc(g: ^Graph) {
	for &tag in g.tags {
		if len(tag.label) > 0 {
			delete(tag.label)
		}
	}
	delete(g.tags)
}

changed_dimensions :: proc(g: ^Graph, new_dims: []DimEntry) -> []u16 {
	result := make([dynamic]u16, 0, len(new_dims))
	defer delete(result)

	for &dim in new_dims {
		found := false
		for &tag in g.tags {
			if tag.dim_index == dim.index {
				found = true
				break
			}
		}
		if !found {
			append(&result, dim.index)
		}
	}

	if len(result) == 0 {
		return {}
	}

	out := make([]u16, len(result))
	for i in 0 ..< len(result) {
		out[i] = result[i]
	}
	return out
}
