package gnn

import "../graph"

snapshot_from_graph :: proc(g: ^graph.Graph) -> (GraphSnapshot, bool) {
	if graph.thought_count(g) == 0 {
		return {}, false
	}
	snap := build_snapshot(g)
	return snap, true
}

forward_alloc :: proc(model: ^MPNN, snap: ^GraphSnapshot) -> ForwardCache {
	cache: ForwardCache
	forward(model, snap, &cache)
	return cache
}

forward_cache_release :: proc(cache: ^ForwardCache) {
	release_cache(cache)
}


strand_forward_query :: proc(
	s: ^StrandMPNN,
	base: ^MPNN,
	snap: ^GraphSnapshot,
	cache: ^ForwardCache,
) {
	if cache.final_hidden == nil {return}
	strand_forward(s, cache.final_hidden, snap, cache)
}

snapshot_release :: proc(snap: ^GraphSnapshot) {
	release_snapshot(snap)
}
