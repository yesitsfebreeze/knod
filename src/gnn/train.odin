package gnn

import "core:math"

import "../graph"
import log "../logger"

// adaptive_steps computes the number of training steps given the current
// graph size.  Steps scale linearly from step_max down to step_min as the
// thought count grows from 0 to MATURITY_THRESHOLD.
adaptive_steps :: proc(n, step_max, step_min: int) -> int {
	maturity := min(f32(n) / f32(MATURITY_THRESHOLD), 1.0)
	s := f32(step_max) - maturity * f32(step_max - step_min)
	return max(step_min, int(s + 0.5)) // round to nearest
}

build_snapshot :: proc(g: ^graph.Graph) -> GraphSnapshot {
	snap: GraphSnapshot

	snap.num_nodes = graph.thought_count(g)
	snap.num_edges = graph.edge_count(g)

	if snap.num_nodes == 0 {
		return snap
	}

	id_to_idx := make(map[u64]int)
	defer delete(id_to_idx)

	snap.node_ids = make([]u64, snap.num_nodes)
	snap.node_embeddings = make([]f32, snap.num_nodes * EMBEDDING_DIM)

	idx := 0
	for thought_id, &thought in g.thoughts {
		id_to_idx[thought_id] = idx
		snap.node_ids[idx] = thought_id
		for d in 0 ..< EMBEDDING_DIM {
			snap.node_embeddings[idx * EMBEDDING_DIM + d] = thought.embedding[d]
		}
		idx += 1
	}

	if snap.num_edges > 0 {
		snap.edge_embeddings = make([]f32, snap.num_edges * EMBEDDING_DIM)
		snap.edge_src = make([]int, snap.num_edges)
		snap.edge_dst = make([]int, snap.num_edges)
		snap.edge_weights = make([]f32, snap.num_edges)

		for e_idx in 0 ..< snap.num_edges {
			edge := &g.edges[e_idx]
			src_idx, src_ok := id_to_idx[edge.source_id]
			dst_idx, dst_ok := id_to_idx[edge.target_id]

			if !src_ok || !dst_ok {
				snap.edge_src[e_idx] = 0
				snap.edge_dst[e_idx] = 0
				snap.edge_weights[e_idx] = 0
				continue
			}

			snap.edge_src[e_idx] = src_idx
			snap.edge_dst[e_idx] = dst_idx
			snap.edge_weights[e_idx] = edge.weight

			for d in 0 ..< EMBEDDING_DIM {
				snap.edge_embeddings[e_idx * EMBEDDING_DIM + d] = edge.embedding[d]
			}
		}

		snap.incoming_offsets = make([]int, snap.num_nodes + 1)
		for e_idx in 0 ..< snap.num_edges {
			dst := snap.edge_dst[e_idx]
			snap.incoming_offsets[dst + 1] += 1
		}
		for i in 1 ..= snap.num_nodes {
			snap.incoming_offsets[i] += snap.incoming_offsets[i - 1]
		}
		snap.incoming_edges = make([]int, snap.num_edges)
		counters := make([]int, snap.num_nodes)
		defer delete(counters)
		for e_idx in 0 ..< snap.num_edges {
			dst := snap.edge_dst[e_idx]
			pos := snap.incoming_offsets[dst] + counters[dst]
			snap.incoming_edges[pos] = e_idx
			counters[dst] += 1
		}
	}

	return snap
}

release_snapshot :: proc(snap: ^GraphSnapshot) {
	if snap.node_embeddings != nil {delete(snap.node_embeddings)}
	if snap.edge_embeddings != nil {delete(snap.edge_embeddings)}
	if snap.edge_src != nil {delete(snap.edge_src)}
	if snap.edge_dst != nil {delete(snap.edge_dst)}
	if snap.edge_weights != nil {delete(snap.edge_weights)}
	if snap.node_ids != nil {delete(snap.node_ids)}
	if snap.incoming_offsets != nil {delete(snap.incoming_offsets)}
	if snap.incoming_edges != nil {delete(snap.incoming_edges)}
}

build_masked_snapshot :: proc(full_snap: ^GraphSnapshot, rng: ^u64) -> (GraphSnapshot, []bool) {
	mask := make([]bool, full_snap.num_edges)
	num_masked := 0

	for e in 0 ..< full_snap.num_edges {
		if random_f32(rng) < EDGE_MASK_RATIO {
			mask[e] = true
			num_masked += 1
		}
	}

	if num_masked == 0 || num_masked == full_snap.num_edges {
		for e in 0 ..< full_snap.num_edges {mask[e] = false}
		return full_snap^, mask
	}

	kept := full_snap.num_edges - num_masked
	snap: GraphSnapshot
	snap.num_nodes = full_snap.num_nodes
	snap.num_edges = kept

	snap.node_ids = full_snap.node_ids
	snap.node_embeddings = full_snap.node_embeddings

	snap.edge_src = make([]int, kept)
	snap.edge_dst = make([]int, kept)
	snap.edge_weights = make([]f32, kept)
	snap.edge_embeddings = make([]f32, kept * EMBEDDING_DIM)

	ki := 0
	for e in 0 ..< full_snap.num_edges {
		if mask[e] {continue}
		snap.edge_src[ki] = full_snap.edge_src[e]
		snap.edge_dst[ki] = full_snap.edge_dst[e]
		snap.edge_weights[ki] = full_snap.edge_weights[e]
		for d in 0 ..< EMBEDDING_DIM {
			snap.edge_embeddings[ki * EMBEDDING_DIM + d] =
				full_snap.edge_embeddings[e * EMBEDDING_DIM + d]
		}
		ki += 1
	}

	snap.incoming_offsets = make([]int, snap.num_nodes + 1)
	for e_idx in 0 ..< kept {
		dst := snap.edge_dst[e_idx]
		snap.incoming_offsets[dst + 1] += 1
	}
	for i in 1 ..= snap.num_nodes {
		snap.incoming_offsets[i] += snap.incoming_offsets[i - 1]
	}
	snap.incoming_edges = make([]int, kept)
	counters := make([]int, snap.num_nodes)
	defer delete(counters)
	for e_idx in 0 ..< kept {
		dst := snap.edge_dst[e_idx]
		pos := snap.incoming_offsets[dst] + counters[dst]
		snap.incoming_edges[pos] = e_idx
		counters[dst] += 1
	}

	return snap, mask
}

release_masked_snapshot :: proc(snap: ^GraphSnapshot) {
	if snap.edge_src != nil {delete(snap.edge_src)}
	if snap.edge_dst != nil {delete(snap.edge_dst)}
	if snap.edge_weights != nil {delete(snap.edge_weights)}
	if snap.edge_embeddings != nil {delete(snap.edge_embeddings)}
	if snap.incoming_offsets != nil {delete(snap.incoming_offsets)}
	if snap.incoming_edges != nil {delete(snap.incoming_edges)}
}

train_step :: proc(
	model: ^MPNN,
	strand: ^StrandMPNN,
	g: ^graph.Graph,
	learning_rate: f32,
) -> (
	f32,
	f32,
) {
	full_snap := build_snapshot(g)
	defer release_snapshot(&full_snap)

	if full_snap.num_nodes < 2 {
		return 0, 0
	}

	rng_mask := model.rng
	masked_snap, mask := build_masked_snapshot(&full_snap, &rng_mask)
	defer delete(mask)

	any_masked := false
	for m in mask {if m {any_masked = true; break}}

	defer if any_masked {release_masked_snapshot(&masked_snap)}

	cache: ForwardCache
	forward(model, &masked_snap, &cache)

	if strand != nil && cache.final_hidden != nil {
		strand_forward(strand, cache.final_hidden, &masked_snap, &cache)
	}

	H := model.hidden_dim
	N := masked_snap.num_nodes
	hidden := cache.strand_hidden if cache.strand_hidden != nil else cache.final_hidden
	if cache.relevance_scores != nil {delete(cache.relevance_scores)}
	cache.relevance_scores = make([]f32, N)
	for nn in 0 ..< N {
		score: f32 = model.b_relevance
		for i in 0 ..< H {
			score += hidden[nn * H + i] * model.w_relevance[i]
		}
		cache.relevance_scores[nn] = score
	}

	defer release_cache(&cache)

	zero_grads(model)
	if strand != nil {strand_zero_grads(strand)}

	link_loss, rel_loss := compute_loss_and_backward_strand(
		model,
		strand,
		&full_snap,
		&masked_snap,
		&cache,
		mask,
	)

	adamw_step(model, learning_rate)
	if strand != nil {strand_adamw_step(strand, learning_rate)}

	return link_loss, rel_loss
}

train_on_graph :: proc(model: ^MPNN, strand: ^StrandMPNN, g: ^graph.Graph, steps: int) {
	n := graph.thought_count(g)
	if n < 2 {
		return
	}

	maturity := min(f32(n) / 1024.0, 1.0)
	lr := ADAPT_LR_MAX * math.pow(ADAPT_LR_MIN / ADAPT_LR_MAX, maturity)

	for step in 0 ..< steps {
		link_loss, rel_loss := train_step(model, strand, g, lr)

		if step == 0 || step == steps - 1 {
			log.info(
				"[gnn] step %d/%d: link_loss=%.4f rel_loss=%.4f lr=%.6f",
				step + 1,
				steps,
				link_loss,
				rel_loss,
				lr,
			)
		}
	}
}

train_strand :: proc(base: ^MPNN, strand: ^StrandMPNN, g: ^graph.Graph, steps: int) {
	n := graph.thought_count(g)
	if n < 2 {return}

	maturity := min(f32(n) / 1024.0, 1.0)
	lr := ADAPT_LR_MAX * math.pow(ADAPT_LR_MIN / ADAPT_LR_MAX, maturity)

	// Build snapshot once — graph is unchanged across training steps.
	full_snap := build_snapshot(g)
	defer release_snapshot(&full_snap)

	if full_snap.num_nodes < 2 {return}

	for step in 0 ..< steps {
		rng_mask := base.rng
		masked_snap, mask := build_masked_snapshot(&full_snap, &rng_mask)

		any_masked := false
		for m in mask {if m {any_masked = true; break}}

		cache: ForwardCache
		forward(base, &masked_snap, &cache)

		if cache.final_hidden != nil {
			strand_forward(strand, cache.final_hidden, &masked_snap, &cache)
		}

		H := base.hidden_dim
		N := masked_snap.num_nodes
		hidden := cache.strand_hidden if cache.strand_hidden != nil else cache.final_hidden
		if cache.relevance_scores != nil {delete(cache.relevance_scores)}
		cache.relevance_scores = make([]f32, N)
		for nn in 0 ..< N {
			score: f32 = base.b_relevance
			for i in 0 ..< H {
				score += hidden[nn * H + i] * base.w_relevance[i]
			}
			cache.relevance_scores[nn] = score
		}

		strand_zero_grads(strand)
		compute_strand_only_backward(base, strand, &full_snap, &masked_snap, &cache, mask)

		strand_adamw_step(strand, lr)

		release_cache(&cache)
		if any_masked {release_masked_snapshot(&masked_snap)}
		delete(mask)

		base.rng = rng_mask

		if step == 0 || step == steps - 1 {
			log.info("[gnn] strand step %d/%d lr=%.6f", step + 1, steps, lr)
		}
	}
}

train_base_refine :: proc(base: ^MPNN, strand: ^StrandMPNN, g: ^graph.Graph, steps: int) {
	n := graph.thought_count(g)
	if n < 2 {return}

	lr := BASE_LR

	// Build snapshot once — graph is unchanged across training steps.
	full_snap := build_snapshot(g)
	defer release_snapshot(&full_snap)

	if full_snap.num_nodes < 2 {return}

	for step in 0 ..< steps {
		rng_mask := base.rng
		masked_snap, mask := build_masked_snapshot(&full_snap, &rng_mask)

		any_masked := false
		for m in mask {if m {any_masked = true; break}}

		cache: ForwardCache
		forward(base, &masked_snap, &cache)

		if strand != nil && cache.final_hidden != nil {
			strand_forward(strand, cache.final_hidden, &masked_snap, &cache)
		}

		H := base.hidden_dim
		N := masked_snap.num_nodes
		hidden := cache.strand_hidden if cache.strand_hidden != nil else cache.final_hidden
		if cache.relevance_scores != nil {delete(cache.relevance_scores)}
		cache.relevance_scores = make([]f32, N)
		for nn in 0 ..< N {
			score: f32 = base.b_relevance
			for i in 0 ..< H {
				score += hidden[nn * H + i] * base.w_relevance[i]
			}
			cache.relevance_scores[nn] = score
		}

		zero_grads(base)
		compute_base_only_backward(base, strand, &full_snap, &masked_snap, &cache, mask)

		adamw_step(base, lr)

		release_cache(&cache)
		if any_masked {release_masked_snapshot(&masked_snap)}
		delete(mask)

		base.rng = rng_mask

		if step == 0 || step == steps - 1 {
			log.info("[gnn] base refine step %d/%d lr=%.6f", step + 1, steps, lr)
		}
	}
}

compute_loss_and_backward_strand :: proc(
	model: ^MPNN,
	strand: ^StrandMPNN,
	full_snap: ^GraphSnapshot,
	masked_snap: ^GraphSnapshot,
	cache: ^ForwardCache,
	mask: []bool,
) -> (
	link_loss: f32,
	rel_loss: f32,
) {
	H := model.hidden_dim
	N := masked_snap.num_nodes

	hidden := cache.strand_hidden if cache.strand_hidden != nil else cache.final_hidden

	link_loss = link_prediction_loss_hidden(hidden, full_snap, H, nil)
	rng_copy := model.rng
	rel_loss = relevance_ranking_loss(cache, masked_snap, &rng_copy)

	E_masked := masked_snap.num_edges
	d_relevance := make([]f32, N)
	defer delete(d_relevance)

	if E_masked > 0 && N > 1 {
		rng_copy2 := model.rng
		count: f32 = 0
		for e in 0 ..< E_masked {
			dst := masked_snap.edge_dst[e]
			neg := int(random_u32(&rng_copy2)) % N
			if neg == dst {neg = (neg + 1) % N}

			pos_score := cache.relevance_scores[dst]
			neg_score := cache.relevance_scores[neg]

			if MARGIN - (pos_score - neg_score) > 0 {
				d_relevance[dst] -= 1.0
				d_relevance[neg] += 1.0
			}
			count += 1
		}
		if count > 0 {
			scale := RELEVANCE_WEIGHT / count
			for i in 0 ..< N {d_relevance[i] *= scale}
		}
	}

	d_hidden := make([]f32, N * H)
	defer delete(d_hidden)

	ensure_grads(model)
	grads := get_grad_slices(model)
	defer delete(grads.layers)

	db_rel: f32 = 0
	for nn in 0 ..< N {
		db_rel += d_relevance[nn]
		for i in 0 ..< H {
			grads.w_relevance[i] += hidden[nn * H + i] * d_relevance[nn]
			d_hidden[nn * H + i] += model.w_relevance[i] * d_relevance[nn]
		}
	}
	model.grads_memory[model.num_parameters - 1] += db_rel

	E_full := full_snap.num_edges
	if E_full > 0 {
		count := f32(E_full)
		for e in 0 ..< E_full {
			src := full_snap.edge_src[e]
			dst := full_snap.edge_dst[e]

			dot: f32 = 0
			for i in 0 ..< H {
				dot += hidden[src * H + i] * hidden[dst * H + i]
			}
			pred := 1.0 / (1.0 + math.exp(-dot))
			target := full_snap.edge_weights[e]

			d_pred := 2.0 * (pred - target) / count
			d_dot := d_pred * pred * (1.0 - pred)
			d_dot *= LINK_PRED_WEIGHT

			for i in 0 ..< H {
				d_hidden[src * H + i] += d_dot * hidden[dst * H + i]
				d_hidden[dst * H + i] += d_dot * hidden[src * H + i]
			}
		}
	}

	if strand != nil && cache.strand_hidden != nil {
		d_base := strand_backward(
			strand,
			masked_snap,
			&cache.strand_cache,
			d_hidden,
			cache.final_hidden,
			cache.edge_hidden,
		)
		d_rel_zero := make([]f32, N)
		backward(model, masked_snap, cache, d_relevance = d_rel_zero, d_final_link = d_base)
		delete(d_rel_zero)
		delete(d_base)
	} else {
		backward(model, masked_snap, cache, d_relevance, nil)
	}

	model.rng = rng_copy
	return
}

compute_strand_only_backward :: proc(
	base: ^MPNN,
	strand: ^StrandMPNN,
	full_snap: ^GraphSnapshot,
	masked_snap: ^GraphSnapshot,
	cache: ^ForwardCache,
	mask: []bool,
) {
	H := base.hidden_dim
	N := masked_snap.num_nodes

	hidden := cache.strand_hidden if cache.strand_hidden != nil else cache.final_hidden

	E_masked := masked_snap.num_edges
	d_relevance := make([]f32, N)
	defer delete(d_relevance)

	if E_masked > 0 && N > 1 {
		rng_copy := base.rng
		count: f32 = 0
		for e in 0 ..< E_masked {
			dst := masked_snap.edge_dst[e]
			neg := int(random_u32(&rng_copy)) % N
			if neg == dst {neg = (neg + 1) % N}

			pos_score := cache.relevance_scores[dst]
			neg_score := cache.relevance_scores[neg]

			if MARGIN - (pos_score - neg_score) > 0 {
				d_relevance[dst] -= 1.0
				d_relevance[neg] += 1.0
			}
			count += 1
		}
		if count > 0 {
			scale := RELEVANCE_WEIGHT / count
			for i in 0 ..< N {d_relevance[i] *= scale}
		}
	}

	d_hidden := make([]f32, N * H)
	defer delete(d_hidden)

	for nn in 0 ..< N {
		for i in 0 ..< H {
			d_hidden[nn * H + i] += base.w_relevance[i] * d_relevance[nn]
		}
	}

	E_full := full_snap.num_edges
	if E_full > 0 {
		count := f32(E_full)
		for e in 0 ..< E_full {
			src := full_snap.edge_src[e]
			dst := full_snap.edge_dst[e]

			dot: f32 = 0
			for i in 0 ..< H {
				dot += hidden[src * H + i] * hidden[dst * H + i]
			}
			pred := 1.0 / (1.0 + math.exp(-dot))
			target := full_snap.edge_weights[e]

			d_pred := 2.0 * (pred - target) / count
			d_dot := d_pred * pred * (1.0 - pred)
			d_dot *= LINK_PRED_WEIGHT

			for i in 0 ..< H {
				d_hidden[src * H + i] += d_dot * hidden[dst * H + i]
				d_hidden[dst * H + i] += d_dot * hidden[src * H + i]
			}
		}
	}

	if cache.strand_hidden != nil {
		d_base := strand_backward(
			strand,
			masked_snap,
			&cache.strand_cache,
			d_hidden,
			cache.final_hidden,
			cache.edge_hidden,
		)
		delete(d_base)
	}
}

compute_base_only_backward :: proc(
	base: ^MPNN,
	strand: ^StrandMPNN,
	full_snap: ^GraphSnapshot,
	masked_snap: ^GraphSnapshot,
	cache: ^ForwardCache,
	mask: []bool,
) {
	H := base.hidden_dim
	N := masked_snap.num_nodes

	hidden := cache.strand_hidden if cache.strand_hidden != nil else cache.final_hidden

	E_masked := masked_snap.num_edges
	d_relevance := make([]f32, N)
	defer delete(d_relevance)

	if E_masked > 0 && N > 1 {
		rng_copy := base.rng
		count: f32 = 0
		for e in 0 ..< E_masked {
			dst := masked_snap.edge_dst[e]
			neg := int(random_u32(&rng_copy)) % N
			if neg == dst {neg = (neg + 1) % N}

			pos_score := cache.relevance_scores[dst]
			neg_score := cache.relevance_scores[neg]

			if MARGIN - (pos_score - neg_score) > 0 {
				d_relevance[dst] -= 1.0
				d_relevance[neg] += 1.0
			}
			count += 1
		}
		if count > 0 {
			scale := RELEVANCE_WEIGHT / count
			for i in 0 ..< N {d_relevance[i] *= scale}
		}
	}

	d_hidden := make([]f32, N * H)
	defer delete(d_hidden)

	ensure_grads(base)
	grads := get_grad_slices(base)
	defer delete(grads.layers)

	db_rel: f32 = 0
	for nn in 0 ..< N {
		db_rel += d_relevance[nn]
		for i in 0 ..< H {
			grads.w_relevance[i] += hidden[nn * H + i] * d_relevance[nn]
			d_hidden[nn * H + i] += base.w_relevance[i] * d_relevance[nn]
		}
	}
	base.grads_memory[base.num_parameters - 1] += db_rel

	E_full := full_snap.num_edges
	if E_full > 0 {
		count := f32(E_full)
		for e in 0 ..< E_full {
			src := full_snap.edge_src[e]
			dst := full_snap.edge_dst[e]

			dot: f32 = 0
			for i in 0 ..< H {
				dot += hidden[src * H + i] * hidden[dst * H + i]
			}
			pred := 1.0 / (1.0 + math.exp(-dot))
			target := full_snap.edge_weights[e]

			d_pred := 2.0 * (pred - target) / count
			d_dot := d_pred * pred * (1.0 - pred)
			d_dot *= LINK_PRED_WEIGHT

			for i in 0 ..< H {
				d_hidden[src * H + i] += d_dot * hidden[dst * H + i]
				d_hidden[dst * H + i] += d_dot * hidden[src * H + i]
			}
		}
	}

	if strand != nil && cache.strand_hidden != nil {
		d_base := strand_backward(
			strand,
			masked_snap,
			&cache.strand_cache,
			d_hidden,
			cache.final_hidden,
			cache.edge_hidden,
		)
		d_rel_zero := make([]f32, N)
		backward(base, masked_snap, cache, d_relevance = d_rel_zero, d_final_link = d_base)
		delete(d_rel_zero)
		delete(d_base)
	} else {
		backward(base, masked_snap, cache, d_relevance, nil)
	}
}

link_prediction_loss_hidden :: proc(
	hidden: []f32,
	snap: ^GraphSnapshot,
	H: int,
	mask: []bool,
) -> f32 {
	if snap.num_edges == 0 {return 0}

	loss: f32 = 0
	count := 0
	for e in 0 ..< snap.num_edges {
		if mask != nil && mask[e] {continue}

		src := snap.edge_src[e]
		dst := snap.edge_dst[e]

		dot: f32 = 0
		for i in 0 ..< H {
			dot += hidden[src * H + i] * hidden[dst * H + i]
		}
		pred := 1.0 / (1.0 + math.exp(-dot))
		target := snap.edge_weights[e]

		diff := pred - target
		loss += diff * diff
		count += 1
	}

	if count == 0 {return 0}
	return loss / f32(count)
}
