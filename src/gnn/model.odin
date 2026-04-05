package gnn

import log "../logger"
import "core:math"
import "core:mem"
import "core:os"

random_u32 :: proc(state: ^u64) -> u32 {
	state^ ~= state^ >> 12
	state^ ~= state^ << 25
	state^ ~= state^ >> 27
	return u32((state^ * 0x2545F4914F6CDD1D) >> 32)
}

random_f32 :: proc(state: ^u64) -> f32 {
	return f32(random_u32(state) >> 8) / 16777216.0
}

count_parameters :: proc(hidden_dim, num_layers: int) -> int {
	H := hidden_dim
	E := EMBEDDING_DIM

	total := 0

	total += H * E + H
	total += H * E + H

	per_layer := H * 3 * H + H + H * 2 * H + H + H + H
	total += per_layer * num_layers

	total += H + 1

	return total
}

create :: proc(
	model: ^MPNN,
	hidden_dim: int = DEFAULT_HIDDEN_DIM,
	num_layers: int = DEFAULT_NUM_LAYERS,
) {
	model.hidden_dim = hidden_dim
	model.num_layers = num_layers
	model.rng = RNG_SEED
	model.adam_t = 0

	H := hidden_dim
	E := EMBEDDING_DIM

	model.num_parameters = count_parameters(H, num_layers)
	model.params_memory = make([]f32, model.num_parameters)

	offset := 0
	point :: proc(mem: []f32, offset: ^int, size: int) -> []f32 {
		s := mem[offset^:][:size]
		offset^ += size
		return s
	}

	model.w_node_proj = point(model.params_memory, &offset, H * E)
	model.b_node_proj = point(model.params_memory, &offset, H)
	model.w_edge_proj = point(model.params_memory, &offset, H * E)
	model.b_edge_proj = point(model.params_memory, &offset, H)

	model.layers = make([]MPNNLayerWeights, num_layers)
	for l in 0 ..< num_layers {
		model.layers[l].w_msg = point(model.params_memory, &offset, H * 3 * H)
		model.layers[l].b_msg = point(model.params_memory, &offset, H)
		model.layers[l].w_upd = point(model.params_memory, &offset, H * 2 * H)
		model.layers[l].b_upd = point(model.params_memory, &offset, H)
		model.layers[l].ln_w = point(model.params_memory, &offset, H)
		model.layers[l].ln_b = point(model.params_memory, &offset, H)
	}

	model.w_relevance = point(model.params_memory, &offset, H)
	assert(offset == model.num_parameters - 1, "parameter count mismatch in create")
	offset += 1

	scale := 1.0 / math.sqrt(f32(H))
	for i in 0 ..< model.num_parameters {
		model.params_memory[i] = (random_f32(&model.rng) * 2.0 - 1.0) * scale
	}

	for l in 0 ..< num_layers {
		for i in 0 ..< H {model.layers[l].ln_w[i] = 1.0}
		for i in 0 ..< H {model.layers[l].ln_b[i] = 0.0}
	}

	model.b_relevance = 0
	model.params_memory[model.num_parameters - 1] = 0

	for i in 0 ..< H {model.b_node_proj[i] = 0}
	for i in 0 ..< H {model.b_edge_proj[i] = 0}
	for l in 0 ..< num_layers {
		for i in 0 ..< H {model.layers[l].b_msg[i] = 0}
		for i in 0 ..< H {model.layers[l].b_upd[i] = 0}
	}

	log.info(
		"[gnn] created MPNN: %d layers, hidden=%d, %d params",
		num_layers,
		hidden_dim,
		model.num_parameters,
	)
}

release :: proc(model: ^MPNN) {
	if model.params_memory != nil {delete(model.params_memory)}
	if model.grads_memory != nil {delete(model.grads_memory)}
	if model.m_memory != nil {delete(model.m_memory)}
	if model.v_memory != nil {delete(model.v_memory)}
	if model.layers != nil {delete(model.layers)}
}

ensure_grads :: proc(model: ^MPNN) {
	if model.grads_memory != nil {return}
	model.grads_memory = make([]f32, model.num_parameters)
}

zero_grads :: proc(model: ^MPNN) {
	if model.grads_memory == nil {return}
	for i in 0 ..< model.num_parameters {model.grads_memory[i] = 0}
}

GradSlices :: struct {
	w_node_proj: []f32,
	b_node_proj: []f32,
	w_edge_proj: []f32,
	b_edge_proj: []f32,
	layers:      []MPNNLayerWeights,
	w_relevance: []f32,
}

get_grad_slices :: proc(model: ^MPNN) -> GradSlices {
	H := model.hidden_dim
	E := EMBEDDING_DIM

	g: GradSlices
	offset := 0
	point :: proc(mem: []f32, offset: ^int, size: int) -> []f32 {
		s := mem[offset^:][:size]
		offset^ += size
		return s
	}

	g.w_node_proj = point(model.grads_memory, &offset, H * E)
	g.b_node_proj = point(model.grads_memory, &offset, H)
	g.w_edge_proj = point(model.grads_memory, &offset, H * E)
	g.b_edge_proj = point(model.grads_memory, &offset, H)

	g.layers = make([]MPNNLayerWeights, model.num_layers)
	for l in 0 ..< model.num_layers {
		g.layers[l].w_msg = point(model.grads_memory, &offset, H * 3 * H)
		g.layers[l].b_msg = point(model.grads_memory, &offset, H)
		g.layers[l].w_upd = point(model.grads_memory, &offset, H * 2 * H)
		g.layers[l].b_upd = point(model.grads_memory, &offset, H)
		g.layers[l].ln_w = point(model.grads_memory, &offset, H)
		g.layers[l].ln_b = point(model.grads_memory, &offset, H)
	}

	g.w_relevance = point(model.grads_memory, &offset, H)
	return g
}

forward :: proc(model: ^MPNN, snap: ^GraphSnapshot, cache: ^ForwardCache) {
	H := model.hidden_dim
	N := snap.num_nodes
	E_count := snap.num_edges

	if N == 0 {return}

	cache.node_hidden = make([]f32, N * H)
	linear_forward(
		cache.node_hidden,
		snap.node_embeddings,
		model.w_node_proj,
		model.b_node_proj,
		N,
		EMBEDDING_DIM,
		H,
	)

	cache.edge_hidden = make([]f32, max(E_count, 1) * H)
	if E_count > 0 {
		linear_forward(
			cache.edge_hidden,
			snap.edge_embeddings,
			model.w_edge_proj,
			model.b_edge_proj,
			E_count,
			EMBEDDING_DIM,
			H,
		)
	}

	cache.layer_caches = make([]MPNNLayerActivations, model.num_layers)

	current := make([]f32, N * H)
	copy(current, cache.node_hidden)

	for l in 0 ..< model.num_layers {
		lc := &cache.layer_caches[l]
		lw := &model.layers[l]

		if E_count == 0 {
			lc.output = make([]f32, N * H)
			copy(lc.output, current)
			continue
		}

		msg_input := make([]f32, E_count * 3 * H)
		defer delete(msg_input)

		src_gathered := make([]f32, E_count * H)
		dst_gathered := make([]f32, E_count * H)
		defer delete(src_gathered)
		defer delete(dst_gathered)

		for e in 0 ..< E_count {
			src := snap.edge_src[e]
			dst := snap.edge_dst[e]
			for i in 0 ..< H {
				src_gathered[e * H + i] = current[src * H + i]
				dst_gathered[e * H + i] = current[dst * H + i]
			}
		}

		concat3_forward(msg_input, src_gathered, dst_gathered, cache.edge_hidden, E_count, H)

		lc.messages_pre = make([]f32, E_count * H)
		linear_forward(lc.messages_pre, msg_input, lw.w_msg, lw.b_msg, E_count, 3 * H, H)

		lc.messages = make([]f32, E_count * H)
		relu_forward(lc.messages, lc.messages_pre, E_count * H)

		lc.aggregated = make([]f32, N * H)
		for i in 0 ..< N * H {lc.aggregated[i] = 0}
		scatter_add(lc.aggregated, lc.messages, snap.edge_dst, E_count, H)

		update_input := make([]f32, N * 2 * H)
		defer delete(update_input)
		concat2_forward(update_input, current, lc.aggregated, N, H)

		lc.update_pre = make([]f32, N * H)
		linear_forward(lc.update_pre, update_input, lw.w_upd, lw.b_upd, N, 2 * H, H)

		lc.update_relu = make([]f32, N * H)
		relu_forward(lc.update_relu, lc.update_pre, N * H)

		lc.ln_mean = make([]f32, N)
		lc.ln_rstd = make([]f32, N)
		lc.output = make([]f32, N * H)
		layernorm_forward(
			lc.output,
			lc.ln_mean,
			lc.ln_rstd,
			lc.update_relu,
			lw.ln_w,
			lw.ln_b,
			N,
			H,
		)

		delete(current)
		current = make([]f32, N * H)
		copy(current, lc.output)
	}

	cache.final_hidden = current

	cache.relevance_scores = make([]f32, N)
	for n in 0 ..< N {
		score: f32 = model.b_relevance
		for i in 0 ..< H {
			score += cache.final_hidden[n * H + i] * model.w_relevance[i]
		}
		cache.relevance_scores[n] = score
	}
}

release_cache :: proc(cache: ^ForwardCache) {
	if cache.node_hidden != nil {delete(cache.node_hidden)}
	if cache.edge_hidden != nil {delete(cache.edge_hidden)}
	if cache.final_hidden != nil {delete(cache.final_hidden)}
	if cache.relevance_scores != nil {delete(cache.relevance_scores)}
	if cache.strand_hidden != nil {delete(cache.strand_hidden)}
	release_layer_activations :: proc(lc: ^MPNNLayerActivations) {
		if lc.messages != nil {delete(lc.messages)}
		if lc.messages_pre != nil {delete(lc.messages_pre)}
		if lc.aggregated != nil {delete(lc.aggregated)}
		if lc.update_pre != nil {delete(lc.update_pre)}
		if lc.update_relu != nil {delete(lc.update_relu)}
		if lc.ln_mean != nil {delete(lc.ln_mean)}
		if lc.ln_rstd != nil {delete(lc.ln_rstd)}
		if lc.output != nil {delete(lc.output)}
	}
	if cache.layer_caches != nil {
		for &lc in cache.layer_caches {
			release_layer_activations(&lc)
		}
		delete(cache.layer_caches)
	}
	release_layer_activations(&cache.strand_cache)
}

backward :: proc(
	model: ^MPNN,
	snap: ^GraphSnapshot,
	cache: ^ForwardCache,
	d_relevance: []f32,
	d_final_link: []f32 = nil,
) {
	H := model.hidden_dim
	N := snap.num_nodes
	E_count := snap.num_edges

	if N == 0 {return}

	ensure_grads(model)
	grads := get_grad_slices(model)
	defer delete(grads.layers)

	d_final := make([]f32, N * H)
	defer delete(d_final)

	db_rel: f32 = 0
	for n in 0 ..< N {
		db_rel += d_relevance[n]
		for i in 0 ..< H {
			grads.w_relevance[i] += cache.final_hidden[n * H + i] * d_relevance[n]
			d_final[n * H + i] += model.w_relevance[i] * d_relevance[n]
		}
	}
	model.grads_memory[model.num_parameters - 1] += db_rel

	if d_final_link != nil {
		for i in 0 ..< N * H {
			d_final[i] += d_final_link[i]
		}
	}

	d_current := make([]f32, N * H)
	copy(d_current, d_final)

	d_edge_hidden_total: []f32
	if E_count > 0 {
		d_edge_hidden_total = make([]f32, E_count * H)
	}

	for li in 0 ..< model.num_layers {
		l := model.num_layers - 1 - li
		lc := &cache.layer_caches[l]
		lw := &model.layers[l]
		lg := &grads.layers[l]

		if E_count == 0 {
			continue
		}

		input_h: []f32
		if l == 0 {
			input_h = cache.node_hidden
		} else {
			input_h = cache.layer_caches[l - 1].output
		}

		d_update_relu := make([]f32, N * H)
		defer delete(d_update_relu)
		layernorm_backward(
			d_update_relu,
			lg.ln_w,
			lg.ln_b,
			d_current,
			lc.update_relu,
			lw.ln_w,
			lc.ln_mean,
			lc.ln_rstd,
			N,
			H,
		)

		d_update_pre := make([]f32, N * H)
		defer delete(d_update_pre)
		relu_backward(d_update_pre, d_update_relu, lc.update_pre, N * H)

		d_update_input := make([]f32, N * 2 * H)
		defer delete(d_update_input)

		update_input := make([]f32, N * 2 * H)
		defer delete(update_input)
		concat2_forward(update_input, input_h, lc.aggregated, N, H)

		linear_backward(
			d_update_input,
			lg.w_upd,
			lg.b_upd,
			d_update_pre,
			update_input,
			lw.w_upd,
			N,
			2 * H,
			H,
		)

		d_input_h := make([]f32, N * H)
		d_aggregated := make([]f32, N * H)
		defer delete(d_input_h)
		defer delete(d_aggregated)
		concat2_backward(d_input_h, d_aggregated, d_update_input, N, H)

		d_messages := make([]f32, E_count * H)
		defer delete(d_messages)
		scatter_add_backward(d_messages, d_aggregated, snap.edge_dst, E_count, H)

		d_messages_pre := make([]f32, E_count * H)
		defer delete(d_messages_pre)
		relu_backward(d_messages_pre, d_messages, lc.messages_pre, E_count * H)

		src_gathered := make([]f32, E_count * H)
		dst_gathered := make([]f32, E_count * H)
		defer delete(src_gathered)
		defer delete(dst_gathered)
		for e in 0 ..< E_count {
			src := snap.edge_src[e]
			dst := snap.edge_dst[e]
			for i in 0 ..< H {
				src_gathered[e * H + i] = input_h[src * H + i]
				dst_gathered[e * H + i] = input_h[dst * H + i]
			}
		}
		msg_input := make([]f32, E_count * 3 * H)
		defer delete(msg_input)
		concat3_forward(msg_input, src_gathered, dst_gathered, cache.edge_hidden, E_count, H)

		d_msg_input := make([]f32, E_count * 3 * H)
		defer delete(d_msg_input)
		linear_backward(
			d_msg_input,
			lg.w_msg,
			lg.b_msg,
			d_messages_pre,
			msg_input,
			lw.w_msg,
			E_count,
			3 * H,
			H,
		)

		d_src_gathered := make([]f32, E_count * H)
		d_dst_gathered := make([]f32, E_count * H)
		d_edge_hidden := make([]f32, E_count * H)
		defer delete(d_src_gathered)
		defer delete(d_dst_gathered)
		defer delete(d_edge_hidden)
		concat3_backward(d_src_gathered, d_dst_gathered, d_edge_hidden, d_msg_input, E_count, H)

		for i in 0 ..< E_count * H {
			d_edge_hidden_total[i] += d_edge_hidden[i]
		}

		new_d_current := make([]f32, N * H)
		for i in 0 ..< N * H {new_d_current[i] += d_input_h[i]}
		for e in 0 ..< E_count {
			src := snap.edge_src[e]
			dst := snap.edge_dst[e]
			for i in 0 ..< H {
				new_d_current[src * H + i] += d_src_gathered[e * H + i]
				new_d_current[dst * H + i] += d_dst_gathered[e * H + i]
			}
		}

		delete(d_current)
		d_current = new_d_current
	}

	dummy_d_node_emb := make([]f32, N * EMBEDDING_DIM)
	defer delete(dummy_d_node_emb)
	linear_backward(
		dummy_d_node_emb,
		grads.w_node_proj,
		grads.b_node_proj,
		d_current,
		snap.node_embeddings,
		model.w_node_proj,
		N,
		EMBEDDING_DIM,
		H,
	)

	if E_count > 0 && d_edge_hidden_total != nil {
		dummy_d_edge_emb := make([]f32, E_count * EMBEDDING_DIM)
		defer delete(dummy_d_edge_emb)
		linear_backward(
			dummy_d_edge_emb,
			grads.w_edge_proj,
			grads.b_edge_proj,
			d_edge_hidden_total,
			snap.edge_embeddings,
			model.w_edge_proj,
			E_count,
			EMBEDDING_DIM,
			H,
		)
		delete(d_edge_hidden_total)
	}

	delete(d_current)
}

link_prediction_loss :: proc(
	cache: ^ForwardCache,
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
			dot += cache.final_hidden[src * H + i] * cache.final_hidden[dst * H + i]
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

relevance_ranking_loss :: proc(cache: ^ForwardCache, snap: ^GraphSnapshot, rng: ^u64) -> f32 {
	if snap.num_edges == 0 || snap.num_nodes < 2 {return 0}

	loss: f32 = 0
	count := 0

	for e in 0 ..< snap.num_edges {
		dst := snap.edge_dst[e]
		neg := int(random_u32(rng)) % snap.num_nodes
		if neg == dst {neg = (neg + 1) % snap.num_nodes}

		pos_score := cache.relevance_scores[dst]
		neg_score := cache.relevance_scores[neg]

		margin_loss := max(0, MARGIN - (pos_score - neg_score))
		loss += margin_loss
		count += 1
	}

	if count == 0 {return 0}
	return loss / f32(count)
}

compute_loss_and_backward_masked :: proc(
	model: ^MPNN,
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

	link_loss = link_prediction_loss(cache, full_snap, H, nil)

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
			for i in 0 ..< N {
				d_relevance[i] *= scale
			}
		}
	}

	E_full := full_snap.num_edges
	d_final_link := make([]f32, N * H)
	defer delete(d_final_link)

	if E_full > 0 {
		count := f32(E_full)
		for e in 0 ..< E_full {
			src := full_snap.edge_src[e]
			dst := full_snap.edge_dst[e]

			dot: f32 = 0
			for i in 0 ..< H {
				dot += cache.final_hidden[src * H + i] * cache.final_hidden[dst * H + i]
			}
			pred := 1.0 / (1.0 + math.exp(-dot))
			target := full_snap.edge_weights[e]

			d_pred := 2.0 * (pred - target) / count
			d_dot := d_pred * pred * (1.0 - pred)
			d_dot *= LINK_PRED_WEIGHT

			for i in 0 ..< H {
				d_final_link[src * H + i] += d_dot * cache.final_hidden[dst * H + i]
				d_final_link[dst * H + i] += d_dot * cache.final_hidden[src * H + i]
			}
		}
	}

	backward(model, masked_snap, cache, d_relevance, d_final_link)

	model.rng = rng_copy
	return
}

adamw_step :: proc(model: ^MPNN, learning_rate: f32) {
	if model.grads_memory == nil {return}

	if model.m_memory == nil {
		model.m_memory = make([]f32, model.num_parameters)
		model.v_memory = make([]f32, model.num_parameters)
	}

	model.adam_t += 1
	t := model.adam_t

	for i in 0 ..< model.num_parameters {
		param := model.params_memory[i]
		grad := model.grads_memory[i]

		m := BETA1 * model.m_memory[i] + (1.0 - BETA1) * grad
		v := BETA2 * model.v_memory[i] + (1.0 - BETA2) * grad * grad

		m_hat := m / (1.0 - math.pow(BETA1, f32(t)))
		v_hat := v / (1.0 - math.pow(BETA2, f32(t)))

		model.m_memory[i] = m
		model.v_memory[i] = v
		model.params_memory[i] -=
			learning_rate * (m_hat / (math.sqrt(v_hat) + ADAM_EPS) + WEIGHT_DECAY * param)
	}

	model.b_relevance = model.params_memory[model.num_parameters - 1]
}

score_nodes :: proc(
	base: ^MPNN,
	strand: ^StrandMPNN,
	snap: ^GraphSnapshot,
	query_embedding: ^Embedding,
	k: int,
) -> []ScoredNode {
	if snap.num_nodes == 0 || k <= 0 {return {}}

	cache: ForwardCache
	forward(base, snap, &cache)

	if strand != nil && cache.final_hidden != nil {
		strand_forward(strand, cache.final_hidden, snap, &cache)
	}

	hidden := cache.strand_hidden if cache.strand_hidden != nil else cache.final_hidden
	H := base.hidden_dim
	N := snap.num_nodes
	if cache.relevance_scores == nil {
		cache.relevance_scores = make([]f32, N)
	}
	for nn in 0 ..< N {
		score: f32 = base.b_relevance
		for i in 0 ..< H {
			score += hidden[nn * H + i] * base.w_relevance[i]
		}
		cache.relevance_scores[nn] = score
	}

	defer release_cache(&cache)

	n := min(k, N)

	results := make([dynamic]ScoredNode, 0, N)
	defer delete(results)

	for i in 0 ..< N {
		dot: f32 = 0
		norm_a: f32 = 0
		norm_b: f32 = 0
		for d in 0 ..< EMBEDDING_DIM {
			a := query_embedding[d]
			b_val := snap.node_embeddings[i * EMBEDDING_DIM + d]
			dot += a * b_val
			norm_a += a * a
			norm_b += b_val * b_val
		}
		denom := math.sqrt(norm_a) * math.sqrt(norm_b)
		cos_sim: f32 = 0
		if denom > 0 {cos_sim = dot / denom}

		gnn_score := cache.relevance_scores[i]
		combined := 0.5 * cos_sim + 0.5 * gnn_score

		append(&results, ScoredNode{node_idx = i, node_id = snap.node_ids[i], score = combined})
	}

	for i in 1 ..< len(results) {
		j := i
		for j > 0 && results[j].score > results[j - 1].score {
			results[j], results[j - 1] = results[j - 1], results[j]
			j -= 1
		}
	}

	out := make([]ScoredNode, n)
	for i in 0 ..< n {
		out[i] = results[i]
	}
	return out
}

save_checkpoint :: proc(model: ^MPNN, path: string) -> bool {
	if model.params_memory == nil {
		return false
	}

	fd, err := os.open(path, os.O_WRONLY | os.O_CREATE | os.O_TRUNC, 0o644)
	if err != os.ERROR_NONE {
		log.err("[gnn] could not open %s for saving", path)
		return false
	}
	defer os.close(fd)

	header: [8]i32
	header[0] = GNN_MAGIC
	header[1] = GNN_VERSION
	header[2] = i32(model.hidden_dim)
	header[3] = i32(model.num_layers)
	header[4] = i32(model.num_parameters)
	header[5] = i32(model.adam_t)
	header[6] = i32(model.rng >> 32)
	header[7] = i32(model.rng & 0xFFFFFFFF)

	written, _ := os.write(fd, mem.slice_to_bytes(header[:]))
	if written != size_of(header) {
		log.err("[gnn] header write failed")
		return false
	}

	param_bytes := mem.slice_to_bytes(model.params_memory)
	written2, _ := os.write(fd, param_bytes)
	if written2 != len(param_bytes) {
		log.err("[gnn] params write failed")
		return false
	}

	if model.m_memory != nil && model.v_memory != nil {
		m_bytes := mem.slice_to_bytes(model.m_memory)
		v_bytes := mem.slice_to_bytes(model.v_memory)
		os.write(fd, m_bytes)
		os.write(fd, v_bytes)
	}

	log.info("[gnn] checkpoint saved: %d params, step %d", model.num_parameters, model.adam_t)
	return true
}

load_checkpoint :: proc(model: ^MPNN, path: string) -> bool {
	data, ok := os.read_entire_file(path)
	if !ok {return false}
	defer delete(data)

	header_size := 8 * size_of(i32)
	if len(data) < header_size {
		log.err("[gnn] checkpoint too small")
		return false
	}

	header := (cast([^]i32)raw_data(data))[:8]
	if header[0] != GNN_MAGIC {
		log.err("[gnn] bad magic in checkpoint")
		return false
	}
	if header[1] != GNN_VERSION {
		log.err("[gnn] unsupported checkpoint version %d", header[1])
		return false
	}

	hidden_dim := int(header[2])
	num_layers := int(header[3])
	num_params := int(header[4])
	adam_t := int(header[5])
	rng := (u64(u32(header[6])) << 32) | u64(u32(header[7]))

	create(model, hidden_dim, num_layers)
	model.adam_t = adam_t
	model.rng = rng

	if model.num_parameters != num_params {
		log.err("[gnn] param count mismatch: file=%d, model=%d", num_params, model.num_parameters)
		return false
	}

	param_bytes := num_params * size_of(f32)
	if len(data) < header_size + param_bytes {
		log.err("[gnn] checkpoint truncated")
		return false
	}
	mem.copy(raw_data(model.params_memory), &data[header_size], param_bytes)

	model.b_relevance = model.params_memory[model.num_parameters - 1]

	optim_offset := header_size + param_bytes
	optim_size := 2 * param_bytes
	if len(data) >= optim_offset + optim_size {
		model.m_memory = make([]f32, num_params)
		model.v_memory = make([]f32, num_params)
		mem.copy(raw_data(model.m_memory), &data[optim_offset], param_bytes)
		mem.copy(raw_data(model.v_memory), &data[optim_offset + param_bytes], param_bytes)
	}

	log.info(
		"[gnn] checkpoint loaded: %d layers, hidden=%d, %d params, step %d",
		num_layers,
		hidden_dim,
		num_params,
		adam_t,
	)
	return true
}


strand_count_parameters :: proc(H: int) -> int {
	return H * 3 * H + H + H * 2 * H + H + H + H
}

strand_create :: proc(s: ^StrandMPNN, hidden_dim: int) {
	H := hidden_dim
	s.hidden_dim = hidden_dim
	s.adam_t = 0

	s.num_parameters = strand_count_parameters(H)
	s.params_memory = make([]f32, s.num_parameters)

	offset := 0
	point :: proc(mem_buf: []f32, off: ^int, size: int) -> []f32 {
		sl := mem_buf[off^:][:size]
		off^ += size
		return sl
	}

	s.layer.w_msg = point(s.params_memory, &offset, H * 3 * H)
	s.layer.b_msg = point(s.params_memory, &offset, H)
	s.layer.w_upd = point(s.params_memory, &offset, H * 2 * H)
	s.layer.b_upd = point(s.params_memory, &offset, H)
	s.layer.ln_w = point(s.params_memory, &offset, H)
	s.layer.ln_b = point(s.params_memory, &offset, H)
	assert(offset == s.num_parameters, "strand parameter count mismatch")

	rng: u64 = RNG_SEED + 12345
	scale := 1.0 / math.sqrt(f32(H))
	for i in 0 ..< s.num_parameters {
		s.params_memory[i] = (random_f32(&rng) * 2.0 - 1.0) * scale
	}
	for i in 0 ..< H {s.layer.ln_w[i] = 1.0}
	for i in 0 ..< H {s.layer.ln_b[i] = 0.0}
	for i in 0 ..< H {s.layer.b_msg[i] = 0}
	for i in 0 ..< H {s.layer.b_upd[i] = 0}

	log.info("[gnn] strand created: hidden=%d, %d params", hidden_dim, s.num_parameters)
}

strand_release :: proc(s: ^StrandMPNN) {
	if s.params_memory != nil {delete(s.params_memory)}
	if s.grads_memory != nil {delete(s.grads_memory)}
	if s.m_memory != nil {delete(s.m_memory)}
	if s.v_memory != nil {delete(s.v_memory)}
}

strand_ensure_grads :: proc(s: ^StrandMPNN) {
	if s.grads_memory != nil {return}
	s.grads_memory = make([]f32, s.num_parameters)
}

strand_zero_grads :: proc(s: ^StrandMPNN) {
	if s.grads_memory == nil {return}
	for i in 0 ..< s.num_parameters {s.grads_memory[i] = 0}
}

StrandGradSlices :: struct {
	w_msg: []f32,
	b_msg: []f32,
	w_upd: []f32,
	b_upd: []f32,
	ln_w:  []f32,
	ln_b:  []f32,
}

strand_get_grad_slices :: proc(s: ^StrandMPNN) -> StrandGradSlices {
	H := s.hidden_dim
	g: StrandGradSlices
	offset := 0
	point :: proc(mem_buf: []f32, off: ^int, size: int) -> []f32 {
		sl := mem_buf[off^:][:size]
		off^ += size
		return sl
	}
	g.w_msg = point(s.grads_memory, &offset, H * 3 * H)
	g.b_msg = point(s.grads_memory, &offset, H)
	g.w_upd = point(s.grads_memory, &offset, H * 2 * H)
	g.b_upd = point(s.grads_memory, &offset, H)
	g.ln_w = point(s.grads_memory, &offset, H)
	g.ln_b = point(s.grads_memory, &offset, H)
	return g
}

strand_forward :: proc(
	s: ^StrandMPNN,
	base_hidden: []f32,
	snap: ^GraphSnapshot,
	cache: ^ForwardCache,
) {
	H := s.hidden_dim
	N := snap.num_nodes
	E_count := snap.num_edges

	if N == 0 {return}

	lc := &cache.strand_cache
	lw := &s.layer

	if E_count == 0 {
		cache.strand_hidden = make([]f32, N * H)
		copy(cache.strand_hidden, base_hidden)
		lc.output = make([]f32, N * H)
		copy(lc.output, base_hidden)
		return
	}

	msg_input := make([]f32, E_count * 3 * H)
	defer delete(msg_input)

	src_gathered := make([]f32, E_count * H)
	dst_gathered := make([]f32, E_count * H)
	defer delete(src_gathered)
	defer delete(dst_gathered)

	for e in 0 ..< E_count {
		src := snap.edge_src[e]
		dst := snap.edge_dst[e]
		for i in 0 ..< H {
			src_gathered[e * H + i] = base_hidden[src * H + i]
			dst_gathered[e * H + i] = base_hidden[dst * H + i]
		}
	}

	concat3_forward(msg_input, src_gathered, dst_gathered, cache.edge_hidden, E_count, H)

	lc.messages_pre = make([]f32, E_count * H)
	linear_forward(lc.messages_pre, msg_input, lw.w_msg, lw.b_msg, E_count, 3 * H, H)

	lc.messages = make([]f32, E_count * H)
	relu_forward(lc.messages, lc.messages_pre, E_count * H)

	lc.aggregated = make([]f32, N * H)
	for i in 0 ..< N * H {lc.aggregated[i] = 0}
	scatter_add(lc.aggregated, lc.messages, snap.edge_dst, E_count, H)

	update_input := make([]f32, N * 2 * H)
	defer delete(update_input)
	concat2_forward(update_input, base_hidden, lc.aggregated, N, H)

	lc.update_pre = make([]f32, N * H)
	linear_forward(lc.update_pre, update_input, lw.w_upd, lw.b_upd, N, 2 * H, H)

	lc.update_relu = make([]f32, N * H)
	relu_forward(lc.update_relu, lc.update_pre, N * H)

	lc.ln_mean = make([]f32, N)
	lc.ln_rstd = make([]f32, N)
	lc.output = make([]f32, N * H)
	layernorm_forward(lc.output, lc.ln_mean, lc.ln_rstd, lc.update_relu, lw.ln_w, lw.ln_b, N, H)

	cache.strand_hidden = make([]f32, N * H)
	copy(cache.strand_hidden, lc.output)
}

strand_backward :: proc(
	s: ^StrandMPNN,
	snap: ^GraphSnapshot,
	layer_cache: ^MPNNLayerActivations,
	d_output: []f32,
	base_hidden: []f32,
	edge_hidden: []f32,
) -> []f32 {
	H := s.hidden_dim
	N := snap.num_nodes
	E_count := snap.num_edges

	if N == 0 || E_count == 0 {
		d_base := make([]f32, max(N * H, 1))
		for i in 0 ..< N * H {d_base[i] = d_output[i]}
		return d_base
	}

	strand_ensure_grads(s)
	sg := strand_get_grad_slices(s)
	lw := &s.layer
	lc := layer_cache

	d_update_relu := make([]f32, N * H)
	defer delete(d_update_relu)
	layernorm_backward(
		d_update_relu,
		sg.ln_w,
		sg.ln_b,
		d_output,
		lc.update_relu,
		lw.ln_w,
		lc.ln_mean,
		lc.ln_rstd,
		N,
		H,
	)

	d_update_pre := make([]f32, N * H)
	defer delete(d_update_pre)
	relu_backward(d_update_pre, d_update_relu, lc.update_pre, N * H)

	d_update_input := make([]f32, N * 2 * H)
	defer delete(d_update_input)

	update_input := make([]f32, N * 2 * H)
	defer delete(update_input)
	concat2_forward(update_input, base_hidden, lc.aggregated, N, H)

	linear_backward(
		d_update_input,
		sg.w_upd,
		sg.b_upd,
		d_update_pre,
		update_input,
		lw.w_upd,
		N,
		2 * H,
		H,
	)

	d_base_from_update := make([]f32, N * H)
	d_aggregated := make([]f32, N * H)
	defer delete(d_base_from_update)
	defer delete(d_aggregated)
	concat2_backward(d_base_from_update, d_aggregated, d_update_input, N, H)

	d_messages := make([]f32, E_count * H)
	defer delete(d_messages)
	scatter_add_backward(d_messages, d_aggregated, snap.edge_dst, E_count, H)

	d_messages_pre := make([]f32, E_count * H)
	defer delete(d_messages_pre)
	relu_backward(d_messages_pre, d_messages, lc.messages_pre, E_count * H)

	src_gathered := make([]f32, E_count * H)
	dst_gathered := make([]f32, E_count * H)
	defer delete(src_gathered)
	defer delete(dst_gathered)
	for e in 0 ..< E_count {
		src := snap.edge_src[e]
		dst := snap.edge_dst[e]
		for i in 0 ..< H {
			src_gathered[e * H + i] = base_hidden[src * H + i]
			dst_gathered[e * H + i] = base_hidden[dst * H + i]
		}
	}
	msg_input := make([]f32, E_count * 3 * H)
	defer delete(msg_input)
	concat3_forward(msg_input, src_gathered, dst_gathered, edge_hidden, E_count, H)

	d_msg_input := make([]f32, E_count * 3 * H)
	defer delete(d_msg_input)
	linear_backward(
		d_msg_input,
		sg.w_msg,
		sg.b_msg,
		d_messages_pre,
		msg_input,
		lw.w_msg,
		E_count,
		3 * H,
		H,
	)

	d_src_gathered := make([]f32, E_count * H)
	d_dst_gathered := make([]f32, E_count * H)
	d_edge_hidden := make([]f32, E_count * H)
	defer delete(d_src_gathered)
	defer delete(d_dst_gathered)
	defer delete(d_edge_hidden)
	concat3_backward(d_src_gathered, d_dst_gathered, d_edge_hidden, d_msg_input, E_count, H)

	d_base := make([]f32, N * H)
	for i in 0 ..< N * H {d_base[i] = d_base_from_update[i]}
	for e in 0 ..< E_count {
		src := snap.edge_src[e]
		dst := snap.edge_dst[e]
		for i in 0 ..< H {
			d_base[src * H + i] += d_src_gathered[e * H + i]
			d_base[dst * H + i] += d_dst_gathered[e * H + i]
		}
	}

	return d_base
}

strand_adamw_step :: proc(s: ^StrandMPNN, lr: f32) {
	if s.grads_memory == nil {return}

	if s.m_memory == nil {
		s.m_memory = make([]f32, s.num_parameters)
		s.v_memory = make([]f32, s.num_parameters)
	}

	s.adam_t += 1
	t := s.adam_t

	for i in 0 ..< s.num_parameters {
		param := s.params_memory[i]
		grad := s.grads_memory[i]

		m := BETA1 * s.m_memory[i] + (1.0 - BETA1) * grad
		v := BETA2 * s.v_memory[i] + (1.0 - BETA2) * grad * grad

		m_hat := m / (1.0 - math.pow(BETA1, f32(t)))
		v_hat := v / (1.0 - math.pow(BETA2, f32(t)))

		s.m_memory[i] = m
		s.v_memory[i] = v
		s.params_memory[i] -= lr * (m_hat / (math.sqrt(v_hat) + ADAM_EPS) + WEIGHT_DECAY * param)
	}
}

strand_save :: proc(s: ^StrandMPNN, fd: os.Handle) -> bool {
	if s.params_memory == nil {return false}

	header: [6]i32
	header[0] = STRAND_MAGIC
	header[1] = STRAND_VERSION
	header[2] = i32(s.hidden_dim)
	header[3] = i32(s.num_parameters)
	header[4] = i32(s.adam_t)
	header[5] = 0

	written, _ := os.write(fd, mem.slice_to_bytes(header[:]))
	if written != size_of(header) {
		log.err("[gnn] strand header write failed")
		return false
	}

	param_bytes := mem.slice_to_bytes(s.params_memory)
	written2, _ := os.write(fd, param_bytes)
	if written2 != len(param_bytes) {
		log.err("[gnn] strand params write failed")
		return false
	}

	if s.m_memory != nil && s.v_memory != nil {
		m_bytes := mem.slice_to_bytes(s.m_memory)
		v_bytes := mem.slice_to_bytes(s.v_memory)
		os.write(fd, m_bytes)
		os.write(fd, v_bytes)
	}

	log.info("[gnn] strand saved: %d params, step %d", s.num_parameters, s.adam_t)
	return true
}

strand_load :: proc(s: ^StrandMPNN, data: []u8, offset: ^int) -> bool {
	header_size := 6 * size_of(i32)
	if offset^ + header_size > len(data) {
		log.err("[gnn] strand checkpoint too small")
		return false
	}

	header := (cast([^]i32)raw_data(data[offset^:]))[:6]
	if header[0] != STRAND_MAGIC {
		log.err("[gnn] bad magic in strand checkpoint")
		return false
	}
	if header[1] != STRAND_VERSION {
		log.err("[gnn] unsupported strand checkpoint version %d", header[1])
		return false
	}

	hidden_dim := int(header[2])
	num_params := int(header[3])
	adam_t := int(header[4])
	offset^ += header_size

	strand_create(s, hidden_dim)
	s.adam_t = adam_t

	if s.num_parameters != num_params {
		log.err(
			"[gnn] strand param count mismatch: file=%d, model=%d",
			num_params,
			s.num_parameters,
		)
		return false
	}

	param_bytes := num_params * size_of(f32)
	if offset^ + param_bytes > len(data) {
		log.err("[gnn] strand checkpoint truncated")
		return false
	}
	mem.copy(raw_data(s.params_memory), &data[offset^], param_bytes)
	offset^ += param_bytes

	optim_size := 2 * param_bytes
	if offset^ + optim_size <= len(data) {
		s.m_memory = make([]f32, num_params)
		s.v_memory = make([]f32, num_params)
		mem.copy(raw_data(s.m_memory), &data[offset^], param_bytes)
		offset^ += param_bytes
		mem.copy(raw_data(s.v_memory), &data[offset^], param_bytes)
		offset^ += param_bytes
	}

	log.info("[gnn] strand loaded: hidden=%d, %d params, step %d", hidden_dim, num_params, adam_t)
	return true
}

strand_save_bytes :: proc(s: ^StrandMPNN) -> []u8 {
	if s.params_memory == nil {return nil}

	header_size := 6 * size_of(i32)
	param_bytes := s.num_parameters * size_of(f32)
	optim_size := 0
	if s.m_memory != nil && s.v_memory != nil {
		optim_size = 2 * param_bytes
	}
	total := header_size + param_bytes + optim_size
	buf := make([]u8, total)

	header := (cast([^]i32)raw_data(buf))[:6]
	header[0] = STRAND_MAGIC
	header[1] = STRAND_VERSION
	header[2] = i32(s.hidden_dim)
	header[3] = i32(s.num_parameters)
	header[4] = i32(s.adam_t)
	header[5] = 0

	mem.copy(&buf[header_size], raw_data(s.params_memory), param_bytes)

	if s.m_memory != nil && s.v_memory != nil {
		mem.copy(&buf[header_size + param_bytes], raw_data(s.m_memory), param_bytes)
		mem.copy(&buf[header_size + 2 * param_bytes], raw_data(s.v_memory), param_bytes)
	}

	return buf
}
