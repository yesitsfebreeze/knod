package gnn


Embedding :: [EMBEDDING_DIM]f32

MPNNLayerWeights :: struct {
	w_msg: []f32, // [H, 3*H]
	b_msg: []f32, // [H]
	w_upd: []f32, // [H, 2*H]
	b_upd: []f32, // [H]
	ln_w:  []f32, // [H]
	ln_b:  []f32, // [H]
}

MPNNLayerActivations :: struct {
	messages:     []f32,
	messages_pre: []f32,
	aggregated:   []f32,
	update_pre:   []f32,
	update_relu:  []f32,
	ln_mean:      []f32,
	ln_rstd:      []f32,
	output:       []f32,
}

MPNN :: struct {
	hidden_dim:     int,
	num_layers:     int,
	w_node_proj:    []f32, // [H, EMBEDDING_DIM]
	b_node_proj:    []f32, // [H]
	w_edge_proj:    []f32, // [H, EMBEDDING_DIM]
	b_edge_proj:    []f32, // [H]
	layers:         []MPNNLayerWeights,
	w_relevance:    []f32, // [H] -- applied to strand_hidden when strand present
	b_relevance:    f32,
	params_memory:  []f32,
	num_parameters: int,
	grads_memory:   []f32,
	m_memory:       []f32,
	v_memory:       []f32,
	adam_t:         int,
	rng:            u64,
}


StrandMPNN :: struct {
	hidden_dim:     int,
	layer:          MPNNLayerWeights,
	params_memory:  []f32,
	num_parameters: int,
	grads_memory:   []f32,
	m_memory:       []f32,
	v_memory:       []f32,
	adam_t:         int,
}

GraphSnapshot :: struct {
	num_nodes:        int,
	num_edges:        int,
	node_embeddings:  []f32, // num_nodes * EMBEDDING_DIM
	edge_embeddings:  []f32, // num_edges * EMBEDDING_DIM
	edge_src:         []int,
	edge_dst:         []int,
	edge_weights:     []f32,
	node_ids:         []u64,
	incoming_offsets: []int, // num_nodes + 1 (CSR)
	incoming_edges:   []int,
}

ForwardCache :: struct {
	node_hidden:      []f32, // num_nodes * H
	edge_hidden:      []f32, // num_edges * H
	layer_caches:     []MPNNLayerActivations,
	final_hidden:     []f32, // num_nodes * H  (base output)
	strand_cache:     MPNNLayerActivations,
	strand_hidden:    []f32, // num_nodes * H  (nil => use final_hidden)
	relevance_scores: []f32, // num_nodes
}

ScoredNode :: struct {
	node_idx: int,
	node_id:  u64,
	score:    f32,
}
