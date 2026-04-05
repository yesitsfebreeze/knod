package gnn

import "../graph"
import "core:math"
import "core:os"
import "core:testing"

EPSILON :: 1e-4

@(private = "file")
approx_eq :: proc(a, b: f32, eps: f32 = EPSILON) -> bool {
	return math.abs(a - b) < eps
}

@(private = "file")
make_test_graph :: proc() -> graph.Graph {
	g: graph.Graph
	graph.create(&g)

	emb1: graph.Embedding
	emb1[0] = 1.0; emb1[1] = 0.0
	emb2: graph.Embedding
	emb2[0] = 0.0; emb2[1] = 1.0
	emb3: graph.Embedding
	emb3[0] = 0.7; emb3[1] = 0.7
	emb4: graph.Embedding
	emb4[0] = -1.0; emb4[1] = 0.0

	graph.add_thought(&g, "A", "t:1", emb1, 100)
	graph.add_thought(&g, "B", "t:2", emb2, 200)
	graph.add_thought(&g, "C", "t:3", emb3, 300)
	graph.add_thought(&g, "D", "t:4", emb4, 400)

	edge_emb1: graph.Embedding
	edge_emb1[2] = 1.0
	edge_emb2: graph.Embedding
	edge_emb2[3] = 1.0
	edge_emb3: graph.Embedding
	edge_emb3[4] = 1.0

	graph.add_edge(&g, 1, 2, 0.8, "A relates to B", edge_emb1, 500)
	graph.add_edge(&g, 2, 3, 0.6, "B relates to C", edge_emb2, 600)
	graph.add_edge(&g, 1, 3, 0.9, "A strongly relates to C", edge_emb3, 700)

	return g
}


@(test)
test_create_model :: proc(t: ^testing.T) {
	model: MPNN
	create(&model, 32, 2)
	defer release(&model)

	testing.expect_value(t, model.hidden_dim, 32)
	testing.expect_value(t, model.num_layers, 2)
	testing.expect(t, model.num_parameters > 0, "should have parameters")
	testing.expect(t, model.params_memory != nil, "params should be allocated")

	expected := count_parameters(32, 2)
	testing.expect_value(t, model.num_parameters, expected)
}


@(test)
test_build_snapshot :: proc(t: ^testing.T) {
	g := make_test_graph()
	defer graph.release(&g)

	snap := build_snapshot(&g)
	defer release_snapshot(&snap)

	testing.expect_value(t, snap.num_nodes, 4)
	testing.expect_value(t, snap.num_edges, 3)
	testing.expect(t, snap.node_ids != nil, "node_ids should be allocated")
	testing.expect(t, snap.edge_src != nil, "edge_src should be allocated")
	testing.expect(t, snap.edge_dst != nil, "edge_dst should be allocated")
	testing.expect(t, snap.edge_weights != nil, "edge_weights should be allocated")
	testing.expect(t, snap.incoming_offsets != nil, "incoming_offsets should be allocated")
}


@(test)
test_build_snapshot_empty :: proc(t: ^testing.T) {
	g: graph.Graph
	graph.create(&g)
	defer graph.release(&g)

	snap := build_snapshot(&g)
	defer release_snapshot(&snap)

	testing.expect_value(t, snap.num_nodes, 0)
	testing.expect_value(t, snap.num_edges, 0)
}


@(test)
test_forward_pass :: proc(t: ^testing.T) {
	model: MPNN
	create(&model, 16, 2)
	defer release(&model)

	g := make_test_graph()
	defer graph.release(&g)

	snap := build_snapshot(&g)
	defer release_snapshot(&snap)

	cache: ForwardCache
	forward(&model, &snap, &cache)
	defer release_cache(&cache)

	testing.expect(t, cache.final_hidden != nil, "final_hidden should be allocated")
	testing.expect(t, cache.relevance_scores != nil, "relevance_scores should be allocated")

	scores_ok := true
	for i in 0 ..< snap.num_nodes {
		if math.is_nan(cache.relevance_scores[i]) || math.is_inf(cache.relevance_scores[i]) {
			scores_ok = false
			break
		}
	}
	testing.expect(t, scores_ok, "relevance scores should be finite")
}


@(test)
test_forward_no_edges :: proc(t: ^testing.T) {
	model: MPNN
	create(&model, 16, 2)
	defer release(&model)

	g: graph.Graph
	graph.create(&g)
	defer graph.release(&g)

	emb1: graph.Embedding
	emb1[0] = 1.0
	emb2: graph.Embedding
	emb2[1] = 1.0
	graph.add_thought(&g, "A", "t:1", emb1, 100)
	graph.add_thought(&g, "B", "t:2", emb2, 200)

	snap := build_snapshot(&g)
	defer release_snapshot(&snap)

	cache: ForwardCache
	forward(&model, &snap, &cache)
	defer release_cache(&cache)

	testing.expect(t, cache.relevance_scores != nil, "should produce scores even with no edges")
}


@(test)
test_train_step :: proc(t: ^testing.T) {
	model: MPNN
	create(&model, 16, 2)
	defer release(&model)

	strand: StrandMPNN
	strand_create(&strand, 16)
	defer strand_release(&strand)

	g := make_test_graph()
	defer graph.release(&g)

	link_loss, rel_loss := train_step(&model, &strand, &g, 1e-3)

	testing.expect(t, link_loss >= 0, "link loss should be non-negative")
	testing.expect(t, rel_loss >= 0, "relevance loss should be non-negative")

	testing.expect(t, model.grads_memory != nil, "grads should be allocated")

	testing.expect(t, model.m_memory != nil, "adam m should be allocated")
	testing.expect(t, model.v_memory != nil, "adam v should be allocated")
	testing.expect_value(t, model.adam_t, 1)
}

@(test)
test_training_reduces_loss :: proc(t: ^testing.T) {
	model: MPNN
	create(&model, 16, 2)
	defer release(&model)

	strand: StrandMPNN
	strand_create(&strand, 16)
	defer strand_release(&strand)

	g := make_test_graph()
	defer graph.release(&g)

	snap := build_snapshot(&g)
	cache: ForwardCache
	forward(&model, &snap, &cache)
	initial_link_loss := link_prediction_loss(&cache, &snap, model.hidden_dim, nil)
	release_cache(&cache)
	release_snapshot(&snap)

	for _ in 0 ..< 20 {
		train_step(&model, &strand, &g, 1e-3)
	}

	snap2 := build_snapshot(&g)
	cache2: ForwardCache
	forward(&model, &snap2, &cache2)
	final_link_loss := link_prediction_loss(&cache2, &snap2, model.hidden_dim, nil)
	release_cache(&cache2)
	release_snapshot(&snap2)

	testing.expect(
		t,
		final_link_loss <= initial_link_loss + 0.1,
		"loss should not increase significantly after training",
	)
}


@(test)
test_score_nodes :: proc(t: ^testing.T) {
	model: MPNN
	create(&model, 16, 2)
	defer release(&model)

	strand: StrandMPNN
	strand_create(&strand, 16)
	defer strand_release(&strand)

	g := make_test_graph()
	defer graph.release(&g)

	for _ in 0 ..< 5 {
		train_step(&model, &strand, &g, 1e-3)
	}

	snap := build_snapshot(&g)
	defer release_snapshot(&snap)

	query: Embedding
	query[0] = 1.0

	results := score_nodes(&model, &strand, &snap, &query, 3)
	defer delete(results)

	testing.expect_value(t, len(results), 3)

	for i in 1 ..< len(results) {
		testing.expect(
			t,
			results[i].score <= results[i - 1].score,
			"results should be sorted by score descending",
		)
	}
}


@(test)
test_checkpoint :: proc(t: ^testing.T) {
	model: MPNN
	create(&model, 16, 2)
	defer release(&model)

	strand: StrandMPNN
	strand_create(&strand, 16)
	defer strand_release(&strand)

	g := make_test_graph()
	defer graph.release(&g)

	train_step(&model, &strand, &g, 1e-3)
	train_step(&model, &strand, &g, 1e-3)

	path := "test_gnn_checkpoint.bin"
	save_ok := save_checkpoint(&model, path)
	testing.expect(t, save_ok, "save should succeed")

	model2: MPNN
	load_ok := load_checkpoint(&model2, path)
	defer release(&model2)
	testing.expect(t, load_ok, "load should succeed")

	testing.expect_value(t, model2.hidden_dim, model.hidden_dim)
	testing.expect_value(t, model2.num_layers, model.num_layers)
	testing.expect_value(t, model2.num_parameters, model.num_parameters)
	testing.expect_value(t, model2.adam_t, model.adam_t)

	params_match := true
	for i in 0 ..< model.num_parameters {
		if !approx_eq(model.params_memory[i], model2.params_memory[i]) {
			params_match = false
			break
		}
	}
	testing.expect(t, params_match, "loaded params should match saved params")

	os.remove(path)
}


@(test)
test_linear_forward :: proc(t: ^testing.T) {
	inp := []f32{1, 2, 3, 4, 5, 6}
	weight := []f32{1, 0, 0, 0, 1, 0}
	bias := []f32{0.5, 0.5}
	out := make([]f32, 4)
	defer delete(out)

	linear_forward(out, inp, weight, bias, 2, 3, 2)

	testing.expect(t, approx_eq(out[0], 1.5), "linear out[0] should be 1.5")
	testing.expect(t, approx_eq(out[1], 2.5), "linear out[1] should be 2.5")
}

@(test)
test_relu :: proc(t: ^testing.T) {
	inp := []f32{-1, 0, 1, -0.5, 2}
	out := make([]f32, 5)
	defer delete(out)

	relu_forward(out, inp, 5)

	testing.expect(t, approx_eq(out[0], 0), "relu(-1) = 0")
	testing.expect(t, approx_eq(out[1], 0), "relu(0) = 0")
	testing.expect(t, approx_eq(out[2], 1), "relu(1) = 1")
	testing.expect(t, approx_eq(out[3], 0), "relu(-0.5) = 0")
	testing.expect(t, approx_eq(out[4], 2), "relu(2) = 2")
}

@(test)
test_scatter_add :: proc(t: ^testing.T) {
	messages := []f32{1, 2, 3, 4, 5, 6}
	edge_dst := []int{0, 1, 0}
	out := make([]f32, 4)
	defer delete(out)
	for i in 0 ..< 4 {out[i] = 0}

	scatter_add(out, messages, edge_dst, 3, 2)

	testing.expect(t, approx_eq(out[0], 6), "scatter node0 dim0")
	testing.expect(t, approx_eq(out[1], 8), "scatter node0 dim1")
	testing.expect(t, approx_eq(out[2], 3), "scatter node1 dim0")
	testing.expect(t, approx_eq(out[3], 4), "scatter node1 dim1")
}


@(test)
test_strand_create :: proc(t: ^testing.T) {
	strand: StrandMPNN
	strand_create(&strand, 16)
	defer strand_release(&strand)

	testing.expect_value(t, strand.hidden_dim, 16)
	testing.expect(t, strand.num_parameters > 0, "strand should have parameters")
	testing.expect(t, strand.params_memory != nil, "strand params should be allocated")

	expected := strand_count_parameters(16)
	testing.expect_value(t, strand.num_parameters, expected)
}

@(test)
test_strand_forward :: proc(t: ^testing.T) {
	model: MPNN
	create(&model, 16, 2)
	defer release(&model)

	strand: StrandMPNN
	strand_create(&strand, 16)
	defer strand_release(&strand)

	g := make_test_graph()
	defer graph.release(&g)

	snap := build_snapshot(&g)
	defer release_snapshot(&snap)

	cache: ForwardCache
	forward(&model, &snap, &cache)
	defer release_cache(&cache)

	strand_forward(&strand, cache.final_hidden, &snap, &cache)

	testing.expect(t, cache.strand_hidden != nil, "strand_hidden should be allocated")

	scores_ok := true
	H := model.hidden_dim
	for i in 0 ..< snap.num_nodes * H {
		if math.is_nan(cache.strand_hidden[i]) || math.is_inf(cache.strand_hidden[i]) {
			scores_ok = false
			break
		}
	}
	testing.expect(t, scores_ok, "strand_hidden values should be finite")
}

@(test)
test_strand_forward_no_edges :: proc(t: ^testing.T) {
	model: MPNN
	create(&model, 16, 2)
	defer release(&model)

	strand: StrandMPNN
	strand_create(&strand, 16)
	defer strand_release(&strand)

	g: graph.Graph
	graph.create(&g)
	defer graph.release(&g)

	emb1: graph.Embedding
	emb1[0] = 1.0
	emb2: graph.Embedding
	emb2[1] = 1.0
	graph.add_thought(&g, "A", "t:1", emb1, 100)
	graph.add_thought(&g, "B", "t:2", emb2, 200)

	snap := build_snapshot(&g)
	defer release_snapshot(&snap)

	cache: ForwardCache
	forward(&model, &snap, &cache)
	defer release_cache(&cache)

	strand_forward(&strand, cache.final_hidden, &snap, &cache)

	testing.expect(
		t,
		cache.strand_hidden != nil,
		"strand_hidden should be allocated even with no edges",
	)

	H := model.hidden_dim
	match := true
	for i in 0 ..< snap.num_nodes * H {
		if !approx_eq(cache.strand_hidden[i], cache.final_hidden[i]) {
			match = false
			break
		}
	}
	testing.expect(t, match, "strand output should match base output when no edges")
}

@(test)
test_strand_checkpoint :: proc(t: ^testing.T) {
	strand: StrandMPNN
	strand_create(&strand, 16)
	defer strand_release(&strand)

	model: MPNN
	create(&model, 16, 2)
	defer release(&model)

	g := make_test_graph()
	defer graph.release(&g)

	train_step(&model, &strand, &g, 1e-3)

	bytes := strand_save_bytes(&strand)
	testing.expect(t, bytes != nil, "strand_save_bytes should return data")
	defer delete(bytes)

	strand2: StrandMPNN
	defer strand_release(&strand2)
	off := 0
	load_ok := strand_load(&strand2, bytes, &off)
	testing.expect(t, load_ok, "strand_load should succeed")

	testing.expect_value(t, strand2.hidden_dim, strand.hidden_dim)
	testing.expect_value(t, strand2.num_parameters, strand.num_parameters)
	testing.expect_value(t, strand2.adam_t, strand.adam_t)

	params_match := true
	for i in 0 ..< strand.num_parameters {
		if !approx_eq(strand.params_memory[i], strand2.params_memory[i]) {
			params_match = false
			break
		}
	}
	testing.expect(t, params_match, "loaded strand params should match saved")
}

@(test)
test_train_strand :: proc(t: ^testing.T) {
	model: MPNN
	create(&model, 16, 2)
	defer release(&model)

	strand: StrandMPNN
	strand_create(&strand, 16)
	defer strand_release(&strand)

	g := make_test_graph()
	defer graph.release(&g)

	train_strand(&model, &strand, &g, 5)

	testing.expect(
		t,
		strand.grads_memory != nil,
		"strand grads should be allocated after training",
	)
	testing.expect(t, strand.m_memory != nil, "strand adam m should be allocated")
	testing.expect(t, strand.v_memory != nil, "strand adam v should be allocated")
	testing.expect_value(t, strand.adam_t, 5)
}

@(test)
test_train_base_refine :: proc(t: ^testing.T) {
	model: MPNN
	create(&model, 16, 2)
	defer release(&model)

	strand: StrandMPNN
	strand_create(&strand, 16)
	defer strand_release(&strand)

	g := make_test_graph()
	defer graph.release(&g)

	initial_adam_t := model.adam_t

	train_base_refine(&model, &strand, &g, 3)

	testing.expect(t, model.grads_memory != nil, "base grads should be allocated after refine")
	testing.expect_value(t, model.adam_t, initial_adam_t + 3)
}

@(test)
test_score_nodes_two_model :: proc(t: ^testing.T) {
	model: MPNN
	create(&model, 16, 2)
	defer release(&model)

	strand: StrandMPNN
	strand_create(&strand, 16)
	defer strand_release(&strand)

	g := make_test_graph()
	defer graph.release(&g)

	train_strand(&model, &strand, &g, 3)
	train_base_refine(&model, &strand, &g, 2)

	snap := build_snapshot(&g)
	defer release_snapshot(&snap)

	query: Embedding
	query[0] = 1.0

	results := score_nodes(&model, &strand, &snap, &query, 4)
	defer delete(results)

	testing.expect_value(t, len(results), 4)

	for i in 1 ..< len(results) {
		testing.expect(
			t,
			results[i].score <= results[i - 1].score,
			"results should be sorted by score descending",
		)
	}

	results_base := score_nodes(&model, nil, &snap, &query, 4)
	defer delete(results_base)

	testing.expect_value(t, len(results_base), 4)
}
