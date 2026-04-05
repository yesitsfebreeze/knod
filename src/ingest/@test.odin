package ingest

import "core:strings"
import "core:testing"

import "../graph"
import "../provider"

Mock_State :: struct {
	decompose_result:        []string,
	embed_calls:             int,
	batch_link_calls:        int,
	batch_link_return_empty: bool, // When true, batch_link_reason returns no connections.
	label_calls:             int,
}

@(private = "file")
mock_embed_text :: proc(self: ^provider.Provider, text: string) -> (provider.Embedding, bool) {
	state := cast(^Mock_State)self.state
	state.embed_calls += 1

	emb: provider.Embedding
	emb[0] = f32(len(text)) / 100.0
	emb[1] = 0.5
	return emb, true
}

@(private = "file")
mock_decompose_text :: proc(
	self: ^provider.Provider,
	text: string,
	prompt: string,
	descriptor: string = "",
) -> (
	[]string,
	bool,
) {
	state := cast(^Mock_State)self.state
	if state.decompose_result == nil {
		return nil, false
	}

	result := make([]string, len(state.decompose_result))
	for s, i in state.decompose_result {
		result[i] = strings.clone(s)
	}
	return result, true
}

@(private = "file")
mock_batch_link_reason :: proc(
	self: ^provider.Provider,
	new_thought: string,
	candidates: []string,
) -> (
	[]provider.Batch_Link_Result,
	bool,
) {
	state := cast(^Mock_State)self.state
	state.batch_link_calls += 1

	if state.batch_link_return_empty {
		return make([]provider.Batch_Link_Result, 0), true
	}

	results := make([]provider.Batch_Link_Result, len(candidates))
	for i in 0 ..< len(candidates) {
		results[i] = provider.Batch_Link_Result {
			index     = i,
			reasoning = strings.clone("test batch connection"),
			weight    = 0.8,
		}
	}
	return results, true
}

@(private = "file")
mock_generate_answer :: proc(
	self: ^provider.Provider,
	query: string,
	context_text: string,
) -> (
	string,
	bool,
) {
	return strings.clone("mock answer"), true
}

@(private = "file")
mock_label_dimensions :: proc(
	self: ^provider.Provider,
	purpose: string,
	dim_thoughts: [][]string,
) -> (
	[]string,
	bool,
) {
	state := cast(^Mock_State)self.state
	state.label_calls += 1

	result := make([]string, len(dim_thoughts))
	for i in 0 ..< len(dim_thoughts) {
		result[i] = strings.clone("mock-label")
	}
	return result, true
}

@(private = "file")
mock_embed_texts :: proc(
	self: ^provider.Provider,
	texts: []string,
) -> (
	[]provider.Embedding,
	bool,
) {
	state := cast(^Mock_State)self.state
	result := make([]provider.Embedding, len(texts))
	for text, i in texts {
		state.embed_calls += 1
		result[i][0] = f32(len(text)) / 100.0
		result[i][1] = 0.5
	}
	return result, true
}

@(private = "file")
make_mock_provider :: proc(state: ^Mock_State) -> provider.Provider {
	return provider.Provider {
		embed_text = mock_embed_text,
		embed_texts = mock_embed_texts,
		decompose_text = mock_decompose_text,
		batch_link_reason = mock_batch_link_reason,
		generate_answer = mock_generate_answer,
		label_dimensions = mock_label_dimensions,
		state = state,
	}
}


@(test)
test_ingest_empty_text :: proc(t: ^testing.T) {
	g: graph.Graph
	graph.create(&g)
	defer graph.release(&g)

	state := Mock_State{}
	p := make_mock_provider(&state)

	result := ingest(&g, &p, "")
	testing.expect_value(t, result, 0)
	testing.expect_value(t, graph.thought_count(&g), 0)
}

@(test)
test_ingest_decompose_failure :: proc(t: ^testing.T) {
	g: graph.Graph
	graph.create(&g)
	defer graph.release(&g)

	state := Mock_State {
		decompose_result = nil,
	}
	p := make_mock_provider(&state)

	result := ingest(&g, &p, "some text")
	testing.expect_value(t, result, -1)
}

@(test)
test_ingest_single_thought :: proc(t: ^testing.T) {
	g: graph.Graph
	graph.create(&g)
	defer graph.release(&g)

	thoughts := []string{"The sky is blue"}
	state := Mock_State {
		decompose_result = thoughts,
	}
	p := make_mock_provider(&state)

	result := ingest(&g, &p, "The sky is blue in the daytime")
	testing.expect_value(t, result, 1)
	testing.expect_value(t, graph.thought_count(&g), 1)
	testing.expect_value(t, state.embed_calls, 1)
}

@(test)
test_ingest_drops_unconnected_when_mature :: proc(t: ^testing.T) {
	g: graph.Graph
	graph.create(&g)
	defer graph.release(&g)

	test_cfg := Config {
		max_similar        = 5,
		edge_threshold     = 0.0,
		maturity_threshold = 0,
	}

	emb: graph.Embedding
	emb[1] = 0.5
	for i in 0 ..< 6 {
		emb[0] = f32(i) * 0.1
		graph.add_thought(&g, "seed thought", "seed", emb, i64(1000 + i))
	}
	testing.expect_value(t, graph.thought_count(&g), 6)

	thoughts := []string{"Not relevant", "Also not relevant"}
	state := Mock_State {
		decompose_result        = thoughts,
		batch_link_return_empty = true,
	}
	p := make_mock_provider(&state)

	result := ingest(&g, &p, "some text about nothing", test_cfg)
	testing.expect_value(t, result, 0)
	testing.expect_value(t, graph.thought_count(&g), 6)
}

@(test)
test_ingest_keeps_unconnected_when_immature :: proc(t: ^testing.T) {
	g: graph.Graph
	graph.create(&g)
	defer graph.release(&g)

	thoughts := []string{"A new thought"}
	state := Mock_State {
		decompose_result        = thoughts,
		batch_link_return_empty = true,
	}
	p := make_mock_provider(&state)

	result := ingest(&g, &p, "some text")
	testing.expect_value(t, result, 1)
	testing.expect_value(t, graph.thought_count(&g), 1)
}

@(test)
test_ingest_creates_edges :: proc(t: ^testing.T) {
	g: graph.Graph
	graph.create(&g)
	defer graph.release(&g)

	emb: graph.Embedding
	emb[0] = 0.15
	emb[1] = 0.5
	graph.add_thought(&g, "Existing thought", "seed:1", emb, 1000)

	thoughts := []string{"The sky is blue"}
	state := Mock_State {
		decompose_result = thoughts,
	}
	p := make_mock_provider(&state)

	test_cfg := Config {
		max_similar        = 5,
		edge_threshold     = 0.0,
		maturity_threshold = 50,
	}

	result := ingest(&g, &p, "The sky is blue", test_cfg)
	testing.expect_value(t, result, 1)
	testing.expect_value(t, graph.thought_count(&g), 2)
	testing.expect(t, graph.edge_count(&g) > 0, "should have created at least one edge")
	testing.expect(t, state.batch_link_calls > 0, "should have called batch_link_reason")
}

@(test)
test_default_config :: proc(t: ^testing.T) {
	testing.expect_value(t, DEFAULT_CONFIG.max_similar, 5)
	testing.expect(
		t,
		DEFAULT_CONFIG.edge_threshold > 0.29 && DEFAULT_CONFIG.edge_threshold < 0.31,
		"default edge_threshold ~0.3",
	)
	testing.expect_value(t, DEFAULT_CONFIG.maturity_threshold, 50)
}
