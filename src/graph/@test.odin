package graph

import "core:math"
import "core:os"
import "core:testing"

EPSILON :: 1e-4

@(private = "file")
approx_eq :: proc(a, b: f32, eps: f32 = EPSILON) -> bool {
	return math.abs(a - b) < eps
}

@(private = "file")
make_embedding :: proc(dim_val: f32, dim_idx: int = 0) -> Embedding {
	emb: Embedding
	emb[dim_idx] = dim_val
	return emb
}

@(test)
test_create_and_free :: proc(t: ^testing.T) {
	g: Graph
	create(&g)
	defer release(&g)

	testing.expect_value(t, thought_count(&g), 0)
	testing.expect_value(t, edge_count(&g), 0)
}

@(test)
test_add_thought :: proc(t: ^testing.T) {
	g: Graph
	create(&g)
	defer release(&g)

	emb := make_embedding(1.0)
	id := add_thought(&g, "The sky is blue", "test:1", emb, 1000)

	testing.expect_value(t, id, 1)
	testing.expect_value(t, thought_count(&g), 1)

	thought := get_thought(&g, id)
	testing.expect(t, thought != nil, "thought should exist")
	testing.expect(t, thought.text == "The sky is blue", "text should match")
	testing.expect(t, thought.source_id == "test:1", "source_id should match")
	testing.expect_value(t, thought.created_at, i64(1000))
	testing.expect_value(t, thought.access_count, u32(0))
}

@(test)
test_add_multiple_thoughts :: proc(t: ^testing.T) {
	g: Graph
	create(&g)
	defer release(&g)

	id1 := add_thought(&g, "First", "t:1", make_embedding(1.0), 100)
	id2 := add_thought(&g, "Second", "t:2", make_embedding(2.0), 200)
	id3 := add_thought(&g, "Third", "t:3", make_embedding(3.0), 300)

	testing.expect_value(t, id1, 1)
	testing.expect_value(t, id2, 2)
	testing.expect_value(t, id3, 3)
	testing.expect_value(t, thought_count(&g), 3)
}

@(test)
test_add_edge :: proc(t: ^testing.T) {
	g: Graph
	create(&g)
	defer release(&g)

	id1 := add_thought(&g, "A", "t:1", make_embedding(1.0), 100)
	id2 := add_thought(&g, "B", "t:2", make_embedding(2.0), 200)

	ok := add_edge(&g, id1, id2, 0.8, "A supports B", make_embedding(0.5), 300)
	testing.expect(t, ok, "edge should be added successfully")
	testing.expect_value(t, edge_count(&g), 1)
}

@(test)
test_add_edge_invalid_thoughts :: proc(t: ^testing.T) {
	g: Graph
	create(&g)
	defer release(&g)

	id1 := add_thought(&g, "A", "t:1", make_embedding(1.0), 100)

	ok := add_edge(&g, id1, 999, 0.5, "invalid", make_embedding(0.0), 100)
	testing.expect(t, !ok, "edge to nonexistent thought should fail")
	testing.expect_value(t, edge_count(&g), 0)
}

@(test)
test_adjacency :: proc(t: ^testing.T) {
	g: Graph
	create(&g)
	defer release(&g)

	id1 := add_thought(&g, "A", "t:1", make_embedding(1.0), 100)
	id2 := add_thought(&g, "B", "t:2", make_embedding(2.0), 200)
	id3 := add_thought(&g, "C", "t:3", make_embedding(3.0), 300)

	add_edge(&g, id1, id2, 0.8, "A->B", make_embedding(0.1), 400)
	add_edge(&g, id1, id3, 0.6, "A->C", make_embedding(0.2), 500)
	add_edge(&g, id2, id3, 0.9, "B->C", make_embedding(0.3), 600)

	out := outgoing(&g, id1)
	defer delete(out)
	testing.expect_value(t, len(out), 2)

	inc := incoming(&g, id3)
	defer delete(inc)
	testing.expect_value(t, len(inc), 2)

	out2 := outgoing(&g, id2)
	defer delete(out2)
	testing.expect_value(t, len(out2), 1)

	inc2 := incoming(&g, id2)
	defer delete(inc2)
	testing.expect_value(t, len(inc2), 1)
}

@(test)
test_touch :: proc(t: ^testing.T) {
	g: Graph
	create(&g)
	defer release(&g)

	id := add_thought(&g, "A", "t:1", make_embedding(1.0), 100)
	touch(&g, id, 500)
	touch(&g, id, 600)

	thought := get_thought(&g, id)
	testing.expect_value(t, thought.access_count, u32(2))
	testing.expect_value(t, thought.last_accessed, i64(600))
}

@(test)
test_cosine_identical :: proc(t: ^testing.T) {
	emb := make_embedding(1.0)
	sim := cosine_similarity(&emb, &emb)
	testing.expect(t, approx_eq(sim, 1.0), "identical embeddings should have similarity 1.0")
}

@(test)
test_cosine_orthogonal :: proc(t: ^testing.T) {
	a := make_embedding(1.0, 0)
	b := make_embedding(1.0, 1)
	sim := cosine_similarity(&a, &b)
	testing.expect(t, approx_eq(sim, 0.0), "orthogonal embeddings should have similarity 0.0")
}

@(test)
test_cosine_opposite :: proc(t: ^testing.T) {
	a := make_embedding(1.0, 0)
	b := make_embedding(-1.0, 0)
	sim := cosine_similarity(&a, &b)
	testing.expect(t, approx_eq(sim, -1.0), "opposite embeddings should have similarity -1.0")
}

@(test)
test_find_thoughts :: proc(t: ^testing.T) {
	g: Graph
	create(&g)
	defer release(&g)

	add_thought(&g, "A", "t:1", make_embedding(1.0, 0), 100)
	add_thought(&g, "B", "t:2", make_embedding(1.0, 1), 200)
	add_thought(&g, "C", "t:3", make_embedding(1.0, 0), 300)

	query := make_embedding(1.0, 0)
	results := find_thoughts(&g, &query, 2)
	defer delete(results)

	testing.expect_value(t, len(results), 2)
	testing.expect(t, approx_eq(results[0].score, 1.0), "top result should have similarity 1.0")
	testing.expect(
		t,
		approx_eq(results[1].score, 1.0),
		"second result should also have similarity 1.0",
	)
}

@(test)
test_find_thoughts_empty_graph :: proc(t: ^testing.T) {
	g: Graph
	create(&g)
	defer release(&g)

	query := make_embedding(1.0)
	results := find_thoughts(&g, &query, 5)
	testing.expect_value(t, len(results), 0)
}

@(test)
test_find_edges :: proc(t: ^testing.T) {
	g: Graph
	create(&g)
	defer release(&g)

	id1 := add_thought(&g, "A", "t:1", make_embedding(1.0, 0), 100)
	id2 := add_thought(&g, "B", "t:2", make_embedding(1.0, 1), 200)
	id3 := add_thought(&g, "C", "t:3", make_embedding(1.0, 2), 300)

	add_edge(&g, id1, id2, 0.8, "color relation", make_embedding(1.0, 3), 400)
	add_edge(&g, id2, id3, 0.9, "size relation", make_embedding(1.0, 4), 500)

	query := make_embedding(1.0, 3)
	results := find_edges(&g, &query, 1)
	defer delete(results)

	testing.expect_value(t, len(results), 1)
	testing.expect_value(t, results[0].edge_index, 0)
	testing.expect(t, approx_eq(results[0].score, 1.0), "matching edge should have similarity 1.0")
}

@(test)
test_save_and_load :: proc(t: ^testing.T) {
	path := "_test_graph.bin"
	defer os.remove(path)

	g: Graph
	create(&g)

	id1 := add_thought(&g, "The sky is blue", "src:1", make_embedding(1.0, 0), 1000)
	id2 := add_thought(&g, "Water is wet", "src:2", make_embedding(1.0, 1), 2000)
	add_edge(&g, id1, id2, 0.7, "both natural", make_embedding(0.5, 2), 3000)
	touch(&g, id1, 4000)

	ok := save(&g, path)
	testing.expect(t, ok, "save should succeed")
	release(&g)

	g2: Graph
	create(&g2)
	defer release(&g2)

	ok2 := load(&g2, path)
	testing.expect(t, ok2, "load should succeed")

	testing.expect_value(t, thought_count(&g2), 2)
	testing.expect_value(t, edge_count(&g2), 1)
	testing.expect_value(t, g2.next_id, 3)

	t1 := get_thought(&g2, id1)
	testing.expect(t, t1 != nil, "thought 1 should exist")
	testing.expect(t, t1.text == "The sky is blue", "text should survive round-trip")
	testing.expect(t, t1.source_id == "src:1", "source_id should survive round-trip")
	testing.expect_value(t, t1.access_count, u32(1))
	testing.expect_value(t, t1.last_accessed, i64(4000))

	t2 := get_thought(&g2, id2)
	testing.expect(t, t2 != nil, "thought 2 should exist")
	testing.expect(t, t2.text == "Water is wet", "text should survive round-trip")

	testing.expect_value(t, g2.edges[0].source_id, id1)
	testing.expect_value(t, g2.edges[0].target_id, id2)
	testing.expect(t, approx_eq(g2.edges[0].weight, 0.7), "weight should survive round-trip")
	testing.expect(
		t,
		g2.edges[0].reasoning == "both natural",
		"reasoning should survive round-trip",
	)

	out := outgoing(&g2, id1)
	defer delete(out)
	testing.expect_value(t, len(out), 1)

	inc := incoming(&g2, id2)
	defer delete(inc)
	testing.expect_value(t, len(inc), 1)
}

@(test)
test_load_nonexistent :: proc(t: ^testing.T) {
	g: Graph
	create(&g)
	defer release(&g)

	ok := load(&g, "_nonexistent_file.bin")
	testing.expect(t, !ok, "loading nonexistent file should fail")
}

@(test)
test_save_empty_graph :: proc(t: ^testing.T) {
	path := "_test_empty_graph.bin"
	defer os.remove(path)

	g: Graph
	create(&g)

	ok := save(&g, path)
	testing.expect(t, ok, "saving empty graph should succeed")
	release(&g)

	g2: Graph
	create(&g2)
	defer release(&g2)

	ok2 := load(&g2, path)
	testing.expect(t, ok2, "loading empty graph should succeed")
	testing.expect_value(t, thought_count(&g2), 0)
	testing.expect_value(t, edge_count(&g2), 0)
}


@(test)
test_profile_running_mean :: proc(t: ^testing.T) {
	g: Graph
	create(&g)
	defer release(&g)

	emb1 := make_embedding(1.0, 0)
	add_thought(&g, "A", "t:1", emb1, 100)

	testing.expect_value(t, g.profile_count, 1)
	testing.expect(t, approx_eq(g.profile[0], 1.0), "profile[0] should be 1.0 after first thought")

	emb2 := make_embedding(1.0, 1)
	add_thought(&g, "B", "t:2", emb2, 200)

	testing.expect_value(t, g.profile_count, 2)
	testing.expect(t, approx_eq(g.profile[0], 0.5), "profile[0] should be 0.5 after two thoughts")
	testing.expect(t, approx_eq(g.profile[1], 0.5), "profile[1] should be 0.5 after two thoughts")

	emb3 := make_embedding(1.0, 0)
	add_thought(&g, "C", "t:3", emb3, 300)

	testing.expect_value(t, g.profile_count, 3)
	testing.expect(
		t,
		approx_eq(g.profile[0], 2.0 / 3.0),
		"profile[0] should be 2/3 after three thoughts",
	)
	testing.expect(
		t,
		approx_eq(g.profile[1], 1.0 / 3.0),
		"profile[1] should be 1/3 after three thoughts",
	)
}

@(test)
test_top_dimensions_basic :: proc(t: ^testing.T) {
	g: Graph
	create(&g)
	defer release(&g)

	emb1: Embedding
	emb1[5] = 1.0
	emb1[10] = 0.5
	emb1[20] = 0.2
	add_thought(&g, "A", "t:1", emb1, 100)

	dims := top_dimensions(&g, 3)
	defer delete(dims)

	testing.expect_value(t, len(dims), 3)
	testing.expect_value(t, dims[0].index, 5)
	testing.expect(t, approx_eq(dims[0].value, 1.0), "top dim value should be 1.0")
	testing.expect_value(t, dims[1].index, 10)
	testing.expect(t, approx_eq(dims[1].value, 0.5), "second dim value should be 0.5")
	testing.expect_value(t, dims[2].index, 20)
	testing.expect(t, approx_eq(dims[2].value, 0.2), "third dim value should be 0.2")
}

@(test)
test_top_dimensions_empty_graph :: proc(t: ^testing.T) {
	g: Graph
	create(&g)
	defer release(&g)

	dims := top_dimensions(&g, 5)
	testing.expect_value(t, len(dims), 0)
}

@(test)
test_set_get_tags :: proc(t: ^testing.T) {
	g: Graph
	create(&g)
	defer release(&g)

	tags := []Tag{{dim_index = 5, label = "science"}, {dim_index = 10, label = "physics"}}
	set_tags(&g, tags)

	result := get_tags(&g)
	testing.expect_value(t, len(result), 2)
	testing.expect_value(t, result[0].dim_index, 5)
	testing.expect(t, result[0].label == "science", "first tag label should be 'science'")
	testing.expect_value(t, result[1].dim_index, 10)
	testing.expect(t, result[1].label == "physics", "second tag label should be 'physics'")
}

@(test)
test_set_tags_replaces :: proc(t: ^testing.T) {
	g: Graph
	create(&g)
	defer release(&g)

	tags1 := []Tag{{dim_index = 1, label = "old"}}
	set_tags(&g, tags1)
	testing.expect_value(t, len(get_tags(&g)), 1)

	tags2 := []Tag{{dim_index = 2, label = "new1"}, {dim_index = 3, label = "new2"}}
	set_tags(&g, tags2)
	testing.expect_value(t, len(get_tags(&g)), 2)
	testing.expect_value(t, get_tags(&g)[0].dim_index, 2)
}

@(test)
test_changed_dimensions :: proc(t: ^testing.T) {
	g: Graph
	create(&g)
	defer release(&g)

	tags := []Tag{{dim_index = 5, label = "known"}, {dim_index = 10, label = "also-known"}}
	set_tags(&g, tags)

	new_dims := []DimEntry {
		{index = 5, value = 1.0},
		{index = 15, value = 0.8},
		{index = 20, value = 0.3},
	}

	changed := changed_dimensions(&g, new_dims)
	defer delete(changed)

	testing.expect_value(t, len(changed), 2)
	testing.expect_value(t, changed[0], 15)
	testing.expect_value(t, changed[1], 20)
}

@(test)
test_changed_dimensions_all_known :: proc(t: ^testing.T) {
	g: Graph
	create(&g)
	defer release(&g)

	tags := []Tag{{dim_index = 5, label = "known"}}
	set_tags(&g, tags)

	new_dims := []DimEntry{{index = 5, value = 1.0}}
	changed := changed_dimensions(&g, new_dims)
	testing.expect_value(t, len(changed), 0)
}

@(test)
test_profile_tags_persistence :: proc(t: ^testing.T) {
	path := "_test_profile_tags.bin"
	defer os.remove(path)

	g: Graph
	create(&g)

	emb1: Embedding
	emb1[5] = 1.0
	emb1[10] = 0.5
	add_thought(&g, "Alpha", "src:1", emb1, 1000)

	emb2: Embedding
	emb2[5] = 0.8
	emb2[20] = 0.3
	add_thought(&g, "Beta", "src:2", emb2, 2000)

	tags := []Tag{{dim_index = 5, label = "science"}, {dim_index = 10, label = "physics"}}
	set_tags(&g, tags)

	ok := save(&g, path)
	testing.expect(t, ok, "save should succeed")
	saved_profile_count := g.profile_count
	saved_profile_5 := g.profile[5]
	saved_profile_10 := g.profile[10]
	saved_profile_20 := g.profile[20]
	release(&g)

	g2: Graph
	create(&g2)
	defer release(&g2)

	ok2 := load(&g2, path)
	testing.expect(t, ok2, "load should succeed")

	testing.expect_value(t, g2.profile_count, saved_profile_count)
	testing.expect(
		t,
		approx_eq(g2.profile[5], saved_profile_5),
		"profile[5] should survive round-trip",
	)
	testing.expect(
		t,
		approx_eq(g2.profile[10], saved_profile_10),
		"profile[10] should survive round-trip",
	)
	testing.expect(
		t,
		approx_eq(g2.profile[20], saved_profile_20),
		"profile[20] should survive round-trip",
	)

	loaded_tags := get_tags(&g2)
	testing.expect_value(t, len(loaded_tags), 2)
	testing.expect_value(t, loaded_tags[0].dim_index, 5)
	testing.expect(t, loaded_tags[0].label == "science", "tag label should survive round-trip")
	testing.expect_value(t, loaded_tags[1].dim_index, 10)
	testing.expect(t, loaded_tags[1].label == "physics", "tag label should survive round-trip")

	testing.expect_value(t, thought_count(&g2), 2)
}

@(test)
test_set_get_descriptor :: proc(t: ^testing.T) {
	g: Graph
	create(&g)
	defer release(&g)

	testing.expect_value(t, descriptor_count(&g), 0)

	set_descriptor(&g, "jira", "This is a Jira ticket.")
	testing.expect_value(t, descriptor_count(&g), 1)

	d := get_descriptor(&g, "jira")
	testing.expect(t, d != nil, "descriptor should exist")
	testing.expect(t, d.name == "jira", "name should match")
	testing.expect(t, d.text == "This is a Jira ticket.", "text should match")

	missing := get_descriptor(&g, "nonexistent")
	testing.expect(t, missing == nil, "missing descriptor should return nil")
}

@(test)
test_remove_descriptor :: proc(t: ^testing.T) {
	g: Graph
	create(&g)
	defer release(&g)

	set_descriptor(&g, "slack", "A Slack thread.")
	testing.expect_value(t, descriptor_count(&g), 1)

	ok := remove_descriptor(&g, "slack")
	testing.expect(t, ok, "remove should return true for existing descriptor")
	testing.expect_value(t, descriptor_count(&g), 0)

	ok2 := remove_descriptor(&g, "slack")
	testing.expect(t, !ok2, "remove should return false for missing descriptor")
}

@(test)
test_set_descriptor_replaces :: proc(t: ^testing.T) {
	g: Graph
	create(&g)
	defer release(&g)

	set_descriptor(&g, "jira", "old text")
	set_descriptor(&g, "jira", "new text")

	testing.expect_value(t, descriptor_count(&g), 1)
	d := get_descriptor(&g, "jira")
	testing.expect(t, d != nil, "descriptor should exist after replace")
	testing.expect(t, d.text == "new text", "text should be updated")
}

@(test)
test_descriptor_persistence :: proc(t: ^testing.T) {
	path := "_test_descriptor.bin"
	defer os.remove(path)

	g: Graph
	create(&g)

	set_descriptor(
		&g,
		"jira",
		"This input is a Jira ticket with fields: Summary, Description, Acceptance Criteria.",
	)
	set_descriptor(
		&g,
		"slack",
		"This is a Slack thread. The first message is the topic; replies are discussion.",
	)

	ok := save(&g, path)
	testing.expect(t, ok, "save should succeed")
	release(&g)

	g2: Graph
	create(&g2)
	defer release(&g2)

	ok2 := load(&g2, path)
	testing.expect(t, ok2, "load should succeed")

	testing.expect_value(t, descriptor_count(&g2), 2)

	jira := get_descriptor(&g2, "jira")
	testing.expect(t, jira != nil, "jira descriptor should survive round-trip")
	testing.expect(
		t,
		jira.text ==
		"This input is a Jira ticket with fields: Summary, Description, Acceptance Criteria.",
		"jira text should survive round-trip",
	)

	slack := get_descriptor(&g2, "slack")
	testing.expect(t, slack != nil, "slack descriptor should survive round-trip")
	testing.expect(
		t,
		slack.text ==
		"This is a Slack thread. The first message is the topic; replies are discussion.",
		"slack text should survive round-trip",
	)
}
