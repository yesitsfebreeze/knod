package provider

import "../graph"

// Embedding dimension must match the graph module.
EMBEDDING_DIM :: graph.EMBEDDING_DIM
Embedding :: graph.Embedding

// Provider configuration. Caller sources all values (env vars, CLI, etc.).
Config :: struct {
	api_key:         string,
	base_url:        string,
	embedding_model: string,
	chat_model:      string,
	timeout_ms:      int,
}

DEFAULT_CONFIG :: Config {
	api_key         = "",
	base_url        = "https://api.openai.com/v1",
	embedding_model = "text-embedding-3-small",
	chat_model      = "gpt-4o-mini",
	timeout_ms      = 30_000,
}

// Link result returned by batch_link_reason.
Link_Result :: struct {
	reasoning: string,
	weight:    f32,
}

// Batch link result: one entry per candidate thought.
// index is the position in the candidates array that was passed in.
Batch_Link_Result :: struct {
	index:     int,
	reasoning: string,
	weight:    f32,
}

Provider :: struct {
	// Embed a single text string into a fixed-size vector.
	embed_text:        proc(self: ^Provider, text: string) -> (Embedding, bool),

	// Embed multiple text strings in a single API call.
	// Returns a slice of embeddings in the same order as inputs.
	// Caller must delete the returned slice.
	embed_texts:       proc(self: ^Provider, texts: []string) -> ([]Embedding, bool),

	// Decompose raw text into a list of atomic thought strings.
	// descriptor is an optional hint that describes the structure of the
	// incoming text (e.g. "this is a Jira ticket with fields X, Y, Z").
	// Pass "" for no descriptor.
	decompose_text:    proc(
		self: ^Provider,
		text: string,
		prompt: string,
		descriptor: string = "",
	) -> (
		[]string,
		bool,
	),

	// Evaluate whether a thought is worth storing for this node.
	evaluate_thought:  proc(self: ^Provider, thought: string, prompt: string) -> (bool, bool),

	// Generate reasoning for why two thoughts should be connected, plus a weight.
	link_reason:       proc(
		self: ^Provider,
		thought_a: string,
		thought_b: string,
	) -> (
		Link_Result,
		bool,
	),

	// Batch version: given a new thought and a list of candidate neighbors,
	// return which candidates should be linked, with reasoning and weight.
	// One LLM call replaces N separate link_reason calls.
	batch_link_reason: proc(
		self: ^Provider,
		new_thought: string,
		candidates: []string,
	) -> (
		[]Batch_Link_Result,
		bool,
	),

	// Generate an answer from retrieved context.
	generate_answer:   proc(
		self: ^Provider,
		query: string,
		context_text: string,
	) -> (
		string,
		bool,
	),

	// Label embedding dimensions. Given the node's purpose and groups of
	// representative thoughts per dimension, return a single-word label for
	// each dimension. dim_thoughts[i] is a slice of thought texts that score
	// highest on the i-th dimension being labeled.
	label_dimensions:  proc(
		self: ^Provider,
		purpose: string,
		dim_thoughts: [][]string,
	) -> (
		[]string,
		bool,
	),

	// Given a list of thought texts that form a coherent cluster, suggest a
	// short kebab-case store name (e.g. "quantum-computing") and a one-sentence
	// purpose string for the new specialist store.
	suggest_store:     proc(
		self: ^Provider,
		thoughts: []string,
	) -> (
		name: string,
		purpose: string,
		ok: bool,
	),

	// Provider-specific state (opaque to callers).
	state:             rawptr,
}

// Package-level config, set via configure().
cfg: Config = DEFAULT_CONFIG

configure :: proc(c: Config) {
	cfg = c
}
