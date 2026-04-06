package provider

import "../graph"


EMBEDDING_DIM :: graph.EMBEDDING_DIM
Embedding :: graph.Embedding


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


Link_Result :: struct {
	reasoning: string,
	weight:    f32,
}


Batch_Link_Result :: struct {
	index:     int,
	reasoning: string,
	weight:    f32,
}

Provider :: struct {
	embed_text:        proc(self: ^Provider, text: string) -> (Embedding, bool),
	embed_texts:       proc(self: ^Provider, texts: []string) -> ([]Embedding, bool),
	decompose_text:    proc(
		self: ^Provider,
		text: string,
		prompt: string,
		descriptor: string = "",
	) -> (
		[]string,
		bool,
	),
	evaluate_thought:  proc(self: ^Provider, thought: string, prompt: string) -> (bool, bool),
	link_reason:       proc(
		self: ^Provider,
		thought_a: string,
		thought_b: string,
	) -> (
		Link_Result,
		bool,
	),
	batch_link_reason: proc(
		self: ^Provider,
		new_thought: string,
		candidates: []string,
	) -> (
		[]Batch_Link_Result,
		bool,
	),
	generate_answer:   proc(
		self: ^Provider,
		query: string,
		context_text: string,
	) -> (
		string,
		bool,
	),
	label_dimensions:  proc(
		self: ^Provider,
		purpose: string,
		dim_thoughts: [][]string,
	) -> (
		[]string,
		bool,
	),
	suggest_store:     proc(
		self: ^Provider,
		thoughts: []string,
	) -> (
		name: string,
		purpose: string,
		ok: bool,
	),
	state:             rawptr,
}


cfg: Config = DEFAULT_CONFIG

configure :: proc(c: Config) {
	cfg = c
}
