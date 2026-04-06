package provider

import "core:encoding/json"
import "core:fmt"
import "core:strings"

import http "../http"
import client "../http/client"
import log "../logger"

Embedding_Data :: struct {
	embedding: []f64 `json:"embedding"`,
	index:     int `json:"index"`,
}

Embedding_Response :: struct {
	data: []Embedding_Data `json:"data"`,
}

Chat_Choice :: struct {
	message: Chat_Message `json:"message"`,
	index:   int `json:"index"`,
}

Chat_Message :: struct {
	role:    string `json:"role"`,
	content: string `json:"content"`,
}

Chat_Response :: struct {
	choices: []Chat_Choice `json:"choices"`,
}

OpenAI_State :: struct {
	api_key:         string,
	base_url:        string,
	embedding_model: string,
	chat_model:      string,
}

openai_create :: proc() -> Provider {
	state := new(OpenAI_State)
	state.api_key = cfg.api_key
	state.base_url = cfg.base_url
	state.embedding_model = cfg.embedding_model
	state.chat_model = cfg.chat_model

	return Provider {
		embed_text = openai_embed_text,
		embed_texts = openai_embed_texts,
		decompose_text = openai_decompose_text,
		batch_link_reason = openai_batch_link_reason,
		generate_answer = openai_generate_answer,
		label_dimensions = openai_label_dimensions,
		suggest_store = openai_suggest_store,
		state = state,
	}
}

openai_destroy :: proc(p: ^Provider) {
	if p.state != nil {
		free(cast(^OpenAI_State)p.state)
		p.state = nil
	}
}

@(private = "file")
openai_post :: proc(state: ^OpenAI_State, endpoint: string, payload: any) -> (string, bool) {
	url := fmt.aprintf("%s%s", state.base_url, endpoint)
	defer delete(url)

	log.info("[provider] POST %s", endpoint)

	req: client.Request
	client.request_init(&req, .Post)
	defer client.request_destroy(&req)

	auth := fmt.aprintf("Bearer %s", state.api_key)
	defer delete(auth)
	http.headers_set_unsafe(&req.headers, "authorization", auth)

	marshal_err := client.with_json(&req, payload)
	if marshal_err != nil {
		log.err("[provider] JSON marshal error for %s", endpoint)
		return "", false
	}

	log.info("[provider] sending request to %s...", endpoint)
	res, req_err := client.request(&req, url)
	if req_err != nil {
		log.err("[provider] HTTP request failed for %s: %v", endpoint, req_err)
		return "", false
	}

	log.info("[provider] %s returned status %v", endpoint, res.status)

	if res.status != .OK {
		log.err("[provider] %s returned status %v", endpoint, res.status)
		body, was_alloc, _ := client.response_body(&res)
		defer client.response_destroy(&res, body, was_alloc)
		if plain, ok := body.(client.Body_Plain); ok {
			log.err("[provider] response: %s", plain)
		}
		return "", false
	}

	body, was_alloc, body_err := client.response_body(&res)
	if body_err != nil {
		log.err("[provider] failed to read response body for %s", endpoint)
		client.response_destroy(&res)
		return "", false
	}

	plain, ok := body.(client.Body_Plain)
	if !ok {
		log.err("[provider] unexpected body type for %s", endpoint)
		client.response_destroy(&res, body, was_alloc)
		return "", false
	}

	result := strings.clone(plain)
	client.response_destroy(&res, body, was_alloc)
	return result, true
}

@(private = "file")
openai_chat :: proc(
	state: ^OpenAI_State,
	system_prompt: string,
	user_content: string,
	json_mode: bool = false,
) -> (
	string,
	bool,
) {
	Msg :: struct {
		role:    string `json:"role"`,
		content: string `json:"content"`,
	}

	Response_Format :: struct {
		type: string `json:"type"`,
	}

	Payload :: struct {
		model:           string `json:"model"`,
		messages:        []Msg `json:"messages"`,
		temperature:     f64 `json:"temperature"`,
		response_format: Maybe(Response_Format) `json:"response_format,omitempty"`,
	}

	messages := []Msg {
		{role = "system", content = system_prompt},
		{role = "user", content = user_content},
	}

	payload := Payload {
		model           = state.chat_model,
		messages        = messages,
		temperature     = 0.3,
		response_format = json_mode ? Response_Format{type = "json_object"} : nil,
	}

	body, ok := openai_post(state, "/chat/completions", payload)
	if !ok {
		return "", false
	}
	defer delete(body)

	chat_res: Chat_Response
	unmarshal_err := json.unmarshal_string(body, &chat_res)
	if unmarshal_err != nil {
		log.err("[provider] failed to parse chat response")
		return "", false
	}
	defer delete(chat_res.choices)

	if len(chat_res.choices) == 0 {
		log.err("[provider] chat response has no choices")
		return "", false
	}

	content := chat_res.choices[0].message.content
	if json_mode {
		content = strip_json_fences(content)
	}

	return strings.clone(content), true
}

@(private = "file")
strip_json_fences :: proc(s: string) -> string {
	trimmed := strings.trim_space(s)

	fence_json := "```json"
	fence_plain := "```"
	fence_close := "```"

	start: int
	if strings.has_prefix(trimmed, fence_json) {
		start = len(fence_json)
	} else if strings.has_prefix(trimmed, fence_plain) {
		start = len(fence_plain)
	} else {
		return s
	}

	if start < len(trimmed) && trimmed[start] == '\n' {
		start += 1
	} else if start < len(trimmed) && trimmed[start] == '\r' {
		start += 1
		if start < len(trimmed) && trimmed[start] == '\n' {
			start += 1
		}
	}

	end := strings.last_index(trimmed, fence_close)
	if end <= start {
		return s
	}

	inner := strings.trim_space(trimmed[start:end])
	return inner
}

openai_embed_text :: proc(self: ^Provider, text: string) -> (Embedding, bool) {
	state := cast(^OpenAI_State)self.state

	Payload :: struct {
		model: string `json:"model"`,
		input: string `json:"input"`,
	}

	payload := Payload {
		model = state.embedding_model,
		input = text,
	}

	body, ok := openai_post(state, "/embeddings", payload)
	if !ok {
		return {}, false
	}
	defer delete(body)

	embed_res: Embedding_Response
	unmarshal_err := json.unmarshal_string(body, &embed_res)
	if unmarshal_err != nil {
		log.err("[provider] failed to parse embedding response")
		return {}, false
	}
	defer {
		for &d in embed_res.data {
			delete(d.embedding)
		}
		delete(embed_res.data)
	}

	if len(embed_res.data) == 0 || len(embed_res.data[0].embedding) != EMBEDDING_DIM {
		log.err(
			"[provider] embedding response has wrong dimensions: got %d, expected %d",
			len(embed_res.data) > 0 ? len(embed_res.data[0].embedding) : 0,
			EMBEDDING_DIM,
		)
		return {}, false
	}

	result: Embedding
	for i in 0 ..< EMBEDDING_DIM {
		result[i] = f32(embed_res.data[0].embedding[i])
	}

	return result, true
}

openai_embed_texts :: proc(self: ^Provider, texts: []string) -> ([]Embedding, bool) {
	if len(texts) == 0 {
		return make([]Embedding, 0), true
	}

	if len(texts) == 1 {
		emb, ok := openai_embed_text(self, texts[0])
		if !ok {
			return nil, false
		}
		result := make([]Embedding, 1)
		result[0] = emb
		return result, true
	}

	state := cast(^OpenAI_State)self.state

	Payload :: struct {
		model: string `json:"model"`,
		input: []string `json:"input"`,
	}

	payload := Payload {
		model = state.embedding_model,
		input = texts,
	}

	body, ok := openai_post(state, "/embeddings", payload)
	if !ok {
		return nil, false
	}
	defer delete(body)

	embed_res: Embedding_Response
	unmarshal_err := json.unmarshal_string(body, &embed_res)
	if unmarshal_err != nil {
		log.err("[provider] failed to parse batch embedding response")
		return nil, false
	}
	defer {
		for &d in embed_res.data {
			delete(d.embedding)
		}
		delete(embed_res.data)
	}

	if len(embed_res.data) != len(texts) {
		log.err(
			"[provider] batch embedding: expected %d results, got %d",
			len(texts),
			len(embed_res.data),
		)
		return nil, false
	}

	results := make([]Embedding, len(texts))
	for &d in embed_res.data {
		if d.index < 0 || d.index >= len(texts) {
			log.err("[provider] batch embedding: out-of-range index %d", d.index)
			delete(results)
			return nil, false
		}
		if len(d.embedding) != EMBEDDING_DIM {
			log.err("[provider] batch embedding: wrong dim at index %d", d.index)
			delete(results)
			return nil, false
		}
		for i in 0 ..< EMBEDDING_DIM {
			results[d.index][i] = f32(d.embedding[i])
		}
	}

	return results, true
}

openai_decompose_text :: proc(
	self: ^Provider,
	text: string,
	prompt: string,
	descriptor: string = "",
) -> (
	[]string,
	bool,
) {
	state := cast(^OpenAI_State)self.state

	system: string
	if len(descriptor) > 0 {
		system = fmt.aprintf(
			`You are a knowledge decomposition system. %s

%s

Decompose the following text into atomic, self-contained thoughts.
Each thought must stand alone without requiring context from the original text.
Respond with a JSON object: {"thoughts": ["thought 1", "thought 2"]}`,
			prompt,
			descriptor,
		)
	} else {
		system = fmt.aprintf(
			`You are a knowledge decomposition system. %s

Decompose the following text into atomic, self-contained thoughts.
Each thought must stand alone without requiring context from the original text.
Respond with a JSON object: {"thoughts": ["thought 1", "thought 2"]}`,
			prompt,
		)
	}
	defer delete(system)

	content, ok := openai_chat(state, system, text, json_mode = true)
	if !ok {
		return nil, false
	}
	defer delete(content)

	Decompose_Response :: struct {
		thoughts: [dynamic]string `json:"thoughts"`,
	}

	parsed: Decompose_Response
	unmarshal_err := json.unmarshal_string(content, &parsed)
	if unmarshal_err != nil {
		log.err("[provider] failed to parse decomposition response: %s", content)
		return nil, false
	}

	result := make([]string, len(parsed.thoughts))
	for t, i in parsed.thoughts {
		result[i] = strings.clone(t)
	}
	delete(parsed.thoughts)

	return result, true
}

openai_batch_link_reason :: proc(
	self: ^Provider,
	new_thought: string,
	candidates: []string,
) -> (
	[]Batch_Link_Result,
	bool,
) {
	state := cast(^OpenAI_State)self.state

	if len(candidates) == 0 {
		return nil, true
	}

	system := `You are a knowledge graph reasoning system.
Given a NEW thought and a numbered list of EXISTING thoughts, determine which existing thoughts should be connected to the new thought.
For each connection worth making, provide reasoning and a weight (0.0-1.0).
Respond with a JSON object: {"links": [{"index": 0, "reasoning": "why connected", "weight": 0.85}]}
Only include entries for thoughts that SHOULD be connected. Omit thoughts with no meaningful connection.
If none should be connected, respond with: {"links": []}`

	b := strings.builder_make()
	defer strings.builder_destroy(&b)

	strings.write_string(&b, "NEW THOUGHT: ")
	strings.write_string(&b, new_thought)
	strings.write_string(&b, "\n\nEXISTING THOUGHTS:\n")

	for c, i in candidates {
		line := fmt.aprintf("[%d] %s\n", i, c)
		strings.write_string(&b, line)
		delete(line)
	}

	user_content := strings.to_string(b)
	content, ok := openai_chat(state, system, user_content, json_mode = true)
	if !ok {
		return nil, false
	}
	defer delete(content)

	Raw_Batch_Link :: struct {
		index:     int `json:"index"`,
		reasoning: string `json:"reasoning"`,
		weight:    f64 `json:"weight"`,
	}

	Batch_Response :: struct {
		links: [dynamic]Raw_Batch_Link `json:"links"`,
	}

	parsed: Batch_Response
	unmarshal_err := json.unmarshal_string(content, &parsed)
	if unmarshal_err != nil {
		log.err("[provider] failed to parse batch_link_reason response: %s", content)
		return nil, false
	}
	defer delete(parsed.links)

	results := make([dynamic]Batch_Link_Result, 0, len(parsed.links))
	for &raw in parsed.links {
		if raw.index < 0 || raw.index >= len(candidates) {
			continue
		}
		append(
			&results,
			Batch_Link_Result {
				index = raw.index,
				reasoning = strings.clone(raw.reasoning),
				weight = f32(raw.weight),
			},
		)
	}

	result_slice := make([]Batch_Link_Result, len(results))
	copy(result_slice, results[:])
	delete(results)

	return result_slice, true
}

openai_generate_answer :: proc(
	self: ^Provider,
	query: string,
	context_text: string,
) -> (
	string,
	bool,
) {
	state := cast(^OpenAI_State)self.state

	system := `You are a knowledge retrieval system. Answer the user's question using ONLY the provided context.
If the context doesn't contain enough information, say so honestly.
Be concise and direct.`

	b := strings.builder_make()
	defer strings.builder_destroy(&b)
	strings.write_string(&b, "Context:\n")
	strings.write_string(&b, context_text)
	strings.write_string(&b, "\n\nQuestion: ")
	strings.write_string(&b, query)
	user_content := strings.to_string(b)

	return openai_chat(state, system, user_content)
}

openai_label_dimensions :: proc(
	self: ^Provider,
	purpose: string,
	dim_thoughts: [][]string,
) -> (
	[]string,
	bool,
) {
	state := cast(^OpenAI_State)self.state

	if len(dim_thoughts) == 0 {
		return nil, true
	}

	system := fmt.aprintf(
		`You are a knowledge taxonomy system for a specialist node with purpose: "%s"

You will be given numbered groups of representative thoughts. Each group corresponds to one semantic dimension of this node's knowledge.
For each group, provide a single word or short phrase (max 3 words) that captures the common theme.
Respond with a JSON object: {"labels": ["physics", "color theory", "mammals"]}`,
		purpose,
	)
	defer delete(system)

	b := strings.builder_make()
	defer strings.builder_destroy(&b)

	for group, i in dim_thoughts {
		line := fmt.aprintf("Group %d:\n", i)
		strings.write_string(&b, line)
		delete(line)
		for thought in group {
			strings.write_string(&b, "  - ")
			strings.write_string(&b, thought)
			strings.write_string(&b, "\n")
		}
		strings.write_string(&b, "\n")
	}

	user_content := strings.to_string(b)
	content, ok := openai_chat(state, system, user_content, json_mode = true)
	if !ok {
		return nil, false
	}
	defer delete(content)

	Labels_Response :: struct {
		labels: [dynamic]string `json:"labels"`,
	}

	parsed: Labels_Response
	unmarshal_err := json.unmarshal_string(content, &parsed)
	if unmarshal_err != nil {
		log.err("[provider] failed to parse label_dimensions response: %s", content)
		return nil, false
	}

	if len(parsed.labels) == 0 {
		log.err("[provider] label_dimensions returned 0 labels")
		delete(parsed.labels)
		return nil, false
	}

	if len(parsed.labels) != len(dim_thoughts) {
		log.warn(
			"[provider] label_dimensions returned %d labels, expected %d — using partial results",
			len(parsed.labels),
			len(dim_thoughts),
		)
	}

	take := min(len(parsed.labels), len(dim_thoughts))
	result := make([]string, take)
	for i in 0 ..< take {
		result[i] = strings.clone(parsed.labels[i])
	}
	for &l in parsed.labels {
		delete(l)
	}
	delete(parsed.labels)

	return result, true
}

openai_suggest_store :: proc(
	self: ^Provider,
	thoughts: []string,
) -> (
	name: string,
	purpose: string,
	ok: bool,
) {
	state := cast(^OpenAI_State)self.state

	if len(thoughts) == 0 {
		return "", "", false
	}

	system := `You are a knowledge organisation assistant.
You will receive a list of related thoughts that have emerged from a shared topic.
Suggest a short, descriptive kebab-case store name (2-4 words, e.g. "quantum-computing") and a one-sentence purpose string that captures the specialist domain.
Respond with a JSON object: {"name": "store-name", "purpose": "One sentence describing the specialist domain."}`

	b := strings.builder_make()
	defer strings.builder_destroy(&b)

	strings.write_string(&b, "CLUSTER THOUGHTS:\n")
	for t, i in thoughts {
		line := fmt.aprintf("[%d] %s\n", i, t)
		strings.write_string(&b, line)
		delete(line)
	}

	user_content := strings.to_string(b)
	content, chat_ok := openai_chat(state, system, user_content, json_mode = true)
	if !chat_ok {
		return "", "", false
	}
	defer delete(content)

	Suggest_Response :: struct {
		name:    string `json:"name"`,
		purpose: string `json:"purpose"`,
	}

	parsed: Suggest_Response
	unmarshal_err := json.unmarshal_string(content, &parsed)
	if unmarshal_err != nil {
		log.err("[provider] failed to parse suggest_store response: %s", content)
		return "", "", false
	}

	if len(parsed.name) == 0 || len(parsed.purpose) == 0 {
		log.err("[provider] suggest_store returned empty name or purpose")
		delete(parsed.name)
		delete(parsed.purpose)
		return "", "", false
	}

	return strings.clone(parsed.name), strings.clone(parsed.purpose), true
}
