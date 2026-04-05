package provider

import "core:encoding/json"
import "core:testing"

@(test)
test_default_config :: proc(t: ^testing.T) {
	testing.expect(t, DEFAULT_CONFIG.base_url == "https://api.openai.com/v1", "default base_url")
	testing.expect(t, DEFAULT_CONFIG.embedding_model == "text-embedding-3-small", "default embedding model")
	testing.expect(t, DEFAULT_CONFIG.chat_model == "gpt-4o-mini", "default chat model")
	testing.expect(t, DEFAULT_CONFIG.timeout_ms == 30_000, "default timeout")
}

@(test)
test_configure :: proc(t: ^testing.T) {
	old := cfg
	defer { cfg = old }

	configure(Config{
		api_key         = "test-key",
		base_url        = "https://custom.api.com/v1",
		embedding_model = "custom-embed",
		chat_model      = "custom-chat",
		timeout_ms      = 5_000,
	})

	testing.expect(t, cfg.api_key == "test-key", "api_key should be set")
	testing.expect(t, cfg.base_url == "https://custom.api.com/v1", "base_url should be set")
	testing.expect(t, cfg.embedding_model == "custom-embed", "embedding_model should be set")
	testing.expect(t, cfg.chat_model == "custom-chat", "chat_model should be set")
	testing.expect_value(t, cfg.timeout_ms, 5_000)
}

@(test)
test_openai_create_destroy :: proc(t: ^testing.T) {
	old := cfg
	defer { cfg = old }

	configure(Config{
		api_key         = "sk-test",
		base_url        = "https://api.openai.com/v1",
		embedding_model = "text-embedding-3-small",
		chat_model      = "gpt-4o-mini",
		timeout_ms      = 10_000,
	})

	p := openai_create()
	defer openai_destroy(&p)

	testing.expect(t, p.embed_text != nil, "embed_text should be set")
	testing.expect(t, p.decompose_text != nil, "decompose_text should be set")
	testing.expect(t, p.batch_link_reason != nil, "batch_link_reason should be set")
	testing.expect(t, p.generate_answer != nil, "generate_answer should be set")
	testing.expect(t, p.label_dimensions != nil, "label_dimensions should be set")
	testing.expect(t, p.state != nil, "state should be set")

	state := cast(^OpenAI_State)p.state
	testing.expect(t, state.api_key == "sk-test", "state api_key")
	testing.expect(t, state.base_url == "https://api.openai.com/v1", "state base_url")
	testing.expect(t, state.embedding_model == "text-embedding-3-small", "state embedding_model")
	testing.expect(t, state.chat_model == "gpt-4o-mini", "state chat_model")
}

@(test)
test_parse_embedding_response :: proc(t: ^testing.T) {
	body := `{"data":[{"embedding":[0.1,0.2,0.3],"index":0}]}`

	embed_res: Embedding_Response
	err := json.unmarshal_string(body, &embed_res)
	defer {
		for &d in embed_res.data {
			delete(d.embedding)
		}
		delete(embed_res.data)
	}

	testing.expect(t, err == nil, "should parse without error")
	testing.expect_value(t, len(embed_res.data), 1)
	testing.expect_value(t, len(embed_res.data[0].embedding), 3)
	testing.expect(t, embed_res.data[0].embedding[0] > 0.09 && embed_res.data[0].embedding[0] < 0.11, "first value ~0.1")
	testing.expect(t, embed_res.data[0].embedding[1] > 0.19 && embed_res.data[0].embedding[1] < 0.21, "second value ~0.2")
}

@(test)
test_parse_chat_response :: proc(t: ^testing.T) {
	body := `{"choices":[{"message":{"role":"assistant","content":"Hello world"},"index":0}]}`

	chat_res: Chat_Response
	err := json.unmarshal_string(body, &chat_res)
	defer {
		for &c in chat_res.choices {
			delete(c.message.role)
			delete(c.message.content)
		}
		delete(chat_res.choices)
	}

	testing.expect(t, err == nil, "should parse without error")
	testing.expect_value(t, len(chat_res.choices), 1)
	testing.expect(t, chat_res.choices[0].message.role == "assistant", "role should be assistant")
	testing.expect(t, chat_res.choices[0].message.content == "Hello world", "content should match")
}

@(test)
test_parse_empty_chat_response :: proc(t: ^testing.T) {
	body := `{"choices":[]}`

	chat_res: Chat_Response
	err := json.unmarshal_string(body, &chat_res)
	defer delete(chat_res.choices)

	testing.expect(t, err == nil, "should parse without error")
	testing.expect_value(t, len(chat_res.choices), 0)
}

@(test)
test_embedding_dim_matches_graph :: proc(t: ^testing.T) {
	// Verify our EMBEDDING_DIM matches what the graph module defines.
	testing.expect_value(t, EMBEDDING_DIM, 1536)
}

@(test)
test_parse_link_response :: proc(t: ^testing.T) {
	body := `{"reasoning":"Both relate to physics","weight":0.85}`

	Raw_Link :: struct {
		reasoning: string `json:"reasoning"`,
		weight:    f64    `json:"weight"`,
	}

	raw: Raw_Link
	err := json.unmarshal_string(body, &raw)
	defer delete(raw.reasoning)

	testing.expect(t, err == nil, "should parse without error")
	testing.expect(t, raw.reasoning == "Both relate to physics", "reasoning should match")
	testing.expect(t, raw.weight > 0.84 && raw.weight < 0.86, "weight should be ~0.85")
}

@(test)
test_parse_label_dimensions_response :: proc(t: ^testing.T) {
	body := `{"labels":["physics","color theory","mammals"]}`

	Labels_Response :: struct {
		labels: [dynamic]string `json:"labels"`,
	}

	parsed: Labels_Response
	err := json.unmarshal_string(body, &parsed)
	defer {
		for &l in parsed.labels {
			delete(l)
		}
		delete(parsed.labels)
	}

	testing.expect(t, err == nil, "should parse without error")
	testing.expect_value(t, len(parsed.labels), 3)
	testing.expect(t, parsed.labels[0] == "physics", "first label should be 'physics'")
	testing.expect(t, parsed.labels[1] == "color theory", "second label should be 'color theory'")
	testing.expect(t, parsed.labels[2] == "mammals", "third label should be 'mammals'")
}
