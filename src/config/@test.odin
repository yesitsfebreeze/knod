package config

import "core:math"
import "core:testing"

EPSILON :: 1e-4

@(private = "file")
approx_eq :: proc(a, b: f32, eps: f32 = EPSILON) -> bool {
	return math.abs(a - b) < eps
}

@(test)
test_defaults :: proc(t: ^testing.T) {
	testing.expect(t, DEFAULT.base_url == "https://api.openai.com/v1", "base_url")
	testing.expect(t, DEFAULT.embedding_model == "text-embedding-3-small", "embedding_model")
	testing.expect(t, DEFAULT.chat_model == "gpt-4o-mini", "chat_model")
	testing.expect_value(t, DEFAULT.timeout_ms, 30_000)
	testing.expect_value(t, DEFAULT.tcp_port, 7999)
	testing.expect_value(t, DEFAULT.http_port, 8080)
	testing.expect_value(t, DEFAULT.find_k, 10)
	testing.expect_value(t, DEFAULT.max_similar, 5)
	testing.expect(t, approx_eq(DEFAULT.edge_threshold, 0.3), "edge_threshold")
}

@(test)
test_parse_full :: proc(t: ^testing.T) {
	content := `# test config
api_key = sk-test-key
base_url = https://custom.api.com/v1
embedding_model = custom-embed
chat_model = custom-chat
timeout_ms = 5000
tcp_port = 8000
http_port = 9000
graph_path = custom.graph
max_thoughts = 1000
max_edges = 5000
edge_decay = 0.01
similarity_threshold = 0.5
find_k = 20
max_similar = 10
edge_threshold = 0.4
`

	cfg, ok := parse(content)
	defer release(&cfg)
	testing.expect(t, ok, "should parse successfully")

	testing.expect(t, cfg.api_key == "sk-test-key", "api_key")
	testing.expect(t, cfg.base_url == "https://custom.api.com/v1", "base_url")
	testing.expect(t, cfg.embedding_model == "custom-embed", "embedding_model")
	testing.expect(t, cfg.chat_model == "custom-chat", "chat_model")
	testing.expect_value(t, cfg.timeout_ms, 5000)
	testing.expect_value(t, cfg.tcp_port, 8000)
	testing.expect_value(t, cfg.http_port, 9000)
	testing.expect(t, cfg.graph_path == "custom.graph", "graph_path")
	testing.expect_value(t, cfg.max_thoughts, 1000)
	testing.expect_value(t, cfg.max_edges, 5000)
	testing.expect(t, approx_eq(cfg.edge_decay, 0.01), "edge_decay")
	testing.expect(t, approx_eq(cfg.similarity_threshold, 0.5), "similarity_threshold")
	testing.expect_value(t, cfg.find_k, 20)
	testing.expect_value(t, cfg.max_similar, 10)
	testing.expect(t, approx_eq(cfg.edge_threshold, 0.4), "edge_threshold")
}

@(test)
test_parse_partial :: proc(t: ^testing.T) {
	content := `api_key = sk-partial
tcp_port = 9999
`

	cfg, ok := parse(content)
	defer release(&cfg)
	testing.expect(t, ok, "should parse successfully")

	testing.expect(t, cfg.api_key == "sk-partial", "api_key overridden")
	testing.expect_value(t, cfg.tcp_port, 9999)

	// Defaults should remain.
	testing.expect(t, cfg.base_url == DEFAULT.base_url, "base_url default")
	testing.expect(t, cfg.chat_model == DEFAULT.chat_model, "chat_model default")
	testing.expect_value(t, cfg.timeout_ms, DEFAULT.timeout_ms)
	testing.expect_value(t, cfg.http_port, DEFAULT.http_port)
}

@(test)
test_parse_comments_and_blanks :: proc(t: ^testing.T) {
	content := `
# This is a comment
   # Indented comment

api_key = sk-comments

# Another comment
tcp_port = 1234

`

	cfg, ok := parse(content)
	defer release(&cfg)
	testing.expect(t, ok, "should parse successfully")
	testing.expect(t, cfg.api_key == "sk-comments", "api_key")
	testing.expect_value(t, cfg.tcp_port, 1234)
}

@(test)
test_parse_empty :: proc(t: ^testing.T) {
	cfg, ok := parse("")
	defer release(&cfg)
	testing.expect(t, ok, "empty content should still succeed")

	// All defaults.
	testing.expect(t, cfg.base_url == DEFAULT.base_url, "base_url default")
	testing.expect_value(t, cfg.tcp_port, DEFAULT.tcp_port)
}

@(test)
test_parse_spaces_around_equals :: proc(t: ^testing.T) {
	content := `api_key=no-spaces
base_url  =  lots-of-spaces
chat_model =trailing
`

	cfg, ok := parse(content)
	defer release(&cfg)
	testing.expect(t, ok, "should parse")
	testing.expect(t, cfg.api_key == "no-spaces", "no spaces")
	testing.expect(t, cfg.base_url == "lots-of-spaces", "extra spaces")
	testing.expect(t, cfg.chat_model == "trailing", "trailing space")
}

@(test)
test_parse_value_with_equals :: proc(t: ^testing.T) {
	// Value contains '=' (e.g., base64 in API key).
	content := `api_key = sk-abc=def=ghi
`

	cfg, ok := parse(content)
	defer release(&cfg)
	testing.expect(t, ok, "should parse")
	testing.expect(t, cfg.api_key == "sk-abc=def=ghi", "value with equals signs")
}

@(test)
test_parse_unknown_keys_ignored :: proc(t: ^testing.T) {
	content := `api_key = sk-test
unknown_key = some_value
another_bad = 42
tcp_port = 5555
`

	cfg, ok := parse(content)
	defer release(&cfg)
	testing.expect(t, ok, "should parse")
	testing.expect(t, cfg.api_key == "sk-test", "known key parsed")
	testing.expect_value(t, cfg.tcp_port, 5555)
}

@(test)
test_config_path_not_empty :: proc(t: ^testing.T) {
	path := config_path()
	defer delete(path)
	testing.expect(t, len(path) > 0, "config_path should resolve on this system")
}

@(test)
test_default_contents_parseable :: proc(t: ^testing.T) {
	contents := default_contents()
	cfg, ok := parse(contents)
	defer release(&cfg)
	testing.expect(t, ok, "default contents should parse")
	testing.expect(t, cfg.api_key == "", "default api_key should be empty")
}

@(test)
test_parse_malformed_lines :: proc(t: ^testing.T) {
	content := `api_key = sk-good
this line has no equals sign
= value with no key
tcp_port = 3000
`

	cfg, ok := parse(content)
	defer release(&cfg)
	testing.expect(t, ok, "should parse, skipping malformed lines")
	testing.expect(t, cfg.api_key == "sk-good", "good line parsed")
	testing.expect_value(t, cfg.tcp_port, 3000)
}
