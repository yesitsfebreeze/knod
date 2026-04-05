package registry

import "core:testing"

@(test)
test_parse_empty :: proc(t: ^testing.T) {
	r, ok := parse("")
	defer release(&r)
	testing.expect(t, !ok || len(r.stores) == 0, "empty content should have no stores")
	testing.expect(t, len(r.knids) == 0, "empty content should have no knids")
}

@(test)
test_parse_stores :: proc(t: ^testing.T) {
	content := `medical = /data/medical.graph
legal = /data/legal.graph`
	r, ok := parse(content)
	defer release(&r)
	testing.expect(t, ok, "should parse ok")
	testing.expect(t, len(r.stores) == 2, "should have 2 stores")
	testing.expect(t, r.stores[0].name == "medical", "first store name")
	testing.expect(t, r.stores[0].path == "/data/medical.graph", "first store path")
	testing.expect(t, r.stores[1].name == "legal", "second store name")
}

@(test)
test_parse_with_knids :: proc(t: ^testing.T) {
	content := `medical = /data/medical.graph
legal = /data/legal.graph
anatomy = /data/anatomy.graph

[health]
medical
anatomy

[professional]
medical
legal`
	r, ok := parse(content)
	defer release(&r)
	testing.expect(t, ok, "should parse ok")
	testing.expect(t, len(r.stores) == 3, "should have 3 stores")
	testing.expect(t, len(r.knids) == 2, "should have 2 knids")
	testing.expect(t, r.knids[0].name == "health", "first knid name")
	testing.expect(t, len(r.knids[0].members) == 2, "health should have 2 members")
	testing.expect(t, r.knids[0].members[0] == "medical", "health first member")
	testing.expect(t, r.knids[0].members[1] == "anatomy", "health second member")
	testing.expect(t, r.knids[1].name == "professional", "second knid name")
	testing.expect(t, len(r.knids[1].members) == 2, "professional should have 2 members")
}

@(test)
test_parse_with_comments :: proc(t: ^testing.T) {
	content := `# stores
medical = /data/medical.graph
# another comment
legal = /data/legal.graph`
	r, ok := parse(content)
	defer release(&r)
	testing.expect(t, ok, "should parse ok")
	testing.expect(t, len(r.stores) == 2, "comments should be ignored")
}

@(test)
test_serialize_roundtrip :: proc(t: ^testing.T) {
	content := `medical = /data/medical.graph
legal = /data/legal.graph

[health]
medical
`
	r, ok := parse(content)
	defer release(&r)
	testing.expect(t, ok, "initial parse should work")

	serialized := serialize(&r)
	defer delete(serialized)

	r2, ok2 := parse(serialized)
	defer release(&r2)
	testing.expect(t, ok2, "re-parse should work")
	testing.expect(t, len(r2.stores) == len(r.stores), "store count should match")
	testing.expect(t, len(r2.knids) == len(r.knids), "knid count should match")
}

@(test)
test_find_store :: proc(t: ^testing.T) {
	r: Registry
	r.stores = make([dynamic]Store)
	r.knids = make([dynamic]Knid)
	defer release(&r)

	add_store(&r, "medical", "/data/medical.graph")
	add_store(&r, "legal", "/data/legal.graph")

	s := find_store(&r, "medical")
	testing.expect(t, s != nil, "should find medical")
	testing.expect(t, s.path == "/data/medical.graph", "path should match")

	s2 := find_store(&r, "nonexistent")
	testing.expect(t, s2 == nil, "should not find nonexistent")
}

@(test)
test_add_store_duplicate :: proc(t: ^testing.T) {
	r: Registry
	r.stores = make([dynamic]Store)
	r.knids = make([dynamic]Knid)
	defer release(&r)

	ok1 := add_store(&r, "medical", "/data/medical.graph")
	testing.expect(t, ok1, "first add should work")

	ok2 := add_store(&r, "medical", "/data/other.graph")
	testing.expect(t, !ok2, "duplicate add should fail")
	testing.expect(t, len(r.stores) == 1, "should still have 1 store")
}

@(test)
test_knid_operations :: proc(t: ^testing.T) {
	r: Registry
	r.stores = make([dynamic]Store)
	r.knids = make([dynamic]Knid)
	defer release(&r)

	add_store(&r, "medical", "/data/medical.graph")
	add_store(&r, "legal", "/data/legal.graph")

	ok := add_knid(&r, "health")
	testing.expect(t, ok, "should create knid")

	ok2 := add_knid(&r, "health")
	testing.expect(t, !ok2, "duplicate knid should fail")

	ok3 := knid_add_store(&r, "health", "medical")
	testing.expect(t, ok3, "should add store to knid")

	ok4 := knid_add_store(&r, "health", "nonexistent")
	testing.expect(t, !ok4, "should not add nonexistent store")

	ok5 := knid_add_store(&r, "health", "medical")
	testing.expect(t, !ok5, "duplicate add to knid should fail")

	stores := knid_stores(&r, "health")
	defer delete(stores)
	testing.expect(t, len(stores) == 1, "should have 1 store in knid")
	testing.expect(t, stores[0].name == "medical", "should be medical")

	ok6 := knid_remove_store(&r, "health", "medical")
	testing.expect(t, ok6, "should remove from knid")

	stores2 := knid_stores(&r, "health")
	defer delete(stores2)
	testing.expect(t, len(stores2) == 0, "knid should be empty after removal")
}
