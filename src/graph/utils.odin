package graph

read_i32 :: proc(data: []u8, off: ^int) -> i32 {
	if off^ + 4 > len(data) {return 0}
	val := (^i32)(raw_data(data[off^:]))^
	off^ += 4
	return val
}

read_u16 :: proc(data: []u8, off: ^int) -> u16 {
	if off^ + 2 > len(data) {return 0}
	val := (^u16)(raw_data(data[off^:]))^
	off^ += 2
	return val
}

read_u32 :: proc(data: []u8, off: ^int) -> u32 {
	if off^ + 4 > len(data) {return 0}
	val := (^u32)(raw_data(data[off^:]))^
	off^ += 4
	return val
}

read_u64 :: proc(data: []u8, off: ^int) -> u64 {
	if off^ + 8 > len(data) {return 0}
	val := (^u64)(raw_data(data[off^:]))^
	off^ += 8
	return val
}

read_i64 :: proc(data: []u8, off: ^int) -> i64 {
	if off^ + 8 > len(data) {return 0}
	val := (^i64)(raw_data(data[off^:]))^
	off^ += 8
	return val
}

read_f32 :: proc(data: []u8, off: ^int) -> f32 {
	if off^ + 4 > len(data) {return 0.0}
	val := (^f32)(raw_data(data[off^:]))^
	off^ += 4
	return val
}

read_str :: proc(data: []u8, off: ^int) -> string {
	slen := int(read_i32(data, off))
	if slen <= 0 || off^ + slen > len(data) {
		return ""
	}
	buf := make([]u8, slen)
	copy(buf, data[off^:off^ + slen])
	off^ += slen
	return string(buf)
}
