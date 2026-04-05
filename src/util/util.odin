package util

clone_string :: proc(s: string) -> string {
	if len(s) == 0 {
		return ""
	}
	buf := make([]u8, len(s))
	copy(buf, transmute([]u8)s)
	return string(buf)
}
