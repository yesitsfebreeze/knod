package util

import "core:os"
import os2 "core:os/os2"
import "core:path/filepath"

clone_string :: proc(s: string) -> string {
	if len(s) == 0 {
		return ""
	}
	buf := make([]u8, len(s))
	copy(buf, transmute([]u8)s)
	return string(buf)
}

// exe_dir returns the directory containing the running executable.
// Caller must delete the returned string.
exe_dir :: proc() -> string {
	exe_path, err := os2.get_executable_path(context.allocator)
	if err != nil {
		return ""
	}
	defer delete(exe_path)
	return filepath.dir(exe_path)
}

// Return the user's home directory. Caller must delete the returned string.
home_dir :: proc() -> string {
	home := os.get_env("USERPROFILE")
	if len(home) == 0 {
		home = os.get_env("HOME")
	}
	return home
}

// Recursively create a directory and all parent directories.
ensure_dir :: proc(path: string) {
	if os.exists(path) {
		return
	}
	parent := filepath.dir(path)
	if len(parent) > 0 && parent != path {
		defer delete(parent)
		ensure_dir(parent)
	}
	os.make_directory(path)
}

// Shared constants
STRAND_EXTENSION :: ".strand"
EDGE_SCORE_DISCOUNT: f32 : 0.8
LIMBO_THRESHOLD: f32 : 0.75
KNOD_CONFIG_DIR :: ".config/knod"
