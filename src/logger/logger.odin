package logger

import "core:fmt"
import "core:os"
import "core:path/filepath"
import "core:time"

CHANNEL_COUNT :: 3
Channel :: enum u8 {
	KNOD,
	PERF,
	ERR,
}

Level :: enum u8 {
	DEBUG,
	INFO,
	WARN,
	ERROR,
}

@(private)
channel_files: [CHANNEL_COUNT]os.Handle

@(private)
channel_open: [CHANNEL_COUNT]bool

@(private)
channel_names := [CHANNEL_COUNT]string{"knod", "performance", "error"}

@(private)
level_tags := [4]string{"DBG", "INF", "WRN", "ERR"}

@(private)
channel_tags := [CHANNEL_COUNT]string{"knod", "perf", "err "}

init :: proc() {
	exe_dir := filepath.dir(os.args[0])
	log_dir := filepath.join({exe_dir, "logs"})
	os.make_directory(log_dir)

	for i in 0 ..< CHANNEL_COUNT {
		path := filepath.join({log_dir, fmt.tprintf("%s.log", channel_names[i])})
		fd, err := os.open(path, os.O_WRONLY | os.O_CREATE | os.O_APPEND, 0o644)
		if err == os.ERROR_NONE {
			channel_files[i] = fd
			channel_open[i] = true
		} else {
			fmt.printf("[log] warning: could not open %s: %v\n", path, err)
		}
	}
}

shutdown :: proc() {
	for i in 0 ..< CHANNEL_COUNT {
		if channel_open[i] {
			os.close(channel_files[i])
			channel_open[i] = false
		}
	}
}

logf :: proc(ch: Channel, level: Level, format: string, args: ..any) {
	now := time.now()
	y, mon, d := time.date(now)
	h, m, s := time.clock(now)

	msg := fmt.tprintf(format, ..args)

	ch_idx := int(ch)
	lvl_idx := int(level)

	fmt.printf(
		"%4d-%02d-%02d %02d:%02d:%02d [%s] [%s] %s\n",
		y,
		mon,
		d,
		h,
		m,
		s,
		channel_tags[ch_idx],
		level_tags[lvl_idx],
		msg,
	)

	if channel_open[ch_idx] {
		line := fmt.tprintf(
			"%4d-%02d-%02d %02d:%02d:%02d [%s] %s\n",
			y,
			mon,
			d,
			h,
			m,
			s,
			level_tags[lvl_idx],
			msg,
		)
		os.write(channel_files[ch_idx], transmute([]u8)line)
	}

	if level >= .WARN && ch != .ERR && channel_open[int(Channel.ERR)] {
		line := fmt.tprintf(
			"%4d-%02d-%02d %02d:%02d:%02d [%s] [%s] %s\n",
			y,
			mon,
			d,
			h,
			m,
			s,
			channel_tags[ch_idx],
			level_tags[lvl_idx],
			msg,
		)
		os.write(channel_files[int(Channel.ERR)], transmute([]u8)line)
	}
}

info :: proc(format: string, args: ..any) {
	logf(.KNOD, .INFO, format, ..args)
}

warn :: proc(format: string, args: ..any) {
	logf(.KNOD, .WARN, format, ..args)
}

err :: proc(format: string, args: ..any) {
	logf(.ERR, .ERROR, format, ..args)
}

perf :: proc(format: string, args: ..any) {
	logf(.PERF, .INFO, format, ..args)
}

debug :: proc(format: string, args: ..any) {
	logf(.KNOD, .DEBUG, format, ..args)
}

