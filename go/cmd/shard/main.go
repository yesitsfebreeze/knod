package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
)

// shardsDir is the one place all shard DBs live.
func shardsDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		panic("cannot resolve home directory: " + err.Error())
	}
	return filepath.Join(home, ".shards")
}

// shardPath returns the DB path for a named shard.
func shardPath(name string) string {
	return filepath.Join(shardsDir(), name+".shard")
}

func main() {
	// Child mode: spawned by parent, owns exactly one shard DB.
	// Invoked as: shard --child=<name>
	childName := flag.String("child", "", "Run as child process for this shard name")
	flag.Parse()

	if *childName != "" {
		runChild(*childName, shardPath(*childName))
		return
	}

	// Parent mode: dispatch subcommands.
	args := flag.Args()
	if len(args) == 0 {
		runServe()
		return
	}

	switch args[0] {
	case "serve":
		runServe()
	case "new":
		if len(args) < 2 {
			fmt.Fprintln(os.Stderr, "usage: shard new <name>")
			os.Exit(1)
		}
		runNew(args[1])
	case "list":
		runList()
	case "ask":
		if len(args) < 2 {
			fmt.Fprintln(os.Stderr, "usage: shard ask <query>")
			os.Exit(1)
		}
		runAsk(args[1])
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n", args[0])
		fmt.Fprintln(os.Stderr, "commands: serve, new <name>, list, ask <query>")
		os.Exit(1)
	}
}

func runServe() {
	fmt.Println("serve: not yet implemented")
}

func runNew(name string) {
	path := shardPath(name)
	fmt.Printf("new shard '%s' → %s\n", name, path)
}

func runList() {
	dir := shardsDir()
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			fmt.Println("no shards yet")
			return
		}
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	for _, e := range entries {
		if filepath.Ext(e.Name()) == ".shard" {
			name := e.Name()[:len(e.Name())-6]
			fmt.Printf("  %s → %s\n", name, filepath.Join(dir, e.Name()))
		}
	}
}

func runAsk(query string) {
	fmt.Printf("ask %q: not yet implemented\n", query)
}

func runChild(name, dbPath string) {
	fmt.Printf("child: shard=%s path=%s\n", name, dbPath)
}
