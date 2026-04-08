"""CLI entry point — run server (HTTP + TCP + MCP), ingest files, ask questions."""

import argparse
import logging
import signal
import sys
import threading
from pathlib import Path

from .config import Config


def main():
	parser = argparse.ArgumentParser(prog="shard", description="Knowledge graph with GNN")
	parser.add_argument("-v", "--verbose", action="store_true")
	sub = parser.add_subparsers(dest="command")

	# serve — starts all protocols
	serve_cmd = sub.add_parser("serve", help="Start HTTP + TCP + MCP server")
	serve_cmd.add_argument("--port", type=int, default=None, help="HTTP port")
	serve_cmd.add_argument("--tcp-port", type=int, default=None, help="TCP port")
	serve_cmd.add_argument("--graph", type=str, default=None)
	serve_cmd.add_argument("--no-http", action="store_true", help="Disable HTTP")
	serve_cmd.add_argument("--no-tcp", action="store_true", help="Disable TCP")
	serve_cmd.add_argument("--no-mcp", action="store_true", help="Disable MCP")
	serve_cmd.add_argument(
		"--mcp-transport",
		choices=["stdio", "sse", "streamable-http"],
		default="stdio",
		help="MCP transport (default: stdio)",
	)
	serve_cmd.add_argument("--mcp-port", type=int, default=None, help="MCP HTTP/SSE port (default: mcp_port in config)")

	# mcp — standalone MCP server
	mcp_cmd = sub.add_parser("mcp", help="Run as standalone MCP server")
	mcp_cmd.add_argument(
		"--transport",
		choices=["stdio", "sse", "streamable-http"],
		default="stdio",
		help="Transport to use (default: stdio)",
	)
	mcp_cmd.add_argument("--port", type=int, default=None, help="Port for HTTP/SSE transport")
	mcp_cmd.add_argument("--host", type=str, default="127.0.0.1", help="Host for HTTP/SSE transport")

	# ingest
	ingest_cmd = sub.add_parser("ingest", help="Ingest a text file")
	ingest_cmd.add_argument("file", type=str, help="Path to text file")
	ingest_cmd.add_argument("--graph", type=str, default=None)
	ingest_cmd.add_argument("--descriptor", type=str, default="")
	ingest_cmd.add_argument("--cluster", type=str, default=None, help="Scope to cluster")

	# ask
	ask_cmd = sub.add_parser("ask", help="Ask a question")
	ask_cmd.add_argument("query", type=str)
	ask_cmd.add_argument("--graph", type=str, default=None)
	ask_cmd.add_argument("--cluster", type=str, default=None, help="Scope to cluster")

	# explore
	explore_cmd = sub.add_parser("explore", help="Show graph stats")
	explore_cmd.add_argument("--graph", type=str, default=None)

	# ingest-corpus
	corpus_cmd = sub.add_parser("ingest-corpus", help="Ingest all files in corpus/")
	corpus_cmd.add_argument("--dir", type=str, default="corpus")
	corpus_cmd.add_argument("--graph", type=str, default=None)

	# new — create a new shard interactively
	new_cmd = sub.add_parser("new", help="Create a new shard")
	new_cmd.add_argument("--cluster", type=str, default=None, help="Add to cluster")

	# register — register an existing shard file
	register_cmd = sub.add_parser("register", help="Register an existing shard file")
	register_cmd.add_argument("path", type=str, help="Path to .shard file")
	register_cmd.add_argument("--cluster", type=str, default=None, help="Add to cluster")

	# list — list registered stores
	list_cmd = sub.add_parser("list", help="List registered stores")
	list_cmd.add_argument("--cluster", type=str, default=None, help="Filter by cluster")

	# cluster — manage clusterings
	cluster_cmd = sub.add_parser("cluster", help="Manage clusterings")
	cluster_sub = cluster_cmd.add_subparsers(dest="cluster_command")
	cluster_new_cmd = cluster_sub.add_parser("new", help="Create a new cluster")
	cluster_new_cmd.add_argument("name", type=str)
	cluster_add_cmd = cluster_sub.add_parser("add", help="Add store to cluster")
	cluster_add_cmd.add_argument("name", type=str, help="cluster name")
	cluster_add_cmd.add_argument("store", type=str, help="Store name")
	cluster_remove_cmd = cluster_sub.add_parser("remove", help="Remove store from cluster")
	cluster_remove_cmd.add_argument("name", type=str, help="cluster name")
	cluster_remove_cmd.add_argument("store", type=str, help="Store name")
	cluster_list_cmd = cluster_sub.add_parser("list", help="List clusters or stores in a cluster")
	cluster_list_cmd.add_argument("name", type=str, nargs="?", default=None)

	# migrate — rename store files to SHA-256 hashed names
	sub.add_parser("migrate", help="Rename store files to SHA-256 hashed names")

	args = parser.parse_args()

	logging.basicConfig(
		level=logging.DEBUG if args.verbose else logging.INFO,
		format="%(asctime)s %(levelname)s %(name)s: %(message)s",
	)

	cfg = Config.load()
	if hasattr(args, "graph") and args.graph:
		cfg.graph_path = args.graph
	if hasattr(args, "port") and args.port:
		cfg.http_port = args.port
	if hasattr(args, "tcp_port") and args.tcp_port:
		cfg.tcp_port = args.tcp_port

	if args.command == "serve":
		_do_serve(cfg, args)

	elif args.command == "mcp":
		_do_mcp(cfg, args)

	elif args.command == "ingest":
		_do_ingest_file(cfg, args.file, args.descriptor, cluster=getattr(args, "cluster", None))

	elif args.command == "ask":
		_do_ask(cfg, args.query, cluster=getattr(args, "cluster", None))

	elif args.command == "explore":
		_do_explore(cfg)

	elif args.command == "ingest-corpus":
		_do_ingest_corpus(cfg, args.dir)

	elif args.command == "new":
		_do_new(cfg, cluster=getattr(args, "cluster", None))

	elif args.command == "register":
		_do_register(cfg, args.path, cluster=getattr(args, "cluster", None))

	elif args.command == "list":
		_do_list(cfg, cluster=getattr(args, "cluster", None))

	elif args.command == "cluster":
		_do_cluster(cfg, args)

	elif args.command == "migrate":
		_do_migrate()

	else:
		parser.print_help()


def _do_serve(cfg: Config, args):
	"""Start all enabled protocols sharing one Handler."""
	from .handler import Handler

	handler = Handler(cfg)
	handler.init()

	log = logging.getLogger(__name__)
	stop = threading.Event()

	# TCP
	tcp_server = None
	if not args.no_tcp:
		from .protocol import _tcp

		tcp_server = _tcp(handler, port=cfg.tcp_port)

	# MCP (background thread — all transports run here so HTTP can own the main thread)
	mcp_thread = None
	if not args.no_mcp:
		from .protocol import _mcp

		mcp_transport = args.mcp_transport
		mcp_port = args.mcp_port or cfg.mcp_port
		mcp_server = _mcp(handler, port=mcp_port)

		def run_mcp():
			try:
				mcp_server.run(transport=mcp_transport)
			except Exception:
				log.exception("MCP server exited")

		mcp_thread = threading.Thread(target=run_mcp, daemon=True)
		mcp_thread.start()
		if mcp_transport == "stdio":
			log.info("mcp: stdio transport active")
		else:
			log.info("mcp: %s transport active on :%d", mcp_transport, mcp_port)

	# HTTP (blocks on main thread via uvicorn)
	if not args.no_http:
		import uvicorn
		from .protocol import http

		_http = http(handler)

		def shutdown_handler(signum, frame):
			stop.set()
			if tcp_server:
				tcp_server.stop()
			handler.shutdown()
			sys.exit(0)

		signal.signal(signal.SIGINT, shutdown_handler)
		signal.signal(signal.SIGTERM, shutdown_handler)

		log.info("http: starting on :%d", cfg.http_port)
		log_config = {
			"version": 1,
			"disable_existing_loggers": False,
			"formatters": {"default": {"()": "logging.Formatter", "fmt": "%(levelname)s %(message)s"}},
			"handlers": {"default": {"class": "logging.StreamHandler", "formatter": "default", "stream": "ext://sys.stderr"}},
			"root": {"level": "WARNING", "handlers": ["default"]},
		}
		uvicorn.run(_http, host="0.0.0.0", port=cfg.http_port, log_level="warning", log_config=log_config)
	else:
		log.info("http: disabled")

		# No HTTP → block on stop event
		def shutdown_handler(signum, frame):
			stop.set()

		signal.signal(signal.SIGINT, shutdown_handler)
		signal.signal(signal.SIGTERM, shutdown_handler)
		stop.wait()

	if tcp_server:
		tcp_server.stop()
	handler.shutdown()


def _do_mcp(cfg: Config, args):
	"""Run standalone MCP server (stdio, sse, or streamable-http)."""
	from .handler import Handler
	from .protocol import _mcp

	transport = getattr(args, "transport", "stdio")
	port = getattr(args, "port", None) or cfg.mcp_port
	host = getattr(args, "host", "127.0.0.1")

	handler = Handler(cfg)
	handler.init()
	mcp = _mcp(handler, host=host, port=port)
	if transport != "stdio":
		log = logging.getLogger(__name__)
		log.info("mcp: %s transport on http://%s:%d/mcp", transport, host, port)
	mcp.run(transport=transport)


def _load_handler(cfg: Config):
	from .handler import Handler

	handler = Handler(cfg)
	handler.init()
	return handler


def _do_ingest_file(cfg: Config, filepath: str, descriptor: str = "", cluster: str | None = None):
	log = logging.getLogger(__name__)
	text = Path(filepath).read_text(encoding="utf-8")
	handler = _load_handler(cfg)

	if cluster:
		store_names = handler.registry.stores_in_cluster(cluster)
		if not store_names:
			log.warning("No stores in cluster '%s'", cluster)
			handler.shutdown()
			return
		for sname in store_names:
			try:
				n = handler.ingest_into_shard(sname, text, source=Path(filepath).stem, descriptor=descriptor)
				log.info("%s: %d thoughts committed", sname, n)
			except KeyError:
				log.info("%s: not loaded, skipping", sname)
	else:
		stats = handler.ingest_sync(text, source=Path(filepath).stem, descriptor=descriptor)
		log.info("Ingested from %s — %d thoughts, %d edges", filepath, stats["thoughts"], stats["edges"])

	handler.shutdown()


def _do_ask(cfg: Config, query: str, cluster: str | None = None):
	handler = _load_handler(cfg)

	answer, sources = handler.ask(query, cluster=cluster)

	print(f"\n{answer}\n")
	print("--- Sources ---")
	for s in sources:
		print(f"  [{s['similarity']:.3f}] {s['text'][:80]}...")

	handler.shutdown()


def _do_explore(cfg: Config):
	log = logging.getLogger(__name__)
	base = Path(cfg.graph_path).with_suffix("")
	shard_file = base.with_suffix(".shard")

	if not shard_file.exists():
		log.warning("No graph found")
		return

	handler = _load_handler(cfg)
	info = handler.graph_info
	log.info("Purpose: %s", info["purpose"] or "(none)")
	log.info("Thoughts: %d", info["thought_count"])
	log.info("Edges: %d", info["edge_count"])
	log.info("Maturity: %.2f", info["maturity"])
	if info["descriptors"]:
		log.info("Descriptors: %s", ", ".join(info["descriptors"].keys()))


def _do_ingest_corpus(cfg: Config, corpus_dir: str):
	log = logging.getLogger(__name__)
	corpus = Path(corpus_dir)
	if not corpus.is_dir():
		log.error("Directory not found: %s", corpus_dir)
		sys.exit(1)

	files = sorted(corpus.glob("*.txt"))
	if not files:
		log.warning("No .txt files in %s", corpus_dir)
		sys.exit(1)

	handler = _load_handler(cfg)

	ingested = handler.ingested_sources()

	for i, f in enumerate(files, 1):
		if f.name == "manifest.txt":
			continue
		if f.stem in ingested:
			log.info("[%d/%d] %s (already ingested, skipping)", i, len(files), f.name)
			continue
		log.info("[%d/%d] %s", i, len(files), f.name)
		text = f.read_text(encoding="utf-8")
		try:
			stats = handler.ingest_sync(text, source=f.stem)
			log.info("  → %d thoughts, %d edges", stats["thoughts"], stats["edges"])
			handler.save()
		except Exception as exc:
			log.error("  ✗ %s", exc)
			handler.save()
			continue

	handler.shutdown()
	info = handler.graph_info
	log.info("Done. Graph: %d thoughts, %d edges", info["thought_count"], info["edge_count"])


def _do_new(cfg: Config, cluster: str | None = None):
	log = logging.getLogger(__name__)
	purpose = input("Purpose: ").strip()
	if not purpose:
		log.warning("Purpose is required")
		return

	name = input("Name: ").strip()
	if not name:
		log.warning("Name is required")
		return

	location = input(f"Location [{Path.cwd()}]: ").strip()
	if not location:
		location = str(Path.cwd())

	handler = _load_handler(cfg)
	graph_path = handler.create_shard(name, purpose, location, cluster=cluster)

	if cluster:
		log.info("Added to cluster '%s'", cluster)

	log.info("Created shard '%s' at %s", name, graph_path)
	handler.shutdown()


def _do_register(cfg: Config, path: str, cluster: str | None = None):
	log = logging.getLogger(__name__)
	from .registry import Registry
	from .shard.store import read_shard_metadata

	graph_path = Path(path)
	if not graph_path.exists():
		log.error("File not found: %s", path)
		sys.exit(1)

	try:
		meta = read_shard_metadata(str(graph_path))
		name = meta.get("name") or graph_path.stem
		purpose = meta.get("purpose", "")
	except Exception:
		log.error("Invalid graph file: %s", path)
		sys.exit(1)

	registry = Registry()
	registry.register(str(graph_path.resolve()))

	if cluster:
		registry.add_to_cluster(cluster, name)
		log.info("Added to cluster '%s'", cluster)

	log.info("Registered '%s' (purpose: %s)", name, purpose or "(none)")


def _do_list(cfg: Config, cluster: str | None = None):
	log = logging.getLogger(__name__)
	from .registry import Registry

	registry = Registry()

	if cluster:
		members = registry.stores_in_cluster(cluster)
		if not members:
			log.info("No stores in cluster '%s'", cluster)
			return
		log.info("Stores in cluster '%s':", cluster)
		for name in sorted(members):
			entry = registry.stores.get(name, {})
			print(f"  {name} = {entry.get('path', '(unknown)')}")
	else:
		stores = registry.list_stores()
		if not stores:
			log.info("No registered stores")
			return
		for name, entry in stores.items():
			print(f"  {name} = {entry['path']}")


def _do_cluster(cfg: Config, args):
	log = logging.getLogger(__name__)
	from .registry import Registry

	registry = Registry()
	cmd = getattr(args, "cluster_command", None)

	if cmd == "new":
		name = args.name
		if name in registry.clusters:
			log.info("cluster '%s' already exists", name)
			return
		registry.clusters[name] = set()
		registry.save()
		log.info("Created cluster '%s'", name)

	elif cmd == "add":
		if args.store not in registry.stores:
			log.warning("Store '%s' is not registered", args.store)
			return
		registry.add_to_cluster(args.name, args.store)
		log.info("Added '%s' to cluster '%s'", args.store, args.name)

	elif cmd == "remove":
		if registry.remove_from_cluster(args.name, args.store):
			log.info("Removed '%s' from cluster '%s'", args.store, args.name)
		else:
			log.warning("'%s' not found in cluster '%s'", args.store, args.name)

	elif cmd == "list":
		if args.name:
			members = registry.stores_in_cluster(args.name)
			if not members:
				log.info("No stores in cluster '%s'", args.name)
			else:
				log.info("cluster '%s':", args.name)
				for m in sorted(members):
					print(f"  {m}")
		else:
			clusters = registry.list_clusters()
			if not clusters:
				log.info("No clusters defined")
			else:
				for name, members in clusters.items():
					print(f"  [{name}] {', '.join(sorted(members)) or '(empty)'}")

	else:
		print("Usage: py_shard cluster {new|add|remove|list}")


def _do_migrate():
	log = logging.getLogger(__name__)
	from .registry import Registry

	registry = Registry()
	n = registry.migrate_to_hashed()
	if n:
		log.info("Migrated %d store(s) to hashed filenames", n)
	else:
		log.info("All stores already use hashed filenames (nothing to migrate)")


if __name__ == "__main__":
	main()
