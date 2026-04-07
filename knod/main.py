"""CLI entry point — run server (HTTP + TCP + MCP), ingest files, ask questions."""

import argparse
import logging
import signal
import sys
import threading
from pathlib import Path

from .config import Config


def main():
	parser = argparse.ArgumentParser(prog="knod", description="Knowledge graph with GNN")
	parser.add_argument("-v", "--verbose", action="store_true")
	sub = parser.add_subparsers(dest="command")

	# serve — starts all protocols
	serve_cmd = sub.add_parser("serve", help="Start HTTP + TCP + MCP server")
	serve_cmd.add_argument("--port", type=int, default=None, help="HTTP port")
	serve_cmd.add_argument("--tcp-port", type=int, default=None, help="TCP port")
	serve_cmd.add_argument("--graph", type=str, default=None)
	serve_cmd.add_argument("--no-http", action="store_true", help="Disable HTTP")
	serve_cmd.add_argument("--no-tcp", action="store_true", help="Disable TCP")
	serve_cmd.add_argument("--no-mcp", action="store_true", help="Disable MCP (stdio)")

	# mcp — standalone MCP server (stdio only, for use in MCP client configs)
	sub.add_parser("mcp", help="Run as MCP stdio server")

	# ingest
	ingest_cmd = sub.add_parser("ingest", help="Ingest a text file")
	ingest_cmd.add_argument("file", type=str, help="Path to text file")
	ingest_cmd.add_argument("--graph", type=str, default=None)
	ingest_cmd.add_argument("--descriptor", type=str, default="")
	ingest_cmd.add_argument("--knid", type=str, default=None, help="Scope to knid group")

	# ask
	ask_cmd = sub.add_parser("ask", help="Ask a question")
	ask_cmd.add_argument("query", type=str)
	ask_cmd.add_argument("--graph", type=str, default=None)
	ask_cmd.add_argument("--knid", type=str, default=None, help="Scope to knid group")

	# explore
	explore_cmd = sub.add_parser("explore", help="Show graph stats")
	explore_cmd.add_argument("--graph", type=str, default=None)

	# ingest-corpus
	corpus_cmd = sub.add_parser("ingest-corpus", help="Ingest all files in corpus/")
	corpus_cmd.add_argument("--dir", type=str, default="corpus")
	corpus_cmd.add_argument("--graph", type=str, default=None)

	# new — create a new strand interactively
	new_cmd = sub.add_parser("new", help="Create a new strand graph")
	new_cmd.add_argument("--knid", type=str, default=None, help="Add to knid group")

	# register — register an existing graph file
	register_cmd = sub.add_parser("register", help="Register an existing graph file")
	register_cmd.add_argument("path", type=str, help="Path to .knod file")
	register_cmd.add_argument("--knid", type=str, default=None, help="Add to knid group")

	# list — list registered stores
	list_cmd = sub.add_parser("list", help="List registered stores")
	list_cmd.add_argument("--knid", type=str, default=None, help="Filter by knid group")

	# knid — manage knid groupings
	knid_cmd = sub.add_parser("knid", help="Manage knid groupings")
	knid_sub = knid_cmd.add_subparsers(dest="knid_command")
	knid_new_cmd = knid_sub.add_parser("new", help="Create a new knid")
	knid_new_cmd.add_argument("name", type=str)
	knid_add_cmd = knid_sub.add_parser("add", help="Add store to knid")
	knid_add_cmd.add_argument("name", type=str, help="Knid name")
	knid_add_cmd.add_argument("store", type=str, help="Store name")
	knid_remove_cmd = knid_sub.add_parser("remove", help="Remove store from knid")
	knid_remove_cmd.add_argument("name", type=str, help="Knid name")
	knid_remove_cmd.add_argument("store", type=str, help="Store name")
	knid_list_cmd = knid_sub.add_parser("list", help="List knids or stores in a knid")
	knid_list_cmd.add_argument("name", type=str, nargs="?", default=None)

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
		_do_mcp_stdio(cfg)

	elif args.command == "ingest":
		_do_ingest_file(cfg, args.file, args.descriptor, knid=getattr(args, "knid", None))

	elif args.command == "ask":
		_do_ask(cfg, args.query, knid=getattr(args, "knid", None))

	elif args.command == "explore":
		_do_explore(cfg)

	elif args.command == "ingest-corpus":
		_do_ingest_corpus(cfg, args.dir)

	elif args.command == "new":
		_do_new(cfg, knid=getattr(args, "knid", None))

	elif args.command == "register":
		_do_register(cfg, args.path, knid=getattr(args, "knid", None))

	elif args.command == "list":
		_do_list(cfg, knid=getattr(args, "knid", None))

	elif args.command == "knid":
		_do_knid(cfg, args)

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

	# MCP stdio (background thread)
	mcp_thread = None
	if not args.no_mcp:
		from .protocol import _mcp

		mcp_server = _mcp(handler)

		def run_mcp():
			try:
				mcp_server.run(transport="stdio")
			except Exception:
				log.exception("MCP server exited")

		mcp_thread = threading.Thread(target=run_mcp, daemon=True)
		mcp_thread.start()
		log.info("mcp: stdio transport active")

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


def _do_mcp_stdio(cfg: Config):
	"""Run standalone MCP server over stdio (for MCP client configs)."""
	from .handler import Handler
	from .protocol import _mcp

	handler = Handler(cfg)
	handler.init()
	mcp = _mcp(handler)
	mcp.run(transport="stdio")


def _load_handler(cfg: Config):
	from .handler import Handler

	handler = Handler(cfg)
	handler.init()
	return handler


def _do_ingest_file(cfg: Config, filepath: str, descriptor: str = "", knid: str | None = None):
	log = logging.getLogger(__name__)
	text = Path(filepath).read_text(encoding="utf-8")
	handler = _load_handler(cfg)

	if knid:
		store_names = handler.registry.stores_in_knid(knid)
		if not store_names:
			log.warning("No stores in knid '%s'", knid)
			handler.shutdown()
			return
		for sname in store_names:
			try:
				n = handler.ingest_into_strand(sname, text, source=Path(filepath).stem, descriptor=descriptor)
				log.info("%s: %d thoughts committed", sname, n)
			except KeyError:
				log.info("%s: not loaded, skipping", sname)
	else:
		stats = handler.ingest_sync(text, source=Path(filepath).stem, descriptor=descriptor)
		log.info("Ingested from %s — %d thoughts, %d edges", filepath, stats["thoughts"], stats["edges"])

	handler.shutdown()


def _do_ask(cfg: Config, query: str, knid: str | None = None):
	handler = _load_handler(cfg)

	answer, sources = handler.ask(query, knid=knid)

	print(f"\n{answer}\n")
	print("--- Sources ---")
	for s in sources:
		print(f"  [{s['similarity']:.3f}] {s['text'][:80]}...")

	handler.shutdown()


def _do_explore(cfg: Config):
	log = logging.getLogger(__name__)
	base = Path(cfg.graph_path).with_suffix("")
	knod_file = base.with_suffix(".knod")

	if not knod_file.exists():
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


def _do_new(cfg: Config, knid: str | None = None):
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
	graph_path = handler.create_strand(name, purpose, location, knid=knid)

	if knid:
		log.info("Added to knid '%s'", knid)

	log.info("Created strand '%s' at %s", name, graph_path)
	handler.shutdown()


def _do_register(cfg: Config, path: str, knid: str | None = None):
	log = logging.getLogger(__name__)
	from .registry import Registry
	from .strand.store import read_knod_metadata

	graph_path = Path(path)
	if not graph_path.exists():
		log.error("File not found: %s", path)
		sys.exit(1)

	try:
		meta = read_knod_metadata(str(graph_path))
		name = meta.get("name") or graph_path.stem
		purpose = meta.get("purpose", "")
	except Exception:
		log.error("Invalid graph file: %s", path)
		sys.exit(1)

	registry = Registry()
	registry.register(str(graph_path.resolve()))

	if knid:
		registry.add_to_knid(knid, name)
		log.info("Added to knid '%s'", knid)

	log.info("Registered '%s' (purpose: %s)", name, purpose or "(none)")


def _do_list(cfg: Config, knid: str | None = None):
	log = logging.getLogger(__name__)
	from .registry import Registry

	registry = Registry()

	if knid:
		members = registry.stores_in_knid(knid)
		if not members:
			log.info("No stores in knid '%s'", knid)
			return
		log.info("Stores in knid '%s':", knid)
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


def _do_knid(cfg: Config, args):
	log = logging.getLogger(__name__)
	from .registry import Registry

	registry = Registry()
	cmd = getattr(args, "knid_command", None)

	if cmd == "new":
		name = args.name
		if name in registry.knids:
			log.info("Knid '%s' already exists", name)
			return
		registry.knids[name] = set()
		registry.save()
		log.info("Created knid '%s'", name)

	elif cmd == "add":
		if args.store not in registry.stores:
			log.warning("Store '%s' is not registered", args.store)
			return
		registry.add_to_knid(args.name, args.store)
		log.info("Added '%s' to knid '%s'", args.store, args.name)

	elif cmd == "remove":
		if registry.remove_from_knid(args.name, args.store):
			log.info("Removed '%s' from knid '%s'", args.store, args.name)
		else:
			log.warning("'%s' not found in knid '%s'", args.store, args.name)

	elif cmd == "list":
		if args.name:
			members = registry.stores_in_knid(args.name)
			if not members:
				log.info("No stores in knid '%s'", args.name)
			else:
				log.info("Knid '%s':", args.name)
				for m in sorted(members):
					print(f"  {m}")
		else:
			knids = registry.list_knids()
			if not knids:
				log.info("No knids defined")
			else:
				for name, members in knids.items():
					print(f"  [{name}] {', '.join(sorted(members)) or '(empty)'}")

	else:
		print("Usage: py_knod knid {new|add|remove|list}")


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
