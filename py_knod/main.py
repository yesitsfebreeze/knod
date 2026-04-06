"""CLI entry point — run server (HTTP + TCP + MCP), ingest files, ask questions."""

import argparse
import logging
import signal
import sys
import threading
from pathlib import Path

from .config import Config


def main():
	parser = argparse.ArgumentParser(prog="py_knod", description="Knowledge graph with GNN")
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

	# new — create a new specialist interactively
	new_cmd = sub.add_parser("new", help="Create a new specialist graph")
	new_cmd.add_argument("--knid", type=str, default=None, help="Add to knid group")

	# register — register an existing graph file
	register_cmd = sub.add_parser("register", help="Register an existing graph file")
	register_cmd.add_argument("path", type=str, help="Path to .graph file")
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
		uvicorn.run(_http, host="0.0.0.0", port=cfg.http_port, log_level="warning")
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
	text = Path(filepath).read_text(encoding="utf-8")
	handler = _load_handler(cfg)

	if knid:
		# Ingest into all stores in the knid
		store_names = handler.registry.stores_in_knid(knid)
		if not store_names:
			print(f"No stores in knid '{knid}'")
			handler.shutdown()
			return
		for sname in store_names:
			if sname in handler._specialists:
				spec = handler._specialists[sname]
				from .ingest import Ingester

				ingester = Ingester(spec.graph, handler.provider, cfg)
				committed = ingester.ingest(text, source=Path(filepath).stem, descriptor=descriptor)
				print(f"  {sname}: {len(committed)} thoughts committed")
	else:
		stats = handler.handle_ingest(text, source=Path(filepath).stem, descriptor=descriptor)
		print(f"Ingested from {filepath}")
		print(f"Graph: {stats['thoughts']} thoughts, {stats['edges']} edges")

	handler.shutdown()


def _do_ask(cfg: Config, query: str, knid: str | None = None):
	handler = _load_handler(cfg)

	if knid:
		# Scope ask to stores in the knid
		from .retrieval import cosine_scores, edge_scores, gnn_scores, merge, deduplicate, answer as gen_answer

		store_names = handler.registry.stores_in_knid(knid)
		query_emb = handler.provider.embed_text(query)
		all_scored = []
		for sname in store_names:
			if sname in handler._specialists:
				spec = handler._specialists[sname]
				all_scored.append(handler._score_specialist(query_emb, spec.graph, spec.model, spec.strand))
		# Deduplicate across knid stores
		scored = deduplicate(all_scored, cfg.top_k)
		if not scored:
			print("No relevant knowledge found in knid.")
			handler.shutdown()
			return
		answer, sources = gen_answer(query, scored, handler.provider)
	else:
		answer, sources = handler.handle_ask(query)

	print(f"\n{answer}\n")
	print("--- Sources ---")
	for s in sources:
		print(f"  [{s['similarity']:.3f}] {s['text'][:80]}...")

	handler.shutdown()


def _do_explore(cfg: Config):
	base = Path(cfg.graph_path).with_suffix("")
	graph_file = base.with_suffix(".graph")

	if not graph_file.exists():
		print("No graph found.")
		return

	handler = _load_handler(cfg)
	g = handler.graph
	print(f"Purpose: {g.purpose or '(none)'}")
	print(f"Thoughts: {g.num_thoughts}")
	print(f"Edges: {g.num_edges}")
	print(f"Maturity: {g.maturity:.2f}")
	if g.descriptors:
		print(f"Descriptors: {', '.join(g.descriptors.keys())}")


def _do_ingest_corpus(cfg: Config, corpus_dir: str):
	corpus = Path(corpus_dir)
	if not corpus.is_dir():
		print(f"Directory not found: {corpus_dir}")
		sys.exit(1)

	files = sorted(corpus.glob("*.txt"))
	if not files:
		print(f"No .txt files in {corpus_dir}")
		sys.exit(1)

	handler = _load_handler(cfg)

	for i, f in enumerate(files, 1):
		if f.name == "manifest.txt":
			continue
		print(f"[{i}/{len(files)}] {f.name}")
		text = f.read_text(encoding="utf-8")
		stats = handler.handle_ingest(text, source=f.stem)
		print(f"  → {stats['thoughts']} thoughts, {stats['edges']} edges")

	handler.shutdown()
	print(f"\nDone. Graph: {handler.graph.num_thoughts} thoughts, {handler.graph.num_edges} edges")


def _do_new(cfg: Config, knid: str | None = None):
	"""Interactive: create a new specialist graph."""
	from .registry import Registry
	from .specialist import Graph, KnodMPNN, StrandLayer, save_all

	purpose = input("Purpose: ").strip()
	if not purpose:
		print("Purpose is required.")
		return

	name = input("Name: ").strip()
	if not name:
		print("Name is required.")
		return

	location = input(f"Location [{Path.cwd()}]: ").strip()
	if not location:
		location = str(Path.cwd())

	safe_name = name.replace(" ", "_").lower()
	base = Path(location) / safe_name

	graph = Graph(purpose=purpose, max_thoughts=cfg.max_thoughts, max_edges=cfg.max_edges)
	model = KnodMPNN(cfg)
	strand = StrandLayer(cfg.hidden_dim)
	save_all(graph, model, strand, base)

	registry = Registry()
	graph_path = str(base.with_suffix(".graph"))
	registry.register(name, graph_path, purpose)

	if knid:
		registry.add_to_knid(knid, name)
		print(f"Added to knid '{knid}'")

	print(f"Created specialist '{name}' at {graph_path}")


def _do_register(cfg: Config, path: str, knid: str | None = None):
	"""Register an existing graph file."""
	import pickle
	from .registry import Registry

	graph_path = Path(path)
	if not graph_path.exists():
		print(f"File not found: {path}")
		sys.exit(1)

	# Validate: try to load and check it's a valid graph
	try:
		with open(graph_path, "rb") as f:
			state = pickle.load(f)
		purpose = state.get("purpose", "")
	except Exception:
		print(f"Invalid graph file: {path}")
		sys.exit(1)

	name = graph_path.stem
	registry = Registry()
	registry.register(name, str(graph_path.resolve()), purpose)

	if knid:
		registry.add_to_knid(knid, name)
		print(f"Added to knid '{knid}'")

	print(f"Registered '{name}' (purpose: {purpose or '(none)'})")


def _do_list(cfg: Config, knid: str | None = None):
	"""List registered stores, optionally filtered by knid."""
	from .registry import Registry

	registry = Registry()

	if knid:
		members = registry.stores_in_knid(knid)
		if not members:
			print(f"No stores in knid '{knid}'")
			return
		print(f"Stores in knid '{knid}':")
		for name in sorted(members):
			entry = registry.stores.get(name, {})
			print(f"  {name} = {entry.get('path', '(unknown)')}")
	else:
		stores = registry.list_stores()
		if not stores:
			print("No registered stores.")
			return
		for name, entry in stores.items():
			print(f"  {name} = {entry['path']}")


def _do_knid(cfg: Config, args):
	"""Handle knid subcommands."""
	from .registry import Registry

	registry = Registry()
	cmd = getattr(args, "knid_command", None)

	if cmd == "new":
		name = args.name
		if name in registry.knids:
			print(f"Knid '{name}' already exists")
			return
		registry.knids[name] = set()
		registry.save()
		print(f"Created knid '{name}'")

	elif cmd == "add":
		if args.store not in registry.stores:
			print(f"Store '{args.store}' is not registered")
			return
		registry.add_to_knid(args.name, args.store)
		print(f"Added '{args.store}' to knid '{args.name}'")

	elif cmd == "remove":
		if registry.remove_from_knid(args.name, args.store):
			print(f"Removed '{args.store}' from knid '{args.name}'")
		else:
			print(f"'{args.store}' not found in knid '{args.name}'")

	elif cmd == "list":
		if args.name:
			members = registry.stores_in_knid(args.name)
			if not members:
				print(f"No stores in knid '{args.name}'")
			else:
				print(f"Knid '{args.name}':")
				for m in sorted(members):
					print(f"  {m}")
		else:
			knids = registry.list_knids()
			if not knids:
				print("No knids defined.")
			else:
				for name, members in knids.items():
					print(f"  [{name}] {', '.join(sorted(members)) or '(empty)'}")

	else:
		print("Usage: py_knod knid {new|add|remove|list}")


if __name__ == "__main__":
	main()
