import hashlib
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def store_hash(name: str) -> str:
	return hashlib.sha256(name.lower().strip().encode("utf-8")).hexdigest()


def store_path(store_dir: str | Path, name: str) -> Path:
	return Path(store_dir) / f"{store_hash(name)}.shard"


class Registry:
	def __init__(self):
		self._path = Path.home() / ".config" / "shard" / "stores"
		self.stores: dict[str, dict[str, str]] = {}  # name → {"path", "purpose"}
		self.knids: dict[str, set[str]] = {}
		self._load()

	def _read_metadata(self, path: str) -> dict | None:
		try:
			from .strand.store import read_shard_metadata

			meta = read_shard_metadata(path)
			return meta
		except Exception:
			log.warning("Failed to read metadata from %s", path, exc_info=True)
			return None

	def _load(self):
		if not self._path.exists():
			return
		current_knid = None
		paths: list[str] = []  # collect paths first, index after
		for line in self._path.read_text(encoding="utf-8").splitlines():
			line = line.strip()
			if not line or line.startswith("#"):
				continue
			# Knid section header
			if line.startswith("[") and line.endswith("]"):
				current_knid = line[1:-1].strip()
				if current_knid not in self.knids:
					self.knids[current_knid] = set()
				continue
			if current_knid is not None:
				# Lines inside a knid section are store names
				self.knids[current_knid].add(line)
			else:
				# Legacy format: name = path
				if "=" in line:
					_, _, path = line.partition("=")
					path = path.strip()
				else:
					path = line
				if path:
					resolved = str(Path(path).resolve())
					paths.append(resolved)

		# Index each path by reading .shard metadata
		for resolved in paths:
			if not Path(resolved).exists():
				log.info("pruning stale store (file missing): %s", resolved)
				continue
			meta = self._read_metadata(resolved)
			if meta is None:
				log.warning("skipping unreadable store: %s", resolved)
				continue
			name = meta.get("name") or Path(resolved).stem
			purpose = meta.get("purpose", "")
			self.stores[name] = {"path": resolved, "purpose": purpose}

		# Prune knid members that don't correspond to any loaded store
		changed = False
		for members in self.knids.values():
			stale = members - set(self.stores.keys())
			if stale:
				members -= stale
				changed = True
		if changed:
			self.knids = {k: v for k, v in self.knids.items() if v}
			self.save()

	def _append(self, path: str):
		self._path.parent.mkdir(parents=True, exist_ok=True)
		with self._path.open("a", encoding="utf-8") as f:
			f.write(f"{path}\n")

	def save(self):
		self._path.parent.mkdir(parents=True, exist_ok=True)
		lines = []
		# Store entries — just paths
		seen_paths = set()
		for entry in self.stores.values():
			p = entry["path"]
			if p not in seen_paths:
				lines.append(p)
				seen_paths.add(p)
		# Knid sections
		for knid_name, members in self.knids.items():
			lines.append("")
			lines.append(f"[{knid_name}]")
			for member in sorted(members):
				lines.append(member)
		self._path.write_text("\n".join(lines) + "\n", encoding="utf-8")

	def register(self, path: str, name: str = "", purpose: str = ""):
		resolved = str(Path(path).resolve())
		if not name or not purpose:
			meta = self._read_metadata(resolved)
			if meta:
				name = name or meta.get("name") or Path(resolved).stem
				purpose = purpose or meta.get("purpose", "")
			else:
				name = name or Path(resolved).stem
		self.stores[name] = {"path": resolved, "purpose": purpose}
		self._append(resolved)

	def unregister(self, name: str):
		self.stores.pop(name, None)
		# Remove from all knids
		for members in self.knids.values():
			members.discard(name)
		self.save()

	def list_stores(self) -> dict[str, dict[str, str]]:
		return dict(self.stores)

	# ---- knid management ----

	def add_to_knid(self, knid_name: str, store_name: str):
		if knid_name not in self.knids:
			self.knids[knid_name] = set()
		self.knids[knid_name].add(store_name)
		self.save()

	def remove_from_knid(self, knid_name: str, store_name: str) -> bool:
		if knid_name not in self.knids:
			return False
		members = self.knids[knid_name]
		if store_name not in members:
			return False
		members.discard(store_name)
		if not members:
			del self.knids[knid_name]
		self.save()
		return True

	def list_knids(self) -> dict[str, set[str]]:
		return dict(self.knids)

	def stores_in_knid(self, knid_name: str) -> set[str]:
		return set(self.knids.get(knid_name, set()))

	def migrate_to_hashed(self):
		import re

		migrated = 0
		for name, entry in list(self.stores.items()):
			old = Path(entry["path"])
			if not old.exists():
				continue
			stem = old.stem
			# Already hashed — 64 hex chars
			if re.fullmatch(r"[0-9a-f]{64}", stem):
				continue
			new = store_path(old.parent, name)
			if new.exists():
				log.warning("Hash collision during migrate: %s → %s (target exists)", old, new)
				continue
			# Rename the main .shard file
			old.rename(new)
			entry["path"] = str(new)
			# Rename companion files (.pt)
			for suffix in (".pt",):
				companion = old.with_suffix(suffix)
				if companion.exists():
					companion.rename(new.with_suffix(suffix))
			migrated += 1
			log.info("Migrated '%s': %s → %s", name, old.name, new.name)
		if migrated:
			self.save()
		return migrated
