import hashlib
import logging
from pathlib import Path

from .shard.store import read_shard_metadata

log = logging.getLogger(__name__)


def store_hash(name: str) -> str:
	return hashlib.sha256(name.lower().strip().encode("utf-8")).hexdigest()


def store_path(store_dir: str | Path, name: str) -> Path:
	return Path(store_dir) / f"{store_hash(name)}.shard"


class Registry:
	def __init__(self):
		self._path = Path.home() / ".config" / "shard" / "stores"
		self.stores: dict[str, dict[str, str]] = {}  # name → {"path", "purpose"}
		self.clusters: dict[str, set[str]] = {}
		self._load()

	def _read_metadata(self, path: str) -> dict | None:
		try:
			return read_shard_metadata(path)
		except Exception:
			log.warning("Failed to read metadata from %s", path, exc_info=True)
			return None

	def _load(self):
		if not self._path.exists():
			return
		current_cluster = None
		paths: list[str] = []  # collect paths first, index after
		for line in self._path.read_text(encoding="utf-8").splitlines():
			line = line.strip()
			if not line or line.startswith("#"):
				continue
			# Cluster section header
			if line.startswith("[") and line.endswith("]"):
				current_cluster = line[1:-1].strip()
				if current_cluster not in self.clusters:
					self.clusters[current_cluster] = set()
				continue
			if current_cluster is not None:
				# Lines inside a cluster section are store names
				self.clusters[current_cluster].add(line)
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
			origin = meta.get("origin", "")
			self.stores[name] = {"path": resolved, "purpose": purpose, "origin": origin}

		# Prune cluster members that don't correspond to any loaded store
		changed = False
		for members in self.clusters.values():
			stale = members - set(self.stores.keys())
			if stale:
				members -= stale
				changed = True
		if changed:
			self.clusters = {k: v for k, v in self.clusters.items() if v}
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
		# cluster sections
		for cluster_name, members in self.clusters.items():
			lines.append("")
			lines.append(f"[{cluster_name}]")
			for member in sorted(members):
				lines.append(member)
		self._path.write_text("\n".join(lines) + "\n", encoding="utf-8")

	def register(self, path: str, name: str = "", purpose: str = ""):
		resolved = str(Path(path).resolve())
		origin = ""
		if not name or not purpose:
			meta = self._read_metadata(resolved)
			if meta:
				name = name or meta.get("name") or Path(resolved).stem
				purpose = purpose or meta.get("purpose", "")
				origin = meta.get("origin", "")
			else:
				name = name or Path(resolved).stem
		self.stores[name] = {"path": resolved, "purpose": purpose, "origin": origin}
		self._append(resolved)

	def unregister(self, name: str):
		self.stores.pop(name, None)
		# Remove from all clusters
		for members in self.clusters.values():
			members.discard(name)
		self.save()

	def list_stores(self) -> dict[str, dict[str, str]]:
		return dict(self.stores)

	# ---- cluster management ----

	def add_to_cluster(self, cluster_name: str, store_name: str):
		if cluster_name not in self.clusters:
			self.clusters[cluster_name] = set()
		self.clusters[cluster_name].add(store_name)
		self.save()

	def remove_from_cluster(self, cluster_name: str, store_name: str) -> bool:
		if cluster_name not in self.clusters:
			return False
		members = self.clusters[cluster_name]
		if store_name not in members:
			return False
		members.discard(store_name)
		if not members:
			del self.clusters[cluster_name]
		self.save()
		return True

	def list_clusters(self) -> dict[str, set[str]]:
		return dict(self.clusters)

	def stores_in_cluster(self, cluster_name: str) -> set[str]:
		return set(self.clusters.get(cluster_name, set()))

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
