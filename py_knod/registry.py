"""Specialist registry — store list + knid groupings at ~/.config/knod/stores.

Format (INI-style):
  name = /path/to/graph.graph
  other = /path/to/other.graph

  [health]
  medical
  anatomy

  [marine]
  sea_turtles
"""

import logging
from pathlib import Path

log = logging.getLogger(__name__)


class Registry:
	"""Manages specialist entries and knid groupings.

	Persists to ~/.config/knod/stores in INI-like format.
	Top-level lines are ``name = /path/to/graph.graph`` entries.
	``[knid_name]`` sections list store names belonging to that knid.
	"""

	def __init__(self):
		self._path = Path.home() / ".config" / "knod" / "stores"
		self.stores: dict[str, dict[str, str]] = {}
		self.knids: dict[str, set[str]] = {}
		self._load()

	def _load(self):
		if not self._path.exists():
			return
		current_knid = None
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
			elif "=" in line:
				name, _, path = line.partition("=")
				name = name.strip()
				path = path.strip()
				if name and path:
					self.stores[name] = {"path": path, "purpose": ""}

	def save(self):
		self._path.parent.mkdir(parents=True, exist_ok=True)
		lines = []
		# Store entries
		for name, entry in self.stores.items():
			lines.append(f"{name} = {entry['path']}")
		# Knid sections
		for knid_name, members in self.knids.items():
			lines.append("")
			lines.append(f"[{knid_name}]")
			for member in sorted(members):
				lines.append(member)
		self._path.write_text("\n".join(lines) + "\n", encoding="utf-8")

	def register(self, name: str, path: str, purpose: str = ""):
		self.stores[name] = {"path": path, "purpose": purpose}
		self.save()

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
		"""Add a store to a knid group. Creates the knid if it doesn't exist."""
		if knid_name not in self.knids:
			self.knids[knid_name] = set()
		self.knids[knid_name].add(store_name)
		self.save()

	def remove_from_knid(self, knid_name: str, store_name: str) -> bool:
		"""Remove a store from a knid group. Returns True if found."""
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
		"""Return all knid groupings."""
		return dict(self.knids)

	def stores_in_knid(self, knid_name: str) -> set[str]:
		"""Return store names in a specific knid."""
		return set(self.knids.get(knid_name, set()))
