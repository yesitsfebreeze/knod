"""LIMBO — background scan every 60 s.

Rejected thoughts (no links + mature store) accumulate here.
Periodically clustered; clusters are promoted to existing or new Shards.
"""

from .scan import find_clusters
from .promote import promote_cluster, bootstrap_thoughts

__all__ = ["find_clusters", "promote_cluster", "bootstrap_thoughts"]
