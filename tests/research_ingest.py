"""Ingest deep research on AI improvement methods and content storage into knod."""

import time
import sys

from knod.config import Config
from knod.handler import Handler

RESEARCH_ARTICLES = {
	"continual_learning": (
		"Incremental learning is a method of machine learning where input data is "
		"continuously used to extend the existing model's knowledge without retraining "
		"from scratch. It represents a dynamic technique of supervised and unsupervised "
		"learning applied when training data arrives gradually over time or exceeds "
		"system memory limits. The aim is for the learning model to adapt to new data "
		"without forgetting existing knowledge, known as catastrophic forgetting. "
		"Stable incremental machine learning algorithms learn representations that are "
		"not even partially forgotten over time. Examples include decision trees like "
		"IDE4 and ID5R, neural networks like RBF networks, Learn++, Fuzzy ARTMAP, "
		"TopoART, and incremental SVMs. Fuzzy ART and TopoART are stable incremental "
		"learners that never forget previously learned representations. Incremental "
		"algorithms are frequently applied to data streams and big data, addressing "
		"issues in data availability and resource scarcity."
	),
	"knowledge_graph_embedding": (
		"Knowledge graph embedding (KGE) is a machine learning task of learning "
		"low-dimensional representations of a knowledge graph's entities and relations "
		"while preserving semantic meaning. A knowledge graph G consists of entities E, "
		"relations R, and facts F where each fact is a triple (head, relation, tail). "
		"The embedding translates each entity and relation into a vector of dimension d. "
		"KGE is characterized by four aspects: representation space, scoring function, "
		"encoding models, and additional information. Models include tensor decomposition "
		"approaches like DistMult, ComplEx, ANALOGY, and SimplE. Geometric models include "
		"TransE using vector sum h+r=t, TransH using hyperplanes, TransR using separate "
		"entity and relation spaces, and RotatE representing relations as rotations in "
		"complex space via Euler's identity. Deep learning models use convolutional or "
		"recurrent neural networks for embedding. Performance is measured via Hits@K, "
		"Mean Rank, and Mean Reciprocal Rank. Applications include link prediction, "
		"triple classification, entity recognition, clustering, and relation extraction."
	),
	"retrieval_augmented_generation": (
		"Retrieval-augmented generation (RAG) enables large language models to retrieve "
		"and incorporate new information from external data sources before generating "
		"responses. RAG improves LLMs by pulling relevant text from databases, documents, "
		"or web sources rather than relying solely on static training data. This reduces "
		"AI hallucinations and the need to retrain models. The process involves converting "
		"reference data into embeddings stored in vector databases, then using document "
		"retrieval to find relevant context for user queries. Key improvements include "
		"encoder optimizations with approximate nearest neighbor searches, retriever-centric "
		"methods using supervised optimization minimizing KL divergence, and language model "
		"redesigns like Retro achieving comparable performance with 25x smaller networks. "
		"Chunking strategies include fixed-length with overlap, syntax-based sentence "
		"splitting, and file-format-aware chunking. Hybrid search combines vector database "
		"searches with traditional text searches. RAG does not fully eliminate hallucinations "
		"since LLMs can misinterpret context from correct sources. RAG poisoning occurs "
		"when models retrieve correct but misleading sources."
	),
	"gnn_advanced_architectures": (
		"Graph neural networks use pairwise message passing where nodes iteratively "
		"update representations by exchanging information with neighbors. The architecture "
		"implements permutation equivariant layers, local pooling for downsampling, and "
		"global pooling for fixed-size graph representation. A Message Passing Neural "
		"Network layer computes new node representations by combining current features "
		"with aggregated neighbor messages. Graph Convolutional Networks use spectral "
		"filters with symmetric normalization of the adjacency matrix. Graph Attention "
		"Networks introduce learnable attention coefficients measuring node importance, "
		"computed via LeakyReLU activation and multi-head attention. Gated Graph Sequence "
		"Neural Networks extend GNNs with GRU cells for sequential output. Stacking n "
		"MPNN layers allows n-hop communication but risks oversmoothing where node "
		"representations become indistinguishable, and oversquashing from compressing "
		"long-range dependencies into fixed-size representations. Countermeasures include "
		"skip connections, gated update rules, jumping knowledge, and fully-adjacent final "
		"layers treating the graph as complete."
	),
	"ai_self_improvement": (
		"AI self-improvement encompasses several paradigms for autonomous knowledge "
		"enhancement. Recursive self-improvement allows an AI system to modify its own "
		"architecture and learning algorithms to become more capable. Meta-learning or "
		"learning-to-learn trains models to adapt quickly to new tasks from few examples. "
		"Reinforcement learning from human feedback (RLHF) uses human preference signals "
		"to align model behavior. Self-supervised learning generates training signals from "
		"unlabeled data through pretext tasks like masked language modeling, next token "
		"prediction, and contrastive learning. Active learning selects the most informative "
		"samples for labeling, reducing annotation costs. Curriculum learning orders "
		"training examples from easy to hard, mimicking human education. Elastic Weight "
		"Consolidation prevents catastrophic forgetting by constraining important weights "
		"when learning new tasks. Progressive neural networks freeze old columns and add "
		"new capacity for new tasks. Experience replay stores and replays past experiences "
		"to maintain knowledge while learning new information."
	),
	"content_storage_methods": (
		"Modern content storage methods for AI systems span multiple paradigms. Vector "
		"databases like Pinecone, Milvus, Weaviate, and FAISS store high-dimensional "
		"embeddings for approximate nearest neighbor search with sub-millisecond latency. "
		"Graph databases like Neo4j and Amazon Neptune store content as nodes and edges "
		"enabling traversal queries and relationship-centric retrieval. Hybrid storage "
		"combines vector search with structured graph relationships for richer content "
		"retrieval. Content-addressable storage uses cryptographic hashes to deduplicate "
		"and retrieve content by its fingerprint rather than location. Hierarchical storage "
		"organizes content in multi-level structures with summarization at each level. "
		"Temporal storage tracks content evolution over time enabling version-aware "
		"retrieval. Distributed hash tables spread content across nodes for fault-tolerant "
		"decentralized storage. Knowledge vaults combine information extraction with "
		"knowledge base construction to automatically populate knowledge graphs from "
		"web content."
	),
	"embedding_innovations": (
		"Recent embedding space innovations improve how AI systems represent and store "
		"knowledge. Matryoshka Representation Learning trains embeddings that work at "
		"multiple dimensionalities by optimizing loss at several truncation points "
		"simultaneously, enabling flexible trade-offs between accuracy and efficiency. "
		"Hyperbolic embeddings represent hierarchical structures more naturally than "
		"Euclidean spaces by encoding tree-like relationships in hyperbolic geometry "
		"where distance grows exponentially with depth. Sparse embeddings from models "
		"like SPLADE combine the interpretability of bag-of-words with learned "
		"representations. Late interaction models like ColBERT store per-token embeddings "
		"and compute fine-grained similarity at query time for better retrieval accuracy. "
		"Mixture of Experts routing allows embeddings to be processed by specialized "
		"sub-networks based on content type. Contrastive learning with hard negatives "
		"produces more discriminative embeddings by focusing on difficult similar pairs. "
		"Quantized embeddings using product quantization reduce storage requirements "
		"while maintaining retrieval quality. Binary embeddings compress vectors to "
		"single bits per dimension for ultra-fast hamming distance comparisons."
	),
	"graph_knowledge_persistence": (
		"Persisting knowledge in graph structures requires balancing fidelity, efficiency, "
		"and queryability. Binary serialization formats store graph topology and node "
		"features compactly using tagged sections with magic numbers and version headers "
		"for backward compatibility. Incremental persistence writes only changed portions "
		"of the graph to reduce I/O overhead during frequent saves. Write-ahead logging "
		"ensures durability by recording mutations before applying them to the main store. "
		"Snapshot isolation provides consistent reads while concurrent writes proceed. "
		"Graph partitioning using vertex-cut or edge-cut strategies distributes large "
		"graphs across multiple storage nodes while minimizing cross-partition edges. "
		"Property graphs extend basic node-edge structures with typed properties and "
		"labels enabling rich metadata storage. RDF triple stores use subject-predicate-"
		"object triples with URI-based identifiers for global knowledge interoperability. "
		"Graph compression techniques like grammar-based compression and community-aware "
		"encoding reduce storage footprint while maintaining traversal performance."
	),
}

QUESTIONS = [
	(
		"What are the main approaches to prevent catastrophic forgetting?",
		["forget", "elastic", "weight", "incremental", "replay", "progressive", "stable"],
	),
	(
		"How does RAG improve large language model accuracy?",
		["retriev", "augment", "hallucin", "vector", "database", "document", "context"],
	),
	(
		"What are the differences between TransE and RotatE for knowledge graph embedding?",
		["trans", "rotat", "vector", "complex", "relation", "embed", "space"],
	),
	(
		"What modern storage methods exist for AI content systems?",
		["vector", "graph", "hash", "hierarch", "temporal", "storage", "database"],
	),
	(
		"How do graph attention networks differ from graph convolutional networks?",
		["attention", "coefficient", "learnable", "weight", "gcn", "gat", "spectral"],
	),
	(
		"What are the latest innovations in embedding representations?",
		["matryoshka", "hyperbol", "sparse", "quantiz", "contrastive", "colbert", "binary"],
	),
]


def main():
	cfg = Config.load()
	if not cfg.api_key:
		print("ERROR: no API key found.")
		sys.exit(1)

	handler = Handler(cfg)
	handler.init()

	# Phase 1: Ingest all research articles
	print("\n=== Ingesting Research Articles ===")
	for i, (name, text) in enumerate(RESEARCH_ARTICLES.items()):
		t0 = time.time()
		r = handler.ingest_sync(text, source=name)
		dt = time.time() - t0
		c = r["committed"]
		e = r["edges"]
		d = r["deduplicated"]
		print(f"  [{i + 1}/{len(RESEARCH_ARTICLES)}] {name}: {c} committed, {e} edges, {d} dedup ({dt:.1f}s)")

	g = handler.graph
	print(f"\n  Total: {g.num_thoughts} thoughts, {g.num_edges} edges, maturity={g.maturity:.3f}")
	print(f"  Limbo: {len(g.limbo)}")
	sources = set(t.source for t in g.thoughts.values())
	print(f"  Sources: {sources}")

	# Phase 2: Ask questions about the new knowledge
	print("\n=== Testing Questions on Research Knowledge ===")
	passed = 0
	failed = 0
	for q, hint_words in QUESTIONS:
		answer_text, srcs = handler.ask(q)
		has_answer = len(answer_text) > 10
		answer_lower = answer_text.lower()
		relevant = any(w in answer_lower for w in hint_words)

		print(f"\n  Q: {q}")
		if len(answer_text) > 300:
			print(f"  A: {answer_text[:300]}...")
		else:
			print(f"  A: {answer_text}")

		if has_answer and relevant:
			passed += 1
			print("  PASS - relevant answer")
		else:
			failed += 1
			print(f"  FAIL - expected one of: {hint_words}")
		print(f"  Sources: {len(srcs)}")

	total = passed + failed
	print(f"\n{'=' * 60}")
	print(f"RESULTS: {passed}/{total} questions answered relevantly")
	print(f"{'=' * 60}")


if __name__ == "__main__":
	main()
