// Package gnn implements a small MPNN (message-passing neural network)
// for graph navigation and thought scoring.
//
// Architecture:
//   - 3 message-passing layers
//   - 512 hidden dim
//   - Input: thought embedding (1536) + aggregated neighbor embeddings (1536)
//   - Output: relevance score per thought
//
// No external ML library. Pure Go matrix math.
// Weights are loaded from a .gnn file alongside the .shard graph.
// Training happens externally (Python); this package does inference only.
package gnn

import "math"

const (
	InputDim  = 1536
	HiddenDim = 512
	NumLayers = 3
)

// Layer is one message-passing step: project → aggregate → normalize.
type Layer struct {
	W [][]float32 // [HiddenDim][InputDim or HiddenDim]
	B []float32   // [HiddenDim]
}

// MPNN holds the full model weights.
type MPNN struct {
	Layers [NumLayers]*Layer
	ScoreW []float32 // [HiddenDim] — dot with hidden to get scalar score
}

// Score returns a relevance score for a thought given its embedding
// and the mean embedding of its neighbors.
func (m *MPNN) Score(thoughtEmb, neighborMean []float32) float32 {
	// Concatenate thought + neighbor mean as input
	input := make([]float32, InputDim*2)
	copy(input[:InputDim], thoughtEmb)
	copy(input[InputDim:], neighborMean)

	hidden := input
	for i, layer := range m.Layers {
		hidden = layer.forward(hidden, i == 0)
	}

	// Final dot product with score head
	var score float32
	for i, w := range m.ScoreW {
		score += w * hidden[i]
	}
	return sigmoid(score)
}

func (l *Layer) forward(input []float32, first bool) []float32 {
	out := make([]float32, HiddenDim)
	inDim := len(input)
	for i := 0; i < HiddenDim; i++ {
		var sum float32
		for j := 0; j < inDim && j < len(l.W[i]); j++ {
			sum += l.W[i][j] * input[j]
		}
		sum += l.B[i]
		out[i] = relu(sum)
	}
	layerNorm(out)
	return out
}

func relu(x float32) float32 {
	if x < 0 {
		return 0
	}
	return x
}

func sigmoid(x float32) float32 {
	return float32(1.0 / (1.0 + math.Exp(float64(-x))))
}

func layerNorm(v []float32) {
	var mean float32
	for _, x := range v {
		mean += x
	}
	mean /= float32(len(v))

	var variance float32
	for _, x := range v {
		d := x - mean
		variance += d * d
	}
	variance /= float32(len(v))
	std := float32(math.Sqrt(float64(variance) + 1e-5))

	for i := range v {
		v[i] = (v[i] - mean) / std
	}
}

// MeanEmbedding computes the element-wise mean of a set of embeddings.
func MeanEmbedding(embeddings [][]float32) []float32 {
	if len(embeddings) == 0 {
		return make([]float32, InputDim)
	}
	out := make([]float32, len(embeddings[0]))
	for _, emb := range embeddings {
		for i, v := range emb {
			out[i] += v
		}
	}
	n := float32(len(embeddings))
	for i := range out {
		out[i] /= n
	}
	return out
}
