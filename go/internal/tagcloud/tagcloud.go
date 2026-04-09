// Package tagcloud derives a specialist's broadcast identity from its embedding space.
//
// Pipeline:
//  1. Cluster the thought embeddings (k-means)
//  2. For each cluster, collect representative thought texts
//  3. Call the LLM to name each cluster → tag
//  4. Emit { tag, tag_score } pairs for broadcasting
//
// Tags are never written by hand. They reflect what the graph actually learned.
package tagcloud

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"net/http"
)

const defaultK = 8 // number of tag clusters

// Tag is one entry in the broadcast cloud.
type Tag struct {
	Name  string  `json:"tag"`
	Score float32 `json:"tag_score"` // cluster cohesion: 0–1
}

// Thought is the minimal input this package needs.
type Thought struct {
	Text      string
	Embedding []float32
}

// Extract derives tags from a set of thoughts.
// It clusters the embeddings, then names each cluster via the LLM API.
func Extract(thoughts []Thought, apiKey, model string) ([]Tag, error) {
	if len(thoughts) == 0 {
		return nil, nil
	}

	k := defaultK
	if len(thoughts) < k {
		k = len(thoughts)
	}

	clusters := kmeans(thoughts, k)
	tags := make([]Tag, 0, len(clusters))

	for _, c := range clusters {
		if len(c.members) == 0 {
			continue
		}
		name, err := nameCluster(c.members, apiKey, model)
		if err != nil {
			return nil, err
		}
		tags = append(tags, Tag{
			Name:  name,
			Score: c.cohesion,
		})
	}
	return tags, nil
}

// --- k-means clustering ---

type cluster struct {
	centroid []float32
	members  []Thought
	cohesion float32 // mean cosine similarity of members to centroid
}

func kmeans(thoughts []Thought, k int) []cluster {
	dim := len(thoughts[0].Embedding)
	centroids := initCentroids(thoughts, k)

	for iter := 0; iter < 20; iter++ {
		// Assign
		assignments := make([]int, len(thoughts))
		for i, t := range thoughts {
			best, bestSim := 0, float32(-1)
			for j, c := range centroids {
				s := cosine(t.Embedding, c)
				if s > bestSim {
					bestSim = s
					best = j
				}
			}
			assignments[i] = best
		}

		// Recompute centroids
		newCentroids := make([][]float32, k)
		counts := make([]int, k)
		for j := range newCentroids {
			newCentroids[j] = make([]float32, dim)
		}
		for i, t := range thoughts {
			j := assignments[i]
			counts[j]++
			for d, v := range t.Embedding {
				newCentroids[j][d] += v
			}
		}
		for j := range newCentroids {
			if counts[j] > 0 {
				n := float32(counts[j])
				for d := range newCentroids[j] {
					newCentroids[j][d] /= n
				}
			}
		}
		centroids = newCentroids
	}

	// Build final clusters
	clusters := make([]cluster, k)
	for j := range clusters {
		clusters[j].centroid = centroids[j]
	}
	for _, t := range thoughts {
		best, bestSim := 0, float32(-1)
		for j, c := range centroids {
			s := cosine(t.Embedding, c)
			if s > bestSim {
				bestSim = s
				best = j
			}
		}
		clusters[best].members = append(clusters[best].members, t)
	}

	// Compute cohesion per cluster
	for j := range clusters {
		c := &clusters[j]
		if len(c.members) == 0 {
			continue
		}
		var sum float32
		for _, t := range c.members {
			sum += cosine(t.Embedding, c.centroid)
		}
		c.cohesion = sum / float32(len(c.members))
	}

	return clusters
}

func initCentroids(thoughts []Thought, k int) [][]float32 {
	perm := rand.Perm(len(thoughts))
	centroids := make([][]float32, k)
	for i := 0; i < k; i++ {
		src := thoughts[perm[i]].Embedding
		c := make([]float32, len(src))
		copy(c, src)
		centroids[i] = c
	}
	return centroids
}

// --- LLM cluster naming ---

func nameCluster(members []Thought, apiKey, model string) (string, error) {
	// Pick up to 5 representative texts
	limit := 5
	if len(members) < limit {
		limit = len(members)
	}
	texts := make([]string, limit)
	for i := 0; i < limit; i++ {
		texts[i] = members[i].Text
	}

	prompt := "Given these related statements, respond with a single short tag (2-4 words) that names their common topic. Return only the tag, nothing else.\n\n"
	for _, t := range texts {
		prompt += "- " + t + "\n"
	}

	body := map[string]any{
		"model": model,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
		"max_tokens": 20,
	}
	data, _ := json.Marshal(body)

	req, err := http.NewRequest("POST", "https://api.openai.com/v1/chat/completions", bytes.NewReader(data))
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var result struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", err
	}
	if len(result.Choices) == 0 {
		return "", fmt.Errorf("no choices in LLM response")
	}
	return result.Choices[0].Message.Content, nil
}

// --- math helpers ---

func cosine(a, b []float32) float32 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return float32(dot / (math.Sqrt(na) * math.Sqrt(nb)))
}
