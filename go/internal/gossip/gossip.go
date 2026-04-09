// Package gossip handles UDP broadcast of specialist tag clouds and
// distributed tag queries.
//
// Three packet types share the same multicast/broadcast address+port:
//
//	"announce" — periodic identity broadcast: agent_id, tags, scores
//	"query"    — broadcast tag query, includes reply_addr for unicast responses
//	"response" — unicast reply: scored thought fragments from a matching node
package gossip

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"net"
	"os"
	"sort"
	"strconv"
	"sync"
	"time"
)

const (
	DefaultPort    = 7998
	BroadcastAddr  = "255.255.255.255"
	MaxPacketBytes = 8192
)

// Packet is the unified wire format for all gossip message types.
type Packet struct {
	Type       string     `json:"type"`                  // "announce" | "query" | "response"
	PID        int        `json:"pid"`
	AgentID    string     `json:"agent_id"`
	Tag        string     `json:"tag,omitempty"`         // announce: one tag per packet
	TagScore   float32    `json:"tag_score,omitempty"`
	AgentScore float32    `json:"agent_score,omitempty"`
	HTTPPort   int        `json:"http_port,omitempty"`   // announce: where to reach this agent
	// query fields
	QueryID   string `json:"query_id,omitempty"`
	ReplyAddr string `json:"reply_addr,omitempty"` // "host:port" for unicast responses
	// response fields
	Fragments []Fragment `json:"fragments,omitempty"`
}

// Fragment is one scored thought returned by a query response.
type Fragment struct {
	IP    string  `json:"ip"`
	Text  string  `json:"text"`
	Score float32 `json:"score"`
}

// ScoredFragment is a fragment with its originating agent's scores attached.
type ScoredFragment struct {
	Fragment
	AgentID    string  `json:"agent_id"`
	TagScore   float32 `json:"tag_score"`
	AgentScore float32 `json:"agent_score"`
	Composite  float32 `json:"composite"` // tag_score × agent_score × fragment_score
}

// Broadcaster sends announce packets and can issue distributed tag queries.
type Broadcaster struct {
	agentID  string
	httpPort int
	conn     *net.UDPConn
	addr     *net.UDPAddr
	mu       sync.RWMutex
	tags     []Packet // current announce set (one Packet per tag)
}

func NewBroadcaster(agentID string, httpPort, gossipPort int) (*Broadcaster, error) {
	addr, err := net.ResolveUDPAddr("udp4", net.JoinHostPort(BroadcastAddr, strconv.Itoa(gossipPort)))
	if err != nil {
		return nil, err
	}
	conn, err := net.DialUDP("udp4", nil, addr)
	if err != nil {
		return nil, err
	}
	if err := conn.SetWriteBuffer(64 * 1024); err != nil {
		_ = err // non-fatal
	}
	return &Broadcaster{agentID: agentID, httpPort: httpPort, conn: conn, addr: addr}, nil
}

// UpdateTags replaces the current tag set. Called by the tag-computation loop.
func (b *Broadcaster) UpdateTags(tags []Packet) {
	b.mu.Lock()
	b.tags = tags
	b.mu.Unlock()
}

// CurrentTags returns a snapshot of the current tag set.
func (b *Broadcaster) CurrentTags() []Packet {
	b.mu.RLock()
	defer b.mu.RUnlock()
	out := make([]Packet, len(b.tags))
	copy(out, b.tags)
	return out
}

// Broadcast sends one announce packet per tag in the cloud.
func (b *Broadcaster) Broadcast(tags []Packet) error {
	for _, p := range tags {
		p.Type = "announce"
		p.AgentID = b.agentID
		p.PID = os.Getpid()
		p.HTTPPort = b.httpPort
		data, err := json.Marshal(p)
		if err != nil {
			continue
		}
		if _, err := b.conn.Write(data); err != nil {
			return err
		}
	}
	return nil
}

// QueryNetwork broadcasts a tag query and collects scored fragment responses
// from all nodes that match within the given timeout.
// Results are sorted by composite score (tag_score × agent_score × fragment_score).
func (b *Broadcaster) QueryNetwork(tag string, timeout time.Duration, topK int) ([]ScoredFragment, error) {
	// Open a reply socket on an ephemeral port.
	replyConn, err := net.ListenPacket("udp4", ":0")
	if err != nil {
		return nil, fmt.Errorf("gossip query: open reply socket: %w", err)
	}
	defer replyConn.Close()

	replyPort := replyConn.LocalAddr().(*net.UDPAddr).Port
	localIP := localOutboundIP()
	replyAddr := net.JoinHostPort(localIP, strconv.Itoa(replyPort))

	// Broadcast the query.
	queryID := randomID()
	pkt := Packet{
		Type:      "query",
		AgentID:   b.agentID,
		PID:       os.Getpid(),
		QueryID:   queryID,
		Tag:       tag,
		ReplyAddr: replyAddr,
	}
	data, err := json.Marshal(pkt)
	if err != nil {
		return nil, err
	}
	if _, err := b.conn.Write(data); err != nil {
		return nil, err
	}

	// Collect responses until deadline.
	if err := replyConn.SetDeadline(time.Now().Add(timeout)); err != nil {
		return nil, err
	}

	var frags []ScoredFragment
	buf := make([]byte, MaxPacketBytes)
	for {
		n, _, err := replyConn.ReadFrom(buf)
		if err != nil {
			break // timeout or closed
		}
		var resp Packet
		if err := json.Unmarshal(buf[:n], &resp); err != nil {
			continue
		}
		if resp.QueryID != queryID || resp.Type != "response" {
			continue
		}
		for _, f := range resp.Fragments {
			frags = append(frags, ScoredFragment{
				Fragment:   f,
				AgentID:    resp.AgentID,
				TagScore:   resp.TagScore,
				AgentScore: resp.AgentScore,
				Composite:  resp.TagScore * resp.AgentScore * f.Score,
			})
		}
	}

	sort.Slice(frags, func(i, j int) bool {
		return frags[i].Composite > frags[j].Composite
	})
	if topK > 0 && len(frags) > topK {
		frags = frags[:topK]
	}
	return frags, nil
}

func (b *Broadcaster) Close() { b.conn.Close() }

// BroadcastLoop sends the tag cloud on an interval until stop is closed.
func BroadcastLoop(b *Broadcaster, interval time.Duration, stop <-chan struct{}) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			b.Broadcast(b.CurrentTags())
		case <-stop:
			return
		}
	}
}

// Listener receives gossip packets from other specialists.
type Listener struct {
	conn        *net.UDPConn
	agentID     string
	GetScore    func() float32 // returns this agent's current agent_score
	OnRecv      func(p Packet, senderIP string)
	OnQuery     func(tag string) (tagScore float32, frags []Fragment)
}

func NewListener(agentID string, port int) (*Listener, error) {
	addr, err := net.ResolveUDPAddr("udp4", net.JoinHostPort("", strconv.Itoa(port)))
	if err != nil {
		return nil, err
	}
	conn, err := net.ListenUDP("udp4", addr)
	if err != nil {
		return nil, err
	}
	if err := conn.SetReadBuffer(256 * 1024); err != nil {
		_ = err // non-fatal
	}
	return &Listener{conn: conn, agentID: agentID}, nil
}

// Start reads packets in a loop. Runs until the connection is closed.
func (l *Listener) Start() {
	buf := make([]byte, MaxPacketBytes)
	for {
		n, addr, err := l.conn.ReadFromUDP(buf)
		if err != nil {
			return
		}
		var p Packet
		if err := json.Unmarshal(buf[:n], &p); err != nil {
			continue
		}
		// Ignore own packets.
		if p.AgentID == l.agentID {
			continue
		}

		switch p.Type {
		case "query":
			if l.OnQuery != nil {
				go l.handleQuery(p, addr)
			}
		default: // "announce" or empty type (backward compat)
			if l.OnRecv != nil {
				host := ""
				if addr != nil {
					host = addr.IP.String()
				}
				l.OnRecv(p, host)
			}
		}
	}
}

func (l *Listener) handleQuery(p Packet, senderAddr *net.UDPAddr) {
	tagScore, frags := l.OnQuery(p.Tag)
	if tagScore <= 0 || len(frags) == 0 {
		return
	}

	agentScore := float32(1.0)
	if l.GetScore != nil {
		agentScore = l.GetScore()
	}

	resp := Packet{
		Type:       "response",
		AgentID:    l.agentID,
		PID:        os.Getpid(),
		QueryID:    p.QueryID,
		Tag:        p.Tag,
		TagScore:   tagScore,
		AgentScore: agentScore,
		Fragments:  frags,
	}
	data, err := json.Marshal(resp)
	if err != nil {
		return
	}

	replyAddr, err := net.ResolveUDPAddr("udp4", p.ReplyAddr)
	if err != nil {
		// Fall back: send to sender's port (best effort)
		if senderAddr != nil {
			replyAddr = senderAddr
		} else {
			return
		}
	}

	conn, err := net.DialUDP("udp4", nil, replyAddr)
	if err != nil {
		return
	}
	defer conn.Close()
	conn.Write(data)
}

func (l *Listener) Close() { l.conn.Close() }

// localOutboundIP returns the local IP used for outbound connections.
// Uses a routing-table lookup (no packet sent).
func localOutboundIP() string {
	conn, err := net.Dial("udp4", "8.8.8.8:80")
	if err != nil {
		return "127.0.0.1"
	}
	defer conn.Close()
	return conn.LocalAddr().(*net.UDPAddr).IP.String()
}

func randomID() string {
	return fmt.Sprintf("%016x", rand.Int63())
}
