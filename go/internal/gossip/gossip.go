// Package gossip handles UDP broadcast of specialist tag clouds.
//
// Packet format (one per tag):
//
//	{ agent_id: string, tag: string, tag_score: float32, agent_score: float32 }
//
// Each specialist periodically broadcasts its full tag cloud.
// The registry listens and indexes incoming packets.
package gossip

import (
	"encoding/json"
	"net"
	"time"
)

const (
	DefaultPort    = 7998
	BroadcastAddr  = "255.255.255.255"
	MaxPacketBytes = 1024
)

// Packet is one tag broadcast from a specialist.
type Packet struct {
	AgentID    string  `json:"agent_id"`
	Tag        string  `json:"tag"`
	TagScore   float32 `json:"tag_score"`
	AgentScore float32 `json:"agent_score"`
}

// Broadcaster sends a specialist's tag cloud over UDP.
type Broadcaster struct {
	agentID string
	conn    *net.UDPConn
	addr    *net.UDPAddr
}

func NewBroadcaster(agentID string, port int) (*Broadcaster, error) {
	addr, err := net.ResolveUDPAddr("udp", net.JoinHostPort(BroadcastAddr, itoa(port)))
	if err != nil {
		return nil, err
	}
	conn, err := net.DialUDP("udp", nil, addr)
	if err != nil {
		return nil, err
	}
	return &Broadcaster{agentID: agentID, conn: conn, addr: addr}, nil
}

// Broadcast sends one packet per tag in the cloud.
func (b *Broadcaster) Broadcast(tags []Packet) error {
	for _, p := range tags {
		p.AgentID = b.agentID
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

func (b *Broadcaster) Close() { b.conn.Close() }

// Listener receives tag broadcasts from other specialists.
type Listener struct {
	conn   *net.UDPConn
	OnRecv func(Packet)
}

func NewListener(port int) (*Listener, error) {
	addr, err := net.ResolveUDPAddr("udp", net.JoinHostPort("", itoa(port)))
	if err != nil {
		return nil, err
	}
	conn, err := net.ListenUDP("udp", addr)
	if err != nil {
		return nil, err
	}
	return &Listener{conn: conn}, nil
}

func (l *Listener) Start() {
	buf := make([]byte, MaxPacketBytes)
	for {
		n, _, err := l.conn.ReadFromUDP(buf)
		if err != nil {
			return
		}
		var p Packet
		if err := json.Unmarshal(buf[:n], &p); err != nil {
			continue
		}
		if l.OnRecv != nil {
			l.OnRecv(p)
		}
	}
}

func (l *Listener) Close() { l.conn.Close() }

// BroadcastLoop sends the tag cloud on an interval until stop is closed.
func BroadcastLoop(b *Broadcaster, tags func() []Packet, interval time.Duration, stop <-chan struct{}) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			b.Broadcast(tags())
		case <-stop:
			return
		}
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	buf := make([]byte, 0, 8)
	for n > 0 {
		buf = append([]byte{byte('0' + n%10)}, buf...)
		n /= 10
	}
	return string(buf)
}
