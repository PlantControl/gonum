// Copyright ©2017 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package topo_test

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"slices"
	"strings"

	"plantcontrol.org/v1/gonum/graph/simple"
	"plantcontrol.org/v1/gonum/graph/topo"
)

var systems = []string{
	// Unsatisfiable system.
	`𝑥_a ∨ ¬𝑥_b
¬𝑥_b ∨ 𝑥_f
𝑥_h ∨ 𝑥_i
𝑥_a ∨ 𝑥_b
𝑥_k ∨ 𝑥_c
¬𝑥_f ∨ 𝑥_h
𝑥_c ∨ 𝑥_g
𝑥_f ∨ 𝑥_g
𝑥_h ∨ ¬𝑥_l
¬𝑥_h ∨ 𝑥_i
𝑥_i ∨ 𝑥_b
¬𝑥_i ∨ ¬𝑥_h
𝑥_i ∨ ¬𝑥_c
𝑥_l ∨ 𝑥_d
¬𝑥_j ∨ ¬𝑥_i
¬𝑥_a ∨ ¬𝑥_j
¬𝑥_a ∨ 𝑥_b
¬𝑥_d ∨ 𝑥_e
¬𝑥_k ∨ 𝑥_h
𝑥_l ∨ ¬𝑥_d
𝑥_l ∨ 𝑥_d
𝑥_l ∨ ¬𝑥_f
𝑥_b ∨ 𝑥_d
𝑥_b ∨ ¬𝑥_g
𝑥_d ∨ ¬𝑥_l
¬𝑥_l ∨ ¬𝑥_k
`,
	// Satisfiable system.
	`𝑥_a ∨ ¬𝑥_b
¬𝑥_b ∨ 𝑥_f
𝑥_h ∨ 𝑥_i
𝑥_a ∨ 𝑥_b
𝑥_k ∨ 𝑥_c
¬𝑥_f ∨ 𝑥_h
𝑥_c ∨ 𝑥_g
𝑥_f ∨ 𝑥_g
𝑥_h ∨ ¬𝑥_l
¬𝑥_h ∨ 𝑥_i
𝑥_i ∨ 𝑥_b
¬𝑥_i ∨ 𝑥_e
𝑥_i ∨ ¬𝑥_c
¬𝑥_g ∨ ¬𝑥_a
𝑥_l ∨ 𝑥_f
¬𝑥_j ∨ ¬𝑥_i
¬𝑥_a ∨ ¬𝑥_j
¬𝑥_a ∨ 𝑥_b
¬𝑥_d ∨ 𝑥_e
𝑥_k ∨ ¬𝑥_a
𝑥_k ∨ 𝑥_h
𝑥_l ∨ ¬𝑥_d
𝑥_l ∨ 𝑥_e
𝑥_l ∨ ¬𝑥_f
𝑥_b ∨ 𝑥_d
𝑥_b ∨ ¬𝑥_g
𝑥_d ∨ ¬𝑥_l
𝑥_l ∨ 𝑥_e
`,

	`fun ∨ ¬fun
fun ∨ ¬Gonum
Gonum ∨ Gonum
`,
}

// twoSat returns whether the system described in the data read from r is
// satisfiable and a set of states that satisfies the system.
// The syntax used by twoSat is "𝑥 ∨ 𝑦" where 𝑥 and 𝑦 may be negated by
// leading "¬" characters. twoSat uses the implication graph approach to
// system analysis.
func twoSat(r io.Reader) (state map[string]bool, ok bool) {
	g := simple.NewDirectedGraph()

	sc := bufio.NewScanner(r)
	nodes := make(map[string]node)
	for count := 1; sc.Scan(); count++ {
		line := sc.Text()
		fields := strings.Split(line, "∨")
		if len(fields) != 2 {
			log.Fatalf("failed to parse on line %d %q: invalid syntax", count, line)
		}
		var variables [2]node
		for i, f := range fields {
			f = strings.TrimSpace(f)
			var negate bool
			for strings.Index(f, "¬") == 0 {
				f = strings.TrimPrefix(f, "¬")
				negate = !negate
			}
			n, ok := nodes[f]
			if !ok {
				n = node{
					id:   int64(len(nodes) + 1), // id must not be zero.
					name: f,
				}
				nodes[f] = n
			}
			if negate {
				n = n.negated()
			}
			variables[i] = n
		}

		// Check for tautology.
		if variables[0].negated().ID() == variables[1].ID() {
			for _, v := range variables {
				if g.Node(v.ID()) == nil {
					g.AddNode(v)
				}
			}
			continue
		}

		// Add implications to the graph.
		g.SetEdge(simple.Edge{F: variables[0].negated(), T: variables[1]})
		g.SetEdge(simple.Edge{F: variables[1].negated(), T: variables[0]})
	}

	// Find implication inconsistencies.
	sccs := topo.TarjanSCC(g)
	for _, c := range sccs {
		set := make(map[int64]struct{})
		for _, n := range c {
			id := n.ID()
			if _, ok := set[-id]; ok {
				return nil, false
			}
			set[id] = struct{}{}
		}
	}

	// Assign states.
	state = make(map[string]bool)
unknown:
	for _, c := range sccs {
		for _, n := range c {
			if _, known := state[n.(node).name]; known {
				continue unknown
			}
		}
		for _, n := range c {
			n := n.(node)
			state[n.name] = n.id > 0
		}
	}

	return state, true
}

type node struct {
	id   int64
	name string
}

func (n node) ID() int64     { return n.id }
func (n node) negated() node { return node{-n.id, n.name} }

func ExampleTarjanSCC_twoSAT() {
	for i, s := range systems {
		state, ok := twoSat(strings.NewReader(s))
		if !ok {
			fmt.Printf("system %d is not satisfiable\n", i)
			continue
		}
		var ps []string
		for v, t := range state {
			ps = append(ps, fmt.Sprintf("%s:%t", v, t))
		}
		slices.Sort(ps)
		fmt.Printf("system %d is satisfiable: %s\n", i, strings.Join(ps, " "))
	}

	// Output:
	// system 0 is not satisfiable
	// system 1 is satisfiable: 𝑥_a:true 𝑥_b:true 𝑥_c:true 𝑥_d:true 𝑥_e:true 𝑥_f:true 𝑥_g:false 𝑥_h:true 𝑥_i:true 𝑥_j:false 𝑥_k:true 𝑥_l:true
	// system 2 is satisfiable: Gonum:true fun:true
}
