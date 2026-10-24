// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package gonum

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/lapack/gonum/internal/netlib"
)

// TestDlacn2Netlib compares the Dlacn2 iteration with Netlib on matrices
// with entries -1, -0 and 1, whose products contain signed zeros: DLACN2
// maps -0 to a positive sign.
func TestDlacn2Netlib(t *testing.T) {
	rnd := rand.New(rand.NewPCG(81, 81))
	for n := 1; n <= 8; n++ {
		for k := 0; k < 200; k++ {
			a := make([]float64, n*n)
			for i := range a {
				a[i] = []float64{-1, math.Copysign(0, -1), 1}[rnd.IntN(3)]
			}
			apply := func(kase int, x []float64) {
				y := make([]float64, n)
				for i := range n {
					for j := range n {
						v := a[i*n+j] * x[j]
						if kase != 1 {
							v = a[j*n+i] * x[j]
						}
						if j == 0 {
							y[i] = v
						} else {
							y[i] += v
						}
					}
				}
				copy(x, y)
			}
			gx, gv := make([]float64, n), make([]float64, n)
			gisgn := make([]int, n)
			var gsave [3]int
			gest, gkase := 0.0, 0
			nx, nv := make([]float64, n), make([]float64, n)
			nisgn := make([]int32, n)
			var nsave [3]int32
			nest, nkase := 0.0, 0
			name := fmt.Sprintf("n=%d/k=%d", n, k)
			for step := 0; ; step++ {
				gest, gkase = Implementation{}.Dlacn2(n, gv, gx, gisgn, gest, gkase, &gsave)
				nest, nkase = netlib.Dlacn2(n, nv, nx, nisgn, nest, nkase, &nsave)
				if gkase != nkase || gest != nest && !(math.IsNaN(gest) && math.IsNaN(nest)) {
					t.Fatalf("%s step %d: est=%v kase=%d, want Netlib %v %d", name, step, gest, gkase, nest, nkase)
				}
				if gkase == 0 {
					break
				}
				apply(gkase, gx)
				apply(nkase, nx)
			}
		}
	}
}
