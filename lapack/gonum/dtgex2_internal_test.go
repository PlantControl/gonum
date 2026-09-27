// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"fmt"
	"math"
	"slices"
	"testing"
)

func TestDtgex2WorkspaceQueryRequiresOutput(t *testing.T) {
	defer func() {
		if got := recover(); got != shortWork {
			t.Fatalf("panic=%v, want %q", got, shortWork)
		}
	}()
	Implementation{}.Dtgex2(false, false, 2,
		nil, 2, nil, 2, nil, 1, nil, 1, 0, 1, 1, nil, -1)
}

// Reused scratch must not retain data from earlier (possibly rejected) swaps.
func TestDtgex2ScratchReuse(t *testing.T) {
	var scratch dtgex2Scratch
	for _, x := range [][]float64{scratch.s[:], scratch.t[:], scratch.li[:], scratch.ir[:], scratch.taul[:], scratch.taur[:], scratch.scpy[:], scratch.tcpy[:], scratch.licopy[:], scratch.sylvester.z[:], scratch.sylvester.rhs[:]} {
		for i := range x {
			x[i] = math.NaN()
		}
	}
	for _, blocks := range [][2]int{{2, 2}, {1, 2}, {2, 1}, {1, 1}, {2, 2}} {
		n1, n2 := blocks[0], blocks[1]
		n := n1 + n2 + 2
		a, b := make([]float64, n*n), make([]float64, n*n)
		for i := range n {
			a[i*n+i] = float64(i + 1)
			b[i*n+i] = 1
			for j := i + 1; j < n; j++ {
				a[i*n+j] = 0.2
				b[i*n+j] = 0.1
			}
		}
		for _, block := range [][2]int{{1, n1}, {1 + n1, n2}} {
			off, size := block[0], block[1]
			if size == 2 {
				a[(off+1)*n+off+1] = a[off*n+off]
				a[off*n+off+1] = 1
				a[(off+1)*n+off] = -1
				b[off*n+off+1] = 0
			}
		}
		run := func(reuse bool) (bool, [][]float64) {
			aa, bb := slices.Clone(a), slices.Clone(b)
			q, z := make([]float64, n*n), make([]float64, n*n)
			for i := range n {
				q[i*n+i], z[i*n+i] = 1, 1
			}
			work := make([]float64, max(n*4, 32))
			var ok bool
			if reuse {
				ok = Implementation{}.dtgex2(true, true, n, aa, n, bb, n, q, n, z, n, 1, n1, n2, work, len(work), &scratch)
			} else {
				ok = Implementation{}.Dtgex2(true, true, n, aa, n, bb, n, q, n, z, n, 1, n1, n2, work, len(work))
			}
			return ok, [][]float64{aa, bb, q, z}
		}
		t.Run(fmt.Sprintf("%dx%d", n1, n2), func(t *testing.T) {
			wantOK, want := run(false)
			gotOK, got := run(true)
			if !wantOK || !gotOK {
				t.Fatal("separated blocks failed to swap")
			}
			for i := range want {
				if !slices.Equal(got[i], want[i]) {
					t.Fatalf("matrix %d depends on previous scratch contents", i)
				}
			}
		})
	}
}
