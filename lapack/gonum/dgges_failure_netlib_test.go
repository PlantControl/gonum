// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package gonum

import (
	"fmt"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/lapack/gonum/internal/netlib"
)

func TestDggesNetlibNonfiniteFailure(t *testing.T) {
	// A nonfinite active block deterministically reaches QZ failure without
	// changing the production iteration limit. This is not a finite-input
	// nonconvergence certificate.
	const n = 5
	for _, vectors := range [][2]bool{{false, false}, {true, false}, {false, true}, {true, true}} {
		for _, bscale := range []float64{1, 1e-200, 1e200} {
			t.Run(fmt.Sprintf("Vectors=%v/BScale=%g", vectors, bscale), func(t *testing.T) {
				a := []float64{
					2, 0, 0, 0, 0,
					0, 1, 1, 0, 0,
					0, 1, 2, 1, 0,
					0, 0, math.NaN(), 3, 0,
					0, 0, 0, 0, 9,
				}
				b := []float64{
					-bscale, 0, 0, 0, 0,
					0, bscale, 0, 0, 0,
					0, 0, bscale, 0, 0,
					0, 0, 0, bscale, 0,
					0, 0, 0, 0, bscale,
				}
				ga, gb := append([]float64(nil), a...), append([]float64(nil), b...)
				na, nb := factorColMajor(n, n, a, n, n), factorColMajor(n, n, b, n, n)
				gar, gai, gbet := []float64{11, 12, 13, 14, 15}, []float64{21, 22, 23, 24, 25}, []float64{31, 32, 33, 34, 35}
				nar, nai, nbet := append([]float64(nil), gar...), append([]float64(nil), gai...), append([]float64(nil), gbet...)
				gq, gz, nq, nz := make([]float64, n*n), make([]float64, n*n), make([]float64, n*n), make([]float64, n*n)
				jobs := [2]lapack.SchurComp{lapack.SchurNone, lapack.SchurNone}
				njobs := [2]byte{'N', 'N'}
				for i, want := range vectors {
					if want {
						jobs[i], njobs[i] = lapack.SchurHess, 'V'
					}
				}
				gquery, nquery := make([]float64, 1), make([]float64, 1)
				calls := 0
				selector := func(ar, ai, beta float64) bool { calls++; return ar < 0 }
				impl := Implementation{}
				impl.Dgges(jobs[0], jobs[1], lapack.SortSelected, selector, n, nil, n, nil, n, nil, nil, nil, nil, n, nil, n, gquery, -1, nil)
				_, info := netlib.DggesWork(njobs[0], njobs[1], 'L', n, na, nb, nar, nai, nbet, nq, nz, nquery, -1, make([]int32, n))
				if info != 0 {
					t.Fatalf("Netlib query info=%d", info)
				}
				gw := make([]float64, max(8*n, 6*n+16))
				nw := make([]float64, len(gw))
				gs, gok := impl.Dgges(jobs[0], jobs[1], lapack.SortSelected, selector, n, ga, n, gb, n, gar, gai, gbet, gq, n, gz, n, gw, len(gw), make([]bool, n))
				ns, info := netlib.DggesWork(njobs[0], njobs[1], 'L', n, na, nb, nar, nai, nbet, nq, nz, nw, len(nw), make([]int32, n))
				if gok || info < 1 || info >= n {
					t.Fatalf("expected QZ failure: Go ok=%v, Netlib INFO=%d", gok, info)
				}
				checkDggesFinite(t, "Gonum converged suffix", gar[info:], gai[info:], gbet[info:])
				checkDggesFinite(t, "Netlib converged suffix", nar[info:], nai[info:], nbet[info:])
				compareGeneralizedEigenvalues(t, gar[info:], gai[info:], gbet[info:], nar[info:], nai[info:], nbet[info:])
				if gs != 0 || ns != 0 || calls != 0 {
					t.Fatalf("failure entered sorting: Go sdim=%d, Netlib sdim=%d, selector calls=%d", gs, ns, calls)
				}
				if gw[0] != gquery[0] || nw[0] != nquery[0] {
					t.Fatalf("failure workspace Go=%g want %g, Netlib=%g want %g", gw[0], gquery[0], nw[0], nquery[0])
				}
				// Freeze the observed state of this deterministic reference fixture.
				// LAPACK does not promise these complete arrays on QZ failure;
				// only the converged eigenvalue suffix has a portable guarantee.
				for _, pair := range []struct {
					name      string
					got, want []float64
				}{
					{"A", ga, factorRowMajor(n, n, na, n, n)}, {"B", gb, factorRowMajor(n, n, nb, n, n)}, {"alphar", gar, nar}, {"alphai", gai, nai}, {"beta", gbet, nbet},
					{"Q", gq, factorRowMajor(n, n, nq, n, n)}, {"Z", gz, factorRowMajor(n, n, nz, n, n)},
				} {
					for i, got := range pair.got {
						want := pair.want[i]
						if math.IsNaN(got) && math.IsNaN(want) {
							continue
						}
						if got == want {
							continue
						}
						if math.IsNaN(got) || math.IsNaN(want) || math.IsInf(got, 0) || math.IsInf(want, 0) || math.Abs(got-want) > 1e-13*math.Max(math.Abs(got), math.Abs(want)) {
							t.Errorf("%s[%d]: got %g, Netlib %g", pair.name, i, got, want)
						}
					}
				}
			})
		}
	}
}
