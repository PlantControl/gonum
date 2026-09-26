// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlapack

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/lapack"
)

func DggesBenchmark(b *testing.B, impl Dggeser) {
	rnd := rand.New(rand.NewPCG(1, 1))
	for _, n := range []int{10, 50, 100, 200} {
		aOrig := make([]float64, n*n)
		bOrig := make([]float64, n*n)
		for i := range aOrig {
			aOrig[i] = rnd.NormFloat64()
			bOrig[i] = rnd.NormFloat64()
		}
		for i := 0; i < n; i++ {
			bOrig[i*n+i] += float64(n)
		}
		a := make([]float64, len(aOrig))
		bm := make([]float64, len(bOrig))
		alphar := make([]float64, n)
		alphai := make([]float64, n)
		beta := make([]float64, n)
		work := make([]float64, 1)
		impl.Dgges(lapack.SchurNone, lapack.SchurNone, lapack.SortNone, nil,
			n, nil, n, nil, n, nil, nil, nil, nil, 1, nil, 1, work, -1, nil)
		work = make([]float64, int(work[0]))

		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				b.StopTimer()
				copy(a, aOrig)
				copy(bm, bOrig)
				b.StartTimer()
				_, ok := impl.Dgges(lapack.SchurNone, lapack.SchurNone, lapack.SortNone, nil,
					n, a, n, bm, n, alphar, alphai, beta,
					nil, 1, nil, 1, work, len(work), nil)
				if !ok {
					b.Fatal("Dgges failed")
				}
			}
		})
	}
}

func DggesScaledSortBenchmark(b *testing.B, impl Dggeser) {
	rnd := rand.New(rand.NewPCG(2, 2))
	selector := func(_, _, _ float64) bool { return false }
	for _, n := range []int{10, 50, 100, 200} {
		aOrig := make([]float64, n*n)
		bOrig := make([]float64, n*n)
		for i := range aOrig {
			aOrig[i] = rnd.NormFloat64() * 1e-200
			bOrig[i] = rnd.NormFloat64() * 1e-200
		}
		for i := 0; i < n; i++ {
			bOrig[i*n+i] += float64(n) * 1e-200
		}
		a := make([]float64, len(aOrig))
		bm := make([]float64, len(bOrig))
		alphar := make([]float64, n)
		alphai := make([]float64, n)
		beta := make([]float64, n)
		bwork := make([]bool, n)
		work := make([]float64, 1)
		impl.Dgges(lapack.SchurNone, lapack.SchurNone, lapack.SortSelected, selector,
			n, nil, n, nil, n, nil, nil, nil, nil, 1, nil, 1, work, -1, nil)
		work = make([]float64, int(work[0]))

		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				b.StopTimer()
				copy(a, aOrig)
				copy(bm, bOrig)
				b.StartTimer()
				_, ok := impl.Dgges(lapack.SchurNone, lapack.SchurNone, lapack.SortSelected, selector,
					n, a, n, bm, n, alphar, alphai, beta,
					nil, 1, nil, 1, work, len(work), bwork)
				if !ok {
					b.Fatal("Dgges failed")
				}
			}
		})
	}
}

func DggesIsolatedBenchmark(b *testing.B, impl Dggeser) {
	for _, n := range []int{10, 50, 100, 200, 500} {
		aOrig := make([]float64, n*n)
		bOrig := make([]float64, n*n)
		for i := range n {
			aOrig[i*n+i] = float64(i + 1)
			bOrig[i*n+i] = 1
		}
		a := make([]float64, len(aOrig))
		bm := make([]float64, len(bOrig))
		alphar := make([]float64, n)
		alphai := make([]float64, n)
		beta := make([]float64, n)
		work := make([]float64, 1)
		impl.Dgges(lapack.SchurNone, lapack.SchurNone, lapack.SortNone, nil,
			n, nil, n, nil, n, nil, nil, nil, nil, 1, nil, 1, work, -1, nil)
		work = make([]float64, int(work[0]))

		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				b.StopTimer()
				copy(a, aOrig)
				copy(bm, bOrig)
				b.StartTimer()
				_, ok := impl.Dgges(lapack.SchurNone, lapack.SchurNone, lapack.SortNone, nil,
					n, a, n, bm, n, alphar, alphai, beta,
					nil, 1, nil, 1, work, len(work), nil)
				if !ok {
					b.Fatal("Dgges failed")
				}
			}
		})
	}
}

// DggesComparisonBenchmark compares preallocated Go and reference LAPACK drivers.
// Each uses its native storage layout; input copies are timed for both, while
// layout conversion and workspace queries are excluded. Go allocation counters
// cannot account for C allocations. The Netlib *_work interface reuses scratch.
// newRun prepares a deterministic pencil and driver outside the timed loop.
// Its returned function restores the inputs and returns the selected dimension
// and convergence status; native selects the reference backend.
func DggesComparisonBenchmark(b *testing.B, kinds []string, newRun func(n int, kind, vectors string, sorting, native bool) func() (int, bool)) {
	for _, n := range []int{10, 50, 100, 200} {
		for _, kind := range kinds {
			for _, vectors := range []string{"none", "right", "both"} {
				for _, sorting := range []bool{false, true} {
					for _, native := range []bool{false, true} {
						backend := "Go"
						if native {
							backend = "Netlib"
						}
						b.Run(fmt.Sprintf("n=%d/%s/vectors=%s/sort=%t/%s", n, kind, vectors, sorting, backend), func(b *testing.B) {
							run := newRun(n, kind, vectors, sorting, native)
							sdim, ok := run()
							if !ok {
								b.Fatal("Dgges failed")
							}
							if sorting && (sdim == 0 || sdim == n) {
								b.Fatal("fixture does not split spectrum")
							}
							b.ReportAllocs()
							b.ResetTimer()
							for i := 0; i < b.N; i++ {
								got, ok := run()
								if !ok || got != sdim {
									b.Fatalf("sdim=%d, ok=%t; want %d, true", got, ok, sdim)
								}
							}
						})
					}
				}
			}
		}
	}
}
