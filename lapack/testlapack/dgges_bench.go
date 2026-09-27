// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlapack

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/lapack"
)

// DggesBenchmarkPencil returns a deterministic dense regular matrix pencil.
// n must be a multiple of 8 and at least 8.
func DggesBenchmarkPencil(n int) (a, b []float64) {
	if n < 8 || n%8 != 0 {
		panic("testlapack: invalid Dgges benchmark pencil size")
	}
	a = make([]float64, n*n)
	b = make([]float64, n*n)
	blocks := n / 2
	reals := [...]float64{-2, 0.5, 2, -0.5}
	for block := 0; block < blocks; block++ {
		i := 2 * block
		scale := 1 + 0.1*float64(block+1)/float64(blocks)
		beta := 1 + 0.125*float64(block%4)
		real := reals[block%len(reals)] * scale
		imag := 0.2 * scale
		a[i*n+i] = beta * real
		a[i*n+i+1] = beta * imag
		a[(i+1)*n+i] = -beta * imag
		a[(i+1)*n+i+1] = beta * real
		b[i*n+i] = beta
		b[(i+1)*n+i+1] = beta
	}

	rotateRows := func(x []float64, i, j int, c, s float64) {
		for k := 0; k < n; k++ {
			xi := x[i*n+k]
			xj := x[j*n+k]
			x[i*n+k] = c*xi + s*xj
			x[j*n+k] = c*xj - s*xi
		}
	}
	rotateCols := func(x []float64, i, j int, c, s float64) {
		for k := 0; k < n; k++ {
			xi := x[k*n+i]
			xj := x[k*n+j]
			x[k*n+i] = c*xi + s*xj
			x[k*n+j] = c*xj - s*xi
		}
	}
	for sweep := 0; sweep < 4; sweep++ {
		for i := 0; i < n; i++ {
			j := (i*37 + sweep*11 + 1) % n
			if i == j {
				j = (j + 1) % n
			}
			angle := 0.07 * float64(1+(i+3*sweep)%9)
			c, s := math.Cos(angle), math.Sin(angle)
			rotateRows(a, i, j, c, s)
			rotateRows(b, i, j, c, s)

			j = (i*29 + sweep*13 + 3) % n
			if i == j {
				j = (j + 1) % n
			}
			angle = 0.05 * float64(1+(2*i+sweep)%11)
			c, s = math.Cos(angle), math.Sin(angle)
			rotateCols(a, i, j, c, s)
			rotateCols(b, i, j, c, s)
		}
	}
	return a, b
}

// DggesBenchmarkSelect selects eigenvalues for DggesControlBenchmark.
func DggesBenchmarkSelect(selection byte, ar, ai, beta float64) bool {
	switch selection {
	case 'N':
		return false
	case 'L':
		return ar < 0 && beta > 0 || ar > 0 && beta < 0
	case 'D':
		return math.Hypot(ar, ai) < math.Abs(beta)
	default:
		panic("testlapack: invalid Dgges benchmark selection")
	}
}

// DggesControlBenchmark benchmarks representative Dgges modes on a dense
// regular pencil with known selection counts.
func DggesControlBenchmark(b *testing.B, impl Dggeser) {
	modes := []struct {
		name       string
		jobvsl     lapack.SchurComp
		jobvsr     lapack.SchurComp
		sort       lapack.SchurSort
		selection  byte
		wantSelect bool
	}{
		{name: "form", jobvsl: lapack.SchurNone, jobvsr: lapack.SchurNone, sort: lapack.SortNone, selection: 'N'},
		{name: "vectors", jobvsl: lapack.SchurHess, jobvsr: lapack.SchurHess, sort: lapack.SortNone, selection: 'N'},
		{name: "left", jobvsl: lapack.SchurHess, jobvsr: lapack.SchurHess, sort: lapack.SortSelected, selection: 'L', wantSelect: true},
		{name: "unit", jobvsl: lapack.SchurHess, jobvsr: lapack.SchurHess, sort: lapack.SortSelected, selection: 'D', wantSelect: true},
	}
	for _, mode := range modes {
		for _, n := range []int{32, 64, 128, 256} {
			aOrig, bOrig := DggesBenchmarkPencil(n)
			a := make([]float64, len(aOrig))
			bm := make([]float64, len(bOrig))
			alphar := make([]float64, n)
			alphai := make([]float64, n)
			beta := make([]float64, n)
			var vsl, vsr []float64
			if mode.jobvsl == lapack.SchurHess {
				vsl = make([]float64, n*n)
			}
			if mode.jobvsr == lapack.SchurHess {
				vsr = make([]float64, n*n)
			}
			var selector lapack.SchurSelect
			var bwork []bool
			if mode.sort == lapack.SortSelected {
				selection := mode.selection
				selector = func(ar, ai, beta float64) bool {
					return DggesBenchmarkSelect(selection, ar, ai, beta)
				}
				bwork = make([]bool, n)
			}
			work := make([]float64, 1)
			impl.Dgges(mode.jobvsl, mode.jobvsr, mode.sort, selector,
				n, nil, n, nil, n, nil, nil, nil,
				nil, n, nil, n, work, -1, bwork)
			work = make([]float64, int(work[0]))
			wantSdim := 0
			if mode.wantSelect {
				wantSdim = n / 2
			}

			b.Run(fmt.Sprintf("mode=%s/n=%d", mode.name, n), func(b *testing.B) {
				b.ReportAllocs()
				b.ResetTimer()
				for range b.N {
					copy(a, aOrig)
					copy(bm, bOrig)
					sdim, ok := impl.Dgges(mode.jobvsl, mode.jobvsr, mode.sort, selector,
						n, a, n, bm, n, alphar, alphai, beta,
						vsl, n, vsr, n, work, len(work), bwork)
					if !ok || sdim != wantSdim {
						b.Fatalf("Dgges failed: ok=%t sdim=%d want=%d", ok, sdim, wantSdim)
					}
				}
			})
		}
	}
}

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
