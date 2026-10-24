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

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/lapack/gonum/internal/netlib"
)

func checkScaledNetlib(t *testing.T, name string, got, want, scale float64) {
	t.Helper()
	if math.Abs(got-want) > 1e-9*scale {
		t.Errorf("%s=%g, want Netlib %g (scale %g)", name, got, want, scale)
	}
}

// randomNetlibQuasiTriangular returns a random n×n upper quasi-triangular
// matrix in Schur canonical form.
func randomNetlibQuasiTriangular(n int, rnd *rand.Rand) []float64 {
	a := make([]float64, n*n)
	for i := range n {
		for j := i + 1; j < n; j++ {
			a[i*n+j] = rnd.NormFloat64()
		}
	}
	for j := 0; j < n; {
		if j < n-1 && rnd.Float64() < 0.5 {
			r := rnd.NormFloat64()
			a[j*n+j], a[j*n+j+1] = r, rnd.Float64()+0.5
			a[(j+1)*n+j], a[(j+1)*n+j+1] = -(rnd.Float64() + 0.5), r
			j += 2
			continue
		}
		a[j*n+j] = rnd.NormFloat64()
		j++
	}
	return a
}

// TestDgeesNetlibScaledSelection compares Dgees with Netlib when A is scaled
// outside [smlnum,bignum], where DGEES selects on unscaled eigenvalues,
// recounts sdim and re-standardizes 2×2 blocks whose off-diagonal underflows.
func TestDgeesNetlibScaledSelection(t *testing.T) {
	rnd := rand.New(rand.NewPCG(51, 51))
	type tc struct {
		name      string
		n         int
		a         []float64
		threshold float64
		scale     float64
	}
	cases := []tc{
		{"SelectOnUnscaled", 2, []float64{0.5, 1, 0, 2e138}, 0.45, 2e138},
		{"UnderflowSubdiagonal", 2, []float64{1e-300, 1e-300, -1e-330, 1e-300}, 0, 1e-300},
		{"UnderflowSuperdiagonal", 2, []float64{1e-300, -1e-330, 1e-300, 1e-300}, 0, 1e-300},
		{"UnderflowEmbedded", 4, []float64{
			2e-300, 1e-300, 3e-300, 1e-300,
			0, 1e-300, 1e-300, 2e-300,
			0, -1e-330, 1e-300, 1e-300,
			0, 0, 0, -1e-300,
		}, 0, 1e-300},
	}
	for _, n := range []int{3, 5, 8} {
		for _, c := range []float64{1e-150, 1e150, 1e-300, 1e300} {
			a := make([]float64, n*n)
			for i := range a {
				a[i] = c * rnd.NormFloat64()
			}
			cases = append(cases, tc{fmt.Sprintf("random/n=%d/c=%g", n, c), n, a, 0.3 * c, c})
		}
	}
	for _, c := range cases {
		for _, sort := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/sort=%v", c.name, sort), func(t *testing.T) {
				n := c.n
				ga := append([]float64(nil), c.a...)
				gwr, gwi := make([]float64, n), make([]float64, n)
				gvs := make([]float64, n*n)
				srt := lapack.SortNone
				if sort {
					srt = lapack.SortSelected
				}
				thr := c.threshold
				sel := func(wr, wi float64) bool { return wr < thr }
				work := make([]float64, 1)
				impl := Implementation{}
				impl.Dgees(lapack.SchurHess, srt, sel, n, nil, n, nil, nil, nil, n, work, -1, nil)
				work = make([]float64, int(work[0]))
				gsdim, ok := impl.Dgees(lapack.SchurHess, srt, sel, n, ga, n, gwr, gwi, gvs, n, work, len(work), make([]bool, n))
				na := append([]float64(nil), c.a...)
				nwr, nwi := make([]float64, n), make([]float64, n)
				nvs := make([]float64, n*n)
				nsdim, info := netlib.Dgees('V', sort, thr, n, na, nwr, nwi, nvs)
				if ok != (info == 0) || gsdim != nsdim {
					t.Fatalf("ok=%v sdim=%d, Netlib info=%d sdim=%d", ok, gsdim, info, nsdim)
				}
				for j := range n {
					checkScaledNetlib(t, fmt.Sprintf("wr[%d]", j), gwr[j], nwr[j], c.scale)
					checkScaledNetlib(t, fmt.Sprintf("wi[%d]", j), gwi[j], nwi[j], c.scale)
					if (gwi[j] == 0) != (nwi[j] == 0) {
						t.Errorf("wi[%d]=%g, want Netlib %g", j, gwi[j], nwi[j])
					}
					for i := range n {
						if (ga[i*n+j] == 0) != (na[i*n+j] == 0) {
							t.Errorf("T[%d,%d]=%g, want Netlib %g", i, j, ga[i*n+j], na[i*n+j])
						}
					}
				}
			})
		}
	}
}

// TestDtrsylNetlib compares Dtrsyl with Netlib for all transpose and sign
// combinations, near-singular systems where DLALN2 perturbs a 2×1 or 1×2
// block with the global smin, and right-hand sides near overflow.
func TestDtrsylNetlib(t *testing.T) {
	type tc struct {
		name string
		m, n int
		a, b []float64
		c    []float64
	}
	cases := []tc{
		{"Overflow", 1, 1, []float64{1}, []float64{-(1 - 1e-10)}, []float64{1e290}},
		{"GlobalSmin2x1", 3, 1, []float64{1e10, 0, 0, 0, 1, 1, 0, -1e-7, 1}, []float64{-1}, []float64{1, 2, 3}},
		{"GlobalSmin1x2", 1, 3, []float64{-1}, []float64{1e10, 0, 0, 0, 1, 1, 0, -1e-7, 1}, []float64{1, 2, 3}},
	}
	rnd := rand.New(rand.NewPCG(52, 52))
	for m := 1; m <= 6; m++ {
		for n := 1; n <= 6; n++ {
			c := make([]float64, m*n)
			for i := range c {
				c[i] = rnd.NormFloat64()
			}
			cases = append(cases, tc{fmt.Sprintf("random/m=%d/n=%d", m, n), m, n,
				randomNetlibQuasiTriangular(m, rnd), randomNetlibQuasiTriangular(n, rnd), c})
		}
	}
	for _, c := range cases {
		for _, trana := range []blas.Transpose{blas.NoTrans, blas.Trans} {
			for _, tranb := range []blas.Transpose{blas.NoTrans, blas.Trans} {
				for _, isgn := range []int{1, -1} {
					t.Run(fmt.Sprintf("%s/%c%c/isgn=%d", c.name, trana, tranb, isgn), func(t *testing.T) {
						gc := append([]float64(nil), c.c...)
						gscale, ok := Implementation{}.Dtrsyl(trana, tranb, isgn, c.m, c.n, c.a, c.m, c.b, c.n, gc, c.n)
						nc := append([]float64(nil), c.c...)
						nscale, info := netlib.Dtrsyl(byte(trana), byte(tranb), isgn, c.m, c.n,
							append([]float64(nil), c.a...), append([]float64(nil), c.b...), nc)
						if ok != (info == 0) {
							t.Fatalf("ok=%v, Netlib info=%d", ok, info)
						}
						if math.Abs(gscale-nscale) > 1e-12*nscale {
							t.Errorf("scale=%g, want Netlib %g", gscale, nscale)
						}
						var xmax float64
						for _, v := range nc {
							xmax = math.Max(xmax, math.Abs(v))
						}
						for i := range nc {
							checkScaledNetlib(t, fmt.Sprintf("X[%d]", i), gc[i], nc[i], math.Max(xmax, 1e-300))
						}
					})
				}
			}
		}
	}
}

// TestDtrsenNetlib compares Dtrsen with Netlib including the condition
// estimates s and sep.
func TestDtrsenNetlib(t *testing.T) {
	rnd := rand.New(rand.NewPCG(53, 53))
	for n := 1; n <= 8; n++ {
		for k := 0; k < 20; k++ {
			tt := randomNetlibQuasiTriangular(n, rnd)
			sel := make([]bool, n)
			for i := range sel {
				sel[i] = rnd.Float64() < 0.5
			}
			for _, ijob := range []int{0, 1, 2, 3} {
				t.Run(fmt.Sprintf("n=%d/case=%d/ijob=%d", n, k, ijob), func(t *testing.T) {
					gt := append([]float64(nil), tt...)
					gq := make([]float64, n*n)
					for i := range n {
						gq[i*n+i] = 1
					}
					nq := append([]float64(nil), gq...)
					gwr, gwi := make([]float64, n), make([]float64, n)
					work := make([]float64, 1)
					iwork := make([]int, 1)
					impl := Implementation{}
					impl.Dtrsen(ijob, true, sel, n, gt, n, gq, n, gwr, gwi, work, -1, iwork, -1)
					work = make([]float64, int(work[0]))
					iwork = make([]int, iwork[0])
					gm, gs, gsep, ok := impl.Dtrsen(ijob, true, sel, n, gt, n, gq, n, gwr, gwi, work, len(work), iwork, len(iwork))
					nt := append([]float64(nil), tt...)
					nwr, nwi := make([]float64, n), make([]float64, n)
					nm, ns, nsep, info := netlib.Dtrsen("NESB"[ijob], 'V', sel, n, nt, nq, nwr, nwi)
					if ok != (info == 0) || gm != nm {
						t.Fatalf("ok=%v m=%d, Netlib info=%d m=%d", ok, gm, info, nm)
					}
					for j := range n {
						checkCloseNetlib(t, fmt.Sprintf("wr[%d]", j), gwr[j], nwr[j])
						checkCloseNetlib(t, fmt.Sprintf("wi[%d]", j), gwi[j], nwi[j])
					}
					if ijob == 1 || ijob == 3 {
						checkScaledNetlib(t, "s", gs, ns, ns)
					}
					if ijob >= 2 {
						checkScaledNetlib(t, "sep", gsep, nsep, nsep)
					}
				})
			}
		}
	}
}
