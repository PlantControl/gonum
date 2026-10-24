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

	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/lapack/gonum/internal/netlib"
)

func checkRelNetlib(t *testing.T, name string, got, want, scale float64) {
	t.Helper()
	if math.Abs(got-want) > 1e-10*scale {
		t.Errorf("%s=%g, want Netlib %g (scale %g)", name, got, want, scale)
	}
}

func runDggevPair(t *testing.T, wantv bool, n int, a, b []float64) (gok bool, info int, g, w [3][]float64) {
	t.Helper()
	jobvl, jobvr := lapack.LeftEVNone, lapack.RightEVNone
	job := byte('N')
	if wantv {
		jobvl, jobvr = lapack.LeftEVCompute, lapack.RightEVCompute
		job = 'V'
	}
	for i := range 3 {
		g[i], w[i] = make([]float64, n), make([]float64, n)
	}
	ga, gb := append([]float64(nil), a...), append([]float64(nil), b...)
	work := make([]float64, 1)
	impl := Implementation{}
	impl.Dggev(jobvl, jobvr, n, nil, n, nil, n, nil, nil, nil, nil, n, nil, n, work, -1)
	work = make([]float64, int(work[0]))
	gok = impl.Dggev(jobvl, jobvr, n, ga, n, gb, n, g[0], g[1], g[2],
		make([]float64, n*n), n, make([]float64, n*n), n, work, len(work))
	info = netlib.DggevWork(job, job, n, append([]float64(nil), a...), append([]float64(nil), b...),
		w[0], w[1], w[2], make([]float64, n*n), make([]float64, n*n))
	return gok, info, g, w
}

// TestDggevNetlibScaledPencils compares alphar, alphai and beta individually
// with Netlib when A or B is scaled outside [smlnum,bignum], which exercises
// DGGEV's DLASCL unscaling of each array.
func TestDggevNetlibScaledPencils(t *testing.T) {
	rnd := rand.New(rand.NewPCG(23, 23))
	for _, n := range []int{1, 2, 3, 5, 8} {
		for _, sc := range [][2]float64{
			{1e-200, 1}, {1e200, 1}, {1, 1e-200}, {1, 1e200}, {1e-200, 1e200}, {1e200, 1e-200}, {1e-250, 1e-250},
		} {
			for _, wantv := range []bool{false, true} {
				a, b := make([]float64, n*n), make([]float64, n*n)
				for i := range a {
					a[i] = sc[0] * rnd.NormFloat64()
					b[i] = sc[1] * rnd.NormFloat64()
				}
				if n > 2 {
					for i := range n {
						b[i*n+n-1] = 0
					}
				}
				t.Run(fmt.Sprintf("n=%d/ca=%g/cb=%g/v=%v", n, sc[0], sc[1], wantv), func(t *testing.T) {
					gok, info, g, w := runDggevPair(t, wantv, n, a, b)
					if !gok || info != 0 {
						t.Fatalf("Gonum ok=%v, Netlib info=%d", gok, info)
					}
					for j := range n {
						checkRelNetlib(t, fmt.Sprintf("alphar[%d]", j), g[0][j], w[0][j], sc[0])
						checkRelNetlib(t, fmt.Sprintf("alphai[%d]", j), g[1][j], w[1][j], sc[0])
						checkRelNetlib(t, fmt.Sprintf("beta[%d]", j), g[2][j], w[2][j], sc[1])
					}
				})
			}
		}
	}
}

// TestDggevNetlibFailureUnscales checks that after a QZ failure the
// eigenvalues QZ isolated are unscaled exactly as DGGEV's GO TO 110 does.
func TestDggevNetlibFailureUnscales(t *testing.T) {
	const n = 4
	pencil := func(v, c float64) []float64 {
		m := []float64{2, 1, 1, 1, 1, v, 2, 1, 1, 2, 3, 1, 0, 0, 0, 0.5}
		for i := range m {
			m[i] *= c
		}
		return m
	}
	nan := math.NaN()
	for _, tc := range []struct {
		name string
		a, b []float64
	}{
		{"NaNA/hugeB", pencil(nan, 1), pencil(1, 1e300)},
		{"NaNA/tinyB", pencil(nan, 1), pencil(1, 1e-300)},
		{"hugeA/NaNB", pencil(1, 1e300), pencil(nan, 1)},
		{"tinyA/NaNB", pencil(1, 1e-300), pencil(nan, 1)},
	} {
		for _, wantv := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/v=%v", tc.name, wantv), func(t *testing.T) {
				gok, info, g, w := runDggevPair(t, wantv, n, tc.a, tc.b)
				if gok || info <= 0 || info > n {
					t.Fatalf("Gonum ok=%v, Netlib info=%d, want QZ failure", gok, info)
				}
				for j := info; j < n; j++ {
					for k, name := range []string{"alphar", "alphai", "beta"} {
						if g[k][j] != w[k][j] {
							t.Errorf("%s[%d]=%g, want Netlib %g", name, j, g[k][j], w[k][j])
						}
					}
				}
			})
		}
	}
}
