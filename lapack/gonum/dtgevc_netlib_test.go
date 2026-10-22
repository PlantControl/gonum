// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package gonum

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/lapack/gonum/internal/netlib"
)

func randomNetlibSchurPair(n int, rnd *rand.Rand) (s, p []float64) {
	s = make([]float64, n*n)
	p = make([]float64, n*n)
	for i := range n {
		for j := i + 1; j < n; j++ {
			s[i*n+j] = rnd.NormFloat64()
			p[i*n+j] = rnd.NormFloat64()
		}
	}
	for j := 0; j < n; {
		size := 1
		if j < n-1 && rnd.Float64() < 0.5 {
			size = 2
		}
		fillSchurBlock(s, p, n, j, size, rnd)
		if size == 2 {
			p[j*n+j+1] = 0
		}
		j += size
	}
	return s, p
}

func randomNetlibOrthogonal(n int, rnd *rand.Rand) []float64 {
	q := make([]float64, n*n)
	for i := range q {
		q[i] = rnd.NormFloat64()
	}
	tau := make([]float64, n)
	work := make([]float64, 64*n)
	impl := Implementation{}
	impl.Dgeqrf(n, n, q, n, tau, work, len(work))
	impl.Dorgqr(n, n, n, q, n, tau, work, len(work))
	return q
}

func TestDtgevcNetlibDifferential(t *testing.T) {
	rnd := rand.New(rand.NewPCG(21, 21))
	for n := 1; n <= 8; n++ {
		for k := 0; k < 20; k++ {
			s, p := randomNetlibSchurPair(n, rnd)
			sel := make([]bool, n)
			for i := range sel {
				sel[i] = rnd.Float64() < 0.5
			}
			q := randomNetlibOrthogonal(n, rnd)
			z := randomNetlibOrthogonal(n, rnd)
			for _, c := range []struct {
				howmny lapack.EVHowMany
				name   byte
			}{
				{lapack.EVAll, 'A'},
				{lapack.EVSelected, 'S'},
				{lapack.EVAllMulQ, 'B'},
			} {
				t.Run(fmt.Sprintf("n=%d/case=%d/howmny=%c", n, k, c.name), func(t *testing.T) {
					gvl, gvr := make([]float64, n*n), make([]float64, n*n)
					if c.howmny == lapack.EVAllMulQ {
						copy(gvl, q)
						copy(gvr, z)
					}
					nvl, nvr := append([]float64(nil), gvl...), append([]float64(nil), gvr...)
					gm, ok := Implementation{}.Dtgevc(lapack.EVBoth, c.howmny, sel, n,
						s, n, p, n, gvl, n, gvr, n, n, make([]float64, 6*n))
					nm, info := netlib.Dtgevc('B', c.name, sel, n,
						append([]float64(nil), s...), append([]float64(nil), p...), nvl, nvr, n)
					if !ok || info != 0 {
						t.Fatalf("ok=%v, Netlib info=%d", ok, info)
					}
					if gm != nm {
						t.Fatalf("m=%d, want Netlib %d", gm, nm)
					}
					for i := 0; i < n; i++ {
						for j := 0; j < gm; j++ {
							checkCloseNetlib(t, fmt.Sprintf("VL[%d,%d]", i, j), gvl[i*n+j], nvl[i*n+j])
							checkCloseNetlib(t, fmt.Sprintf("VR[%d,%d]", i, j), gvr[i*n+j], nvr[i*n+j])
						}
					}
				})
			}
		}
	}
}

func TestDggevNetlibDifferential(t *testing.T) {
	rnd := rand.New(rand.NewPCG(22, 22))
	for n := 1; n <= 8; n++ {
		for k := 0; k < 20; k++ {
			a := make([]float64, n*n)
			b := make([]float64, n*n)
			for i := range a {
				a[i] = rnd.NormFloat64()
			}
			if k%2 == 0 {
				for i := range b {
					b[i] = rnd.NormFloat64()
				}
			} else {
				for i := range b {
					b[i] = 0.1 * rnd.NormFloat64()
				}
				for i := range n {
					b[i*n+i] += 1
				}
			}
			t.Run(fmt.Sprintf("n=%d/case=%d", n, k), func(t *testing.T) {
				ga, gb := append([]float64(nil), a...), append([]float64(nil), b...)
				na, nb := append([]float64(nil), a...), append([]float64(nil), b...)
				gar, gai, gbeta := make([]float64, n), make([]float64, n), make([]float64, n)
				nar, nai, nbeta := make([]float64, n), make([]float64, n), make([]float64, n)
				gvl, gvr := make([]float64, n*n), make([]float64, n*n)
				nvl, nvr := make([]float64, n*n), make([]float64, n*n)
				work := make([]float64, 1)
				impl := Implementation{}
				impl.Dggev(lapack.LeftEVCompute, lapack.RightEVCompute, n, ga, n, gb, n,
					gar, gai, gbeta, gvl, n, gvr, n, work, -1)
				work = make([]float64, int(work[0]))
				if !impl.Dggev(lapack.LeftEVCompute, lapack.RightEVCompute, n, ga, n, gb, n,
					gar, gai, gbeta, gvl, n, gvr, n, work, len(work)) {
					t.Fatal("Dggev did not converge")
				}
				if info := netlib.Dggev('V', 'V', n, na, nb, nar, nai, nbeta, nvl, nvr); info != 0 {
					t.Fatalf("Netlib info=%d", info)
				}
				for j := range n {
					checkCloseNetlib(t, fmt.Sprintf("alphar[%d]", j), gar[j], nar[j])
					checkCloseNetlib(t, fmt.Sprintf("alphai[%d]", j), gai[j], nai[j])
					checkCloseNetlib(t, fmt.Sprintf("beta[%d]", j), gbeta[j], nbeta[j])
				}
				if t.Failed() {
					return
				}
				// Eigenvectors agree up to a real sign per eigenvalue, since the
				// signs of Schur vectors are unspecified.
				for _, v := range []struct {
					name string
					g, w []float64
				}{{"VL", gvl, nvl}, {"VR", gvr, nvr}} {
					for j := 0; j < n; j++ {
						nw := 1
						if gai[j] > 0 {
							nw = 2
						}
						var dot float64
						for i := range n {
							for c := j; c < j+nw; c++ {
								dot += v.g[i*n+c] * v.w[i*n+c]
							}
						}
						sign := 1.0
						if dot < 0 {
							sign = -1
						}
						for i := range n {
							for c := j; c < j+nw; c++ {
								checkCloseNetlib(t, fmt.Sprintf("%s[%d,%d]", v.name, i, c), sign*v.g[i*n+c], v.w[i*n+c])
							}
						}
						j += nw - 1
					}
				}
			})
		}
	}
}
