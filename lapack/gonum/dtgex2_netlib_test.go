// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package gonum

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/lapack/gonum/internal/netlib"
)

func TestDtgex2NetlibDifferential(t *testing.T) {
	rnd := rand.New(rand.NewPCG(2, 2))
	for _, blocks := range [][2]int{{1, 1}, {1, 2}, {2, 1}, {2, 2}} {
		n1, n2 := blocks[0], blocks[1]
		t.Run(string(rune('0'+n1))+"x"+string(rune('0'+n2)), func(t *testing.T) {
			for k := 0; k < 100; k++ {
				n := n1 + n2
				a := make([]float64, n*n)
				b := make([]float64, n*n)
				fillSchurBlock(a, b, n, 0, n1, rnd)
				fillSchurBlock(a, b, n, n1, n2, rnd)
				for i := 0; i < n1; i++ {
					for j := n1; j < n; j++ {
						a[i*n+j] = rnd.NormFloat64()
						b[i*n+j] = rnd.NormFloat64()
					}
				}
				ga, gb := append([]float64(nil), a...), append([]float64(nil), b...)
				na, nb := append([]float64(nil), a...), append([]float64(nil), b...)
				q, z := identityData(n), identityData(n)
				work := make([]float64, max(n*n, 2*n*n))
				gok := Implementation{}.Dtgex2(true, true, n, ga, n, gb, n, q, n, z, n, 0, n1, n2, work, len(work))
				info := netlib.Dtgex2(n, na, nb, 0, n1, n2)
				if gok != (info == 0) {
					t.Fatalf("case %d: acceptance mismatch: Gonum=%v Netlib info=%d", k, gok, int(info))
				}
				if gok {
					gar, gai, gbet := schurBlockEigenvalues(ga, gb, n, 0, n2)
					nar, nai, nbet := schurBlockEigenvalues(na, nb, n, 0, n2)
					compareGeneralizedEigenvalues(t, gar, gai, gbet, nar, nai, nbet)
					gar, gai, gbet = schurBlockEigenvalues(ga, gb, n, n2, n1)
					nar, nai, nbet = schurBlockEigenvalues(na, nb, n, n2, n1)
					compareGeneralizedEigenvalues(t, gar, gai, gbet, nar, nai, nbet)
				}
			}
		})
	}
}

func TestDtgex2NetlibRejectedSwap(t *testing.T) {
	const n = 4
	a := []float64{
		-2.1908605110042063, 0.5720105957485505, -0.12695320692010448, 0.11807691769627293,
		-1.0523887992314855, -2.1908605110042063, -0.060129294742167196, -0.0023051154401202423,
		0, 0, -2.1908605110042063, 0.5720105957485505,
		0, 0, -1.0523887992314855, -2.1908605110042063,
	}
	b := []float64{
		1, 0, 0.004174493657635736, -0.14712312362881694,
		0, 1, 0.050958523233562585, 0.017831076666835567,
		0, 0, 1, 0,
		0, 0, 0, 1,
	}
	ga, gb := append([]float64(nil), a...), append([]float64(nil), b...)
	na, nb := append([]float64(nil), a...), append([]float64(nil), b...)
	q, z := identityData(n), identityData(n)
	work := make([]float64, 2*n*n)
	gok := (Implementation{}).Dtgex2(true, true, n, ga, n, gb, n, q, n, z, n, 0, 2, 2, work, len(work))
	info := netlib.Dtgex2(n, na, nb, 0, 2, 2)
	if gok != (info == 0) {
		t.Fatalf("acceptance mismatch: Gonum=%v Netlib info=%d", gok, info)
	}
	if gok {
		t.Fatal("rejection fixture was accepted")
	}
	identity := identityData(n)
	for i := range a {
		if ga[i] != a[i] || gb[i] != b[i] || na[i] != a[i] || nb[i] != b[i] || q[i] != identity[i] || z[i] != identity[i] {
			t.Fatalf("rejected swap modified outputs at %d", i)
		}
	}
}

func TestDtgex2NetlibSeparatedScales(t *testing.T) {
	rnd := rand.New(rand.NewPCG(3, 3))
	for _, blocks := range [][2]int{{1, 1}, {1, 2}, {2, 1}, {2, 2}} {
		n1, n2 := blocks[0], blocks[1]
		for _, scales := range [][2]float64{{1e-300, 1}, {1, 1e-300}, {1e300, 1}, {1, 1e300}} {
			for k := 0; k < 2000; k++ {
				n := n1 + n2
				a := make([]float64, n*n)
				b := make([]float64, n*n)
				fillSchurBlock(a, b, n, 0, n1, rnd)
				fillSchurBlock(a, b, n, n1, n2, rnd)
				for i := 0; i < n1; i++ {
					for j := n1; j < n; j++ {
						a[i*n+j] = rnd.NormFloat64()
						b[i*n+j] = rnd.NormFloat64()
					}
				}
				for i := range a {
					a[i] *= scales[0]
					b[i] *= scales[1]
				}
				ga, gb := append([]float64(nil), a...), append([]float64(nil), b...)
				na, nb := append([]float64(nil), a...), append([]float64(nil), b...)
				q, z := identityData(n), identityData(n)
				work := make([]float64, max(n*n, 2*n*n))
				gok := Implementation{}.Dtgex2(true, true, n, ga, n, gb, n, q, n, z, n, 0, n1, n2, work, len(work))
				info := netlib.Dtgex2(n, na, nb, 0, n1, n2)
				if gok != (info == 0) {
					t.Fatalf("blocks=%dx%d scales=(%g,%g) case=%d: acceptance mismatch: Gonum=%v Netlib info=%d",
						n1, n2, scales[0], scales[1], k, gok, info)
				}
			}
		}
	}
}

func TestDtgex2NetlibEmbeddedBlocks(t *testing.T) {
	rnd := rand.New(rand.NewPCG(5, 5))
	for _, blocks := range [][2]int{{1, 1}, {1, 2}, {2, 1}, {2, 2}} {
		n1, n2 := blocks[0], blocks[1]
		t.Run(fmt.Sprintf("%dx%d", n1, n2), func(t *testing.T) {
			for k := 0; k < 50; k++ {
				n := n1 + n2 + 2
				a, b := make([]float64, n*n), make([]float64, n*n)
				fillSchurBlock(a, b, n, 0, 1, rnd)
				fillSchurBlock(a, b, n, 1, n1, rnd)
				fillSchurBlock(a, b, n, 1+n1, n2, rnd)
				fillSchurBlock(a, b, n, n-1, 1, rnd)
				for i := 0; i < n; i++ {
					for j := i + 1; j < n; j++ {
						inFirst := i >= 1 && j < 1+n1
						inSecond := i >= 1+n1 && j < 1+n1+n2
						if inFirst || inSecond {
							continue
						}
						a[i*n+j] = rnd.NormFloat64()
						b[i*n+j] = rnd.NormFloat64()
					}
				}
				ga, gb := append([]float64(nil), a...), append([]float64(nil), b...)
				na, nb := append([]float64(nil), a...), append([]float64(nil), b...)
				q, z := identityData(n), identityData(n)
				work := make([]float64, max(n*(n1+n2), 2*(n1+n2)*(n1+n2)))
				gok := Implementation{}.Dtgex2(true, true, n, ga, n, gb, n, q, n, z, n, 1, n1, n2, work, len(work))
				info := netlib.Dtgex2(n, na, nb, 1, n1, n2)
				if gok != (info == 0) {
					t.Fatalf("case %d: acceptance mismatch: Gonum=%v Netlib info=%d", k, gok, info)
				}
				if gok {
					checkGeneralizedSchurResult(t, "Gonum embedded", a, b, ga, gb, q, z, n)
					checkGeneralizedSchurStructure(t, "Netlib embedded", na, nb, n)
				}
			}
		})
	}
}

// TestDtgex2NetlibBorderlineAcceptance pins the 2×2/2×2 swap from the
// Riccati pencil in gonum ergo ZHGFWA. Netlib accepts this exact input with
// BQRA21 at 0.98 of THRESHA, so acceptance is decided by rounding; Gonum and
// Netlib must accept relative perturbations at the same rate.
func TestDtgex2NetlibBorderlineAcceptance(t *testing.T) {
	const n = 4
	s0 := []float64{
		-1.24393017849882703e+01, -1.15957338125495760e-01, -1.33648429888031549e-19, -4.84484799880933280e-19,
		9.28837014591931786e-05, -3.79046077665266541e-04, -2.32263680132544724e-14, 2.40403471231674990e-14,
		0, 0, -9.92128366985719157e-01, 1.25225011168894979e-01,
		0, 0, -1.25225011168895006e-01, -9.92128366985719379e-01,
	}
	t0 := []float64{
		9.99999999999999889e-01, 0, -9.03918484410455235e-16, -9.65194924956767279e-17,
		0, 3.09749244897807347e-05, -6.06970289936176144e+01, -5.74151430673537675e+01,
		0, 0, 1.38810358328641961e+01, 0,
		0, 0, 0, 1.09912427393938792e+01,
	}
	rnd := rand.New(rand.NewPCG(7, 9))
	impl := Implementation{}
	work := make([]float64, 2*n*n)
	const trials = 4000
	for _, rel := range []float64{1e-15, 1e-13} {
		var gonumOK, netlibOK int
		for range trials {
			a, b := append([]float64(nil), s0...), append([]float64(nil), t0...)
			for i := range a {
				a[i] *= 1 + rel*rnd.NormFloat64()
				b[i] *= 1 + rel*rnd.NormFloat64()
			}
			ga, gb := append([]float64(nil), a...), append([]float64(nil), b...)
			gok := impl.Dtgex2(false, false, n, ga, n, gb, n, nil, 1, nil, 1, 0, 2, 2, work, len(work))
			nok := netlib.Dtgex2(n, a, b, 0, 2, 2) == 0
			if gok {
				gonumOK++
			}
			if nok {
				netlibOK++
			}
			if gok && nok {
				for _, off := range []int{0, 2} {
					gar, gai, gbet := schurBlockEigenvalues(ga, gb, n, off, 2)
					nar, nai, nbet := schurBlockEigenvalues(a, b, n, off, 2)
					compareGeneralizedEigenvalues(t, gar, gai, gbet, nar, nai, nbet)
				}
			}
		}
		if netlibOK == 0 || netlibOK == trials {
			t.Fatalf("rel=%g: Netlib accepted %d of %d, fixture not borderline", rel, netlibOK, trials)
		}
		if d := gonumOK - netlibOK; d > trials/20 || -d > trials/20 {
			t.Errorf("rel=%g: Gonum accepted %d, Netlib %d of %d", rel, gonumOK, netlibOK, trials)
		}
	}
}
