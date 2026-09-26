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
