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

// dlagv2NetlibPencils returns random and degenerate 2×2 pencils with upper
// triangular B covering the real, complex and |A21|, |B11|, |B22| <= ulp
// branches of DLAGV2 and the scaling branches of DLAG2.
func dlagv2NetlibPencils() [][2][]float64 {
	rnd := rand.New(rand.NewPCG(61, 61))
	var cases [][2][]float64
	for k := 0; k < 400; k++ {
		a := []float64{rnd.NormFloat64(), rnd.NormFloat64(), rnd.NormFloat64(), rnd.NormFloat64()}
		b := []float64{rnd.NormFloat64(), rnd.NormFloat64(), 0, rnd.NormFloat64()}
		switch k % 8 {
		case 1:
			a[2] = 1e-20
		case 2:
			b[0] = 1e-20
		case 3:
			b[3] = 1e-20
		case 4:
			a[1], a[2] = math.Abs(a[1])+0.5, -(math.Abs(a[2]) + 0.5)
			a[3] = a[0]
		case 5:
			s := math.Pow(10, float64(rnd.IntN(600)-300))
			for i := range a {
				a[i] *= s
			}
		case 6:
			s := math.Pow(10, float64(rnd.IntN(600)-300))
			for i := range b {
				b[i] *= s
			}
		case 7:
			b[0], b[3] = 0, 0
		}
		cases = append(cases, [2][]float64{a, b})
	}
	return cases
}

func TestDlagv2Netlib(t *testing.T) {
	for k, c := range dlagv2NetlibPencils() {
		t.Run(fmt.Sprintf("case=%d", k), func(t *testing.T) {
			ga, gb := append([]float64(nil), c[0]...), append([]float64(nil), c[1]...)
			na, nb := append([]float64(nil), c[0]...), append([]float64(nil), c[1]...)
			csq, snq, _, _, csz, snz, s1, s2, ar0, ar1, ai0, ai1 := Implementation{}.Dlagv2(ga, 2, gb, 2)
			nar, nai, nbeta, csl, snl, csr, snr := netlib.Dlagv2(na, nb)
			anorm := math.Max(math.Abs(c[0][0])+math.Abs(c[0][2]), math.Abs(c[0][1])+math.Abs(c[0][3]))
			bnorm := math.Max(math.Abs(c[1][0]), math.Abs(c[1][1])+math.Abs(c[1][3]))
			for i := range 4 {
				checkScaledNetlib(t, fmt.Sprintf("A[%d]", i), ga[i], na[i], math.Max(anorm, 1e-300))
				checkScaledNetlib(t, fmt.Sprintf("B[%d]", i), gb[i], nb[i], math.Max(bnorm, 1e-300))
			}
			for _, v := range []struct {
				name      string
				got, want float64
			}{{"csl", csq, csl}, {"snl", snq, snl}, {"csr", csz, csr}, {"snr", snz, snr}} {
				checkCloseNetlib(t, v.name, v.got, v.want)
			}
			ascale := math.Max(anorm, math.Max(math.Abs(nar[0]), math.Abs(nai[0])))
			checkScaledNetlib(t, "alphar[0]", ar0, nar[0], ascale)
			checkScaledNetlib(t, "alphar[1]", ar1, nar[1], ascale)
			checkScaledNetlib(t, "alphai[0]", ai0, nai[0], ascale)
			checkScaledNetlib(t, "alphai[1]", ai1, nai[1], ascale)
			checkScaledNetlib(t, "beta[0]", s1, nbeta[0], math.Max(bnorm, 1e-300))
			checkScaledNetlib(t, "beta[1]", s2, nbeta[1], math.Max(bnorm, 1e-300))
		})
	}
}

func TestDlag2Netlib(t *testing.T) {
	for k, c := range dlagv2NetlibPencils() {
		for _, safmin := range []float64{dlamchS, 100 * dlamchS} {
			gs1, gs2, gw1, gw2, gwi := Implementation{}.dlag2(c[0], 2, c[1], 2, safmin)
			ns1, ns2, nw1, nw2, nwi := netlib.Dlag2(c[0], c[1], safmin)
			name := fmt.Sprintf("case=%d safmin=%g", k, safmin)
			for _, v := range []struct {
				label     string
				got, want float64
			}{{"scale1", gs1, ns1}, {"scale2", gs2, ns2}, {"wr1", gw1, nw1}, {"wr2", gw2, nw2}, {"wi", gwi, nwi}} {
				if math.Abs(v.got-v.want) > 1e-12*math.Max(math.Abs(v.want), 1e-300) && !(math.IsNaN(v.got) && math.IsNaN(v.want)) {
					t.Errorf("%s: %s=%v, want Netlib %v", name, v.label, v.got, v.want)
				}
			}
		}
	}
}
