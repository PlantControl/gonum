// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package gonum

import (
	"fmt"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/lapack/gonum/internal/netlib"
)

func TestDlagv2NetlibDifferential(t *testing.T) {
	for _, tc := range []struct {
		name                  string
		a, b                  [4]float64
		doubleInfiniteRegular bool
	}{
		{"DeflatedA21", [4]float64{2, 1, 0, 3}, [4]float64{1, 0.5, 0, 1}, false},
		{"SingularB11", [4]float64{2, -1, 3, 4}, [4]float64{0, 0.5, 0, 3}, false},
		{"DoubleInfiniteRegular", [4]float64{2, -1, 3, 4}, [4]float64{0, 2, 0, 3}, true},
		{"SingularB22", [4]float64{2, -1, 3, 4}, [4]float64{3, 2, 0, 0}, false},
		{"Complex", [4]float64{1, 2, -2, 1}, [4]float64{2, 1, 0, 3}, false},
		{"LargeA", [4]float64{2e200, -3e200, 4e200, 5e200}, [4]float64{2, -1, 0, 3}, false},
		{"SmallA", [4]float64{2e-200, -3e-200, 4e-200, 5e-200}, [4]float64{2, -1, 0, 3}, false},
		{"LargeB", [4]float64{2, -3, 4, 5}, [4]float64{2e200, -1e200, 0, 3e200}, false},
		{"SmallB", [4]float64{2, -3, 4, 5}, [4]float64{2e-200, -1e-200, 0, 3e-200}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ga := append([]float64(nil), tc.a[:]...)
			gb := append([]float64(nil), tc.b[:]...)
			gcsq, gsnq, _, _, gcsz, gsnz, gs1, gs2, gar0, gar1, gai0, gai1 := Implementation{}.Dlagv2(ga, 2, gb, 2)
			na := []float64{tc.a[0], tc.a[2], tc.a[1], tc.a[3]}
			nb := []float64{tc.b[0], tc.b[2], tc.b[1], tc.b[3]}
			nar, nai, nscale, ncsl, nsnl, ncsr, nsnr := netlib.Dlagv2(na, 2, nb, 2)
			for name, pair := range map[string][2]float64{
				"csl": {gcsq, ncsl}, "snl": {gsnq, nsnl},
				"csr": {gcsz, ncsr}, "snr": {gsnz, nsnr},
			} {
				checkDlagv2NativeValue(t, name, pair[0], pair[1], 1)
			}
			for i, got := range []struct {
				ar, ai, scale float64
			}{{gar0, gai0, gs1}, {gar1, gai1, gs2}} {
				want := struct{ ar, ai, scale float64 }{nar[i], nai[i], nscale[i]}
				if !finite3(got.ar, got.ai, got.scale) || !finite3(want.ar, want.ai, want.scale) ||
					zeroTriple(got.ar, got.ai, got.scale) || zeroTriple(want.ar, want.ai, want.scale) {
					t.Fatalf("eigenvalue %d nonfinite: Gonum=%v Netlib=%v", i, got, want)
				}
				// A rank-one B with a constant nonzero determinant polynomial has two
				// infinite roots. Tiny roundoff in beta makes their finite proxies ill-conditioned.
				if !tc.doubleInfiniteRegular {
					if d := eigenDistance(got.ar, got.ai, got.scale, want.ar, want.ai, want.scale); !(d <= 2e-14) {
						t.Fatalf("eigenvalue %d mismatch: Gonum=(%g,%g)/%g Netlib=(%g,%g)/%g",
							i, got.ar, got.ai, got.scale, want.ar, want.ai, want.scale)
					}
				}
			}
			aScale := maxAbs4(tc.a)
			bScale := maxAbs4(tc.b)
			for i, got := range ga {
				checkDlagv2NativeValue(t, fmt.Sprintf("A[%d]", i), got, na[i%2*2+i/2], aScale)
			}
			for i, got := range gb {
				checkDlagv2NativeValue(t, fmt.Sprintf("B[%d]", i), got, nb[i%2*2+i/2], bScale)
			}
		})
	}
}

func finite3(x, y, z float64) bool {
	return !math.IsNaN(x) && !math.IsInf(x, 0) && !math.IsNaN(y) && !math.IsInf(y, 0) && !math.IsNaN(z) && !math.IsInf(z, 0)
}

func zeroTriple(x, y, z float64) bool {
	return x == 0 && y == 0 && z == 0
}

func maxAbs4(a [4]float64) float64 {
	return math.Max(math.Max(math.Abs(a[0]), math.Abs(a[1])), math.Max(math.Abs(a[2]), math.Abs(a[3])))
}

func checkDlagv2NativeValue(t *testing.T, name string, got, want, scale float64) {
	t.Helper()
	if math.IsNaN(got) || math.IsInf(got, 0) || math.IsNaN(want) || math.IsInf(want, 0) {
		t.Fatalf("%s nonfinite: Gonum=%g Netlib=%g", name, got, want)
	}
	if math.Abs(got-want) > 5e-14*scale {
		t.Fatalf("%s mismatch: Gonum=%g Netlib=%g", name, got, want)
	}
}
