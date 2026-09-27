// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package gonum

import (
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/lapack/gonum/internal/netlib"
)

func TestDlag2NetlibDifferential(t *testing.T) {
	for _, tc := range []struct {
		name                  string
		a, b                  [4]float64
		doubleInfiniteRegular bool
	}{
		{"Real", [4]float64{2, 1, 0, 3}, [4]float64{1, 0.5, 0, 1}, false},
		{"Complex", [4]float64{1, 2, -2, 1}, [4]float64{2, 1, 0, 3}, false},
		{"SingularB11", [4]float64{2, -1, 3, 4}, [4]float64{0, 0.5, 0, 3}, false},
		{"DoubleInfiniteRegular", [4]float64{2, -1, 3, 4}, [4]float64{0, 2, 0, 3}, true},
		{"SingularB22", [4]float64{2, -1, 3, 4}, [4]float64{3, 2, 0, 0}, false},
		{"LargeA", [4]float64{2e200, -3e200, 4e200, 5e200}, [4]float64{2, -1, 0, 3}, false},
		{"SmallB", [4]float64{2, -3, 4, 5}, [4]float64{2e-200, -1e-200, 0, 3e-200}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			gs1, gs2, gwr1, gwr2, gwi := Implementation{}.Dlag2(tc.a[:], 2, tc.b[:], 2)
			na := []float64{tc.a[0], tc.a[2], tc.a[1], tc.a[3]}
			nb := []float64{tc.b[0], tc.b[2], tc.b[1], tc.b[3]}
			ns1, ns2, nwr1, nwr2, nwi := netlib.Dlag2(na, 2, nb, 2, math.Ldexp(1, -1022))
			got := [2][3]float64{{gwr1, gwi, gs1}, {gwr2, -gwi, gs2}}
			want := [2][3]float64{{nwr1, nwi, ns1}, {nwr2, -nwi, ns2}}
			for i := range got {
				if !finite3(got[i][0], got[i][1], got[i][2]) || !finite3(want[i][0], want[i][1], want[i][2]) ||
					zeroTriple(got[i][0], got[i][1], got[i][2]) || zeroTriple(want[i][0], want[i][1], want[i][2]) {
					t.Fatalf("eigenvalue %d invalid: Gonum=%v Netlib=%v", i, got[i], want[i])
				}
				if tc.doubleInfiniteRegular {
					if resid := dlag2HomogeneousResidual(tc.a, tc.b, got[i]); !(resid <= 2e-14) {
						t.Fatalf("eigenvalue %d residual=%g for double-infinite regular pencil", i, resid)
					}
					continue
				}
				if d := eigenDistance(got[i][0], got[i][1], got[i][2], want[i][0], want[i][1], want[i][2]); !(d <= 2e-14) {
					t.Fatalf("eigenvalue %d mismatch: distance=%g Gonum=%v Netlib=%v", i, d, got[i], want[i])
				}
			}
		})
	}
}

func dlag2HomogeneousResidual(a, b [4]float64, eig [3]float64) float64 {
	matrixScale := math.Max(maxAbs4(a), maxAbs4(b))
	eigenScale := math.Max(math.Hypot(eig[0], eig[1]), math.Abs(eig[2]))
	s := eig[2] / eigenScale
	w := complex(eig[0]/eigenScale, eig[1]/eigenScale)
	c11 := complex(s*a[0]/matrixScale, 0) - w*complex(b[0]/matrixScale, 0)
	c12 := complex(s*a[1]/matrixScale, 0) - w*complex(b[1]/matrixScale, 0)
	c21 := complex(s*a[2]/matrixScale, 0)
	c22 := complex(s*a[3]/matrixScale, 0) - w*complex(b[3]/matrixScale, 0)
	norm := math.Max(cmplxAbs(c11)+cmplxAbs(c21), cmplxAbs(c12)+cmplxAbs(c22))
	if norm == 0 {
		return 0
	}
	return cmplxAbs(c11*c22-c12*c21) / (norm * norm)
}
