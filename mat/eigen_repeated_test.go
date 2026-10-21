// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"math/cmplx"
	"testing"
)

func TestEigenRepeatedNearIdentity(t *testing.T) {
	for _, tc := range []struct {
		name string
		n    int
		a    []float64
	}{
		{"rotation-pairs", 4, []float64{1, 1e-17, 0, 0, -1e-17, 1, 0, 0, 0, 0, 1, 2e-17, 0, 0, -2e-17, 1}},
		{"identity-noise", 5, []float64{
			0.9999999999999999, 3.2161442547704e-17, 2.726644737388092e-17, 2.112518804393819e-17, -2.2722085473535285e-17,
			4.6501897102499e-17, 1, 1.1711771936748702e-18, -1.0118307026710267e-17, 6.339318431175315e-18,
			-5.094585623278755e-18, -3.691018870888581e-18, 1, -7.33488207544749e-17, -6.843465332182895e-18,
			-1.155593995319046e-16, 3.191706726511874e-17, 1.9080943963414913e-16, 1, -1.6353874587388973e-16,
			-9.972624186255242e-18, 3.614619856335848e-18, 4.6318881558612997e-17, 3.510022697888328e-17, 1,
		}},
	} {
		for _, kind := range []EigenKind{EigenRight, EigenLeft} {
			a := NewDense(tc.n, tc.n, tc.a)
			var e Eigen
			func() {
				defer func() {
					if r := recover(); r != nil {
						t.Fatalf("%s kind=%v: panic: %v", tc.name, kind, r)
					}
				}()
				if !e.Factorize(a, kind) {
					t.Fatalf("%s kind=%v: factorization failed", tc.name, kind)
				}
			}()
			vals := e.Values(nil)
			for _, v := range vals {
				if cmplx.Abs(v-1) > 1e-14 {
					t.Errorf("%s: eigenvalue %v, want 1", tc.name, v)
				}
			}
			var v CDense
			if kind == EigenRight {
				e.VectorsTo(&v)
			} else {
				e.LeftVectorsTo(&v)
			}
			for j := 0; j < tc.n; j++ {
				var norm, res float64
				for i := 0; i < tc.n; i++ {
					norm += cmplx.Abs(v.At(i, j))
					var s complex128
					for k := 0; k < tc.n; k++ {
						if kind == EigenRight {
							s += complex(a.At(i, k), 0) * v.At(k, j)
						} else {
							s += complex(a.At(k, i), 0) * v.At(k, j)
						}
					}
					if kind == EigenRight {
						s -= vals[j] * v.At(i, j)
					} else {
						s -= cmplx.Conj(vals[j]) * v.At(i, j)
					}
					res += cmplx.Abs(s)
				}
				if norm == 0 || res > 1e-13*norm {
					t.Errorf("%s kind=%v: vector %d residual %g, norm %g", tc.name, kind, j, res, norm)
				}
			}
		}
	}
}
