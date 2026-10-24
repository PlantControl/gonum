// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package gonum

import (
	"math"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/lapack/gonum/internal/netlib"
	"plantcontrol.org/v1/gonum/lapack/testlapack"
)

// TestNormNetlib compares the D and Z matrix norm routines with Netlib on
// matrices with Inf and NaN in every pair of referenced elements, and on
// finite random matrices.
func TestNormNetlib(t *testing.T) {
	rnd := rand.New(rand.NewPCG(91, 91))
	for _, c := range testlapack.NormCases(Implementation{}) {
		mats, _ := testlapack.NormPlacements(c)
		for range 5 {
			a := make([]float64, c.M*c.N)
			for i := range a {
				a[i] = rnd.NormFloat64()
			}
			mats = append(mats, a)
		}
		uplo, diag := byte(c.Uplo), byte(c.Diag)
		if uplo == 0 {
			uplo = 'U'
		}
		if diag == 0 {
			diag = 'N'
		}
		for _, a := range mats {
			for _, norm := range []lapack.MatrixNorm{lapack.MaxAbs, lapack.MaxColumnSum, lapack.MaxRowSum, lapack.Frobenius} {
				got := c.Norm(norm, a)
				want := netlib.Norm(c.Kind, byte(norm), c.M, c.N, uplo, diag, c.KL, c.KU, a)
				if math.IsNaN(got) != math.IsNaN(want) || !math.IsNaN(want) && math.Abs(got-want) > 1e-14*math.Abs(want) && got != want {
					t.Errorf("%v norm=%c A=%v: got %v, want Netlib %v", c, norm, a, got, want)
				}
			}
		}
	}
}
