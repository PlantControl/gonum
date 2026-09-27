// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package gonum

import (
	"testing"

	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/lapack/gonum/internal/netlib"
)

func TestDlanhsNetlibDifferential(t *testing.T) {
	for _, tc := range []struct {
		name string
		a    []float64
	}{
		{"Zero", make([]float64, 9)},
		{"Ordinary", []float64{2, -3, 4, 5, -6, 7, 99, 8, 9}},
		{"MixedScale", []float64{1e-300, -2, 3e300, 4e-200, -5e200, 6, 99, -7e100, 8e-100}},
	} {
		for _, norm := range []lapack.MatrixNorm{lapack.MaxAbs, lapack.MaxColumnSum, lapack.MaxRowSum, lapack.Frobenius} {
			t.Run(tc.name+"/"+string(norm), func(t *testing.T) {
				work := make([]float64, 3)
				got := Implementation{}.Dlanhs(norm, 3, tc.a, 3, work)
				col := []float64{tc.a[0], tc.a[3], tc.a[6], tc.a[1], tc.a[4], tc.a[7], tc.a[2], tc.a[5], tc.a[8]}
				want := netlib.Dlanhs(byte(norm), 3, col, 3, make([]float64, 3))
				comparePrimitiveFloat(t, "norm", got, want, 2e-15)
			})
		}
	}
}
