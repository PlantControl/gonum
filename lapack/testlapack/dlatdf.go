// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlapack

import (
	"testing"

	"plantcontrol.org/v1/gonum/lapack"
)

type Dlatdfer interface {
	Dlatdf(job lapack.MaximizeNormXJob, n int, z []float64, ldz int, rhs []float64,
		rdsum, rdscal float64, ipiv, jpiv []int) (scale, sum float64)
}

func DlatdfTest(t *testing.T, impl Dlatdfer) {
	const n = 3
	z := []float64{
		1, 0, 0,
		0, 1, 0,
		0, 0, 1,
	}
	rhs := make([]float64, n)
	piv := []int{0, 1, 2}
	scale, sum := impl.Dlatdf(lapack.LocalLookAhead, n, z, n, rhs, 1, 0, piv, piv)
	if scale != 1 || sum != 3 {
		t.Errorf("sum-of-squares representation=(%v,%v), want (1,3)", scale, sum)
	}
	want := []float64{-1, 1, -1}
	for i, v := range rhs {
		if v != want[i] {
			t.Errorf("rhs[%d]=%v, want %v", i, v, want[i])
		}
	}
}
