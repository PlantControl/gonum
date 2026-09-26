// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlapack

import (
	"testing"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/lapack"
)

type Dggbaker interface {
	Dggbak(job lapack.BalanceJob, side blas.Side, n, ilo, ihi int, lscale, rscale []float64, m int, v []float64, ldv int)
}

func DggbakTest(t *testing.T, impl Dggbaker) {
	// Test basic parameter validation and simple cases.

	for _, n := range []int{0, 1, 2, 5, 10} {
		for _, job := range []lapack.BalanceJob{lapack.BalanceNone, lapack.Permute, lapack.Scale, lapack.PermuteScale} {
			for _, side := range []blas.Side{blas.Left, blas.Right} {
				testDggbak(t, impl, n, job, side)
			}
		}
	}
	testDggbakRectangularStride(t, impl)
	testDggbakUnusedInputs(t, impl)
}

func testDggbakUnusedInputs(t *testing.T, impl Dggbaker) {
	impl.Dggbak(lapack.BalanceNone, blas.Right, 3, 0, 2, nil, nil, 2, nil, 2)
	impl.Dggbak(lapack.Scale, blas.Left, 3, 1, 1, nil, nil, 2, nil, 2)
	impl.Dggbak(lapack.Permute, blas.Right, 3, 0, 2, nil, nil, 2, nil, 2)
	impl.Dggbak(lapack.PermuteScale, blas.Left, 1, 0, 0, nil, nil, 1, nil, 1)

	v := []float64{1, 2, 3, 4}
	impl.Dggbak(lapack.Scale, blas.Right, 2, 0, 1, nil, []float64{2, 3}, 2, v, 2)
	if v[0] != 2 || v[1] != 4 || v[2] != 9 || v[3] != 12 {
		t.Fatalf("right scaling produced %v", v)
	}
	v = []float64{1, 2, 3, 4}
	impl.Dggbak(lapack.Scale, blas.Left, 2, 0, 1, []float64{2, 3}, nil, 2, v, 2)
	if v[0] != 2 || v[1] != 4 || v[2] != 9 || v[3] != 12 {
		t.Fatalf("left scaling produced %v", v)
	}
}

func testDggbakRectangularStride(t *testing.T, impl Dggbaker) {
	const n = 5
	const m = 2
	const ldv = 3
	lscale := make([]float64, n)
	rscale := make([]float64, n)
	v := make([]float64, (n-1)*ldv+m)
	for i := range n {
		lscale[i] = 1
		rscale[i] = float64(i + 1)
		for j := range m {
			v[i*ldv+j] = float64(i*m + j + 1)
		}
	}
	want := append([]float64(nil), v...)
	for i := range n {
		for j := range m {
			want[i*ldv+j] *= rscale[i]
		}
	}
	impl.Dggbak(lapack.Scale, blas.Right, n, 0, n-1, lscale, rscale, m, v, ldv)
	for i := range v {
		if v[i] != want[i] {
			t.Fatalf("v[%d]=%g, want %g", i, v[i], want[i])
		}
	}
}

func testDggbak(t *testing.T, impl Dggbaker, n int, job lapack.BalanceJob, side blas.Side) {
	if n == 0 {
		// Test empty case.
		impl.Dggbak(job, side, 0, 0, -1, nil, nil, 0, nil, 1)
		return
	}

	m := n
	ilo, ihi := 0, n-1

	// Initialize scale arrays.
	lscale := make([]float64, n)
	rscale := make([]float64, n)
	for i := 0; i < n; i++ {
		lscale[i] = 1
		rscale[i] = 1
	}

	// Initialize eigenvector matrix as identity.
	v := make([]float64, n*m)
	for i := 0; i < n; i++ {
		v[i*m+i] = 1
	}

	// Call Dggbak.
	impl.Dggbak(job, side, n, ilo, ihi, lscale, rscale, m, v, m)

	// With scale=1 and ilo=0, ihi=n-1, V should be unchanged.
	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			expected := 0.0
			if i == j {
				expected = 1.0
			}
			if v[i*m+j] != expected {
				t.Errorf("n=%d, job=%c, side=%c: v[%d,%d]=%v, want %v",
					n, job, side, i, j, v[i*m+j], expected)
			}
		}
	}
}
