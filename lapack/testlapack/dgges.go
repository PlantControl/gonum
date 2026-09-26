// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlapack

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/lapack"
)

type Dggeser interface {
	Dgges(jobvsl, jobvsr lapack.SchurComp, sort lapack.SchurSort, selctg lapack.SchurSelect,
		n int, a []float64, lda int, b []float64, ldb int,
		alphar, alphai, beta []float64,
		vsl []float64, ldvsl int, vsr []float64, ldvsr int,
		work []float64, lwork int, bwork []bool) (sdim int, ok bool)
}

func DggesTest(t *testing.T, impl Dggeser) {
	rnd := rand.New(rand.NewPCG(1, 1))

	for _, n := range []int{0, 1, 2, 3, 4, 5, 6, 10, 20, 50} {
		for _, jobvsl := range []lapack.SchurComp{lapack.SchurNone, lapack.SchurHess} {
			for _, jobvsr := range []lapack.SchurComp{lapack.SchurNone, lapack.SchurHess} {
				for _, extra := range []int{0, 3} {
					for _, wl := range []worklen{minimumWork, mediumWork, optimumWork} {
						testDgges(t, impl, n, jobvsl, jobvsr, extra, wl, rnd)
					}
				}
			}
		}
	}

	// Test with special matrices.
	tiny := math.SmallestNonzeroFloat64
	for _, tc := range []struct {
		name string
		a, b blas64.General
	}{
		{
			name: "Identity pair",
			a:    eye(3, 3),
			b:    eye(3, 3),
		},
		{
			name: "Diagonal pair",
			a:    Diagonal(5).Matrix(),
			b:    eye(5, 5),
		},
		{
			name: "Dense-Diagonal",
			a: blas64.General{
				Rows: 3, Cols: 3, Stride: 3,
				Data: []float64{
					1, 2, 3,
					4, 5, 6,
					7, 8, 10,
				},
			},
			b: blas64.General{
				Rows: 3, Cols: 3, Stride: 3,
				Data: []float64{
					1, 0, 0,
					0, 2, 0,
					0, 0, 3,
				},
			},
		},
		{
			// Matrix with complex eigenvalues to test 2x2 block standardization.
			// A = [0, -1; 1, 0] has eigenvalues ±i.
			name: "Complex eigenvalues 2x2",
			a: blas64.General{
				Rows: 2, Cols: 2, Stride: 2,
				Data: []float64{
					0, -1,
					1, 0,
				},
			},
			b: eye(2, 2),
		},
		{
			// Larger matrix with complex eigenvalues.
			// Has both real and complex eigenvalues.
			name: "Mixed eigenvalues 4x4",
			a: blas64.General{
				Rows: 4, Cols: 4, Stride: 4,
				Data: []float64{
					1, 0, 0, 0,
					0, 0, -2, 0,
					0, 2, 0, 0,
					0, 0, 0, 3,
				},
			},
			b: eye(4, 4),
		},
		{
			name: "Imbalanced dense pair",
			a: blas64.General{
				Rows: 3, Cols: 3, Stride: 3,
				Data: []float64{
					1, 2e-8, 3e-16,
					4e8, 5, 6e-8,
					7e16, 8e8, 10,
				},
			},
			b: blas64.General{
				Rows: 3, Cols: 3, Stride: 3,
				Data: []float64{
					2, 1e-8, 5e-17,
					-1e8, 3, 2e-8,
					2.5e15, -2e8, 4,
				},
			},
		},
		{
			name: "Dggbal scaling pair",
			a: blas64.General{
				Rows: 2, Cols: 2, Stride: 2,
				Data: []float64{1e-10, 1, 1, 1e10},
			},
			b: blas64.General{
				Rows: 2, Cols: 2, Stride: 2,
				Data: []float64{1, 1, 1, 1},
			},
		},
		{
			name: "Singular B with zero and infinite eigenvalues",
			a: blas64.General{
				Rows: 3, Cols: 3, Stride: 3,
				Data: []float64{
					0, 2, -1,
					0, 3, 4,
					0, 0, 5,
				},
			},
			b: blas64.General{
				Rows: 3, Cols: 3, Stride: 3,
				Data: []float64{
					0, 1, 0,
					0, 0, 2,
					0, 0, 1,
				},
			},
		},
		{
			name: "Nearly singular B",
			a: blas64.General{
				Rows: 3, Cols: 3, Stride: 3,
				Data: []float64{
					1, 2, 0,
					0, 2, 1,
					0, 0, 3,
				},
			},
			b: blas64.General{
				Rows: 3, Cols: 3, Stride: 3,
				Data: []float64{
					1e-300, 1, 0,
					0, 1, 1,
					0, 0, 2,
				},
			},
		},
		{
			name: "Nearly defective pencil",
			a: blas64.General{
				Rows: 2, Cols: 2, Stride: 2,
				Data: []float64{
					1, 1,
					1e-14, 1,
				},
			},
			b: eye(2, 2),
		},
		{
			name: "Clustered conjugate pairs",
			a: blas64.General{
				Rows: 4, Cols: 4, Stride: 4,
				Data: []float64{
					1, -1, 1e-12, 0,
					1, 1, 0, -1e-12,
					0, 0, 1 + 1e-13, -(1 + 1e-13),
					0, 0, 1 + 1e-13, 1 + 1e-13,
				},
			},
			b: eye(4, 4),
		},
		{
			name: "Complex block near splitting",
			a: blas64.General{
				Rows: 2, Cols: 2, Stride: 2,
				Data: []float64{1, -1e-12, 1e-12, 1},
			},
			b: eye(2, 2),
		},
		{
			name: "Very small pencil",
			a: blas64.General{
				Rows: 2, Cols: 2, Stride: 2,
				Data: []float64{2e-300, -1e-300, 1e-300, 2e-300},
			},
			b: blas64.General{
				Rows: 2, Cols: 2, Stride: 2,
				Data: []float64{1e-300, 0, 0, 1e-300},
			},
		},
		{
			name: "Very large pencil",
			a: blas64.General{
				Rows: 2, Cols: 2, Stride: 2,
				Data: []float64{2e300, -1e300, 1e300, 2e300},
			},
			b: blas64.General{
				Rows: 2, Cols: 2, Stride: 2,
				Data: []float64{1e300, 0, 0, 1e300},
			},
		},
		{
			name: "Subnormal pencil",
			a: blas64.General{
				Rows: 2, Cols: 2, Stride: 2,
				Data: []float64{2 * tiny, -tiny, tiny, 2 * tiny},
			},
			b: blas64.General{
				Rows: 2, Cols: 2, Stride: 2,
				Data: []float64{tiny, 0, 0, tiny},
			},
		},
	} {
		testDggesMatrix(t, impl, tc.name, tc.a, tc.b, optimumWork)
	}

	// Test 2x2 block standardization specifically.
	testDggesBlockStandardization(t, impl)
	testDggesTwoByTwoShiftFallback(t, impl)

	// Test sorting.
	testDggesSorting(t, impl)
	testDggesConjugatePairSorting(t, impl)
	testDggesSortingUsesUnscaledEigenvalues(t, impl)
	testDggesSortingPreservesAlphaScale(t, impl)
	testDggesZeroAndInfiniteEigenvalues(t, impl)
	testDggesExtremeScaleEigenvalueRepresentation(t, impl)
	testDggesSelectionRecheck(t, impl)
}

func testDggesSortingPreservesAlphaScale(t *testing.T, impl Dggeser) {
	const n = 2
	a := []float64{1e-300, 0, 0, 1e-200}
	b := []float64{1e-300, 0, 0, 1e-200}
	alphar := make([]float64, n)
	alphai := make([]float64, n)
	beta := make([]float64, n)
	bwork := make([]bool, n)
	work := make([]float64, max(8*n, 6*n+16))
	selector := func(alphar, _, _ float64) bool {
		return alphar > 1e-299
	}
	sdim, ok := impl.Dgges(lapack.SchurNone, lapack.SchurNone, lapack.SortSelected, selector,
		n, a, n, b, n, alphar, alphai, beta, nil, 1, nil, 1, work, len(work), bwork)
	if !ok {
		t.Fatal("scale-sensitive sorting case failed")
	}
	if sdim != 1 {
		t.Fatalf("sdim=%d, want 1 when selector distinguishes homogeneous alpha scales", sdim)
	}
}

func testDggesTwoByTwoShiftFallback(t *testing.T, impl Dggeser) {
	const n = 14
	rnd := rand.New(rand.NewPCG(93, 93))
	var a, b []float64
	for k := 0; k <= 396; k++ {
		a = make([]float64, n*n)
		b = make([]float64, n*n)
		for i := range a {
			exp := rnd.IntN(601) - 300
			a[i] = rnd.NormFloat64() * math.Pow10(exp)
			exp = rnd.IntN(601) - 300
			b[i] = rnd.NormFloat64() * math.Pow10(exp)
		}
	}
	work := make([]float64, 1)
	impl.Dgges(lapack.SchurHess, lapack.SchurHess, lapack.SortNone, nil, n,
		nil, n, nil, n, nil, nil, nil, nil, n, nil, n, work, -1, nil)
	work = make([]float64, int(work[0]))
	_, ok := impl.Dgges(lapack.SchurHess, lapack.SchurHess, lapack.SortNone, nil, n,
		a, n, b, n, make([]float64, n), make([]float64, n), make([]float64, n),
		make([]float64, n*n), n, make([]float64, n*n), n, work, len(work), nil)
	if !ok {
		t.Fatal("badly scaled 2x2 shift fallback did not converge")
	}
}

func testDggesSortingUsesUnscaledEigenvalues(t *testing.T, impl Dggeser) {
	const n = 2
	a := []float64{1e-300, 0, 0, 2e-300}
	b := []float64{1, 0, 0, 1}
	alphar := make([]float64, n)
	alphai := make([]float64, n)
	beta := make([]float64, n)
	bwork := make([]bool, n)
	work := make([]float64, max(8*n, 6*n+16))
	var seen []float64
	selector := func(alphar, _, beta float64) bool {
		seen = append(seen, alphar/beta)
		return math.Abs(alphar/beta) > 1e-200
	}
	sdim, ok := impl.Dgges(lapack.SchurNone, lapack.SchurNone, lapack.SortSelected, selector,
		n, a, n, b, n, alphar, alphai, beta, nil, 1, nil, 1, work, len(work), bwork)
	if !ok {
		t.Fatal("scaled sorting case failed")
	}
	if sdim != 0 {
		t.Fatalf("sdim=%d, want 0 when selector sees unscaled eigenvalues", sdim)
	}
	for i, v := range seen[:n] {
		if math.Abs(v) > 1e-200 {
			t.Errorf("initial selector call %d saw scaled eigenvalue %v", i, v)
		}
	}
}

func testDggesExtremeScaleEigenvalueRepresentation(t *testing.T, impl Dggeser) {
	n := 2
	a := []float64{2e-105, -1e-105, 1e-105, 2e-105}
	b := []float64{1e239, 0, 0, 1e239}
	alphar := make([]float64, n)
	alphai := make([]float64, n)
	beta := make([]float64, n)
	work := make([]float64, 1)
	impl.Dgges(lapack.SchurNone, lapack.SchurNone, lapack.SortNone, nil,
		n, nil, n, nil, n, nil, nil, nil, nil, 1, nil, 1, work, -1, nil)
	work = make([]float64, int(work[0]))
	_, ok := impl.Dgges(lapack.SchurNone, lapack.SchurNone, lapack.SortNone, nil,
		n, a, n, b, n, alphar, alphai, beta, nil, 1, nil, 1, work, len(work), nil)
	if !ok {
		t.Fatal("extreme-scale test failed to converge")
	}
	for i := range n {
		if math.IsInf(alphar[i], 0) || math.IsInf(alphai[i], 0) || math.IsInf(beta[i], 0) ||
			math.IsNaN(alphar[i]) || math.IsNaN(alphai[i]) || math.IsNaN(beta[i]) {
			t.Fatalf("non-finite eigenvalue representation at %d: alpha=(%g,%g), beta=%g", i, alphar[i], alphai[i], beta[i])
		}
		if beta[i] == 0 || alphai[i] == 0 {
			t.Fatalf("lost finite complex eigenvalue at %d: alpha=(%g,%g), beta=%g", i, alphar[i], alphai[i], beta[i])
		}
	}
}

func testDggesZeroAndInfiniteEigenvalues(t *testing.T, impl Dggeser) {
	n := 3
	a := []float64{
		0, 0, 0,
		0, 2, 0,
		0, 0, 3,
	}
	b := []float64{
		0, 0, 0,
		0, 0, 0,
		0, 0, 1,
	}
	alphar := make([]float64, n)
	alphai := make([]float64, n)
	beta := make([]float64, n)
	work := make([]float64, 1)
	impl.Dgges(lapack.SchurNone, lapack.SchurNone, lapack.SortNone, nil,
		n, nil, n, nil, n, nil, nil, nil, nil, 1, nil, 1, work, -1, nil)
	work = make([]float64, int(work[0]))
	_, ok := impl.Dgges(lapack.SchurNone, lapack.SchurNone, lapack.SortNone, nil,
		n, a, n, b, n, alphar, alphai, beta, nil, 1, nil, 1, work, len(work), nil)
	if !ok {
		t.Fatal("zero and infinite eigenvalue test failed to converge")
	}

	var zero, infinite, finite int
	for i := range n {
		switch {
		case beta[i] == 0 && alphar[i] == 0 && alphai[i] == 0:
			zero++
		case beta[i] == 0:
			infinite++
		case alphai[i] == 0 && math.Abs(alphar[i]/beta[i]-3) < 1e-14:
			finite++
		}
	}
	if zero != 1 || infinite != 1 || finite != 1 {
		t.Fatalf("got (zero, infinite, finite-3)=(%d, %d, %d), want (1, 1, 1); alpha=(%v,%v), beta=%v",
			zero, infinite, finite, alphar, alphai, beta)
	}
}

func testDggesConjugatePairSorting(t *testing.T, impl Dggeser) {
	n := 4
	a := []float64{
		-1, 0, 0, 0,
		0, 3, -2, 0,
		0, 2, 3, 0,
		0, 0, 0, 1,
	}
	b := []float64{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
	}
	alphar := make([]float64, n)
	alphai := make([]float64, n)
	beta := make([]float64, n)
	vsl := make([]float64, n*n)
	vsr := make([]float64, n*n)
	bwork := make([]bool, n)
	work := make([]float64, 1)
	impl.Dgges(lapack.SchurHess, lapack.SchurHess, lapack.SortSelected, nil,
		n, nil, n, nil, n, nil, nil, nil, nil, n, nil, n, work, -1, nil)
	work = make([]float64, int(work[0]))
	selector := func(_, alphai, _ float64) bool { return alphai > 0 }
	sdim, ok := impl.Dgges(lapack.SchurHess, lapack.SchurHess, lapack.SortSelected, selector,
		n, a, n, b, n, alphar, alphai, beta, vsl, n, vsr, n, work, len(work), bwork)
	if !ok {
		t.Fatal("conjugate-pair sorting failed")
	}
	if sdim != 2 {
		t.Fatalf("got sdim=%d, want 2", sdim)
	}
	if alphai[0] == 0 || alphai[1] == 0 || alphai[0] != -alphai[1] {
		t.Fatalf("leading block is not a conjugate pair: alphai=%v", alphai)
	}
	for i := sdim; i < n; i++ {
		if alphai[i] != 0 {
			t.Fatalf("unselected eigenvalue %d is complex: alphai=%v", i, alphai)
		}
	}
}

func testDggesSelectionRecheck(t *testing.T, impl Dggeser) {
	n := 2
	a := []float64{1, 0, 0, 2}
	b := []float64{1, 0, 0, 1}
	work := make([]float64, 1)
	impl.Dgges(lapack.SchurNone, lapack.SchurNone, lapack.SortSelected, nil,
		n, nil, n, nil, n, nil, nil, nil, nil, 1, nil, 1, work, -1, nil)
	work = make([]float64, int(work[0]))
	calls := 0
	selector := func(alphar, alphai, beta float64) bool {
		calls++
		if calls <= n {
			return alphar/beta > 1.5
		}
		return alphar/beta < 1.5
	}
	_, ok := impl.Dgges(lapack.SchurNone, lapack.SchurNone, lapack.SortSelected, selector,
		n, a, n, b, n, make([]float64, n), make([]float64, n), make([]float64, n),
		nil, 1, nil, 1, work, len(work), make([]bool, n))
	if ok {
		t.Fatal("sorting succeeded after selected eigenvalues no longer satisfied selector")
	}
}

func testDgges(t *testing.T, impl Dggeser, n int, jobvsl, jobvsr lapack.SchurComp, extra int, wl worklen, rnd *rand.Rand) {
	const tol = 100

	wantvsl := jobvsl == lapack.SchurHess
	wantvsr := jobvsr == lapack.SchurHess

	a := randomGeneral(n, n, n+extra, rnd)
	b := randomGeneral(n, n, n+extra, rnd)
	aCopy := cloneGeneral(a)
	bCopy := cloneGeneral(b)

	alphar := make([]float64, n)
	alphai := make([]float64, n)
	beta := make([]float64, n)

	var vsl, vsr blas64.General
	ldvsl, ldvsr := 1, 1
	if wantvsl {
		ldvsl = max(1, n+extra)
		vsl = nanGeneral(n, n, ldvsl)
	}
	if wantvsr {
		ldvsr = max(1, n+extra)
		vsr = nanGeneral(n, n, ldvsr)
	}

	minwork := 1
	if n > 0 {
		minwork = max(8*n, 6*n+16)
	}
	var lwork int
	switch wl {
	case minimumWork:
		lwork = minwork
	case mediumWork:
		work := make([]float64, 1)
		impl.Dgges(jobvsl, jobvsr, lapack.SortNone, nil, n, nil, max(1, n), nil, max(1, n),
			nil, nil, nil, nil, ldvsl, nil, ldvsr, work, -1, nil)
		lwork = (int(work[0]) + minwork) / 2
		lwork = max(minwork, lwork)
	case optimumWork:
		work := make([]float64, 1)
		impl.Dgges(jobvsl, jobvsr, lapack.SortNone, nil, n, nil, max(1, n), nil, max(1, n),
			nil, nil, nil, nil, ldvsl, nil, ldvsr, work, -1, nil)
		lwork = int(work[0])
	}
	work := make([]float64, lwork)

	prefix := fmt.Sprintf("n=%v, jobvsl=%c, jobvsr=%c, extra=%v, work=%v", n, jobvsl, jobvsr, extra, wl)

	sdim, ok := impl.Dgges(jobvsl, jobvsr, lapack.SortNone, nil, n, a.Data, a.Stride, b.Data, b.Stride,
		alphar, alphai, beta, vsl.Data, ldvsl, vsr.Data, ldvsr, work, lwork, nil)

	if !ok {
		t.Errorf("%v: Dgges failed to converge", prefix)
		return
	}
	if sdim != 0 {
		t.Errorf("%v: unexpected sdim=%v, want 0 for SortNone", prefix, sdim)
	}
	if n == 0 {
		return
	}

	if !isGeneralizedSchurForm(a, b) {
		t.Errorf("%v: (S,T) is not in generalized real Schur form: S=%v T=%v", prefix, a.Data, b.Data)
	}

	if !isUpperTriangular(b) {
		t.Errorf("%v: T is not upper triangular", prefix)
	}

	if !wantvsl && !wantvsr {
		return
	}

	orthoTol := float64(n) * dlamchE
	if orthoTol < 1e-13 {
		orthoTol = 1e-13
	}

	if wantvsl {
		resid := residualOrthogonal(vsl, false)
		if resid > orthoTol {
			t.Errorf("%v: VSL not orthogonal; |I - VSL*VSLᵀ|=%v, want<=%v", prefix, resid, orthoTol)
		}
	}
	if wantvsr {
		resid := residualOrthogonal(vsr, false)
		if resid > orthoTol {
			t.Errorf("%v: VSR not orthogonal; |I - VSR*VSRᵀ|=%v, want<=%v", prefix, resid, orthoTol)
		}
	}

	if wantvsl && wantvsr {
		residA := residualGeneralizedSchur(aCopy, a, vsl, vsr)
		anorm := dlange(lapack.MaxColumnSum, n, n, aCopy.Data, aCopy.Stride)
		if anorm == 0 {
			anorm = 1
		}
		normResA := residA / (anorm * float64(n) * dlamchE)
		if normResA > tol {
			t.Errorf("%v: ||A - VSL*S*VSRᵀ||/(||A||*n*eps)=%v, want<=%v", prefix, normResA, tol)
		}

		residB := residualGeneralizedSchur(bCopy, b, vsl, vsr)
		bnorm := dlange(lapack.MaxColumnSum, n, n, bCopy.Data, bCopy.Stride)
		if bnorm == 0 {
			bnorm = 1
		}
		normResB := residB / (bnorm * float64(n) * dlamchE)
		if normResB > tol {
			t.Errorf("%v: ||B - VSL*T*VSRᵀ||/(||B||*n*eps)=%v, want<=%v", prefix, normResB, tol)
		}
	}
}

func testDggesMatrix(t *testing.T, impl Dggeser, name string, aOrig, bOrig blas64.General, _ worklen) {
	const tol = 100

	n := aOrig.Rows

	a := cloneGeneral(aOrig)
	b := cloneGeneral(bOrig)

	alphar := make([]float64, n)
	alphai := make([]float64, n)
	beta := make([]float64, n)

	ldvsl := max(1, n)
	ldvsr := max(1, n)
	vsl := nanGeneral(n, n, ldvsl)
	vsr := nanGeneral(n, n, ldvsr)

	work := make([]float64, 1)
	impl.Dgges(lapack.SchurHess, lapack.SchurHess, lapack.SortNone, nil, n, nil, max(1, n), nil, max(1, n),
		nil, nil, nil, nil, ldvsl, nil, ldvsr, work, -1, nil)
	lwork := int(work[0])
	work = make([]float64, lwork)

	prefix := name

	sdim, ok := impl.Dgges(lapack.SchurHess, lapack.SchurHess, lapack.SortNone, nil, n, a.Data, a.Stride, b.Data, b.Stride,
		alphar, alphai, beta, vsl.Data, ldvsl, vsr.Data, ldvsr, work, lwork, nil)

	if !ok {
		t.Errorf("%v: Dgges failed to converge", prefix)
		return
	}
	if sdim != 0 {
		t.Errorf("%v: unexpected sdim=%v, want 0 for SortNone", prefix, sdim)
	}
	if n == 0 {
		return
	}

	if !isGeneralizedSchurForm(a, b) {
		t.Errorf("%v: (S,T) is not in generalized real Schur form: S=%v T=%v", prefix, a.Data, b.Data)
	}
	if !isUpperTriangular(b) {
		t.Errorf("%v: T is not upper triangular", prefix)
	}

	orthoTol := float64(n) * dlamchE
	if orthoTol < 1e-13 {
		orthoTol = 1e-13
	}
	residVSL := residualOrthogonal(vsl, false)
	if residVSL > orthoTol {
		t.Errorf("%v: VSL not orthogonal; |I - VSL*VSLᵀ|=%v, want<=%v", prefix, residVSL, orthoTol)
	}
	residVSR := residualOrthogonal(vsr, false)
	if residVSR > orthoTol {
		t.Errorf("%v: VSR not orthogonal; |I - VSR*VSRᵀ|=%v, want<=%v", prefix, residVSR, orthoTol)
	}

	residA := residualGeneralizedSchur(aOrig, a, vsl, vsr)
	anorm := dlange(lapack.MaxColumnSum, n, n, aOrig.Data, aOrig.Stride)
	if anorm == 0 {
		anorm = 1
	}
	normResA := residA / (anorm * float64(n) * dlamchE)
	if normResA > tol {
		t.Errorf("%v: ||A - VSL*S*VSRᵀ||/(||A||*n*eps)=%v, want<=%v", prefix, normResA, tol)
	}

	residB := residualGeneralizedSchur(bOrig, b, vsl, vsr)
	bnorm := dlange(lapack.MaxColumnSum, n, n, bOrig.Data, bOrig.Stride)
	if bnorm == 0 {
		bnorm = 1
	}
	normResB := residB / (bnorm * float64(n) * dlamchE)
	if normResB > tol {
		t.Errorf("%v: ||B - VSL*T*VSRᵀ||/(||B||*n*eps)=%v, want<=%v", prefix, normResB, tol)
	}
}

// residualGeneralizedSchur computes ||M - VSL*S*VSR^T||_1 for the generalized
// Schur factorization M = VSL * S * VSR^T.
func residualGeneralizedSchur(M, S, VSL, VSR blas64.General) float64 {
	n := M.Rows
	if n == 0 {
		return 0
	}

	R := zeros(n, n, n)
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, VSL, S, 0, R)

	R2 := zeros(n, n, n)
	blas64.Gemm(blas.NoTrans, blas.Trans, 1, R, VSR, 0, R2)

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			R2.Data[i*R2.Stride+j] = M.Data[i*M.Stride+j] - R2.Data[i*R2.Stride+j]
		}
	}

	return dlange(lapack.MaxColumnSum, n, n, R2.Data, R2.Stride)
}

func isGeneralizedSchurForm(a, b blas64.General) bool {
	n := a.Rows
	for i := 2; i < n; i++ {
		for j := 0; j < i-1; j++ {
			if a.Data[i*a.Stride+j] != 0 {
				return false
			}
		}
	}
	for i := 0; i < n-1; i++ {
		if a.Data[(i+1)*a.Stride+i] == 0 {
			continue
		}
		if i+2 < n && a.Data[(i+2)*a.Stride+i+1] != 0 {
			return false
		}
		b11 := b.Data[i*b.Stride+i]
		b12 := b.Data[i*b.Stride+i+1]
		b21 := b.Data[(i+1)*b.Stride+i]
		b22 := b.Data[(i+1)*b.Stride+i+1]
		if b12 != 0 || b21 != 0 || b11 <= 0 || b22 <= 0 {
			return false
		}
		a11 := a.Data[i*a.Stride+i]
		a12 := a.Data[i*a.Stride+i+1]
		a21 := a.Data[(i+1)*a.Stride+i]
		a22 := a.Data[(i+1)*a.Stride+i+1]
		as := math.Max(math.Abs(a11), math.Max(math.Abs(a12), math.Max(math.Abs(a21), math.Abs(a22))))
		bs := math.Max(b11, b22)
		if as == 0 || bs == 0 {
			return false
		}
		a11, a12, a21, a22 = a11/as, a12/as, a21/as, a22/as
		b11, b22 = b11/bs, b22/bs
		d := a11*b22 - a22*b11
		discriminant := d*d + 4*b11*b22*a12*a21
		if discriminant >= 0 {
			return false
		}
		i++
	}
	return true
}

// testDggesSorting tests eigenvalue sorting with SortSelected.
func testDggesSorting(t *testing.T, impl Dggeser) {
	const tol = 100

	// Test matrix pair where eigenvalues are clearly separable.
	// A = diag(1, 5, 3), B = I => eigenvalues are 1, 5, 3.
	// Select eigenvalues > 2 (should get 5 and 3).
	n := 3
	a := blas64.General{
		Rows: n, Cols: n, Stride: n,
		Data: []float64{
			1, 0, 0,
			0, 5, 0,
			0, 0, 3,
		},
	}
	b := blas64.General{
		Rows: n, Cols: n, Stride: n,
		Data: []float64{
			1, 0, 0,
			0, 1, 0,
			0, 0, 1,
		},
	}

	aOrig := cloneGeneral(a)
	bOrig := cloneGeneral(b)

	alphar := make([]float64, n)
	alphai := make([]float64, n)
	beta := make([]float64, n)
	vsl := nanGeneral(n, n, n)
	vsr := nanGeneral(n, n, n)
	bwork := make([]bool, n)

	work := make([]float64, 1)
	impl.Dgges(lapack.SchurHess, lapack.SchurHess, lapack.SortNone, nil, n, nil, n, nil, n,
		nil, nil, nil, nil, n, nil, n, work, -1, nil)
	lwork := int(work[0])
	work = make([]float64, lwork)

	// Select eigenvalues where alphar/beta > 2.
	selctg := func(alphar, alphai, beta float64) bool {
		if beta == 0 {
			return false // Infinite eigenvalue.
		}
		return alphar/beta > 2
	}

	sdim, ok := impl.Dgges(lapack.SchurHess, lapack.SchurHess, lapack.SortSelected, selctg, n,
		a.Data, a.Stride, b.Data, b.Stride, alphar, alphai, beta,
		vsl.Data, vsl.Stride, vsr.Data, vsr.Stride, work, lwork, bwork)

	if !ok {
		t.Errorf("Sorting test: Dgges failed to converge")
		return
	}

	// Should have 2 selected eigenvalues (5 and 3).
	if sdim != 2 {
		t.Errorf("Sorting test: sdim=%d, want 2", sdim)
	}

	// Check that first sdim eigenvalues satisfy the selection criterion.
	for i := 0; i < sdim; i++ {
		if beta[i] != 0 && alphar[i]/beta[i] <= 2 {
			t.Errorf("Sorting test: eigenvalue[%d]=%v should be > 2", i, alphar[i]/beta[i])
		}
	}

	// Check that remaining eigenvalues do NOT satisfy the criterion.
	for i := sdim; i < n; i++ {
		if beta[i] != 0 && alphar[i]/beta[i] > 2 {
			t.Errorf("Sorting test: eigenvalue[%d]=%v should be <= 2", i, alphar[i]/beta[i])
		}
	}

	// Check decomposition is still correct.
	orthoTol := float64(n) * dlamchE
	if orthoTol < 1e-13 {
		orthoTol = 1e-13
	}

	residVSL := residualOrthogonal(vsl, false)
	if residVSL > orthoTol {
		t.Errorf("Sorting test: VSL not orthogonal; |I - VSL*VSLᵀ|=%v, want<=%v", residVSL, orthoTol)
	}
	residVSR := residualOrthogonal(vsr, false)
	if residVSR > orthoTol {
		t.Errorf("Sorting test: VSR not orthogonal; |I - VSR*VSRᵀ|=%v, want<=%v", residVSR, orthoTol)
	}

	residA := residualGeneralizedSchur(aOrig, a, vsl, vsr)
	anorm := dlange(lapack.MaxColumnSum, n, n, aOrig.Data, aOrig.Stride)
	if anorm == 0 {
		anorm = 1
	}
	normResA := residA / (anorm * float64(n) * dlamchE)
	if normResA > tol {
		t.Errorf("Sorting test: ||A - VSL*S*VSRᵀ||/(||A||*n*eps)=%v, want<=%v", normResA, tol)
	}

	residB := residualGeneralizedSchur(bOrig, b, vsl, vsr)
	bnorm := dlange(lapack.MaxColumnSum, n, n, bOrig.Data, bOrig.Stride)
	if bnorm == 0 {
		bnorm = 1
	}
	normResB := residB / (bnorm * float64(n) * dlamchE)
	if normResB > tol {
		t.Errorf("Sorting test: ||B - VSL*T*VSRᵀ||/(||B||*n*eps)=%v, want<=%v", normResB, tol)
	}
}

// testDggesBlockStandardization tests that 2x2 blocks in S are properly
// standardized: equal diagonals and H(i+1,i)*H(i,i+1) < 0.
// Also checks that T's 2x2 block is diagonal with positive entries.
func testDggesBlockStandardization(t *testing.T, impl Dggeser) {
	// Matrix with complex eigenvalues ±i.
	// After Schur factorization, should have a 2x2 block with:
	// - Equal diagonals
	// - Product of off-diagonals negative
	n := 2
	a := blas64.General{
		Rows: n, Cols: n, Stride: n,
		Data: []float64{
			0, -1,
			1, 0,
		},
	}
	b := blas64.General{Rows: n, Cols: n, Stride: n, Data: []float64{2, 1, 0, 3}}

	alphar := make([]float64, n)
	alphai := make([]float64, n)
	beta := make([]float64, n)
	vsl := nanGeneral(n, n, n)
	vsr := nanGeneral(n, n, n)

	work := make([]float64, 1)
	impl.Dgges(lapack.SchurHess, lapack.SchurHess, lapack.SortNone, nil, n,
		nil, n, nil, n, nil, nil, nil, nil, n, nil, n, work, -1, nil)
	lwork := int(work[0])
	work = make([]float64, lwork)

	_, ok := impl.Dgges(lapack.SchurHess, lapack.SchurHess, lapack.SortNone, nil, n,
		a.Data, a.Stride, b.Data, b.Stride, alphar, alphai, beta,
		vsl.Data, vsl.Stride, vsr.Data, vsr.Stride, work, lwork, nil)

	if !ok {
		t.Errorf("Block standardization test: Dgges failed to converge")
		return
	}

	// Check that we have complex eigenvalues.
	if alphai[0] == 0 || alphai[1] == 0 {
		t.Errorf("Block standardization test: expected complex eigenvalues, got alphai=%v", alphai)
		return
	}

	// Check S (stored in a) has standardized 2x2 block.
	s00 := a.Data[0]
	s01 := a.Data[1]
	s10 := a.Data[n]
	s11 := a.Data[n+1]

	// Diagonals should be equal (within tolerance).
	const tol = 1e-12
	if diff := s00 - s11; diff > tol || diff < -tol {
		t.Errorf("Block standardization: S diagonals not equal; S[0,0]=%v, S[1,1]=%v, diff=%v",
			s00, s11, diff)
	}

	// Off-diagonal product should be negative (complex eigenvalues).
	if s10*s01 >= 0 {
		t.Errorf("Block standardization: S[1,0]*S[0,1]=%v, want < 0", s10*s01)
	}

	// Check T (stored in b) is diagonal with positive entries.
	t00 := b.Data[0]
	t01 := b.Data[1]
	t10 := b.Data[n]
	t11 := b.Data[n+1]

	if t01 > tol || t01 < -tol {
		t.Errorf("Block standardization: T[0,1]=%v, want 0", t01)
	}
	if t10 > tol || t10 < -tol {
		t.Errorf("Block standardization: T[1,0]=%v, want 0", t10)
	}
	if t00 <= 0 {
		t.Errorf("Block standardization: T[0,0]=%v, want > 0", t00)
	}
	if t11 <= 0 {
		t.Errorf("Block standardization: T[1,1]=%v, want > 0", t11)
	}
}
