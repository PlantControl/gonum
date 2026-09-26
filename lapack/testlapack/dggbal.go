// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlapack

import (
	"math"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/lapack"
)

type Dggbaler interface {
	Dggbal(job lapack.BalanceJob, n int, a []float64, lda int, b []float64, ldb int, lscale, rscale, work []float64) (ilo, ihi int)
}

func DggbalTest(t *testing.T, impl Dggbaler) {
	rnd := rand.New(rand.NewPCG(1, 1))

	for _, n := range []int{0, 1, 2, 3, 4, 5, 10, 20} {
		for _, job := range []lapack.BalanceJob{lapack.BalanceNone, lapack.Permute, lapack.Scale, lapack.PermuteScale} {
			for _, extra := range []int{0, 5} {
				testDggbal(t, impl, n, job, extra, rnd)
			}
		}
	}

	testDggbalSpecial(t, impl)
	testDggbalReferenceScaling(t, impl)
	testDggbalOffDiagonalIsolation(t, impl)
	testDggbalUnusedInputs(t, impl)
}

func testDggbalUnusedInputs(t *testing.T, impl Dggbaler) {
	for _, job := range []lapack.BalanceJob{lapack.Permute, lapack.Scale, lapack.PermuteScale} {
		lscale := []float64{0}
		rscale := []float64{0}
		ilo, ihi := impl.Dggbal(job, 1, nil, 1, nil, 1, lscale, rscale, nil)
		if ilo != 0 || ihi != 0 || lscale[0] != 1 || rscale[0] != 1 {
			t.Errorf("job=%c: range=(%d,%d), scales=(%v,%v), want (0,0) and unit scales",
				job, ilo, ihi, lscale[0], rscale[0])
		}
	}

	a := []float64{1, 0, 0, 2}
	b := []float64{1, 0, 0, 1}
	impl.Dggbal(lapack.Permute, 2, a, 2, b, 2, make([]float64, 2), make([]float64, 2), nil)
}

func testDggbalOffDiagonalIsolation(t *testing.T, impl Dggbaler) {
	a := []float64{0, 1, 2, 3}
	b := make([]float64, 4)
	lscale := make([]float64, 2)
	rscale := make([]float64, 2)
	ilo, ihi := impl.Dggbal(lapack.Permute, 2, a, 2, b, 2,
		lscale, rscale, make([]float64, 12))
	if ilo != 0 || ihi != 0 {
		t.Fatalf("active range=(%d,%d), want (0,0)", ilo, ihi)
	}
	wantA := []float64{2, 3, 0, 1}
	for i, want := range wantA {
		if a[i] != want {
			t.Errorf("A[%d]=%v, want %v", i, a[i], want)
		}
	}
	if lscale[1] != 0 || rscale[1] != 1 {
		t.Errorf("isolated permutation=(%v,%v), want (0,1)", lscale[1], rscale[1])
	}
}

func testDggbalReferenceScaling(t *testing.T, impl Dggbaler) {
	a := []float64{1e-10, 1, 1, 1e10}
	b := []float64{1, 1, 1, 1}
	lscale := make([]float64, 2)
	rscale := make([]float64, 2)
	ilo, ihi := impl.Dggbal(lapack.PermuteScale, 2, a, 2, b, 2,
		lscale, rscale, make([]float64, 12))
	if ilo != 0 || ihi != 1 {
		t.Fatalf("active range=(%d,%d), want (0,1)", ilo, ihi)
	}
	wantScale := []float64{1e3, 1e-3}
	wantA := []float64{1e-4, 1, 1, 1e4}
	wantB := []float64{1e6, 1, 1, 1e-6}
	for i := range 2 {
		if lscale[i] != wantScale[i] || rscale[i] != wantScale[i] {
			t.Errorf("scales[%d]=(%g,%g), want (%g,%g)", i, lscale[i], rscale[i], wantScale[i], wantScale[i])
		}
	}
	for i := range 4 {
		if math.Abs(a[i]-wantA[i]) > 1e-14*math.Abs(wantA[i]) ||
			math.Abs(b[i]-wantB[i]) > 1e-14*math.Abs(wantB[i]) {
			t.Errorf("entry %d: (A,B)=(%g,%g), want (%g,%g)", i, a[i], b[i], wantA[i], wantB[i])
		}
	}
}

func testDggbal(t *testing.T, impl Dggbaler, n int, job lapack.BalanceJob, extra int, rnd *rand.Rand) {
	a := randomGeneral(n, n, n+extra, rnd)
	b := randomGeneral(n, n, n+extra, rnd)

	aOrig := cloneGeneral(a)
	bOrig := cloneGeneral(b)

	lscale := make([]float64, n)
	rscale := make([]float64, n)
	work := make([]float64, max(1, 6*n))

	ilo, ihi := impl.Dggbal(job, n, a.Data, a.Stride, b.Data, b.Stride, lscale, rscale, work)

	if n == 0 {
		if ilo != 0 || ihi != -1 {
			t.Errorf("n=0: ilo=%d, ihi=%d, want ilo=0, ihi=-1", ilo, ihi)
		}
		return
	}

	if ilo < 0 || ilo > ihi || ihi >= n {
		t.Errorf("n=%d, job=%c: invalid ilo=%d, ihi=%d", n, job, ilo, ihi)
	}

	if job == lapack.BalanceNone {
		if ilo != 0 || ihi != n-1 {
			t.Errorf("n=%d, job=N: ilo=%d, ihi=%d, want ilo=0, ihi=%d", n, ilo, ihi, n-1)
		}
		for i := 0; i < n; i++ {
			if lscale[i] != 1 || rscale[i] != 1 {
				t.Errorf("n=%d, job=N: lscale[%d]=%v, rscale[%d]=%v, want 1", n, i, lscale[i], i, rscale[i])
			}
		}
		if !generalEqual(a, aOrig) || !generalEqual(b, bOrig) {
			t.Errorf("n=%d, job=N: matrices modified", n)
		}
		return
	}

	// For permuting jobs, check that isolated rows/columns have zeros.
	if job == lapack.Permute || job == lapack.PermuteScale {
		for j := 0; j < ilo; j++ {
			for i := j + 1; i < n; i++ {
				if a.Data[i*a.Stride+j] != 0 || b.Data[i*b.Stride+j] != 0 {
					t.Errorf("n=%d, job=%c: A[%d,%d]=%v, B[%d,%d]=%v not zero in lower triangle before ilo",
						n, job, i, j, a.Data[i*a.Stride+j], i, j, b.Data[i*b.Stride+j])
				}
			}
		}
		for j := ihi + 1; j < n; j++ {
			for i := j + 1; i < n; i++ {
				if a.Data[i*a.Stride+j] != 0 || b.Data[i*b.Stride+j] != 0 {
					t.Errorf("n=%d, job=%c: A[%d,%d]=%v, B[%d,%d]=%v not zero in lower triangle after ihi",
						n, job, i, j, a.Data[i*a.Stride+j], i, j, b.Data[i*b.Stride+j])
				}
			}
		}
	}
}

func testDggbalSpecial(t *testing.T, impl Dggbaler) {
	// Test with matrices having isolated eigenvalues.

	// Case 1: Last row is zero except diagonal.
	a := blas64.General{
		Rows: 3, Cols: 3, Stride: 3,
		Data: []float64{
			1, 2, 3,
			4, 5, 6,
			0, 0, 7,
		},
	}
	b := blas64.General{
		Rows: 3, Cols: 3, Stride: 3,
		Data: []float64{
			1, 1, 1,
			1, 1, 1,
			0, 0, 1,
		},
	}

	lscale := make([]float64, 3)
	rscale := make([]float64, 3)
	work := make([]float64, 18)

	ilo, ihi := impl.Dggbal(lapack.Permute, 3, a.Data, a.Stride, b.Data, b.Stride, lscale, rscale, work)

	if ihi != 1 {
		t.Errorf("Special case 1: ihi=%d, want 1", ihi)
	}
	if ilo != 0 {
		t.Errorf("Special case 1: ilo=%d, want 0", ilo)
	}

	// Case 2: First column is zero except diagonal.
	a = blas64.General{
		Rows: 3, Cols: 3, Stride: 3,
		Data: []float64{
			1, 2, 3,
			0, 4, 5,
			0, 6, 7,
		},
	}
	b = blas64.General{
		Rows: 3, Cols: 3, Stride: 3,
		Data: []float64{
			1, 1, 1,
			0, 1, 1,
			0, 1, 1,
		},
	}

	lscale = make([]float64, 3)
	rscale = make([]float64, 3)

	ilo, _ = impl.Dggbal(lapack.Permute, 3, a.Data, a.Stride, b.Data, b.Stride, lscale, rscale, work)

	if ilo != 1 {
		t.Errorf("Special case 2: ilo=%d, want 1", ilo)
	}

	// Case 3: Test scaling with badly scaled matrix.
	a = blas64.General{
		Rows: 2, Cols: 2, Stride: 2,
		Data: []float64{
			1e-10, 1,
			1, 1e10,
		},
	}
	b = blas64.General{
		Rows: 2, Cols: 2, Stride: 2,
		Data: []float64{
			1, 1,
			1, 1,
		},
	}

	aOrig := cloneGeneral(a)
	bOrig := cloneGeneral(b)

	lscale = make([]float64, 2)
	rscale = make([]float64, 2)
	work = make([]float64, 12)

	impl.Dggbal(lapack.Scale, 2, a.Data, a.Stride, b.Data, b.Stride, lscale, rscale, work)
	if lscale[0] == 1 && lscale[1] == 1 && rscale[0] == 1 && rscale[1] == 1 {
		t.Error("badly scaled pair was left unscaled")
	}
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			wantA := lscale[i] * aOrig.Data[i*aOrig.Stride+j] * rscale[j]
			wantB := lscale[i] * bOrig.Data[i*bOrig.Stride+j] * rscale[j]
			if a.Data[i*a.Stride+j] != wantA || b.Data[i*b.Stride+j] != wantB {
				t.Errorf("scaled entry (%d,%d)=(%v,%v), want (%v,%v)", i, j,
					a.Data[i*a.Stride+j], b.Data[i*b.Stride+j], wantA, wantB)
			}
		}
	}

	// Check that scaling reduced the condition number.
	rowNormOrig := make([]float64, 2)
	colNormOrig := make([]float64, 2)
	rowNormNew := make([]float64, 2)
	colNormNew := make([]float64, 2)

	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			rowNormOrig[i] += math.Abs(aOrig.Data[i*aOrig.Stride+j]) + math.Abs(bOrig.Data[i*bOrig.Stride+j])
			colNormOrig[j] += math.Abs(aOrig.Data[i*aOrig.Stride+j]) + math.Abs(bOrig.Data[i*bOrig.Stride+j])
			rowNormNew[i] += math.Abs(a.Data[i*a.Stride+j]) + math.Abs(b.Data[i*b.Stride+j])
			colNormNew[j] += math.Abs(a.Data[i*a.Stride+j]) + math.Abs(b.Data[i*b.Stride+j])
		}
	}

	// The balanced matrix should have more uniform row/column norms.
	origRatio := math.Max(rowNormOrig[0], rowNormOrig[1]) / math.Min(rowNormOrig[0], rowNormOrig[1])
	newRatio := math.Max(rowNormNew[0], rowNormNew[1]) / math.Min(rowNormNew[0], rowNormNew[1])

	if newRatio > origRatio {
		t.Logf("Warning: Scaling did not improve row balance: orig ratio=%v, new ratio=%v", origRatio, newRatio)
	}
}
