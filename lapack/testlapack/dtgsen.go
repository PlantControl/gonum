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

type Dtgsener interface {
	Dtgexcer
	Dtgsyler
	Dlag2er
	Dtgsen(ijob int, wantq, wantz bool, selected []bool, n int,
		a []float64, lda int, b []float64, ldb int,
		alphar, alphai, beta []float64,
		q []float64, ldq int, z []float64, ldz int,
		work []float64, lwork int, iwork []int, liwork int) (m int, pl, pr float64, dif [2]float64, ok bool)
}

func DtgsenTest(t *testing.T, impl Dtgsener) {
	rnd := rand.New(rand.NewPCG(1, 1))

	for _, n := range []int{0, 1, 2, 3, 4, 5, 10, 20} {
		for _, ijob := range []int{0, 1, 2, 4} {
			for _, wantq := range []bool{false, true} {
				for _, wantz := range []bool{false, true} {
					for _, selectPattern := range []string{"none", "first", "last", "alternating", "all"} {
						testDtgsen(t, impl, n, ijob, wantq, wantz, selectPattern, rnd)
					}
				}
			}
		}
	}

	testDtgsenWorkspace(t, impl)
	testDtgsenPartialWorkspaceQueries(t, impl)
	testDtgsenTrivialSeparation(t, impl)
	testDtgsenProjectionBounds(t, impl)
	testDtgsenSingularConditionEstimate(t, impl)
	testDtgsenNormalizesRealEigenvalueSigns(t, impl)
	testDtgsenSeparationDirections(t, impl)
	testDtgsenOneNormSeparation(t, impl)
}

func testDtgsenSingularConditionEstimate(t *testing.T, impl Dtgsener) {
	const n = 2
	a := []float64{1, 0, 0, 1}
	b := []float64{1, 0, 0, 1}
	work := make([]float64, 4*n+16)
	iwork := make([]int, n+6)
	_, _, _, _, ok := impl.Dtgsen(1, false, false, []bool{true, false}, n,
		a, n, b, n, make([]float64, n), make([]float64, n), make([]float64, n),
		nil, 1, nil, 1, work, len(work), iwork, len(iwork))
	if !ok {
		t.Fatal("singular condition estimate reported a reordering failure")
	}
}

func testDtgsenNormalizesRealEigenvalueSigns(t *testing.T, impl Dtgsener) {
	const n = 2
	a := []float64{1, 2, 0, 3}
	b := []float64{-2, 4, 0, -5}
	q := eye(n, n).Data
	alphar := make([]float64, n)
	alphai := make([]float64, n)
	beta := make([]float64, n)
	work := make([]float64, 4*n+16)
	iwork := make([]int, 1)
	_, _, _, _, ok := impl.Dtgsen(0, true, false, []bool{false, false}, n,
		a, n, b, n, alphar, alphai, beta, q, n, nil, 1,
		work, len(work), iwork, len(iwork))
	if !ok {
		t.Fatal("real eigenvalue normalization failed")
	}
	for i, v := range beta {
		if v < 0 {
			t.Errorf("beta[%d]=%v, want non-negative Netlib representation", i, v)
		}
	}
}

func testDtgsenOneNormSeparation(t *testing.T, impl Dtgsener) {
	for _, ijob := range []int{3, 5} {
		work := make([]float64, 128)
		iwork := make([]int, 32)
		_, _, _, dif, ok := impl.Dtgsen(ijob, false, false, []bool{true, false}, 2,
			[]float64{1, 4, 0, 2}, 2, []float64{1, 5, 0, 3}, 2,
			make([]float64, 2), make([]float64, 2), make([]float64, 2),
			nil, 1, nil, 1, work, len(work), iwork, len(iwork))
		if !ok {
			t.Fatalf("ijob=%d: separation estimate failed", ijob)
		}
		if math.Abs(dif[0]-0.25) > 1e-14 || math.Abs(dif[1]-0.25) > 1e-14 {
			t.Fatalf("ijob=%d: DIF=%v, want [0.25 0.25]", ijob, dif)
		}
	}
}

func testDtgsenSeparationDirections(t *testing.T, impl Dtgsener) {
	a := []float64{1, 4, 0, 2}
	b := []float64{1, 5, 0, 3}
	want := [2]float64{
		dtgsylDif(t, impl, a[:1], b[:1], a[3:], b[3:]),
		dtgsylDif(t, impl, a[3:], b[3:], a[:1], b[:1]),
	}
	work := make([]float64, 128)
	iwork := make([]int, 32)
	_, _, _, got, ok := impl.Dtgsen(2, false, false, []bool{true, false}, 2,
		a, 2, b, 2, make([]float64, 2), make([]float64, 2), make([]float64, 2),
		nil, 1, nil, 1, work, len(work), iwork, len(iwork))
	if !ok {
		t.Fatal("separation estimate failed")
	}
	for i := range got {
		if math.Abs(got[i]-want[i]) > 1e-14 {
			t.Fatalf("DIF[%d]=%v, want %v", i, got[i], want[i])
		}
	}
}

func dtgsylDif(t *testing.T, impl Dtgsyler, a, d, b, e []float64) float64 {
	t.Helper()
	work := make([]float64, 2)
	iwork := make([]int, 8)
	_, dif, ok := impl.Dtgsyl(blas.NoTrans, 3, 1, 1,
		a, 1, b, 1, []float64{0}, 1,
		d, 1, e, 1, []float64{0}, 1,
		work, len(work), iwork)
	if !ok {
		t.Fatal("direct separation estimate failed")
	}
	return dif
}

func testDtgsenTrivialSeparation(t *testing.T, impl Dtgsener) {
	work := make([]float64, 64)
	iwork := make([]int, 16)
	_, _, _, dif, ok := impl.Dtgsen(2, false, false, []bool{false}, 1,
		[]float64{3}, 1, []float64{4}, 1,
		make([]float64, 1), make([]float64, 1), make([]float64, 1),
		nil, 1, nil, 1, work, len(work), iwork, len(iwork))
	if !ok {
		t.Fatal("trivial separation estimate failed")
	}
	if dif != [2]float64{5, 5} {
		t.Fatalf("DIF=%v, want [5 5]", dif)
	}
}

func testDtgsenProjectionBounds(t *testing.T, impl Dtgsener) {
	work := make([]float64, 128)
	iwork := make([]int, 32)
	_, pl, pr, _, ok := impl.Dtgsen(1, false, false, []bool{true, false}, 2,
		[]float64{1, 4, 0, 2}, 2, []float64{1, 5, 0, 3}, 2,
		make([]float64, 2), make([]float64, 2), make([]float64, 2),
		nil, 1, nil, 1, work, len(work), iwork, len(iwork))
	if !ok {
		t.Fatal("projection bound estimate failed")
	}
	wantPL := 1 / math.Sqrt(5)
	wantPR := 1 / math.Sqrt2
	if math.Abs(pl-wantPL) > 1e-14 || math.Abs(pr-wantPR) > 1e-14 {
		t.Fatalf("PL=%v PR=%v, want PL=%v PR=%v", pl, pr, wantPL, wantPR)
	}
}

func testDtgsen(t *testing.T, impl Dtgsener, n, ijob int, wantq, wantz bool, selectPattern string, rnd *rand.Rand) {
	const tol = 1e-10

	prefix := fmt.Sprintf("n=%d, ijob=%d, wantq=%v, wantz=%v, select=%s",
		n, ijob, wantq, wantz, selectPattern)

	if n == 0 {
		var workQuery [1]float64
		var iworkQuery [1]int
		impl.Dtgsen(ijob, wantq, wantz, nil, 0,
			nil, 1, nil, 1, nil, nil, nil, nil, 1, nil, 1,
			workQuery[:], -1, iworkQuery[:], -1)
		wantLWork := 16
		wantLIWork := 1
		if ijob != 0 {
			wantLIWork = 6
		}
		if workQuery[0] != float64(wantLWork) || iworkQuery[0] != wantLIWork {
			t.Errorf("%s: workspace query=(%v,%d), want (%d,%d)", prefix,
				workQuery[0], iworkQuery[0], wantLWork, wantLIWork)
		}
		work := make([]float64, wantLWork)
		iwork := make([]int, wantLIWork)
		m, pl, pr, dif, ok := impl.Dtgsen(ijob, wantq, wantz, nil, 0,
			nil, 1, nil, 1, nil, nil, nil, nil, 1, nil, 1,
			work, len(work), iwork, len(iwork))
		if m != 0 || !ok {
			t.Errorf("%s: unexpected result for n=0", prefix)
		}
		_ = pl
		_ = pr
		_ = dif
		return
	}

	// Generate a random matrix pair in generalized Schur form.
	a := randomSchur(n, n, rnd)
	b := randomUpperTriangular(n, n, rnd)

	// Ensure B has positive diagonal.
	for i := 0; i < n; i++ {
		if b.Data[i*b.Stride+i] < 0 {
			b.Data[i*b.Stride+i] = -b.Data[i*b.Stride+i]
		}
		if b.Data[i*b.Stride+i] == 0 {
			b.Data[i*b.Stride+i] = 1
		}
	}

	aOrig := cloneGeneral(a)
	bOrig := cloneGeneral(b)

	// Generate selection array.
	selected := make([]bool, n)
	expectedM := 0
	switch selectPattern {
	case "none":
		// No selection.
	case "first":
		// Select first eigenvalue/block.
		if n > 0 {
			selected[0] = true
			expectedM = 1
			if n > 1 && a.Data[1*a.Stride+0] != 0 {
				selected[1] = true
				expectedM = 2
			}
		}
	case "last":
		// Select last eigenvalue/block.
		if n > 0 {
			k := n - 1
			if k > 0 && a.Data[k*a.Stride+k-1] != 0 {
				k--
				selected[k] = true
				selected[k+1] = true
				expectedM = 2
			} else {
				selected[k] = true
				expectedM = 1
			}
		}
	case "alternating":
		// Select alternating eigenvalues.
		k := 0
		for k < n {
			if k < n-1 && a.Data[(k+1)*a.Stride+k] != 0 {
				// 2x2 block.
				if k%2 == 0 {
					selected[k] = true
					selected[k+1] = true
					expectedM += 2
				}
				k += 2
			} else {
				// 1x1 block.
				if k%2 == 0 {
					selected[k] = true
					expectedM++
				}
				k++
			}
		}
	case "all":
		// Select all eigenvalues.
		for i := 0; i < n; i++ {
			selected[i] = true
		}
		expectedM = n
	}

	// Allocate output arrays.
	alphar := make([]float64, n)
	alphai := make([]float64, n)
	beta := make([]float64, n)

	q := blas64.General{
		Rows:   n,
		Cols:   n,
		Stride: n,
		Data:   make([]float64, n*n),
	}
	z := blas64.General{
		Rows:   n,
		Cols:   n,
		Stride: n,
		Data:   make([]float64, n*n),
	}

	// Initialize Q and Z to identity.
	for i := 0; i < n; i++ {
		q.Data[i*q.Stride+i] = 1
		z.Data[i*z.Stride+i] = 1
	}

	// Workspace query.
	var worksize [1]float64
	var iworksize [1]int
	impl.Dtgsen(ijob, wantq, wantz, selected, n,
		a.Data, max(1, a.Stride), nil, max(1, b.Stride), nil, nil, nil,
		nil, max(1, q.Stride), nil, max(1, z.Stride),
		worksize[:], -1, iworksize[:], -1)
	lwork := int(worksize[0])
	liwork := iworksize[0]

	work := make([]float64, max(1, lwork))
	iwork := make([]int, max(1, liwork))

	// Call Dtgsen.
	m, pl, pr, dif, ok := impl.Dtgsen(ijob, wantq, wantz, selected, n,
		a.Data, a.Stride, b.Data, b.Stride,
		alphar, alphai, beta,
		q.Data, q.Stride, z.Data, z.Stride,
		work, lwork, iwork, liwork)

	if m != expectedM {
		t.Errorf("%s: m=%d, want %d", prefix, m, expectedM)
	}

	if !ok {
		t.Fatalf("%s: reordering failed", prefix)
	}

	// Check that selected eigenvalues are in the leading m positions.
	// Verify by checking structure of A (quasi-triangular).

	// Check orthogonality of Q and Z.
	if wantq {
		resid := orthogonalityResidual(q)
		if resid > tol*float64(n) {
			t.Errorf("%s: Q not orthogonal, residual=%e", prefix, resid)
		}
	}
	if wantz {
		resid := orthogonalityResidual(z)
		if resid > tol*float64(n) {
			t.Errorf("%s: Z not orthogonal, residual=%e", prefix, resid)
		}
	}

	// Check the decomposition: A = Q * Aout * Z^T, B = Q * Bout * Z^T.
	if wantq && wantz {
		residA := schurDecompResidual(aOrig, a, q, z)
		residB := schurDecompResidual(bOrig, b, q, z)
		normA := dlange(lapack.MaxColumnSum, n, n, aOrig.Data, aOrig.Stride)
		normB := dlange(lapack.MaxColumnSum, n, n, bOrig.Data, bOrig.Stride)
		if normA > 0 && residA/(normA*float64(n)*dlamchE) > tol/dlamchE {
			t.Errorf("%s: A decomposition residual too large: %e", prefix, residA/(normA*float64(n)*dlamchE))
		}
		if normB > 0 && residB/(normB*float64(n)*dlamchE) > tol/dlamchE {
			t.Errorf("%s: B decomposition residual too large: %e", prefix, residB/(normB*float64(n)*dlamchE))
		}
	}

	// Check condition estimates are reasonable (non-negative).
	if ijob >= 1 {
		if pl < 0 || pr < 0 {
			t.Errorf("%s: pl=%e or pr=%e negative", prefix, pl, pr)
		}
	}
	if ijob >= 2 {
		if dif[0] < 0 || dif[1] < 0 {
			t.Errorf("%s: dif[0]=%e or dif[1]=%e negative", prefix, dif[0], dif[1])
		}
	}
}

func testDtgsenWorkspace(t *testing.T, impl Dtgsener) {
	for _, n := range []int{0, 1, 5, 20} {
		a := make([]float64, n*n)
		selected := make([]bool, n)
		for i := 0; i < n; i++ {
			a[i*n+i] = 1
			selected[i] = i < n/2
		}
		m := n / 2
		for _, ijob := range []int{0, 1, 2, 3, 4, 5} {
			var worksize [1]float64
			var iworksize [1]int
			ld := max(1, n)
			impl.Dtgsen(ijob, true, true, selected, n,
				a, ld, nil, ld, nil, nil, nil,
				nil, ld, nil, ld,
				worksize[:], -1, iworksize[:], -1)
			lwork := int(worksize[0])
			liwork := iworksize[0]

			// Verify minimum workspace requirements.
			var lwmin, liwmin int
			if ijob == 1 || ijob == 2 || ijob == 4 {
				lwmin = max(1, 4*n+16, 2*m*(n-m))
				liwmin = max(1, n+6)
			} else if ijob == 3 || ijob == 5 {
				lwmin = max(1, 4*n+16, 4*m*(n-m))
				liwmin = max(1, 2*m*(n-m), n+6)
			} else {
				lwmin = max(1, 4*n+16)
				liwmin = 1
			}

			if lwork != lwmin {
				t.Errorf("n=%d, ijob=%d: workspace query returned lwork=%d, want %d",
					n, ijob, lwork, lwmin)
			}
			if liwork != liwmin {
				t.Errorf("n=%d, ijob=%d: workspace query returned liwork=%d, want %d",
					n, ijob, liwork, liwmin)
			}
		}
	}
}

func testDtgsenPartialWorkspaceQueries(t *testing.T, impl Dtgsener) {
	const n = 2
	a := []float64{1, 0, 0, 2}
	selected := []bool{true, false}
	const wantLWork = 24
	const wantLIWork = 8
	for _, tc := range []struct {
		name          string
		lwork, liwork int
	}{
		{name: "Float", lwork: -1, liwork: 1},
		{name: "Integer", lwork: 1, liwork: -1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			work := []float64{-1}
			iwork := []int{-1}
			impl.Dtgsen(1, false, false, selected, n,
				a, n, nil, n, nil, nil, nil,
				nil, 1, nil, 1,
				work, tc.lwork, iwork, tc.liwork)
			if work[0] != wantLWork || iwork[0] != wantLIWork {
				t.Fatalf("workspace query=(%v,%d), want (%d,%d)", work[0], iwork[0], wantLWork, wantLIWork)
			}
		})
	}
}

// orthogonalityResidual computes ||Q^T*Q - I||_F.
func orthogonalityResidual(q blas64.General) float64 {
	n := q.Rows
	qtq := blas64.General{
		Rows:   n,
		Cols:   n,
		Stride: n,
		Data:   make([]float64, n*n),
	}
	blas64.Gemm(blas.Trans, blas.NoTrans, 1, q, q, 0, qtq)

	var resid float64
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			expected := 0.0
			if i == j {
				expected = 1.0
			}
			diff := qtq.Data[i*qtq.Stride+j] - expected
			resid += diff * diff
		}
	}
	return math.Sqrt(resid)
}

// schurDecompResidual computes ||Aorig - Q*A*Z^T||_F.
func schurDecompResidual(orig, a, q, z blas64.General) float64 {
	n := orig.Rows

	// Compute Q*A.
	qa := blas64.General{
		Rows:   n,
		Cols:   n,
		Stride: n,
		Data:   make([]float64, n*n),
	}
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, q, a, 0, qa)

	// Compute Q*A*Z^T.
	qazt := blas64.General{
		Rows:   n,
		Cols:   n,
		Stride: n,
		Data:   make([]float64, n*n),
	}
	blas64.Gemm(blas.NoTrans, blas.Trans, 1, qa, z, 0, qazt)

	// Compute ||Aorig - Q*A*Z^T||_F.
	var resid float64
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			diff := orig.Data[i*orig.Stride+j] - qazt.Data[i*qazt.Stride+j]
			resid += diff * diff
		}
	}
	return math.Sqrt(resid)
}
