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
	"plantcontrol.org/v1/gonum/lapack"
)

type Dtgsyler interface {
	Dtgsy2er
	Dtgsyl(trans blas.Transpose, ijob, m, n int,
		a []float64, lda int, b []float64, ldb int, c []float64, ldc int,
		d []float64, ldd int, e []float64, lde int, f []float64, ldf int,
		work []float64, lwork int, iwork []int) (scale, dif float64, ok bool)
}

func DtgsylTest(t *testing.T, impl Dtgsyler) {
	rnd := rand.New(rand.NewPCG(1, 1))

	for _, m := range []int{0, 1, 2, 3, 4, 5, 10, 20} {
		for _, n := range []int{0, 1, 2, 3, 4, 5, 10, 20} {
			for _, trans := range []blas.Transpose{blas.NoTrans, blas.Trans} {
				for _, ijob := range []int{0, 1, 2} {
					for _, extra := range []int{0, 3} {
						testDtgsyl(t, impl, m, n, trans, ijob, extra, rnd)
					}
				}
			}
		}
	}

	testDtgsylWorkspace(t, impl)
	testDtgsylDifIdentity(t, impl)
	testDtgsylScalingAcrossBlocks(t, impl)
	testDtgsylTransposedScalingAcrossBlocks(t, impl)
	testDtgsylTransIgnoresIjob(t, impl)
}

func testDtgsylTransIgnoresIjob(t *testing.T, impl Dtgsyler) {
	for _, ijob := range []int{-1, 5} {
		work := make([]float64, 1)
		impl.Dtgsyl(blas.Trans, ijob, 1, 1,
			nil, 1, nil, 1, nil, 1, nil, 1, nil, 1, nil, 1,
			work, -1, nil)
		if work[0] != 1 {
			t.Errorf("ijob=%d: transposed workspace query=%v, want 1", ijob, work[0])
		}

		c := []float64{1}
		f := []float64{1}
		_, _, ok := impl.Dtgsyl(blas.Trans, ijob, 1, 1,
			[]float64{2}, 1, []float64{3}, 1, c, 1,
			[]float64{5}, 1, []float64{7}, 1, f, 1,
			work, 1, make([]int, 8))
		if !ok {
			t.Errorf("ijob=%d: transposed solve reported coincident eigenvalues", ijob)
		}
	}
}

func testDtgsylTransposedScalingAcrossBlocks(t *testing.T, impl Dtgsyler) {
	const m, n = 3, 1
	a := []float64{3, 0, 0, 0, 4, 0, 0, 0, 5}
	d := []float64{7, 0, 0, 0, 9, 0, 0, 0, 11}
	b := []float64{1}
	e := []float64{2}
	cOrig := []float64{1e100, -2e307, 3e100}
	fOrig := []float64{-3e100, 2e307, -1e100}
	c := append([]float64(nil), cOrig...)
	f := append([]float64(nil), fOrig...)
	scale, _, ok := impl.Dtgsyl(blas.Trans, 0, m, n,
		a, m, b, n, c, n, d, m, e, n, f, n,
		make([]float64, 1), 1, make([]int, m+n+6))
	if !ok {
		t.Fatal("transposed scaled block solve reported coincident eigenvalues")
	}
	if scale >= 1 {
		t.Fatalf("scale=%v, want scaling for extreme transposed right-hand side", scale)
	}
	for i := range m {
		gotC := a[i*m+i]*c[i] + d[i*m+i]*f[i]
		gotF := c[i]*b[0] + f[i]*e[0]
		wantC := scale * cOrig[i]
		wantF := -scale * fOrig[i]
		tol := 1e-12 * math.Max(math.Abs(wantC), math.Abs(wantF))
		if math.Abs(gotC-wantC) > tol || math.Abs(gotF-wantF) > tol {
			t.Errorf("row %d residuals=(%g,%g), want (%g,%g)", i, gotC, gotF, wantC, wantF)
		}
	}
}

func testDtgsylScalingAcrossBlocks(t *testing.T, impl Dtgsyler) {
	const m, n = 3, 1
	a := []float64{
		3, 0, 0,
		0, 4, 0,
		0, 0, 5,
	}
	d := []float64{
		7, 0, 0,
		0, 9, 0,
		0, 0, 11,
	}
	b := []float64{1}
	e := []float64{2}
	cOrig := []float64{1e307, -2e307, 3e100}
	fOrig := []float64{-3e307, 2e307, -1e100}
	c := append([]float64(nil), cOrig...)
	f := append([]float64(nil), fOrig...)
	scale, _, ok := impl.Dtgsyl(blas.NoTrans, 0, m, n,
		a, m, b, n, c, n, d, m, e, n, f, n,
		make([]float64, 1), 1, make([]int, m+n+6))
	if !ok {
		t.Fatal("scaled block solve reported coincident eigenvalues")
	}
	if scale >= 1 {
		t.Fatalf("scale=%v, want scaling for extreme right-hand side", scale)
	}
	for i := range m {
		gotC := a[i*m+i]*c[i] - f[i]*b[0]
		gotF := d[i*m+i]*c[i] - f[i]*e[0]
		wantC := scale * cOrig[i]
		wantF := scale * fOrig[i]
		tol := 1e-12 * math.Max(math.Abs(wantC), math.Abs(wantF))
		if math.Abs(gotC-wantC) > tol || math.Abs(gotF-wantF) > tol {
			t.Errorf("row %d residuals=(%g,%g), want (%g,%g)", i, gotC, gotF, wantC, wantF)
		}
	}
}

func testDtgsylDifIdentity(t *testing.T, impl Dtgsyler) {
	for _, test := range []struct {
		ijob int
		want float64
	}{
		{ijob: 1, want: 2},
		{ijob: 2, want: 2},
		{ijob: 3, want: 2},
		{ijob: 4, want: 2},
	} {
		work := make([]float64, 2)
		iwork := make([]int, 8)
		_, dif, ok := impl.Dtgsyl(blas.NoTrans, test.ijob, 1, 1,
			[]float64{2}, 1, []float64{0}, 1, []float64{0}, 1,
			[]float64{0}, 1, []float64{-2}, 1, []float64{0}, 1,
			work, len(work), iwork)
		if !ok {
			t.Fatalf("ijob=%d: identity operator reported coincident eigenvalues", test.ijob)
		}
		if math.Abs(dif-test.want) > 1e-14 {
			t.Fatalf("ijob=%d: scaled identity operator DIF=%v, want %v", test.ijob, dif, test.want)
		}
	}
}

func testDtgsyl(t *testing.T, impl Dtgsyler, m, n int, trans blas.Transpose, ijob, extra int, rnd *rand.Rand) {
	const tol = 1e-10

	prefix := fmt.Sprintf("m=%v, n=%v, trans=%c, ijob=%d, extra=%v",
		m, n, trans, ijob, extra)

	// Generate random quasi-triangular matrices A (m×m) and upper triangular B (n×n).
	a := randomSchur(m, m+extra, rnd)
	b := randomUpperTriangular(n, n+extra, rnd)

	// Generate random upper triangular matrices D (m×m) and E (n×n).
	d := randomUpperTriangular(m, m+extra, rnd)
	e := randomUpperTriangular(n, n+extra, rnd)

	// Generate random RHS matrices C, F (m×n).
	c := randomGeneral(m, n, n+extra, rnd)
	f := randomGeneral(m, n, n+extra, rnd)
	cOrig := cloneGeneral(c)
	fOrig := cloneGeneral(f)

	// Keep originals for residual check.
	aOrig := cloneGeneral(a)
	bOrig := cloneGeneral(b)
	dOrig := cloneGeneral(d)
	eOrig := cloneGeneral(e)

	// Workspace query.
	var worksize [1]float64
	impl.Dtgsyl(trans, ijob, m, n,
		nil, max(1, a.Stride), nil, max(1, b.Stride), nil, max(1, c.Stride),
		nil, max(1, d.Stride), nil, max(1, e.Stride), nil, max(1, f.Stride),
		worksize[:], -1, nil)
	lwork := int(worksize[0])
	work := make([]float64, max(1, lwork))
	iwork := make([]int, m+n+6)

	// Call Dtgsyl.
	scale, dif, ok := impl.Dtgsyl(trans, ijob, m, n,
		a.Data, a.Stride, b.Data, b.Stride, c.Data, c.Stride,
		d.Data, d.Stride, e.Data, e.Stride, f.Data, f.Stride,
		work, lwork, iwork)

	// Check that A, B, D, E are not modified.
	if !generalEqual(a, aOrig) {
		t.Errorf("%v: A was modified", prefix)
	}
	if !generalEqual(b, bOrig) {
		t.Errorf("%v: B was modified", prefix)
	}
	if !generalEqual(d, dOrig) {
		t.Errorf("%v: D was modified", prefix)
	}
	if !generalEqual(e, eOrig) {
		t.Errorf("%v: E was modified", prefix)
	}

	// Quick return for empty matrices.
	if m == 0 || n == 0 {
		if scale != 1 {
			t.Errorf("%v: scale=%v, want 1 for empty matrix", prefix, scale)
		}
		return
	}

	// Verify the solution by computing the residual.
	resid := dtgsy2Residual(aOrig, bOrig, cOrig, dOrig, eOrig, fOrig, c, f, trans, scale)

	anorm := dlange(lapack.MaxColumnSum, m, m, aOrig.Data, aOrig.Stride)
	bnorm := dlange(lapack.MaxColumnSum, n, n, bOrig.Data, bOrig.Stride)
	dnorm := dlange(lapack.MaxColumnSum, m, m, dOrig.Data, dOrig.Stride)
	enorm := dlange(lapack.MaxColumnSum, n, n, eOrig.Data, eOrig.Stride)
	rnorm := dlange(lapack.MaxColumnSum, m, n, c.Data, c.Stride)
	lnorm := dlange(lapack.MaxColumnSum, m, n, f.Data, f.Stride)
	cnorm := dlange(lapack.MaxColumnSum, m, n, cOrig.Data, cOrig.Stride)
	fnorm := dlange(lapack.MaxColumnSum, m, n, fOrig.Data, fOrig.Stride)

	denom := (anorm+dnorm)*rnorm + (bnorm+enorm)*lnorm + cnorm + fnorm
	if denom == 0 {
		denom = 1
	}
	normRes := resid / (denom * float64(max(m, n)) * dlamchE)

	if normRes > tol/dlamchE {
		t.Errorf("%v: residual too large: %v, want <= %v (ok=%v, scale=%v)", prefix, normRes, tol/dlamchE, ok, scale)
	}

	// Check Dif estimate is non-negative.
	if ijob >= 1 && trans == blas.NoTrans && dif < 0 {
		t.Errorf("%v: dif=%v, want >= 0", prefix, dif)
	}
}

func testDtgsylWorkspace(t *testing.T, impl Dtgsyler) {
	t.Run("MissingQueryOutput", func(t *testing.T) {
		defer func() {
			if recover() == nil {
				t.Fatal("workspace query accepted a missing output slot")
			}
		}()
		impl.Dtgsyl(blas.NoTrans, 0, 1, 1,
			nil, 1, nil, 1, nil, 1, nil, 1, nil, 1, nil, 1,
			nil, -1, nil)
	})

	// Test workspace query.
	for _, m := range []int{1, 5, 10} {
		for _, n := range []int{1, 5, 10} {
			for _, ijob := range []int{0, 1, 2} {
				var worksize [1]float64
				impl.Dtgsyl(blas.NoTrans, ijob, m, n,
					nil, m, nil, n, nil, n,
					nil, m, nil, n, nil, n,
					worksize[:], -1, nil)
				lwork := int(worksize[0])

				var expectedMin int
				if ijob == 1 || ijob == 2 {
					expectedMin = 2 * m * n
				} else {
					expectedMin = 1
				}

				if lwork < expectedMin {
					t.Errorf("m=%d, n=%d, ijob=%d: workspace query returned %d, want >= %d",
						m, n, ijob, lwork, expectedMin)
				}
			}
		}
	}
}
