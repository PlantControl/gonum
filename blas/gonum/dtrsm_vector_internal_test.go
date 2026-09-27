// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"fmt"
	"math"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/blas"
)

func TestDtrsmLeftVector(t *testing.T) {
	for _, shape := range trsmVectorShapes {
		for _, ul := range []blas.Uplo{blas.Upper, blas.Lower} {
			for _, trans := range []blas.Transpose{blas.NoTrans, blas.Trans, blas.ConjTrans} {
				for _, diag := range []blas.Diag{blas.Unit, blas.NonUnit} {
					for _, alpha := range []float64{0, 1, -0.75} {
						name := fmt.Sprintf("m=%d/ldb=%d/%c/%c/%c/alpha=%g", shape.m, shape.ldb, ul, trans, diag, alpha)
						t.Run(name, func(t *testing.T) {
							testDtrsmLeftVector(t, shape.m, shape.ldb, ul, trans, diag, alpha)
						})
					}
				}
			}
		}
	}
}

var trsmVectorShapes = []struct {
	m, ldb int
}{
	{1, 1}, {2, 3}, {63, 1}, {64, 3}, {65, 1}, {128, 3}, {129, 1}, {256, 3},
}

func testDtrsmLeftVector(t *testing.T, m, ldb int, ul blas.Uplo, trans blas.Transpose, diag blas.Diag, alpha float64) {
	t.Helper()
	lda := m + 3
	a := makeDtrsmCandidateA(m, lda, ul, diag)
	for i := 0; i < m; i++ {
		if diag == blas.Unit {
			a[i*lda+i] = math.NaN()
		}
		for j := 0; j < m; j++ {
			if ul == blas.Upper && j < i || ul == blas.Lower && j > i {
				a[i*lda+j] = math.NaN()
			}
		}
	}
	aOrig := slices.Clone(a)
	b := make([]float64, m*ldb)
	for i := range b {
		b[i] = math.Float64frombits(0x3ff0000000000000 + uint64(i%31))
	}
	for i := 0; i < m; i++ {
		b[i*ldb] = float64(i%17-8) / 8
	}
	bOrig := slices.Clone(b)
	got := slices.Clone(b)
	Implementation{}.Dtrsm(blas.Left, ul, trans, diag, m, 1, alpha, a, lda, got, ldb)

	rldb := max(2, ldb)
	ref := make([]float64, m*rldb)
	for i := range ref {
		ref[i] = 9.87654321
	}
	for i := 0; i < m; i++ {
		ref[i*rldb] = bOrig[i*ldb]
		ref[i*rldb+1] = float64(i%13+1) / 16
	}
	Implementation{}.Dtrsm(blas.Left, ul, trans, diag, m, 2, alpha, a, lda, ref, rldb)
	for i := 0; i < m; i++ {
		if !dtrsmCandidateClose(got[i*ldb], ref[i*rldb]) {
			t.Fatalf("row %d: got %g want two-RHS reference %g", i, got[i*ldb], ref[i*rldb])
		}
		for j := 1; j < ldb; j++ {
			if math.Float64bits(got[i*ldb+j]) != math.Float64bits(bOrig[i*ldb+j]) {
				t.Fatalf("B padding [%d,%d] changed", i, j)
			}
		}
	}
	if !slices.EqualFunc(a, aOrig, func(x, y float64) bool { return math.Float64bits(x) == math.Float64bits(y) }) {
		t.Fatal("A changed")
	}
	if alpha != 0 {
		rhs := slices.Clone(bOrig)
		for i := 0; i < m; i++ {
			rhs[i*ldb] *= alpha
		}
		if residual := dtrsmCandidateResidual(ul, trans, diag, m, 1, a, lda, got, ldb, rhs); math.IsNaN(residual) || math.IsInf(residual, 0) || residual > 2e-11 {
			t.Fatalf("relative residual %g", residual)
		}
	}
}

func TestStrsmLeftVector(t *testing.T) {
	for _, shape := range trsmVectorShapes {
		for _, ul := range []blas.Uplo{blas.Upper, blas.Lower} {
			for _, trans := range []blas.Transpose{blas.NoTrans, blas.Trans, blas.ConjTrans} {
				for _, diag := range []blas.Diag{blas.Unit, blas.NonUnit} {
					for _, alpha := range []float32{0, 1, -0.75} {
						name := fmt.Sprintf("m=%d/ldb=%d/%c/%c/%c/alpha=%g", shape.m, shape.ldb, ul, trans, diag, alpha)
						t.Run(name, func(t *testing.T) {
							testStrsmLeftVector(t, shape.m, shape.ldb, ul, trans, diag, alpha)
						})
					}
				}
			}
		}
	}
}

func testStrsmLeftVector(t *testing.T, m, ldb int, ul blas.Uplo, trans blas.Transpose, diag blas.Diag, alpha float32) {
	t.Helper()
	lda := m + 3
	a64 := makeDtrsmCandidateA(m, lda, ul, diag)
	a := make([]float32, len(a64))
	for i, v := range a64 {
		a[i] = float32(v)
	}
	for i := 0; i < m; i++ {
		if diag == blas.Unit {
			a[i*lda+i] = float32(math.NaN())
		}
		for j := 0; j < m; j++ {
			if ul == blas.Upper && j < i || ul == blas.Lower && j > i {
				a[i*lda+j] = float32(math.NaN())
			}
		}
	}
	aOrig := slices.Clone(a)
	b := make([]float32, m*ldb)
	for i := range b {
		b[i] = math.Float32frombits(0x3f800000 + uint32(i%31))
	}
	for i := 0; i < m; i++ {
		b[i*ldb] = float32(i%17-8) / 8
	}
	bOrig := slices.Clone(b)
	got := slices.Clone(b)
	Implementation{}.Strsm(blas.Left, ul, trans, diag, m, 1, alpha, a, lda, got, ldb)

	rldb := max(2, ldb)
	ref := make([]float32, m*rldb)
	for i := range ref {
		ref[i] = 9.876543
	}
	for i := 0; i < m; i++ {
		ref[i*rldb] = bOrig[i*ldb]
		ref[i*rldb+1] = float32(i%13+1) / 16
	}
	Implementation{}.Strsm(blas.Left, ul, trans, diag, m, 2, alpha, a, lda, ref, rldb)
	tol := float32(8 * 0x1p-23 * float32(m))
	for i := 0; i < m; i++ {
		if !strsmBlockedClose(got[i*ldb], ref[i*rldb], tol) {
			t.Fatalf("row %d: got %g want two-RHS reference %g", i, got[i*ldb], ref[i*rldb])
		}
		for j := 1; j < ldb; j++ {
			if math.Float32bits(got[i*ldb+j]) != math.Float32bits(bOrig[i*ldb+j]) {
				t.Fatalf("B padding [%d,%d] changed", i, j)
			}
		}
	}
	if !slices.EqualFunc(a, aOrig, func(x, y float32) bool { return math.Float32bits(x) == math.Float32bits(y) }) {
		t.Fatal("A changed")
	}
	if alpha != 0 {
		rhs := slices.Clone(bOrig)
		for i := 0; i < m; i++ {
			rhs[i*ldb] *= alpha
		}
		if residual := strsmBlockedResidual(ul, trans, diag, m, 1, a, lda, got, ldb, rhs); math.IsNaN(residual) || math.IsInf(residual, 0) || residual > float64(tol) {
			t.Fatalf("relative residual %g", residual)
		}
	}
}

func TestDtrsmLeftVectorExceptional(t *testing.T) {
	testDtrsmLeftVectorBackwardOverflow(t)
	for _, rhs := range []float64{1e-300, 1e300} {
		b := []float64{rhs}
		Implementation{}.Dtrsm(blas.Left, blas.Upper, blas.NoTrans, blas.NonUnit, 1, 1, 1, []float64{2}, 1, b, 1)
		if b[0] != rhs*0.5 {
			t.Fatalf("scale %g: got %g want %g", rhs, b[0], rhs*0.5)
		}
	}
	testDtrsmLeftVectorNonfiniteZero(t)
}

func testDtrsmLeftVectorBackwardOverflow(t *testing.T) {
	const m = 128
	for _, tc := range []struct {
		name  string
		ul    blas.Uplo
		trans blas.Transpose
		set   func([]float64)
		rows  []int
	}{
		{name: "upper-no-trans", ul: blas.Upper, trans: blas.NoTrans, set: func(a []float64) { a[1], a[64] = 1, -1 }, rows: []int{0, 1, 64}},
		{name: "lower-trans", ul: blas.Lower, trans: blas.Trans, set: func(a []float64) { a[64*m], a[65*m] = -1, 1 }, rows: []int{0, 64, 65}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a, b := make([]float64, m*m), make([]float64, m)
			tc.set(a)
			for _, row := range tc.rows {
				b[row] = 1e308
			}
			Implementation{}.Dtrsm(blas.Left, tc.ul, tc.trans, blas.Unit, m, 1, 1, a, m, b, 1)
			if b[0] != 1e308 {
				t.Fatalf("got %g want %g", b[0], 1e308)
			}
		})
	}
}

func testDtrsmLeftVectorNonfiniteZero(t *testing.T) {
	const m = 4
	for _, tc := range []struct {
		ul       blas.Uplo
		tr       blas.Transpose
		s, d, ai int
	}{
		{ul: blas.Lower, tr: blas.NoTrans, s: 0, d: 3, ai: 3 * m},
		{ul: blas.Upper, tr: blas.Trans, s: 0, d: 3, ai: 3},
		{ul: blas.Upper, tr: blas.NoTrans, s: 3, d: 0, ai: 3},
		{ul: blas.Lower, tr: blas.Trans, s: 3, d: 0, ai: 3 * m},
	} {
		for _, source := range []float64{math.NaN(), math.Inf(1)} {
			for _, coefficient := range []float64{0, math.Copysign(0, -1)} {
				a, b := make([]float64, m*m), make([]float64, m)
				b[tc.s], b[tc.d] = source, 1
				a[tc.ai] = coefficient
				Implementation{}.Dtrsm(blas.Left, tc.ul, tc.tr, blas.Unit, m, 1, 1, a, m, b, 1)
				if b[tc.d] != 1 {
					t.Fatalf("%c/%c/source=%g/coefficient=%g: destination=%g", tc.ul, tc.tr, source, coefficient, b[tc.d])
				}
			}
		}
	}
}

func TestStrsmLeftVectorExceptional(t *testing.T) {
	for _, rhs := range []float32{1e-30, 1e30} {
		b := []float32{rhs}
		Implementation{}.Strsm(blas.Left, blas.Upper, blas.NoTrans, blas.NonUnit, 1, 1, 1, []float32{2}, 1, b, 1)
		if b[0] != rhs*0.5 {
			t.Fatalf("scale %g: got %g want %g", rhs, b[0], rhs*0.5)
		}
	}
	const m = 128
	for _, tc := range []struct {
		name  string
		ul    blas.Uplo
		trans blas.Transpose
		set   func([]float32)
		rows  []int
	}{
		{name: "upper-no-trans", ul: blas.Upper, trans: blas.NoTrans, set: func(a []float32) { a[1], a[64] = 1, -1 }, rows: []int{0, 1, 64}},
		{name: "lower-trans", ul: blas.Lower, trans: blas.Trans, set: func(a []float32) { a[64*m], a[65*m] = -1, 1 }, rows: []int{0, 64, 65}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a, b := make([]float32, m*m), make([]float32, m)
			tc.set(a)
			for _, row := range tc.rows {
				b[row] = 2e38
			}
			Implementation{}.Strsm(blas.Left, tc.ul, tc.trans, blas.Unit, m, 1, 1, a, m, b, 1)
			if b[0] != 2e38 {
				t.Fatalf("got %g want %g", b[0], float32(2e38))
			}
		})
	}
	const small = 4
	for _, tc := range []struct {
		ul       blas.Uplo
		tr       blas.Transpose
		s, d, ai int
	}{
		{ul: blas.Lower, tr: blas.NoTrans, s: 0, d: 3, ai: 3 * small},
		{ul: blas.Upper, tr: blas.Trans, s: 0, d: 3, ai: 3},
		{ul: blas.Upper, tr: blas.NoTrans, s: 3, d: 0, ai: 3},
		{ul: blas.Lower, tr: blas.Trans, s: 3, d: 0, ai: 3 * small},
	} {
		for _, source := range []float32{float32(math.NaN()), float32(math.Inf(1))} {
			for _, coefficient := range []float32{0, float32(math.Copysign(0, -1))} {
				a, b := make([]float32, small*small), make([]float32, small)
				b[tc.s], b[tc.d] = source, 1
				a[tc.ai] = coefficient
				Implementation{}.Strsm(blas.Left, tc.ul, tc.tr, blas.Unit, small, 1, 1, a, small, b, 1)
				if b[tc.d] != 1 {
					t.Fatalf("%c/%c/source=%g/coefficient=%g: destination=%g", tc.ul, tc.tr, source, coefficient, b[tc.d])
				}
			}
		}
	}
}

func TestDtrsmLeftVectorSharedBacking(t *testing.T) {
	const m = 65
	const lda = m + 2
	for _, tc := range []struct {
		ul    blas.Uplo
		trans blas.Transpose
	}{
		{ul: blas.Lower, trans: blas.NoTrans},
		{ul: blas.Upper, trans: blas.Trans},
		{ul: blas.Upper, trans: blas.NoTrans},
		{ul: blas.Lower, trans: blas.Trans},
	} {
		t.Run(fmt.Sprintf("%c/%c", tc.ul, tc.trans), func(t *testing.T) {
			storage := make([]float64, m*lda)
			for i := range storage {
				storage[i] = 9.87654321
			}
			for i := 0; i < m; i++ {
				storage[i*lda+i] = 2 + float64(i%7)/16
				for j := 0; j < m; j++ {
					if tc.ul == blas.Upper && j > i || tc.ul == blas.Lower && j < i {
						storage[i*lda+j] = float64((i+j)%11-5) / float64(16*m)
					}
				}
				storage[m+i*lda] = float64(i%17-8) / 8
			}
			orig := slices.Clone(storage)
			refA := slices.Clone(storage)
			refB := make([]float64, 2*m)
			for i := 0; i < m; i++ {
				refB[2*i], refB[2*i+1] = orig[m+i*lda], float64(i%13+1)/16
			}
			Implementation{}.Dtrsm(blas.Left, tc.ul, tc.trans, blas.NonUnit, m, 2, 1, refA, lda, refB, 2)
			Implementation{}.Dtrsm(blas.Left, tc.ul, tc.trans, blas.NonUnit, m, 1, 1, storage, lda, storage[m:], lda)
			for i := range storage {
				if i >= m && (i-m)%lda == 0 {
					row := (i - m) / lda
					if !dtrsmCandidateClose(storage[i], refB[2*row]) {
						t.Fatalf("row %d: got %g want %g", row, storage[i], refB[2*row])
					}
					continue
				}
				if math.Float64bits(storage[i]) != math.Float64bits(orig[i]) {
					t.Fatalf("storage index %d changed", i)
				}
			}
		})
	}
}

func TestStrsmLeftVectorSharedBacking(t *testing.T) {
	const m = 65
	const lda = m + 2
	for _, tc := range []struct {
		ul    blas.Uplo
		trans blas.Transpose
	}{
		{ul: blas.Lower, trans: blas.NoTrans},
		{ul: blas.Upper, trans: blas.Trans},
		{ul: blas.Upper, trans: blas.NoTrans},
		{ul: blas.Lower, trans: blas.Trans},
	} {
		t.Run(fmt.Sprintf("%c/%c", tc.ul, tc.trans), func(t *testing.T) {
			storage := make([]float32, m*lda)
			for i := range storage {
				storage[i] = 9.876543
			}
			for i := 0; i < m; i++ {
				storage[i*lda+i] = 2 + float32(i%7)/16
				for j := 0; j < m; j++ {
					if tc.ul == blas.Upper && j > i || tc.ul == blas.Lower && j < i {
						storage[i*lda+j] = float32((i+j)%11-5) / float32(16*m)
					}
				}
				storage[m+i*lda] = float32(i%17-8) / 8
			}
			orig := slices.Clone(storage)
			refA := slices.Clone(storage)
			refB := make([]float32, 2*m)
			for i := 0; i < m; i++ {
				refB[2*i], refB[2*i+1] = orig[m+i*lda], float32(i%13+1)/16
			}
			Implementation{}.Strsm(blas.Left, tc.ul, tc.trans, blas.NonUnit, m, 2, 1, refA, lda, refB, 2)
			Implementation{}.Strsm(blas.Left, tc.ul, tc.trans, blas.NonUnit, m, 1, 1, storage, lda, storage[m:], lda)
			tol := float32(8 * 0x1p-23 * m)
			for i := range storage {
				if i >= m && (i-m)%lda == 0 {
					row := (i - m) / lda
					if !strsmBlockedClose(storage[i], refB[2*row], tol) {
						t.Fatalf("row %d: got %g want %g", row, storage[i], refB[2*row])
					}
					continue
				}
				if math.Float32bits(storage[i]) != math.Float32bits(orig[i]) {
					t.Fatalf("storage index %d changed", i)
				}
			}
		})
	}
}
