// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package gonum

import (
	"fmt"
	"math"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/blas"
)

type ttFloat interface{ float32 | float64 }
type ttKernel[T ttFloat] func(int, int, int, []T, int, []T, int, []T, int, T) bool
type ttScalar[T ttFloat] func(int, int, int, []T, int, []T, int, []T, int, T)

func TestDgemmTransTransBlocked(t *testing.T) {
	testGemmTransTransBlocked(t, dgemmSerialTransTransBlocked, dgemmSerialTransTrans, dgemmSIMDEqualBits)
}

func TestSgemmTransTransBlocked(t *testing.T) {
	testGemmTransTransBlocked(t, sgemmSerialTransTransBlocked, sgemmSerialTransTrans, sgemmSIMDEqualBits)
}

func testGemmTransTransBlocked[T ttFloat](t *testing.T, kernel ttKernel[T], scalar ttScalar[T], equal func([]T, []T) bool) {
	t.Helper()
	for _, dims := range [][3]int{
		{1, 4, 16}, {2, 3, 16}, {2, 4, 15},
		{2, 4, 16}, {2, 5, 17}, {3, 4, 17}, {3, 5, 16},
		{4, 7, 17}, {5, 8, 31},
	} {
		m, n, k := dims[0], dims[1], dims[2]
		t.Run(fmt.Sprintf("m=%d/n=%d/k=%d", m, n, k), func(t *testing.T) {
			const off = 3
			lda, ldb, ldc := m+3, k+2, n+4
			aStore := ttData[T](off + k*lda + off)
			bStore := ttData[T](off + n*ldb + off)
			cStore := ttData[T](off + m*ldc + off)
			a, b, c := aStore[off:], bStore[off:], cStore[off:]
			want := slices.Clone(cStore)
			scalar(m, n, k, a, lda, b, ldb, want[off:], ldc, T(-0.75))
			aOrig, bOrig, cOrig := slices.Clone(aStore), slices.Clone(bStore), slices.Clone(cStore)
			accepted := kernel(m, n, k, a, lda, b, ldb, c, ldc, T(-0.75))
			wantAccepted := m >= 2 && n >= 4 && k >= 16
			if accepted != wantAccepted {
				t.Fatalf("accepted=%t want %t", accepted, wantAccepted)
			}
			if !equal(aStore, aOrig) || !equal(bStore, bOrig) {
				t.Fatal("input or guard changed")
			}
			if accepted {
				if !equal(cStore, want) {
					t.Fatal("result differs from scalar TT order")
				}
			} else if !equal(cStore, cOrig) {
				t.Fatal("rejected call changed C")
			} else {
				scalar(m, n, k, a, lda, b, ldb, c, ldc, T(-0.75))
				if !equal(cStore, want) {
					t.Fatal("scalar fallback differs after clean rejection")
				}
			}
		})
	}
}

func TestDgemmTransTransBlockedExceptional(t *testing.T) {
	testGemmTransTransExceptional(t, dgemmSerialTransTransBlocked, dgemmSerialTransTrans, dgemmSIMDEqualBits, math.MaxFloat64)
}

func TestSgemmTransTransBlockedExceptional(t *testing.T) {
	testGemmTransTransExceptional(t, sgemmSerialTransTransBlocked, sgemmSerialTransTrans, sgemmSIMDEqualBits, math.MaxFloat32)
}

func testGemmTransTransExceptional[T ttFloat](t *testing.T, kernel ttKernel[T], scalar ttScalar[T], equal func([]T, []T) bool, maxFinite T) {
	t.Helper()
	const m, n, k = 3, 5, 17
	lda, ldb, ldc := m+2, k+3, n+2
	a, b, got := make([]T, k*lda), make([]T, n*ldb), make([]T, m*ldc)
	for i := range a {
		a[i] = T((i%7)-3) / 8
	}
	for i := range b {
		b[i] = T((i%5)-2) / 4
	}
	for i := range got {
		got[i] = T((i%9)-4) / 8
	}
	a[0], a[lda], a[2*lda], a[3*lda] = 0, T(math.Inf(1)), T(math.MaxFloat32), T(-math.MaxFloat32)
	b[0], b[ldb], b[2*ldb], b[3*ldb] = T(math.NaN()), T(math.Inf(-1)), 1, 1
	want := slices.Clone(got)
	scalar(m, n, k, a, lda, b, ldb, want, ldc, T(-0.75))
	if !kernel(m, n, k, a, lda, b, ldb, got, ldc, T(-0.75)) {
		t.Fatal("eligible exceptional case rejected")
	}
	if !equal(got, want) {
		t.Fatal("exceptional classification or accumulation order differs from scalar TT")
	}
	for _, tc := range []struct {
		name  string
		terms [3]T
		bval  T
		c0    T
	}{
		{"SequentialOverflow", [3]T{maxFinite, maxFinite, -maxFinite}, 1, 0},
		{"FiniteCancellation", [3]T{maxFinite, -maxFinite, maxFinite}, 1, 0},
		{"ZeroTimesNaN", [3]T{}, T(math.NaN()), T(math.Copysign(0, -1))},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a, b := make([]T, k*lda), make([]T, n*ldb)
			got := make([]T, m*ldc)
			got[0] = tc.c0
			for l := 0; l < k; l++ {
				b[l] = tc.bval
			}
			for l, v := range tc.terms {
				a[l*lda] = v
			}
			want := slices.Clone(got)
			scalar(m, n, k, a, lda, b, ldb, want, ldc, 1)
			if !kernel(m, n, k, a, lda, b, ldb, got, ldc, 1) {
				t.Fatal("eligible numerical case rejected")
			}
			if !equal(got, want) {
				t.Fatal("exceptional classification, signed zero, or accumulation order differs")
			}
		})
	}
}

func TestDgemmTransTransBlockedAliasing(t *testing.T) {
	testGemmTransTransAliasing(t, dgemmSerialTransTransBlocked, dgemmSerialTransTrans, dgemmSIMDEqualBits)
}

func TestSgemmTransTransBlockedAliasing(t *testing.T) {
	testGemmTransTransAliasing(t, sgemmSerialTransTransBlocked, sgemmSerialTransTrans, sgemmSIMDEqualBits)
}

func testGemmTransTransAliasing[T ttFloat](t *testing.T, kernel ttKernel[T], scalar ttScalar[T], equal func([]T, []T) bool) {
	t.Helper()
	const m, n, k, stride = 3, 5, 17, 24
	t.Run("shared-backing-disjoint", func(t *testing.T) {
		shared := ttData[T](k * stride)
		b := ttData[T](n * stride)
		c := shared[m:]
		before := slices.Clone(shared)
		want := slices.Clone(shared)
		aRef, bRef := slices.Clone(shared), slices.Clone(b)
		scalar(m, n, k, aRef, stride, bRef, stride, want[m:], stride, T(-0.75))
		if !kernel(m, n, k, shared, stride, b, stride, c, stride, T(-0.75)) {
			t.Fatal("rejected disjoint active regions")
		}
		for l := 0; l < k; l++ {
			for i := 0; i < m; i++ {
				if shared[l*stride+i] != before[l*stride+i] {
					t.Fatal("read-only A changed")
				}
			}
		}
		if !equal(shared, want) {
			t.Fatal("shared-backing result differs from scalar TT order")
		}
	})
	for _, tc := range []struct {
		name       string
		aoff, boff int
	}{
		{"C-overlaps-A", 0, 256},
		{"C-overlaps-B", 256, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			store := ttData[T](640)
			a, b, c := store[tc.aoff:], store[tc.boff:], store[1:]
			before := slices.Clone(store)
			if kernel(m, n, k, a, stride, b, stride, c, stride, T(-0.75)) {
				t.Fatal("accepted active overlap")
			}
			if !equal(store, before) {
				t.Fatal("rejected call changed storage")
			}
		})
	}
}

func TestGemmTransTransBlockedPublic(t *testing.T) {
	testGemmTransTransPublic(t, Implementation{}.Dgemm, dgemmSerialTransTrans, dgemmSIMDEqualBits)
	testGemmTransTransPublic(t, Implementation{}.Sgemm, sgemmSerialTransTrans, sgemmSIMDEqualBits)
}

func testGemmTransTransPublic[T ttFloat](t *testing.T, gemm zeroGemmFunc[T], scalar ttScalar[T], equal func([]T, []T) bool) {
	t.Helper()
	const m, n, k = 3, 5, 17
	lda, ldb, ldc := m+2, k+3, n+2
	a, b, c := ttData[T](k*lda), ttData[T](n*ldb), ttData[T](m*ldc)
	want := slices.Clone(c)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			want[i*ldc+j] *= T(-0.5)
		}
	}
	scalar(m, n, k, a, lda, b, ldb, want, ldc, T(-0.75))
	gemm(blas.Trans, blas.ConjTrans, m, n, k, T(-0.75), a, lda, b, ldb, T(-0.5), c, ldc)
	if !equal(c, want) {
		t.Fatal("public TT result differs from scaled scalar result")
	}
}

func ttData[T ttFloat](n int) []T {
	x := make([]T, n)
	for i := range x {
		x[i] = T((i%17)-8) / 16
	}
	return x
}
