// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && arm64 && !safe && !noasm && !gccgo
// +build go1.27,goexperiment.simd,arm64,!safe,!noasm,!gccgo

package gonum

import (
	"fmt"
	"math"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/internal/asm/f64"
)

func TestDgemmNotTransTransSIMDBoundaries(t *testing.T) {
	ks := []int{15, 16, 17, 23, 24, 25, 31, 32, 33, 63, 64, 65, 127, 128, 129, 479, 480, 481}
	for _, k := range ks {
		for _, m := range []int{1, 3} {
			for _, n := range []int{1, 2, 3, 4} {
				for _, trans := range []blas.Transpose{blas.Trans, blas.ConjTrans} {
					for _, coeff := range [][2]float64{{1, 0}, {-0.75, 1}, {0.5, -0.5}} {
						name := fmt.Sprintf("m=%d/n=%d/k=%d/trans=%c/alpha=%g/beta=%g", m, n, k, trans, coeff[0], coeff[1])
						t.Run(name, func(t *testing.T) {
							testDgemmNotTransTransSIMD(t, trans, m, n, k, coeff[0], coeff[1])
						})
					}
				}
			}
		}
	}
}

func testDgemmNotTransTransSIMD(t *testing.T, trans blas.Transpose, m, n, k int, alpha, beta float64) {
	t.Helper()
	const guard = 4
	aoff, boff, coff := 1+k%4, 1+(k+1)%4, 1+(k+2)%4
	lda, ldb, ldc := k+3, k+5, n+4
	aStore := dgemmTransData(aoff + m*lda + guard)
	bStore := dgemmTransData(boff + n*ldb + guard)
	cStore := dgemmTransData(coff + m*ldc + guard)
	a, b, c := aStore[aoff:], bStore[boff:], cStore[coff:]
	aOrig, bOrig, cOrig := slices.Clone(aStore), slices.Clone(bStore), slices.Clone(cStore)
	want := slices.Clone(cStore)
	wantActive := want[coff:]
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			p := i*ldc + j
			cv := wantActive[p]
			if beta == 0 {
				cv = 0
			} else if beta != 1 {
				cv *= beta
			}
			wantActive[p] = cv + alpha*f64.DotUnitary(a[i*lda:i*lda+k], b[j*ldb:j*ldb+k])
		}
	}
	Implementation{}.Dgemm(blas.NoTrans, trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
	if !dgemmTransEqual(cStore, want) {
		for i := range cStore {
			if math.Float64bits(cStore[i]) != math.Float64bits(want[i]) {
				t.Fatalf("C[%d]: got %g (%#x) want %g (%#x)", i, cStore[i], math.Float64bits(cStore[i]), want[i], math.Float64bits(want[i]))
			}
		}
	}
	if !dgemmTransEqual(aStore, aOrig) || !dgemmTransEqual(bStore, bOrig) {
		t.Fatal("read-only input or guard changed")
	}
	for i := 0; i < m; i++ {
		for j := n; j < ldc; j++ {
			p := coff + i*ldc + j
			if math.Float64bits(cStore[p]) != math.Float64bits(cOrig[p]) {
				t.Fatalf("C padding changed at row %d column %d", i, j)
			}
		}
	}
}

func TestDgemmNotTransTransSIMDHelperGate(t *testing.T) {
	for _, tc := range []struct {
		name       string
		m, n, k    int
		wantAccept bool
	}{
		{"m0", 0, 2, 16, false},
		{"n1", 1, 1, 16, false},
		{"k15", 1, 2, 15, false},
		{"minimum", 1, 2, 16, true},
		{"odd-tail", 3, 3, 17, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			lda, ldb, ldc := max(1, tc.k+2), max(1, tc.k+3), max(1, tc.n+2)
			a := dgemmTransData(max(1, tc.m*lda))
			b := dgemmTransData(max(1, tc.n*ldb))
			c := dgemmTransData(max(1, tc.m*ldc))
			before := slices.Clone(c)
			accepted := dgemmSerialNotTransSIMD(tc.m, tc.n, tc.k, a, lda, b, ldb, c, ldc, -0.75)
			if accepted != tc.wantAccept {
				t.Fatalf("accepted=%t want %t", accepted, tc.wantAccept)
			}
			if !accepted && !dgemmTransEqual(c, before) {
				t.Fatal("rejected call mutated C")
			}
		})
	}
}

func TestDgemmNotTransTransSIMDOverlapFallback(t *testing.T) {
	const m, n, k, stride = 3, 3, 17, 24
	for _, tc := range []struct {
		name             string
		aoff, boff, coff int
	}{
		{"A", 0, 96, 4},
		{"B-pair", 128, 0, 4},
		{"B-odd-tail", 128, 0, 2 * stride},
	} {
		t.Run(tc.name, func(t *testing.T) {
			storage := dgemmTransData(256)
			a, b, c := storage[tc.aoff:], storage[tc.boff:], storage[tc.coff:]
			orig := slices.Clone(storage)
			if dgemmSerialNotTransSIMD(m, n, k, a, stride, b, stride, c, stride, -0.75) {
				t.Fatal("accepted overlapping active regions")
			}
			if !dgemmTransEqual(storage, orig) {
				t.Fatal("rejected call mutated shared storage")
			}
			want := slices.Clone(storage)
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					want[tc.coff+i*stride+j] += -0.75 * f64.DotUnitary(
						want[tc.aoff+i*stride:tc.aoff+i*stride+k],
						want[tc.boff+j*stride:tc.boff+j*stride+k],
					)
				}
			}
			dgemmSerial(false, true, m, n, k, a, stride, b, stride, c, stride, -0.75)
			if !dgemmTransEqual(storage, want) {
				t.Fatal("fallback changed original scalar alias order")
			}
		})
	}
}

func TestDgemmNotTransTransSIMDSharedBackingDisjoint(t *testing.T) {
	const m, n, k, stride = 3, 3, 17, 24
	for _, sharedWith := range []string{"A", "B"} {
		t.Run(sharedWith, func(t *testing.T) {
			shared := dgemmTransData(3*stride + 8)
			other := dgemmTransData(3 * stride)
			var a, b []float64
			if sharedWith == "A" {
				a, b = shared, other
			} else {
				a, b = other, shared
			}
			c := shared[k:]
			want := slices.Clone(shared)
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					want[k+i*stride+j] += -0.75 * f64.DotUnitary(
						slices.Clone(a[i*stride:i*stride+k]),
						slices.Clone(b[j*stride:j*stride+k]),
					)
				}
			}
			if !dgemmSerialNotTransSIMD(m, n, k, a, stride, b, stride, c, stride, -0.75) {
				t.Fatal("rejected disjoint active regions in shared backing storage")
			}
			if !dgemmTransEqual(shared, want) {
				t.Fatal("unexpected active result or padding mutation")
			}
		})
	}
}

func TestDgemmNotTransTransSIMDNumerical(t *testing.T) {
	for _, tc := range []struct {
		name  string
		k     int
		alpha float64
		fill  func(a, b, c []float64, lda, ldb, ldc int)
	}{
		{
			name: "lane-cancellation", k: 17, alpha: 1,
			fill: func(a, b, c []float64, lda, ldb, ldc int) {
				a[0], a[8], a[16] = math.MaxFloat64, -math.MaxFloat64, math.MaxFloat64
				for j := 0; j < 3; j++ {
					b[j*ldb], b[j*ldb+8], b[j*ldb+16] = 1, 1, 1
				}
			},
		},
		{
			name: "lane-overflow", k: 16, alpha: 1,
			fill: func(a, b, c []float64, lda, ldb, ldc int) {
				a[0], a[8] = math.MaxFloat64, math.MaxFloat64
				for j := 0; j < 3; j++ {
					b[j*ldb], b[j*ldb+8] = 1, 1
				}
			},
		},
		{
			name: "finite-tree-cancellation", k: 16, alpha: 1,
			fill: func(a, b, c []float64, lda, ldb, ldc int) {
				a[0], a[2], a[4] = math.MaxFloat64, -math.MaxFloat64, math.MaxFloat64
				for j := 0; j < 3; j++ {
					b[j*ldb], b[j*ldb+2], b[j*ldb+4] = 1, 1, 1
				}
			},
		},
		{
			name: "alpha-c-rounding", k: 16, alpha: 1 + 0x1p-27,
			fill: func(a, b, c []float64, lda, ldb, ldc int) {
				a[0], a[8] = 1-0x1p-27, -1
				for j := 0; j < 3; j++ {
					b[j*ldb], b[j*ldb+8] = 1, 1
					c[j] = 0x1p-54
				}
			},
		},
		{
			name: "scalar-tail-fma", k: 17, alpha: 1,
			fill: func(a, b, c []float64, lda, ldb, ldc int) {
				a[0], b[0] = -1, 1
				a[16], b[16] = 1+0x1p-27, 1-0x1p-27
				for j := 1; j < 3; j++ {
					copy(b[j*ldb:j*ldb+17], b[:17])
				}
			},
		},
		{
			name: "subnormal", k: 17, alpha: 1,
			fill: func(a, b, c []float64, lda, ldb, ldc int) {
				a[0], a[8], a[16] = 0.5, -0.5, 1
				for j := 0; j < 3; j++ {
					b[j*ldb], b[j*ldb+8], b[j*ldb+16] = 8*math.SmallestNonzeroFloat64, 4*math.SmallestNonzeroFloat64, math.SmallestNonzeroFloat64
				}
			},
		},
		{
			name: "signed-zero", k: 17, alpha: 1,
			fill: func(a, b, c []float64, lda, ldb, ldc int) {
				for i := range a {
					a[i] = 1
				}
				for i := range b {
					b[i] = math.Copysign(0, -1)
				}
			},
		},
		{
			name: "positive-infinity", k: 17, alpha: 1,
			fill: func(a, b, c []float64, lda, ldb, ldc int) {
				a[0] = math.Inf(1)
				for j := 0; j < 3; j++ {
					b[j*ldb] = 1
				}
			},
		},
		{
			name: "negative-infinity", k: 17, alpha: 1,
			fill: func(a, b, c []float64, lda, ldb, ldc int) {
				a[0] = math.Inf(-1)
				for j := 0; j < 3; j++ {
					b[j*ldb] = 1
				}
			},
		},
		{
			name: "nan", k: 17, alpha: -0.75,
			fill: func(a, b, c []float64, lda, ldb, ldc int) {
				a[0] = math.NaN()
				for j := 0; j < 3; j++ {
					b[j*ldb] = 1
				}
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			const m, n = 1, 3
			lda, ldb, ldc := tc.k+3, tc.k+5, n+4
			a := make([]float64, m*lda)
			b := make([]float64, n*ldb)
			c := make([]float64, m*ldc)
			tc.fill(a, b, c, lda, ldb, ldc)
			want := slices.Clone(c)
			for j := 0; j < n; j++ {
				want[j] += tc.alpha * f64.DotUnitary(a[:tc.k], b[j*ldb:j*ldb+tc.k])
			}
			if !dgemmSerialNotTransSIMD(m, n, tc.k, a, lda, b, ldb, c, ldc, tc.alpha) {
				t.Fatal("helper rejected valid disjoint input")
			}
			for j := 0; j < n; j++ {
				if math.IsNaN(want[j]) {
					if !math.IsNaN(c[j]) {
						t.Fatalf("C[%d]: got %g want NaN", j, c[j])
					}
				} else if math.Float64bits(c[j]) != math.Float64bits(want[j]) {
					t.Fatalf("C[%d]: got %g (%#x) want %g (%#x)", j, c[j], math.Float64bits(c[j]), want[j], math.Float64bits(want[j]))
				}
			}
		})
	}
}

func dgemmTransData(n int) []float64 {
	x := make([]float64, n)
	for i := range x {
		x[i] = float64(i%23-11) / 16
	}
	return x
}

func dgemmTransEqual(x, y []float64) bool {
	return slices.EqualFunc(x, y, func(a, b float64) bool {
		return math.Float64bits(a) == math.Float64bits(b)
	})
}
