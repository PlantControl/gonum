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

type blockedSyrkFunc[T zeroSyrkFloat] func(blas.Uplo, int, int, T, []T, int, T, []T, int) bool

func TestDsyrkBlockedBoundaries(t *testing.T) {
	testSyrkBlockedBoundaries(t, dsyrkBlocked, dgemmSIMDEqualBits)
}

func TestSsyrkBlockedBoundaries(t *testing.T) {
	testSyrkBlockedBoundaries(t, ssyrkBlocked, sgemmSIMDEqualBits)
}

func testSyrkBlockedBoundaries[T zeroSyrkFloat](t *testing.T, blocked blockedSyrkFunc[T], equal func([]T, []T) bool) {
	t.Helper()
	dims := make([][2]int, 0, 15)
	for _, n := range []int{15, 16, 17} {
		for _, k := range []int{15, 16, 17} {
			dims = append(dims, [2]int{n, k})
		}
	}
	dims = append(dims,
		[2]int{15, 63}, [2]int{16, 64}, [2]int{17, 65},
		[2]int{15, 127}, [2]int{16, 128}, [2]int{17, 129},
	)
	for _, dim := range dims {
		n, k := dim[0], dim[1]
		for _, ul := range []blas.Uplo{blas.Upper, blas.Lower} {
			for _, beta := range []T{0, 1, -0.5} {
				t.Run(fmt.Sprintf("n=%d/k=%d/%c/beta=%g", n, k, ul, beta), func(t *testing.T) {
					const off = 3
					lda, ldc := n+3, n+5
					aStore := syrkSIMDData[T](off + k*lda + off)
					cStore := syrkSIMDData[T](off + n*ldc + off)
					a, c := aStore[off:], cStore[off:]
					want := slices.Clone(cStore)
					syrkTransReference(ul, n, k, T(-0.75), a, lda, beta, want[off:], ldc)
					aOrig := slices.Clone(aStore)
					if !blocked(ul, n, k, T(-0.75), a, lda, beta, c, ldc) {
						t.Fatal("disjoint call rejected")
					}
					if !equal(aStore, aOrig) {
						t.Fatal("A or guard changed")
					}
					if !equal(cStore, want) {
						t.Fatal("triangle, tail, or padding differs from sequential reference")
					}
				})
			}
		}
	}
}

func TestDsyrkBlockedAliasing(t *testing.T) {
	testSyrkBlockedAliasing(t, dsyrkBlocked, dgemmSIMDEqualBits)
}

func TestSsyrkBlockedAliasing(t *testing.T) {
	testSyrkBlockedAliasing(t, ssyrkBlocked, sgemmSIMDEqualBits)
}

func testSyrkBlockedAliasing[T zeroSyrkFloat](t *testing.T, blocked blockedSyrkFunc[T], equal func([]T, []T) bool) {
	t.Helper()
	const n, k = 17, 17
	for _, ul := range []blas.Uplo{blas.Upper, blas.Lower} {
		t.Run(string(ul)+"/overlap", func(t *testing.T) {
			const stride = n + 3
			storage := syrkSIMDData[T](k*stride + n*stride)
			a, c := storage, storage[1:]
			before := slices.Clone(storage)
			if blocked(ul, n, k, T(-0.75), a, stride, 0, c, stride) {
				t.Fatal("accepted active C/A overlap")
			}
			if !equal(storage, before) {
				t.Fatal("rejected call modified storage")
			}
		})
		for _, beta := range []T{0, 1, -0.5} {
			t.Run(fmt.Sprintf("%c/shared-disjoint/beta=%g", ul, beta), func(t *testing.T) {
				const stride = 2*n + 3
				storage := syrkSIMDData[T](k * stride)
				a, c := storage, storage[n:]
				want := slices.Clone(storage)
				syrkTransReference(ul, n, k, T(-0.75), slices.Clone(a), stride, beta, want[n:], stride)
				aActive := make([]T, n*k)
				for l := 0; l < k; l++ {
					copy(aActive[l*n:(l+1)*n], a[l*stride:l*stride+n])
				}
				if !blocked(ul, n, k, T(-0.75), a, stride, beta, c, stride) {
					t.Fatal("rejected disjoint active regions in shared backing")
				}
				if !equal(storage, want) {
					t.Fatal("shared-backing result or padding differs")
				}
				for l := 0; l < k; l++ {
					if !equal(a[l*stride:l*stride+n], aActive[l*n:(l+1)*n]) {
						t.Fatal("active A changed")
					}
				}
			})
		}
	}
}

func TestSyrkBlockedPublicZeroSign(t *testing.T) {
	testSyrkBlockedPublicZeroSign(t, Implementation{}.Dsyrk)
	testSyrkBlockedPublicZeroSign(t, Implementation{}.Ssyrk)
}

func TestDsyrkBlockedExceptional(t *testing.T) {
	testSyrkBlockedExceptional(t, dsyrkBlocked)
}

func TestSsyrkBlockedExceptional(t *testing.T) {
	testSyrkBlockedExceptional(t, ssyrkBlocked)
}

func testSyrkBlockedExceptional[T zeroSyrkFloat](t *testing.T, blocked blockedSyrkFunc[T]) {
	t.Helper()
	const n, k = 17, 17
	lda, ldc := n+3, n+5
	for _, ul := range []blas.Uplo{blas.Upper, blas.Lower} {
		for _, tc := range []struct {
			name        string
			alpha, beta T
			prepare     func([]T, int)
		}{
			{
				name:  "ZeroScaleSkipsNonfinite",
				alpha: 1, beta: 1,
				prepare: func(a []T, lda int) {
					if ul == blas.Upper {
						a[0], a[4], a[16] = 0, T(math.NaN()), T(math.Inf(1))
						return
					}
					a[16], a[0], a[4] = 0, T(math.NaN()), T(math.Inf(-1))
				},
			},
			{name: "BetaNaN", alpha: -0.75, beta: T(math.NaN())},
			{name: "BetaInf", alpha: -0.75, beta: T(math.Inf(1))},
			{
				name:  "InfiniteAlphaTimesZero",
				alpha: T(math.Inf(1)), beta: 0,
				prepare: func(a []T, lda int) {
					for i := range a {
						a[i] = 0
					}
				},
			},
		} {
			t.Run(string(ul)+"/"+tc.name, func(t *testing.T) {
				a := syrkSIMDData[T](k * lda)
				c := syrkSIMDData[T](n * ldc)
				if tc.prepare != nil {
					tc.prepare(a, lda)
				}
				want := slices.Clone(c)
				syrkTransReference(ul, n, k, tc.alpha, a, lda, tc.beta, want, ldc)
				if !blocked(ul, n, k, tc.alpha, a, lda, tc.beta, c, ldc) {
					t.Fatal("eligible exceptional case rejected")
				}
				checkSyrkExceptional(t, c, want)
			})
		}
	}
}

func checkSyrkExceptional[T zeroSyrkFloat](t *testing.T, got, want []T) {
	t.Helper()
	for i, g := range got {
		w := want[i]
		if math.IsNaN(float64(w)) {
			if !math.IsNaN(float64(g)) {
				t.Fatalf("index %d: got %g, want NaN", i, g)
			}
			continue
		}
		if math.IsNaN(float64(g)) || math.Float64bits(float64(g)) != math.Float64bits(float64(w)) {
			t.Fatalf("index %d: got %g, want %g", i, g, w)
		}
	}
}

func testSyrkBlockedPublicZeroSign[T zeroSyrkFloat](t *testing.T, syrk zeroSyrkFunc[T]) {
	t.Helper()
	const n, k = 17, 16
	for _, ul := range []blas.Uplo{blas.Upper, blas.Lower} {
		a := make([]T, k*n)
		c := make([]T, n*(n+2))
		for i := range c {
			c[i] = T(math.Copysign(0, -1))
		}
		syrk(ul, blas.Trans, n, k, 1, a, n, 0, c, n+2)
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				active := ul == blas.Upper && j >= i || ul == blas.Lower && j <= i
				bits := math.Float64bits(float64(c[i*(n+2)+j]))
				if active && bits != 0 {
					t.Fatalf("active C[%d,%d] is not +0", i, j)
				}
				if !active && bits != math.Float64bits(math.Copysign(0, -1)) {
					t.Fatalf("inactive C[%d,%d] changed", i, j)
				}
			}
		}
	}
}

func syrkTransReference[T zeroSyrkFloat](ul blas.Uplo, n, k int, alpha T, a []T, lda int, beta T, c []T, ldc int) {
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if ul == blas.Upper && j < i || ul == blas.Lower && j > i {
				continue
			}
			v := c[i*ldc+j]
			if beta == 0 {
				v = 0
			} else if beta != 1 {
				v *= beta
			}
			for l := 0; l < k; l++ {
				if scale := alpha * a[l*lda+i]; scale != 0 {
					v += scale * a[l*lda+j]
				}
			}
			c[i*ldc+j] = v
		}
	}
}

func syrkSIMDData[T zeroSyrkFloat](n int) []T {
	x := make([]T, n)
	for i := range x {
		x[i] = T((i%17)-8) / 16
	}
	return x
}
