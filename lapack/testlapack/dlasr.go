// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlapack

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/floats"
	"plantcontrol.org/v1/gonum/lapack"
)

type Dlasrer interface {
	Dlasr(side blas.Side, pivot lapack.Pivot, direct lapack.Direct, m, n int, c, s, a []float64, lda int)
}

func DlasrTest(t *testing.T, impl Dlasrer) {
	t.Run("LeftTopBackwardRegression", func(t *testing.T) {
		a := []float64{
			1, 2,
			3, 4,
			5, 6,
		}
		impl.Dlasr(blas.Left, lapack.Top, lapack.Backward, 3, 2, []float64{0, 0}, []float64{1, 1}, a, 2)
		want := []float64{
			3, 4,
			-5, -6,
			-1, -2,
		}
		if !slices.Equal(a, want) {
			t.Fatalf("unexpected result: got %v want %v", a, want)
		}
	})

	t.Run("RightVariableBlockBoundary", func(t *testing.T) {
		const m, n, lda = 65, 65, 68
		for _, direct := range []lapack.Direct{lapack.Forward, lapack.Backward} {
			for _, random := range []bool{false, true} {
				name := fmt.Sprintf("direct=%c/random=%t", direct, random)
				t.Run(name, func(t *testing.T) {
					rnd := rand.New(rand.NewPCG(65, 65))
					a := make([]float64, m*lda)
					for i := range a {
						a[i] = math.Inf(1)
					}
					for i := 0; i < m; i++ {
						for j := 0; j < n; j++ {
							a[i*lda+j] = float64((i+1)*n+j+1) / 7
							if random {
								a[i*lda+j] = rnd.Float64()
							}
						}
					}
					c, s := make([]float64, n-1), make([]float64, n-1)
					for j := range c {
						theta := float64(j+1) * math.Pi / float64(2*len(c)+1)
						c[j], s[j] = math.Cos(theta), math.Sin(theta)
					}
					want := slices.Clone(a)
					if direct == lapack.Forward {
						for j := 0; j < n-1; j++ {
							for i := 0; i < m; i++ {
								tmp, tmp2 := want[i*lda+j+1], want[i*lda+j]
								want[i*lda+j+1] = c[j]*tmp - s[j]*tmp2
								want[i*lda+j] = s[j]*tmp + c[j]*tmp2
							}
						}
					} else {
						for j := n - 2; j >= 0; j-- {
							for i := 0; i < m; i++ {
								tmp, tmp2 := want[i*lda+j+1], want[i*lda+j]
								want[i*lda+j+1] = c[j]*tmp - s[j]*tmp2
								want[i*lda+j] = s[j]*tmp + c[j]*tmp2
							}
						}
					}
					impl.Dlasr(blas.Right, lapack.Variable, direct, m, n, c, s, a, lda)
					for i := 0; i < m; i++ {
						if !floats.EqualApprox(a[i*lda:i*lda+n], want[i*lda:i*lda+n], 1e-14) {
							t.Fatalf("unexpected logical row %d: got %v want %v", i, a[i*lda:i*lda+n], want[i*lda:i*lda+n])
						}
						for j := n; j < lda; j++ {
							k := i*lda + j
							if math.Float64bits(a[k]) != math.Float64bits(want[k]) {
								t.Fatalf("padding modified at %d: got %v want %v", k, a[k], want[k])
							}
						}
					}
				})
			}
		}
	})

	rnd := rand.New(rand.NewPCG(1, 1))
	for _, side := range []blas.Side{blas.Left, blas.Right} {
		for _, pivot := range []lapack.Pivot{lapack.Variable, lapack.Top, lapack.Bottom} {
			for _, direct := range []lapack.Direct{lapack.Forward, lapack.Backward} {
				for _, test := range []struct {
					m, n, lda int
					random    bool
				}{
					{5, 5, 0, false},
					{5, 10, 0, false},
					{10, 5, 0, false},
					{5, 5, 20, false},
					{5, 10, 20, false},
					{10, 5, 20, false},
					{31, 5, 8, false},
					{32, 5, 8, false},
					{33, 5, 8, false},
					{5, 5, 0, true},
					{5, 10, 0, true},
					{10, 5, 0, true},
					{5, 5, 20, true},
					{5, 10, 20, true},
					{10, 5, 20, true},
					{31, 5, 8, true},
					{32, 5, 8, true},
					{33, 5, 8, true},
				} {
					m := test.m
					n := test.n
					lda := test.lda
					if lda == 0 {
						lda = n
					}
					// Allocate an m×n matrix A and preserve padding sentinels.
					a := make([]float64, m*lda)
					for i := range a {
						a[i] = math.Inf(1)
					}
					for i := 0; i < m; i++ {
						for j := 0; j < n; j++ {
							a[i*lda+j] = float64((i+1)*n+j+1) / 7
							if test.random {
								a[i*lda+j] = rnd.Float64()
							}
						}
					}

					// Allocate slices for implicitly
					// represented rotation matrices.
					var s, c []float64
					if side == blas.Left {
						s = make([]float64, m-1)
						c = make([]float64, m-1)
					} else {
						s = make([]float64, n-1)
						c = make([]float64, n-1)
					}
					for k := range s {
						theta := float64(k+1) * math.Pi / float64(2*len(s)+1)
						s[k] = math.Sin(theta)
						c[k] = math.Cos(theta)
					}
					aCopy := make([]float64, len(a))
					copy(aCopy, a)

					// Apply plane a sequence of plane
					// rotation in s and c to the matrix A.
					impl.Dlasr(side, pivot, direct, m, n, c, s, a, lda)

					// Compute a reference solution by multiplying A
					// by explicitly formed rotation matrix P.
					pSize := m
					if side == blas.Right {
						pSize = n
					}
					// Allocate matrix P.
					p := blas64.General{
						Rows:   pSize,
						Cols:   pSize,
						Stride: pSize,
						Data:   make([]float64, pSize*pSize),
					}
					// Allocate matrix P_k.
					pk := blas64.General{
						Rows:   pSize,
						Cols:   pSize,
						Stride: pSize,
						Data:   make([]float64, pSize*pSize),
					}
					ptmp := blas64.General{
						Rows:   pSize,
						Cols:   pSize,
						Stride: pSize,
						Data:   make([]float64, pSize*pSize),
					}
					// Initialize P to the identity matrix.
					for i := 0; i < pSize; i++ {
						p.Data[i*p.Stride+i] = 1
						ptmp.Data[i*p.Stride+i] = 1
					}
					// Iterate over the sequence of plane rotations.
					for k := range s {
						// Set P_k to the identity matrix.
						for i := range p.Data {
							pk.Data[i] = 0
						}
						for i := 0; i < pSize; i++ {
							pk.Data[i*p.Stride+i] = 1
						}
						// Set the corresponding elements of P_k.
						switch pivot {
						case lapack.Variable:
							pk.Data[k*p.Stride+k] = c[k]
							pk.Data[k*p.Stride+k+1] = s[k]
							pk.Data[(k+1)*p.Stride+k] = -s[k]
							pk.Data[(k+1)*p.Stride+k+1] = c[k]
						case lapack.Top:
							pk.Data[0] = c[k]
							pk.Data[k+1] = s[k]
							pk.Data[(k+1)*p.Stride] = -s[k]
							pk.Data[(k+1)*p.Stride+k+1] = c[k]
						case lapack.Bottom:
							pk.Data[k*p.Stride+k] = c[k]
							pk.Data[k*p.Stride+pSize-1] = s[k]
							pk.Data[(pSize-1)*p.Stride+k] = -s[k]
							pk.Data[(pSize-1)*p.Stride+pSize-1] = c[k]
						}
						// Compute P <- P_k * P or P <- P * P_k.
						if direct == lapack.Forward {
							blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, pk, ptmp, 0, p)
						} else {
							blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, ptmp, pk, 0, p)
						}
						copy(ptmp.Data, p.Data)
					}

					aMat := blas64.General{
						Rows:   m,
						Cols:   n,
						Stride: lda,
						Data:   make([]float64, m*lda),
					}
					copy(aMat.Data, aCopy)
					newA := blas64.General{
						Rows:   m,
						Cols:   n,
						Stride: lda,
						Data:   make([]float64, m*lda),
					}
					// Compute P * A or A * P.
					if side == blas.Left {
						blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, p, aMat, 0, newA)
					} else {
						blas64.Gemm(blas.NoTrans, blas.Trans, 1, aMat, p, 0, newA)
					}
					// Compare the result from Dlasr with the reference solution.
					for i := 0; i < m; i++ {
						if !floats.EqualApprox(newA.Data[i*lda:i*lda+n], a[i*lda:i*lda+n], 1e-12) {
							t.Errorf("A update mismatch for side=%v pivot=%v direct=%v m=%d n=%d lda=%d random=%t", side, pivot, direct, m, n, lda, test.random)
							break
						}
						if !slices.Equal(a[i*lda+n:(i+1)*lda], aCopy[i*lda+n:(i+1)*lda]) {
							t.Errorf("padding modified for side=%v pivot=%v direct=%v m=%d n=%d lda=%d random=%t", side, pivot, direct, m, n, lda, test.random)
							break
						}
					}
				}
			}
		}
	}

	for _, side := range []blas.Side{blas.Left, blas.Right} {
		for _, pivot := range []lapack.Pivot{lapack.Variable, lapack.Top, lapack.Bottom} {
			for _, direct := range []lapack.Direct{lapack.Forward, lapack.Backward} {
				const m, n, lda = 3, 4, 6
				a := []float64{
					1, 2, 3, 4, math.Inf(1), math.Inf(1),
					5, 6, 7, 8, math.Inf(1), math.Inf(1),
					9, 10, 11, 12, math.Inf(1), math.Inf(1),
				}
				want := slices.Clone(a)
				z := m
				if side == blas.Right {
					z = n
				}
				c := make([]float64, z-1)
				for i := range c {
					c[i] = 1
				}
				impl.Dlasr(side, pivot, direct, m, n, c, make([]float64, z-1), a, lda)
				if !slices.Equal(a, want) {
					t.Errorf("identity rotation modified A for side=%v pivot=%v direct=%v", side, pivot, direct)
				}
			}
		}
	}

	for _, test := range []struct {
		name string
		c, s []float64
		a    []float64
		want []float64
	}{
		{
			name: "IdentityNonFinite",
			c:    []float64{1, 1, 1},
			s:    []float64{0, 0, 0},
			a:    []float64{math.Inf(1), math.NaN(), math.Copysign(0, -1), 4},
			want: []float64{math.Inf(1), math.NaN(), math.Copysign(0, -1), 4},
		},
		{
			name: "MixedIdentityNonFinite",
			c:    []float64{1, 0, 1},
			s:    []float64{0, 1, 0},
			a:    []float64{math.Inf(1), 2, 3, math.NaN()},
			want: []float64{math.Inf(1), 3, -2, math.NaN()},
		},
	} {
		for _, m := range []int{1, 33, 65} {
			for _, direct := range []lapack.Direct{lapack.Forward, lapack.Backward} {
				t.Run(fmt.Sprintf("%s/m=%d/direct=%c", test.name, m, direct), func(t *testing.T) {
					const n, lda = 65, 67
					c, s := make([]float64, n-1), make([]float64, n-1)
					for i := range c {
						c[i] = 1
					}
					copy(c, test.c)
					copy(s, test.s)
					a := make([]float64, m*lda)
					want := make([]float64, m*lda)
					for i := 0; i < m; i++ {
						for j := n; j < lda; j++ {
							a[i*lda+j] = math.Copysign(0, -1)
							want[i*lda+j] = math.Copysign(0, -1)
						}
						copy(a[i*lda:i*lda+n], test.a)
						copy(want[i*lda:i*lda+n], test.want)
					}
					impl.Dlasr(blas.Right, lapack.Variable, direct, m, n, c, s, a, lda)
					for i, w := range want {
						if math.Float64bits(a[i]) != math.Float64bits(w) {
							t.Fatalf("unexpected result at %d: got %v want %v", i, a[i], w)
						}
					}
				})
			}
		}
	}
}
