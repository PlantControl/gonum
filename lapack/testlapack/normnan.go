// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlapack

import (
	"fmt"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/lapack"
)

// Normer is the set of matrix norm routines checked for NaN propagation.
type Normer interface {
	Dlange(norm lapack.MatrixNorm, m, n int, a []float64, lda int, work []float64) float64
	Dlanhs(norm lapack.MatrixNorm, n int, a []float64, lda int, work []float64) float64
	Dlansy(norm lapack.MatrixNorm, uplo blas.Uplo, n int, a []float64, lda int, work []float64) float64
	Dlantr(norm lapack.MatrixNorm, uplo blas.Uplo, diag blas.Diag, m, n int, a []float64, lda int, work []float64) float64
	Dlangb(norm lapack.MatrixNorm, m, n, kl, ku int, ab []float64, ldab int) float64
	Dlansb(norm lapack.MatrixNorm, uplo blas.Uplo, n, kd int, ab []float64, ldab int, work []float64) float64
	Dlantb(norm lapack.MatrixNorm, uplo blas.Uplo, diag blas.Diag, n, k int, a []float64, lda int, work []float64) float64
	Dlanst(norm lapack.MatrixNorm, n int, d, e []float64) float64
	Dlangt(norm lapack.MatrixNorm, n int, dl, d, du []float64) float64
	Zlange(norm lapack.MatrixNorm, m, n int, a []complex128, lda int, work []float64) float64
	Zlanhs(norm lapack.MatrixNorm, n int, a []complex128, lda int, work []float64) float64
}

// NormCase is one configuration of a matrix norm routine applied to the
// dense row-major M×N matrix a. Ref reports whether the routine references
// element (i,j). Kind names the LAPACK routine and Uplo, Diag, KL and KU
// its parameters, so that an oracle can be called on the same matrix.
type NormCase struct {
	Kind   string
	M, N   int
	Uplo   blas.Uplo
	Diag   blas.Diag
	KL, KU int
	Ref    func(i, j int) bool
	Norm   func(norm lapack.MatrixNorm, a []float64) float64
}

func (c NormCase) String() string {
	return fmt.Sprintf("%s/m=%d/n=%d/uplo=%c/diag=%c/kl=%d/ku=%d", c.Kind, c.M, c.N, c.Uplo, c.Diag, c.KL, c.KU)
}

// NormCases returns the norm routine configurations checked by NormNaNTest.
func NormCases(impl Normer) []NormCase {
	const n = 4
	work := make([]float64, 2*n)
	var cases []NormCase
	all := func(int, int) bool { return true }
	for _, mn := range [][2]int{{n, n}, {3, n}, {n, 3}} {
		m, nn := mn[0], mn[1]
		cases = append(cases, NormCase{Kind: "DLANGE", M: m, N: nn, Ref: all,
			Norm: func(norm lapack.MatrixNorm, a []float64) float64 {
				return impl.Dlange(norm, m, nn, a, nn, work)
			}})
		cases = append(cases, NormCase{Kind: "ZLANGE", M: m, N: nn, Ref: all,
			Norm: func(norm lapack.MatrixNorm, a []float64) float64 {
				z := make([]complex128, len(a))
				for i, v := range a {
					z[i] = complex(v, 0.5)
				}
				return impl.Zlange(norm, m, nn, z, nn, work)
			}})
		for _, uplo := range []blas.Uplo{blas.Upper, blas.Lower} {
			for _, diag := range []blas.Diag{blas.NonUnit, blas.Unit} {
				ref := func(i, j int) bool {
					if diag == blas.Unit && i == j {
						return false
					}
					if uplo == blas.Upper {
						return i <= j
					}
					return i >= j
				}
				cases = append(cases, NormCase{Kind: "DLANTR", M: m, N: nn, Uplo: uplo, Diag: diag, Ref: ref,
					Norm: func(norm lapack.MatrixNorm, a []float64) float64 {
						return impl.Dlantr(norm, uplo, diag, m, nn, a, nn, work)
					}})
			}
		}
	}
	hess := func(i, j int) bool { return i <= j+1 }
	cases = append(cases, NormCase{Kind: "DLANHS", M: n, N: n, Ref: hess,
		Norm: func(norm lapack.MatrixNorm, a []float64) float64 {
			return impl.Dlanhs(norm, n, a, n, work)
		}})
	cases = append(cases, NormCase{Kind: "ZLANHS", M: n, N: n, Ref: hess,
		Norm: func(norm lapack.MatrixNorm, a []float64) float64 {
			z := make([]complex128, len(a))
			for i, v := range a {
				z[i] = complex(v, -0.5)
			}
			return impl.Zlanhs(norm, n, z, n, work)
		}})
	for _, uplo := range []blas.Uplo{blas.Upper, blas.Lower} {
		tri := func(i, j int) bool {
			if uplo == blas.Upper {
				return i <= j
			}
			return i >= j
		}
		cases = append(cases, NormCase{Kind: "DLANSY", M: n, N: n, Uplo: uplo, Ref: tri,
			Norm: func(norm lapack.MatrixNorm, a []float64) float64 {
				return impl.Dlansy(norm, uplo, n, a, n, work)
			}})
		for _, kd := range []int{0, 1, 2} {
			band := func(i, j int) bool {
				if uplo == blas.Upper {
					return i <= j && j-i <= kd
				}
				return i >= j && i-j <= kd
			}
			cases = append(cases, NormCase{Kind: "DLANSB", M: n, N: n, Uplo: uplo, KL: kd, KU: kd, Ref: band,
				Norm: func(norm lapack.MatrixNorm, a []float64) float64 {
					ab := symBandFromDense(uplo, n, kd, a)
					return impl.Dlansb(norm, uplo, n, kd, ab, kd+1, work)
				}})
			for _, diag := range []blas.Diag{blas.NonUnit, blas.Unit} {
				ref := func(i, j int) bool { return band(i, j) && !(diag == blas.Unit && i == j) }
				cases = append(cases, NormCase{Kind: "DLANTB", M: n, N: n, Uplo: uplo, Diag: diag, KL: kd, KU: kd, Ref: ref,
					Norm: func(norm lapack.MatrixNorm, a []float64) float64 {
						ab := symBandFromDense(uplo, n, kd, a)
						return impl.Dlantb(norm, uplo, diag, n, kd, ab, kd+1, work)
					}})
			}
		}
	}
	for _, klku := range [][2]int{{0, 0}, {1, 0}, {0, 2}, {1, 2}, {3, 3}} {
		kl, ku := klku[0], klku[1]
		cases = append(cases, NormCase{Kind: "DLANGB", M: n, N: n, KL: kl, KU: ku,
			Ref: func(i, j int) bool { return j-i <= ku && i-j <= kl },
			Norm: func(norm lapack.MatrixNorm, a []float64) float64 {
				ldab := kl + ku + 1
				ab := make([]float64, n*ldab)
				for i := range n {
					for j := max(0, i-kl); j <= min(n-1, i+ku); j++ {
						ab[i*ldab+j-i+kl] = a[i*n+j]
					}
				}
				return impl.Dlangb(norm, n, n, kl, ku, ab, ldab)
			}})
	}
	cases = append(cases, NormCase{Kind: "DLANST", M: n, N: n,
		Ref: func(i, j int) bool { return j == i || j == i+1 },
		Norm: func(norm lapack.MatrixNorm, a []float64) float64 {
			d, e := make([]float64, n), make([]float64, n-1)
			for i := range n {
				d[i] = a[i*n+i]
				if i < n-1 {
					e[i] = a[i*n+i+1]
				}
			}
			return impl.Dlanst(norm, n, d, e)
		}})
	cases = append(cases, NormCase{Kind: "DLANGT", M: n, N: n,
		Ref: func(i, j int) bool { return j-i <= 1 && i-j <= 1 },
		Norm: func(norm lapack.MatrixNorm, a []float64) float64 {
			dl, d, du := make([]float64, n-1), make([]float64, n), make([]float64, n-1)
			for i := range n {
				d[i] = a[i*n+i]
				if i < n-1 {
					du[i] = a[i*n+i+1]
					dl[i] = a[(i+1)*n+i]
				}
			}
			return impl.Dlangt(norm, n, dl, d, du)
		}})
	return cases
}

// symBandFromDense returns the referenced triangle of the band of the
// row-major n×n a in row-major band storage with kd off-diagonals.
func symBandFromDense(uplo blas.Uplo, n, kd int, a []float64) []float64 {
	ldab := kd + 1
	ab := make([]float64, n*ldab)
	for i := range n {
		if uplo == blas.Upper {
			for j := i; j <= min(n-1, i+kd); j++ {
				ab[i*ldab+j-i] = a[i*n+j]
			}
		} else {
			for j := max(0, i-kd); j <= i; j++ {
				ab[i*ldab+kd+j-i] = a[i*n+j]
			}
		}
	}
	return ab
}

// NormPlacements returns, for c, matrices of ones with an Inf at one
// referenced element and a NaN at another, in every order, together with
// whether the result must be NaN. A matrix with a single Inf, whose norm must
// be Inf, is included for each referenced element.
func NormPlacements(c NormCase) (mats [][]float64, wantNaN []bool) {
	var refs [][2]int
	for i := 0; i < c.M; i++ {
		for j := 0; j < c.N; j++ {
			if c.Ref(i, j) {
				refs = append(refs, [2]int{i, j})
			}
		}
	}
	base := func() []float64 {
		a := make([]float64, c.M*c.N)
		for i := range a {
			a[i] = 1
		}
		return a
	}
	for _, p := range refs {
		a := base()
		a[p[0]*c.N+p[1]] = math.Inf(1)
		mats, wantNaN = append(mats, a), append(wantNaN, false)
		for _, q := range refs {
			if q == p {
				continue
			}
			a := base()
			a[p[0]*c.N+p[1]] = math.Inf(-1)
			a[q[0]*c.N+q[1]] = math.NaN()
			mats, wantNaN = append(mats, a), append(wantNaN, true)
		}
	}
	return mats, wantNaN
}

// NormNaNTest checks that, as in LAPACK, every norm of a matrix with a NaN in
// a referenced element is NaN wherever the NaN appears, including next to an
// infinity, and that a single infinity gives an infinite norm.
func NormNaNTest(t *testing.T, impl Normer) {
	for _, c := range NormCases(impl) {
		mats, wantNaN := NormPlacements(c)
		if len(mats) == 0 {
			// A unit triangular band with no off-diagonals references nothing.
			continue
		}
		for k, a := range mats {
			for _, norm := range []lapack.MatrixNorm{lapack.MaxAbs, lapack.MaxColumnSum, lapack.MaxRowSum, lapack.Frobenius} {
				got := c.Norm(norm, a)
				if wantNaN[k] && !math.IsNaN(got) {
					t.Errorf("%v norm=%c A=%v: got %v, want NaN", c, norm, a, got)
				}
				if !wantNaN[k] && !math.IsInf(got, 1) {
					t.Errorf("%v norm=%c A=%v: got %v, want +Inf", c, norm, a, got)
				}
			}
		}
	}
}
