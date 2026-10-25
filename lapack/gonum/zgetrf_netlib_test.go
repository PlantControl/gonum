// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package gonum

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/lapack/gonum/internal/netlib"
)

// zluNetlibMatrix returns a row-major m×n matrix with stride lda for the Z LU
// differentials. kind "random" is a general matrix, so partial pivoting
// swaps rows; "cabs1tie" has columns whose largest |Re|+|Im| ties between a
// row with the larger modulus and an earlier row with the smaller one, which
// IZAMAX resolves to the earlier row; "singular" has a zero column, which
// LAPACK records in INFO but factors past.
func zluNetlibMatrix(kind string, m, n, lda int, rnd *rand.Rand) []complex128 {
	a := make([]complex128, max(1, (m-1)*lda+n))
	for i := range m {
		for j := range n {
			a[i*lda+j] = complex(rnd.NormFloat64(), rnd.NormFloat64())
		}
	}
	switch kind {
	case "cabs1tie":
		for j := 0; j < min(m, n); j++ {
			for i := j; i < m; i++ {
				a[i*lda+j] = complex(0.1*rnd.NormFloat64(), 0.1*rnd.NormFloat64())
			}
		}
		for j := 0; j < min(m, n)-1 && j+2 < m; j += 2 {
			a[(j+1)*lda+j] = 1 + 1i
			a[(j+2)*lda+j] = 1.5 + 0.5i
		}
	case "singular":
		if n > 1 {
			for i := range m {
				a[i*lda+1] = 0
			}
		}
	}
	return a
}

func zluNetlibClose(t *testing.T, name string, got, want []complex128, scale float64) {
	t.Helper()
	for i := range got {
		if cmplx.Abs(got[i]-want[i]) > 1e-12*scale {
			t.Errorf("%s[%d]=%v, want Netlib %v", name, i, got[i], want[i])
			return
		}
	}
}

func zluNetlibScale(a []complex128) float64 {
	s := 1.0
	for _, v := range a {
		s = math.Max(s, cmplx.Abs(v))
	}
	return s
}

// TestZgetrfNetlib compares Zgetrf with Netlib ZGETRF, including the pivot
// indices, on pivoting, tie-breaking and singular matrices and on sizes that
// take the blocked path.
func TestZgetrfNetlib(t *testing.T) {
	rnd := rand.New(rand.NewPCG(71, 71))
	for _, kind := range []string{"random", "cabs1tie", "singular"} {
		for _, dims := range [][2]int{{1, 1}, {2, 2}, {3, 5}, {5, 3}, {8, 8}, {33, 20}, {20, 33}, {100, 100}, {150, 130}} {
			m, n := dims[0], dims[1]
			for _, ldExtra := range []int{0, 3} {
				lda := n + ldExtra
				t.Run(fmt.Sprintf("%s/m=%d/n=%d/lda=%d", kind, m, n, lda), func(t *testing.T) {
					a := zluNetlibMatrix(kind, m, n, lda, rnd)
					ga := append([]complex128(nil), a...)
					na := append([]complex128(nil), a...)
					mn := min(m, n)
					gipiv := make([]int, mn)
					nipiv := make([]int32, max(1, mn))
					ok := Implementation{}.Zgetrf(m, n, ga, lda, gipiv)
					info := netlib.Zgetrf(m, n, na, lda, nipiv)
					if info < 0 || ok != (info == 0) {
						t.Fatalf("ok=%v, Netlib info=%d", ok, info)
					}
					pivoted := false
					for i := range mn {
						if gipiv[i] != int(nipiv[i])-1 {
							t.Fatalf("ipiv=%v, want Netlib %v (1-based)", gipiv, nipiv[:mn])
						}
						pivoted = pivoted || gipiv[i] != i
					}
					if kind != "singular" && mn > 2 && !pivoted {
						t.Fatalf("no row interchanges; test does not pivot")
					}
					if kind == "cabs1tie" && m > 2 && gipiv[0] != 1 {
						t.Fatalf("ipiv[0]=%d, want 1: the |Re|+|Im| tie goes to the first row", gipiv[0])
					}
					zluNetlibClose(t, "LU", ga, na, zluNetlibScale(a))
				})
			}
		}
	}
}

// TestZgetrsNetlib compares Zgetrs with Netlib ZGETRS for each transpose
// option on pivoted factorizations.
func TestZgetrsNetlib(t *testing.T) {
	rnd := rand.New(rand.NewPCG(73, 73))
	for _, trans := range []blas.Transpose{blas.NoTrans, blas.Trans, blas.ConjTrans} {
		for _, kind := range []string{"random", "cabs1tie"} {
			for _, n := range []int{1, 2, 5, 16, 100} {
				for _, nrhs := range []int{1, 3} {
					lda, ldb := n+2, nrhs+1
					t.Run(fmt.Sprintf("trans=%c/%s/n=%d/nrhs=%d", trans, kind, n, nrhs), func(t *testing.T) {
						a := zluNetlibMatrix(kind, n, n, lda, rnd)
						b := zluNetlibMatrix("random", n, nrhs, ldb, rnd)
						ipiv := make([]int, n)
						if !(Implementation{}).Zgetrf(n, n, a, lda, ipiv) {
							t.Fatal("singular test matrix")
						}
						nipiv := make([]int32, n)
						for i, p := range ipiv {
							nipiv[i] = int32(p + 1)
						}
						gb, nb := append([]complex128(nil), b...), append([]complex128(nil), b...)
						Implementation{}.Zgetrs(trans, n, nrhs, a, lda, ipiv, gb, ldb)
						if info := netlib.Zgetrs(byte(trans), n, nrhs, a, lda, nipiv, nb, ldb); info != 0 {
							t.Fatalf("Netlib info=%d", info)
						}
						zluNetlibClose(t, "X", gb, nb, zluNetlibScale(nb))
					})
				}
			}
		}
	}
}

// TestZgesvNetlib compares Zgesv with Netlib ZGESV, including the singular
// case, where ZGESV returns INFO > 0 after factoring and leaves B unchanged.
func TestZgesvNetlib(t *testing.T) {
	rnd := rand.New(rand.NewPCG(79, 79))
	for _, kind := range []string{"random", "cabs1tie", "singular"} {
		for _, n := range []int{1, 2, 5, 16, 100} {
			for _, nrhs := range []int{1, 4} {
				lda, ldb := n+1, nrhs+2
				t.Run(fmt.Sprintf("%s/n=%d/nrhs=%d", kind, n, nrhs), func(t *testing.T) {
					a := zluNetlibMatrix(kind, n, n, lda, rnd)
					b := zluNetlibMatrix("random", n, nrhs, ldb, rnd)
					ga, na := append([]complex128(nil), a...), append([]complex128(nil), a...)
					gb, nb := append([]complex128(nil), b...), append([]complex128(nil), b...)
					gipiv := make([]int, n)
					nipiv := make([]int32, n)
					ok := Implementation{}.Zgesv(n, nrhs, ga, lda, gipiv, gb, ldb)
					info := netlib.Zgesv(n, nrhs, na, lda, nipiv, nb, ldb)
					if info < 0 || ok != (info == 0) {
						t.Fatalf("ok=%v, Netlib info=%d", ok, info)
					}
					for i := range n {
						if gipiv[i] != int(nipiv[i])-1 {
							t.Fatalf("ipiv=%v, want Netlib %v (1-based)", gipiv, nipiv)
						}
					}
					zluNetlibClose(t, "LU", ga, na, zluNetlibScale(a))
					zluNetlibClose(t, "B", gb, nb, zluNetlibScale(nb))
				})
			}
		}
	}
}
