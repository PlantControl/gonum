// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"plantcontrol.org/v1/gonum/blas/blas64"
)

// Dlagv2 computes the generalized Schur factorization of a real 2×2 matrix
// pencil (A,B), where B is upper triangular. The returned left and right
// rotations satisfy Qᵀ*A*Z and Qᵀ*B*Z equal the overwritten matrices, with
// Q=[csq -snq; snq csq] and Z=[csz -snz; snz csz].
//
// The generalized eigenvalues are
//
//	(alphar0+i*alphai0)/scale1 and (alphar1+i*alphai1)/scale2.
//
// The csr and snr results duplicate csz and snz for compatibility with the
// existing internal API.
//
// Dlagv2 is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dlagv2(a []float64, lda int, b []float64, ldb int) (
	csq, snq, csr, snr, csz, snz, scale1, scale2, alphar0, alphar1, alphai0, alphai1 float64) {
	switch {
	case lda < 2:
		panic(badLdA)
	case ldb < 2:
		panic(badLdB)
	case len(a) < lda+2:
		panic(shortA)
	case len(b) < ldb+2:
		panic(shortB)
	}

	safmin := dlamchS
	ulp := dlamchP
	anorm := math.Max(math.Abs(a[0])+math.Abs(a[lda]), math.Max(math.Abs(a[1])+math.Abs(a[lda+1]), safmin))
	ascale := 1 / anorm
	a[0] *= ascale
	a[1] *= ascale
	a[lda] *= ascale
	a[lda+1] *= ascale
	bnorm := math.Max(math.Abs(b[0]), math.Max(math.Abs(b[1])+math.Abs(b[ldb+1]), safmin))
	bscale := 1 / bnorm
	b[0] *= bscale
	b[1] *= bscale
	b[ldb] *= bscale
	b[ldb+1] *= bscale

	bi := blas64.Implementation()
	wi := 0.0
	var wr1 float64
	if math.Abs(a[lda]) <= ulp {
		csq, csz = 1, 1
		a[lda], b[ldb] = 0, 0
	} else if math.Abs(b[0]) <= ulp {
		csq, snq, _ = impl.Dlartg(a[0], a[lda])
		csz = 1
		bi.Drot(2, a, 1, a[lda:], 1, csq, snq)
		bi.Drot(2, b, 1, b[ldb:], 1, csq, snq)
		a[lda], b[0], b[ldb] = 0, 0, 0
	} else if math.Abs(b[ldb+1]) <= ulp {
		csz, snz, _ = impl.Dlartg(a[lda+1], a[lda])
		snz = -snz
		bi.Drot(2, a, lda, a[1:], lda, csz, snz)
		bi.Drot(2, b, ldb, b[1:], ldb, csz, snz)
		csq = 1
		a[lda], b[ldb], b[ldb+1] = 0, 0, 0
	} else {
		scale1, _, wr1, _, wi = impl.Dlag2(a, lda, b, ldb)
		if wi == 0 {
			h1 := scale1*a[0] - wr1*b[0]
			h2 := scale1*a[1] - wr1*b[1]
			h3 := scale1*a[lda+1] - wr1*b[ldb+1]
			if math.Hypot(h1, h2) > math.Hypot(scale1*a[lda], h3) {
				csz, snz, _ = impl.Dlartg(h2, h1)
			} else {
				csz, snz, _ = impl.Dlartg(h3, scale1*a[lda])
			}
			snz = -snz
			bi.Drot(2, a, lda, a[1:], lda, csz, snz)
			bi.Drot(2, b, ldb, b[1:], ldb, csz, snz)

			hanorm := math.Max(math.Abs(a[0])+math.Abs(a[1]), math.Abs(a[lda])+math.Abs(a[lda+1]))
			hbnorm := math.Max(math.Abs(b[0])+math.Abs(b[1]), math.Abs(b[ldb])+math.Abs(b[ldb+1]))
			if scale1*hanorm >= math.Abs(wr1)*hbnorm {
				csq, snq, _ = impl.Dlartg(b[0], b[ldb])
			} else {
				csq, snq, _ = impl.Dlartg(a[0], a[lda])
			}
			bi.Drot(2, a, 1, a[lda:], 1, csq, snq)
			bi.Drot(2, b, 1, b[ldb:], 1, csq, snq)
			a[lda], b[ldb] = 0, 0
		} else {
			_, _, snz, csz, snq, csq = impl.Dlasv2(b[0], b[1], b[ldb+1])
			bi.Drot(2, a, 1, a[lda:], 1, csq, snq)
			bi.Drot(2, b, 1, b[ldb:], 1, csq, snq)
			bi.Drot(2, a, lda, a[1:], lda, csz, snz)
			bi.Drot(2, b, ldb, b[1:], ldb, csz, snz)
			b[ldb], b[1] = 0, 0
		}
	}

	a[0] *= anorm
	a[1] *= anorm
	a[lda] *= anorm
	a[lda+1] *= anorm
	b[0] *= bnorm
	b[1] *= bnorm
	b[ldb] *= bnorm
	b[ldb+1] *= bnorm
	if wi == 0 {
		alphar0, alphar1 = a[0], a[lda+1]
		scale1, scale2 = b[0], b[ldb+1]
	} else {
		alphar0 = anorm * wr1 / scale1 / bnorm
		alphar1 = alphar0
		alphai0 = anorm * wi / scale1 / bnorm
		alphai1 = -alphai0
		scale1, scale2 = 1, 1
	}
	csr, snr = csz, snz
	return
}
