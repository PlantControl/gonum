// Copyright ©2023 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
	blasgonum "plantcontrol.org/v1/gonum/blas/gonum"
	"plantcontrol.org/v1/gonum/lapack"
)

// Dgghrd reduces a pair of real matrices (A,B) to generalized upper Hessenberg
// form using orthogonal transformations, where A is a general matrix and B is
// upper triangular.
//
// This subroutine simultaneously reduces A to a Hessenberg matrix H
//
//	Qᵀ*A*Z = H,
//
// and transforms B to another upper triangular matrix T
//
//	Qᵀ*B*Z = T.
//
// The orthogonal matrices Q and Z are determined as products of Givens
// rotations. They may either be formed explicitly (lapack.OrthoExplicit), or
// they may be postmultiplied into input matrices Q1 and Z1
// (lapack.OrthoPostmul), so that
//
//	Q1 * A * Z1ᵀ = (Q1*Q) * H * (Z1*Z)ᵀ,
//	Q1 * B * Z1ᵀ = (Q1*Q) * T * (Z1*Z)ᵀ.
//
// ilo and ihi determine the block of A that will be reduced. It must hold that
//
//   - 0 <= ilo <= ihi < n      if n > 0,
//   - ilo == 0 and ihi == -1   if n == 0,
//
// otherwise Dgghrd will panic.
//
// Dgghrd is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dgghrd(compq, compz lapack.OrthoComp, n, ilo, ihi int, a []float64, lda int, b []float64, ldb int, q []float64, ldq int, z []float64, ldz int) {
	switch {
	case compq != lapack.OrthoNone && compq != lapack.OrthoExplicit && compq != lapack.OrthoPostmul:
		panic(badOrthoComp)
	case compz != lapack.OrthoNone && compz != lapack.OrthoExplicit && compz != lapack.OrthoPostmul:
		panic(badOrthoComp)
	case n < 0:
		panic(nLT0)
	case ilo < 0 || max(0, n-1) < ilo:
		panic(badIlo)
	case ihi < min(ilo, n-1) || n <= ihi:
		panic(badIhi)
	case lda < max(1, n):
		panic(badLdA)
	case ldb < max(1, n):
		panic(badLdB)
	case (compq != lapack.OrthoNone && ldq < n) || ldq < 1:
		panic(badLdQ)
	case (compz != lapack.OrthoNone && ldz < n) || ldz < 1:
		panic(badLdZ)
	}

	// Quick return if possible.
	if n == 0 {
		return
	}
	if n == 1 {
		switch {
		case compq == lapack.OrthoExplicit && len(q) < 1:
			panic(shortQ)
		case compz == lapack.OrthoExplicit && len(z) < 1:
			panic(shortZ)
		}
		if compq == lapack.OrthoExplicit {
			q[0] = 1
		}
		if compz == lapack.OrthoExplicit {
			z[0] = 1
		}
		return
	}

	switch {
	case len(a) < (n-1)*lda+n:
		panic(shortA)
	case len(b) < (n-1)*ldb+n:
		panic(shortB)
	case compq != lapack.OrthoNone && len(q) < (n-1)*ldq+n:
		panic(shortQ)
	case compz != lapack.OrthoNone && len(z) < (n-1)*ldz+n:
		panic(shortZ)
	}

	if compq == lapack.OrthoExplicit {
		impl.Dlaset(blas.All, n, n, 0, 1, q, ldq)
	}
	if compz == lapack.OrthoExplicit {
		impl.Dlaset(blas.All, n, n, 0, 1, z, ldz)
	}

	// Zero out lower triangle of B.
	for i := 1; i < n; i++ {
		for j := 0; j < i; j++ {
			b[i*ldb+j] = 0
		}
	}
	bi := blas64.Implementation()
	_, defaultBLAS := bi.(blasgonum.Implementation)
	// Reduce A and B.
	for jcol := ilo; jcol <= ihi-2; jcol++ {
		batchVectors := defaultBLAS && n >= 32 && ihi-jcol-1 >= 2 && (compq != lapack.OrthoNone || compz != lapack.OrthoNone)
		if batchVectors {
			const block = 32
			var qc, qs, zc, zs [block]float64
			for jrow := ihi; jrow >= jcol+2; {
				count := min(block, jrow-jcol-1)
				first := jrow
				for k := 0; k < count; k++ {
					// Step 1: rotate rows jrow-1, jrow to kill A[jrow,jcol].
					var c, s float64
					c, s, a[(jrow-1)*lda+jcol] = impl.Dlartg(a[(jrow-1)*lda+jcol], a[jrow*lda+jcol])
					a[jrow*lda+jcol] = 0

					bi.Drot(n-jcol-1, a[(jrow-1)*lda+jcol+1:], 1, a[jrow*lda+jcol+1:], 1, c, s)
					bi.Drot(n+2-jrow-1, b[(jrow-1)*ldb+jrow-1:], 1, b[jrow*ldb+jrow-1:], 1, c, s)
					qc[k], qs[k] = c, s

					// Step 2: rotate columns jrow, jrow-1 to kill B[jrow,jrow-1].
					c, s, b[jrow*ldb+jrow] = impl.Dlartg(b[jrow*ldb+jrow], b[jrow*ldb+jrow-1])
					b[jrow*ldb+jrow-1] = 0

					bi.Drot(ihi+1, a[jrow:], lda, a[jrow-1:], lda, c, s)
					bi.Drot(jrow, b[jrow:], ldb, b[jrow-1:], ldb, c, s)
					zc[k], zs[k] = c, s
					jrow--
				}
				dgghrdReplayVectors(n, first, count, compq, q, ldq, qc[:], qs[:], compz, z, ldz, zc[:], zs[:])
			}
			continue
		}
		for jrow := ihi; jrow >= jcol+2; jrow-- {
			// Step 1: rotate rows jrow-1, jrow to kill A[jrow,jcol].
			var c, s float64
			c, s, a[(jrow-1)*lda+jcol] = impl.Dlartg(a[(jrow-1)*lda+jcol], a[jrow*lda+jcol])
			a[jrow*lda+jcol] = 0

			bi.Drot(n-jcol-1, a[(jrow-1)*lda+jcol+1:], 1, a[jrow*lda+jcol+1:], 1, c, s)
			bi.Drot(n+2-jrow-1, b[(jrow-1)*ldb+jrow-1:], 1, b[jrow*ldb+jrow-1:], 1, c, s)

			if compq != lapack.OrthoNone {
				bi.Drot(n, q[jrow-1:], ldq, q[jrow:], ldq, c, s)
			}

			// Step 2: rotate columns jrow, jrow-1 to kill B[jrow,jrow-1].
			c, s, b[jrow*ldb+jrow] = impl.Dlartg(b[jrow*ldb+jrow], b[jrow*ldb+jrow-1])
			b[jrow*ldb+jrow-1] = 0

			bi.Drot(ihi+1, a[jrow:], lda, a[jrow-1:], lda, c, s)
			bi.Drot(jrow, b[jrow:], ldb, b[jrow-1:], ldb, c, s)

			if compz != lapack.OrthoNone {
				bi.Drot(n, z[jrow:], ldz, z[jrow-1:], ldz, c, s)
			}
		}
	}
}

func dgghrdReplayVectors(n, first, count int, compq lapack.OrthoComp, q []float64, ldq int, qc, qs []float64, compz lapack.OrthoComp, z []float64, ldz int, zc, zs []float64) {
	// Q and Z do not affect the elimination, so replay each rotation sequence in
	// its original order while traversing their rows contiguously.
	if compq != lapack.OrthoNone {
		qc = qc[:count]
		qs = qs[:count]
		i := 0
		for ; i+4 <= n; i += 4 {
			row0 := q[i*ldq+first-count : i*ldq+first+1]
			row1 := q[(i+1)*ldq+first-count : (i+1)*ldq+first+1]
			row2 := q[(i+2)*ldq+first-count : (i+2)*ldq+first+1]
			row3 := q[(i+3)*ldq+first-count : (i+3)*ldq+first+1]
			carry0 := row0[count]
			carry1 := row1[count]
			carry2 := row2[count]
			carry3 := row3[count]
			for k := 0; k < count; k++ {
				c := qc[k]
				s := qs[k]
				j := count - k
				vx0 := row0[j-1]
				vx1 := row1[j-1]
				vx2 := row2[j-1]
				vx3 := row3[j-1]
				low0 := c*vx0 + s*carry0
				low1 := c*vx1 + s*carry1
				low2 := c*vx2 + s*carry2
				low3 := c*vx3 + s*carry3
				row0[j] = c*carry0 - s*vx0
				row1[j] = c*carry1 - s*vx1
				row2[j] = c*carry2 - s*vx2
				row3[j] = c*carry3 - s*vx3
				carry0 = low0
				carry1 = low1
				carry2 = low2
				carry3 = low3
			}
			row0[0] = carry0
			row1[0] = carry1
			row2[0] = carry2
			row3[0] = carry3
		}
		for ; i < n; i++ {
			row := q[i*ldq+first-count : i*ldq+first+1]
			carry := row[count]
			for k := 0; k < count; k++ {
				c := qc[k]
				s := qs[k]
				j := count - k
				vx := row[j-1]
				low := c*vx + s*carry
				row[j] = c*carry - s*vx
				carry = low
			}
			row[0] = carry
		}
	}
	if compz != lapack.OrthoNone {
		zc = zc[:count]
		zs = zs[:count]
		i := 0
		for ; i+4 <= n; i += 4 {
			row0 := z[i*ldz+first-count : i*ldz+first+1]
			row1 := z[(i+1)*ldz+first-count : (i+1)*ldz+first+1]
			row2 := z[(i+2)*ldz+first-count : (i+2)*ldz+first+1]
			row3 := z[(i+3)*ldz+first-count : (i+3)*ldz+first+1]
			carry0 := row0[count]
			carry1 := row1[count]
			carry2 := row2[count]
			carry3 := row3[count]
			for k := 0; k < count; k++ {
				c := zc[k]
				s := zs[k]
				j := count - k
				vy0 := row0[j-1]
				vy1 := row1[j-1]
				vy2 := row2[j-1]
				vy3 := row3[j-1]
				low0 := c*vy0 - s*carry0
				low1 := c*vy1 - s*carry1
				low2 := c*vy2 - s*carry2
				low3 := c*vy3 - s*carry3
				row0[j] = c*carry0 + s*vy0
				row1[j] = c*carry1 + s*vy1
				row2[j] = c*carry2 + s*vy2
				row3[j] = c*carry3 + s*vy3
				carry0 = low0
				carry1 = low1
				carry2 = low2
				carry3 = low3
			}
			row0[0] = carry0
			row1[0] = carry1
			row2[0] = carry2
			row3[0] = carry3
		}
		for ; i < n; i++ {
			row := z[i*ldz+first-count : i*ldz+first+1]
			carry := row[count]
			for k := 0; k < count; k++ {
				c := zc[k]
				s := zs[k]
				j := count - k
				vy := row[j-1]
				low := c*vy - s*carry
				row[j] = c*carry + s*vy
				carry = low
			}
			row[0] = carry
		}
	}
}
