// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
)

// Dtgsyl solves the generalized Sylvester equation using Level 3 BLAS.
//
// When trans is blas.NoTrans, Dtgsyl solves:
//
//	A*R - L*B = scale*C
//	D*R - L*E = scale*F
//
// When trans is blas.Trans, Dtgsyl solves the transposed system:
//
//	Aᵀ*R + Dᵀ*L = scale*C
//	R*Bᵀ + L*Eᵀ = scale*(-F)
//
// where R and L are unknown m×n matrices, (A, D), (B, E), and (C, F) are
// given matrix pairs of size m×m, n×n, and m×n respectively. A and D must
// be upper quasi-triangular (in real Schur form), and B and E must be upper
// triangular.
//
// The ijob parameter specifies what is computed:
//   - ijob=0: solve only
//   - ijob=1: solve and compute a Frobenius norm-based estimate of Dif
//   - ijob=2: solve and compute a 1-norm-based estimate of Dif
//   - ijob=3: compute Dif only (trans must be NoTrans)
//   - ijob=4: compute Dif only using DGECON (trans must be NoTrans)
//
// C and F are overwritten by the solutions R and L.
//
// Dtgsyl returns scale (a scaling factor to avoid overflow), dif (if ijob >= 1),
// and ok=false if the matrix pair (A,D) and (B,E) have common or close eigenvalues.
//
// work must have length at least max(1, lwork). If lwork is -1, a workspace query
// is performed: the optimal workspace size is returned in work[0] and no other
// work is done.
//
// iwork must have length at least m+n+6.
func (impl Implementation) Dtgsyl(trans blas.Transpose, ijob, m, n int,
	a []float64, lda int, b []float64, ldb int, c []float64, ldc int,
	d []float64, ldd int, e []float64, lde int, f []float64, ldf int,
	work []float64, lwork int, iwork []int) (scale, dif float64, ok bool) {

	notran := trans == blas.NoTrans

	switch trans {
	case blas.NoTrans, blas.Trans:
	default:
		panic(badTrans)
	}
	switch {
	case notran && (ijob < 0 || ijob > 4):
		panic("lapack: invalid ijob")
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case lda < max(1, m):
		panic(badLdA)
	case ldb < max(1, n):
		panic(badLdB)
	case ldc < max(1, n):
		panic(badLdC)
	case ldd < max(1, m):
		panic(badLdD)
	case lde < max(1, n):
		panic(badLdE)
	case ldf < max(1, n):
		panic(badLdF)
	case lwork < 1 && lwork != -1:
		panic(badLWork)
	case len(work) < max(1, lwork):
		panic(shortWork)
	}

	// Quick return if possible.
	if m == 0 || n == 0 {
		scale = 1
		dif = 0
		work[0] = 1
		return scale, dif, true
	}

	// Compute workspace.
	lwmin := 1
	if notran && (ijob == 1 || ijob == 2) {
		lwmin = max(1, 2*m*n)
	}

	if lwork == -1 {
		work[0] = float64(lwmin)
		return 1, 0, true
	}

	switch {
	case len(a) < (m-1)*lda+m:
		panic(shortA)
	case len(b) < (n-1)*ldb+n:
		panic(shortB)
	case len(c) < (m-1)*ldc+n:
		panic(shortC)
	case len(d) < (m-1)*ldd+m:
		panic(shortD)
	case len(e) < (n-1)*lde+n:
		panic(shortE)
	case len(f) < (m-1)*ldf+n:
		panic(shortF)
	case len(iwork) < m+n+6:
		panic(shortIWork)
	case lwork < lwmin:
		panic(badLWork)
	}

	bi := blas64.Implementation()

	// Block sizes returned by ILAENV(2, 'DTGSYL') and ILAENV(5, 'DTGSYL').
	const mbBlock, nbBlock = 2, 2

	scale = 1
	dif = 0
	ok = true

	isolve := 1
	ifunc := 0
	if notran {
		if ijob >= 3 {
			ifunc = ijob - 2
			impl.Dlaset(blas.All, m, n, 0, 0, c, ldc)
			impl.Dlaset(blas.All, m, n, 0, 0, f, ldf)
		} else if ijob >= 1 {
			isolve = 2
		}
	}

	var dscale, dsum, scaloc, scale2 float64
	var pq, ppqq int
	var lok bool

	// setDif and swapRounds follow the end of each solve round in DTGSYL.
	setDif := func() {
		if dscale != 0 {
			if ijob == 1 || ijob == 3 {
				dif = math.Sqrt(float64(2*m*n)) / (dscale * math.Sqrt(dsum))
			} else {
				dif = math.Sqrt(float64(pq)) / (dscale * math.Sqrt(dsum))
			}
		}
	}
	swapRounds := func(iround int) {
		if isolve == 2 && iround == 1 {
			if notran {
				ifunc = ijob
			}
			scale2 = scale
			impl.Dlacpy(blas.All, m, n, c, ldc, work, n)
			impl.Dlacpy(blas.All, m, n, f, ldf, work[m*n:], n)
			impl.Dlaset(blas.All, m, n, 0, 0, c, ldc)
			impl.Dlaset(blas.All, m, n, 0, 0, f, ldf)
		} else if isolve == 2 && iround == 2 {
			impl.Dlacpy(blas.All, m, n, work, n, c, ldc)
			impl.Dlacpy(blas.All, m, n, work[m*n:], n, f, ldf)
			scale = scale2
		}
	}

	if mbBlock >= m && nbBlock >= n {
		// Use the unblocked Level 2 solver.
		for iround := 1; iround <= isolve; iround++ {
			dscale, dsum, pq = 0, 1, 0
			scale, dsum, dscale, pq, lok = impl.Dtgsy2(trans, ifunc, m, n,
				a, lda, b, ldb, c, ldc, d, ldd, e, lde, f, ldf, dsum, dscale, iwork)
			if !lok {
				ok = false
			}
			setDif()
			swapRounds(iround)
		}
		work[0] = float64(lwmin)
		return scale, dif, ok
	}

	// Determine the block structure of A and B as DTGSYL does. iwork holds
	// 1-based block starts: A's in iwork[0:p+1] and B's in iwork[p+1:q+1],
	// with indices p and q as in DTGSYL.
	p := 0
	i := 1
	for i <= m {
		p++
		iwork[p-1] = i
		i += mbBlock
		if i >= m {
			break
		}
		if a[(i-1)*lda+i-2] != 0 {
			i++
		}
	}
	iwork[p] = m + 1
	if iwork[p-1] == iwork[p] {
		p--
	}
	q := p + 1
	j := 1
	for j <= n {
		q++
		iwork[q-1] = j
		j += nbBlock
		if j >= n {
			break
		}
		if b[(j-1)*ldb+j-2] != 0 {
			j++
		}
	}
	iwork[q] = n + 1
	if iwork[q-1] == iwork[q] {
		q--
	}

	// rescale applies scaloc to the parts of C and F outside the block
	// rows is:ie and columns js:je (1-based, inclusive).
	rescale := func(is, ie, js, je int) {
		for k := 1; k <= js-1; k++ {
			bi.Dscal(m, scaloc, c[k-1:], ldc)
			bi.Dscal(m, scaloc, f[k-1:], ldf)
		}
		for k := js; k <= je; k++ {
			bi.Dscal(is-1, scaloc, c[k-1:], ldc)
			bi.Dscal(is-1, scaloc, f[k-1:], ldf)
		}
		if m > ie {
			for k := js; k <= je; k++ {
				bi.Dscal(m-ie, scaloc, c[ie*ldc+k-1:], ldc)
				bi.Dscal(m-ie, scaloc, f[ie*ldf+k-1:], ldf)
			}
		}
		for k := je + 1; k <= n; k++ {
			bi.Dscal(m, scaloc, c[k-1:], ldc)
			bi.Dscal(m, scaloc, f[k-1:], ldf)
		}
		scale *= scaloc
	}

	if notran {
		// Solve (I, J)-subsystem
		//  A(I, I) * R(I, J) - L(I, J) * B(J, J) = C(I, J)
		//  D(I, I) * R(I, J) - L(I, J) * E(J, J) = F(I, J)
		// for I = P, P - 1,..., 1; J = 1, 2,..., Q.
		for iround := 1; iround <= isolve; iround++ {
			dscale, dsum, pq, scale = 0, 1, 0, 1
			for jj := p + 2; jj <= q; jj++ {
				js := iwork[jj-1]
				je := iwork[jj] - 1
				nbj := je - js + 1
				for ii := p; ii >= 1; ii-- {
					is := iwork[ii-1]
					ie := iwork[ii] - 1
					mbi := ie - is + 1
					scaloc, dsum, dscale, ppqq, lok = impl.Dtgsy2(trans, ifunc, mbi, nbj,
						a[(is-1)*lda+is-1:], lda, b[(js-1)*ldb+js-1:], ldb, c[(is-1)*ldc+js-1:], ldc,
						d[(is-1)*ldd+is-1:], ldd, e[(js-1)*lde+js-1:], lde, f[(is-1)*ldf+js-1:], ldf,
						dsum, dscale, iwork[q+1:])
					if !lok {
						ok = false
					}
					pq += ppqq
					if scaloc != 1 {
						rescale(is, ie, js, je)
					}

					// Substitute R(I, J) and L(I, J) into remaining equation.
					if ii > 1 {
						bi.Dgemm(blas.NoTrans, blas.NoTrans, is-1, nbj, mbi, -1,
							a[is-1:], lda, c[(is-1)*ldc+js-1:], ldc, 1, c[js-1:], ldc)
						bi.Dgemm(blas.NoTrans, blas.NoTrans, is-1, nbj, mbi, -1,
							d[is-1:], ldd, c[(is-1)*ldc+js-1:], ldc, 1, f[js-1:], ldf)
					}
					if jj < q {
						bi.Dgemm(blas.NoTrans, blas.NoTrans, mbi, n-je, nbj, 1,
							f[(is-1)*ldf+js-1:], ldf, b[(js-1)*ldb+je:], ldb, 1, c[(is-1)*ldc+je:], ldc)
						bi.Dgemm(blas.NoTrans, blas.NoTrans, mbi, n-je, nbj, 1,
							f[(is-1)*ldf+js-1:], ldf, e[(js-1)*lde+je:], lde, 1, f[(is-1)*ldf+je:], ldf)
					}
				}
			}
			setDif()
			swapRounds(iround)
		}
	} else {
		// Solve transposed (I, J)-subsystem
		//  A(I, I)ᵀ * R(I, J)  + D(I, I)ᵀ * L(I, J)  =  C(I, J)
		//  R(I, J)  * B(J, J)ᵀ + L(I, J)  * E(J, J)ᵀ = -F(I, J)
		// for I = 1,2,..., P; J = Q, Q-1,..., 1.
		scale = 1
		for ii := 1; ii <= p; ii++ {
			is := iwork[ii-1]
			ie := iwork[ii] - 1
			mbi := ie - is + 1
			for jj := q; jj >= p+2; jj-- {
				js := iwork[jj-1]
				je := iwork[jj] - 1
				nbj := je - js + 1
				scaloc, dsum, dscale, _, lok = impl.Dtgsy2(trans, ifunc, mbi, nbj,
					a[(is-1)*lda+is-1:], lda, b[(js-1)*ldb+js-1:], ldb, c[(is-1)*ldc+js-1:], ldc,
					d[(is-1)*ldd+is-1:], ldd, e[(js-1)*lde+js-1:], lde, f[(is-1)*ldf+js-1:], ldf,
					dsum, dscale, iwork[q+1:])
				if !lok {
					ok = false
				}
				if scaloc != 1 {
					rescale(is, ie, js, je)
				}

				// Substitute R(I, J) and L(I, J) into remaining equation.
				if jj > p+2 {
					bi.Dgemm(blas.NoTrans, blas.Trans, mbi, js-1, nbj, 1,
						c[(is-1)*ldc+js-1:], ldc, b[js-1:], ldb, 1, f[(is-1)*ldf:], ldf)
					bi.Dgemm(blas.NoTrans, blas.Trans, mbi, js-1, nbj, 1,
						f[(is-1)*ldf+js-1:], ldf, e[js-1:], lde, 1, f[(is-1)*ldf:], ldf)
				}
				if ii < p {
					bi.Dgemm(blas.Trans, blas.NoTrans, m-ie, nbj, mbi, -1,
						a[(is-1)*lda+ie:], lda, c[(is-1)*ldc+js-1:], ldc, 1, c[ie*ldc+js-1:], ldc)
					bi.Dgemm(blas.Trans, blas.NoTrans, m-ie, nbj, mbi, -1,
						d[(is-1)*ldd+ie:], ldd, f[(is-1)*ldf+js-1:], ldf, 1, c[ie*ldc+js-1:], ldc)
				}
			}
		}
	}

	work[0] = float64(lwmin)
	return scale, dif, ok
}
