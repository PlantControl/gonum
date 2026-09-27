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

	scale = 1
	dif = 0
	ok = true

	// Determine block structure of A.
	p := 0
	i := 0
	for i < m {
		if i == m-1 || a[(i+1)*lda+i] == 0 {
			iwork[p] = i + 1
			p++
			i++
		} else {
			iwork[p] = -(i + 1)
			p++
			i += 2
		}
	}
	iwork[p] = m + 1

	// Determine block structure of B.
	q := p + 1
	j := 0
	for j < n {
		if j == n-1 || b[(j+1)*ldb+j] == 0 {
			iwork[q] = j + 1
			q++
			j++
		} else {
			iwork[q] = -(j + 1)
			q++
			j += 2
		}
	}
	iwork[q] = n + 1
	q = q - p - 1

	isolve := 1
	ifunc := 0
	if ijob >= 3 && notran {
		ifunc = ijob - 2
		impl.Dlaset(blas.All, m, n, 0, 0, c, ldc)
		impl.Dlaset(blas.All, m, n, 0, 0, f, ldf)
	} else if ijob >= 1 && notran {
		isolve = 2
	}

	var dscale, dsum, scaloc, scale2 float64
	var pq int
	var lok bool

	for iround := 1; iround <= isolve; iround++ {
		scale = 1
		dscale = 0
		dsum = 1
		pq = 0

		if notran {
			// Solve (I, J) - subsystem:
			//   A[I,I]*R[I,J] - L[I,J]*B[J,J] = C[I,J]
			//   D[I,I]*R[I,J] - L[I,J]*E[J,J] = F[I,J]
			// for I = P, P-1, ..., 1; J = 1, 2, ..., Q.
			for j := p + 1; j <= p+q; j++ {
				js := iwork[j]
				je := abs(iwork[j+1]) - 1
				nbj := je - abs(js) + 1
				if js < 0 {
					js = -js
				}
				js-- // 0-based

				for i := p; i >= 1; i-- {
					is := iwork[i-1]
					ie := abs(iwork[i]) - 1
					mbi := ie - abs(is) + 1
					if is < 0 {
						is = -is
					}
					is-- // 0-based

					// Solve the (I, J)-subsystem using Dtgsy2.
					var ppqq int
					scaloc, dsum, dscale, ppqq, lok = impl.Dtgsy2(trans, ifunc, mbi, nbj,
						a[is*lda+is:], lda, b[js*ldb+js:], ldb, c[is*ldc+js:], ldc,
						d[is*ldd+is:], ldd, e[js*lde+js:], lde, f[is*ldf+js:], ldf,
						dsum, dscale, iwork[p+q+2:])
					pq += ppqq
					if !lok {
						ok = false
					}

					if scaloc != 1 {
						for k := 0; k < js; k++ {
							bi.Dscal(m, scaloc, c[k:], ldc)
							bi.Dscal(m, scaloc, f[k:], ldf)
						}
						for k := js; k < js+nbj; k++ {
							bi.Dscal(is, scaloc, c[k:], ldc)
							bi.Dscal(is, scaloc, f[k:], ldf)
							bi.Dscal(m-is-mbi, scaloc, c[(is+mbi)*ldc+k:], ldc)
							bi.Dscal(m-is-mbi, scaloc, f[(is+mbi)*ldf+k:], ldf)
						}
						for k := js + nbj; k < n; k++ {
							bi.Dscal(m, scaloc, c[k:], ldc)
							bi.Dscal(m, scaloc, f[k:], ldf)
						}
						scale *= scaloc
					}

					// Substitute R[I,J] and L[I,J] into remaining equation.
					if i > 1 {
						bi.Dgemm(blas.NoTrans, blas.NoTrans, is, nbj, mbi, -1,
							a[is:], lda, c[is*ldc+js:], ldc, 1, c[js:], ldc)
						bi.Dgemm(blas.NoTrans, blas.NoTrans, is, nbj, mbi, -1,
							d[is:], ldd, c[is*ldc+js:], ldc, 1, f[js:], ldf)
					}
					if j < p+q {
						bi.Dgemm(blas.NoTrans, blas.NoTrans, mbi, n-js-nbj, nbj, 1,
							f[is*ldf+js:], ldf, b[js*ldb+js+nbj:], ldb, 1, c[is*ldc+js+nbj:], ldc)
						bi.Dgemm(blas.NoTrans, blas.NoTrans, mbi, n-js-nbj, nbj, 1,
							f[is*ldf+js:], ldf, e[js*lde+js+nbj:], lde, 1, f[is*ldf+js+nbj:], ldf)
					}
				}
			}
		} else {
			// Solve transposed (I, J) - subsystem:
			//   Aᵀ[I,I]*R[I,J] + Dᵀ[I,I]*L[I,J] = C[I,J]
			//   R[I,J]*Bᵀ[J,J] + L[I,J]*Eᵀ[J,J] = -F[I,J]
			// for I = 1, 2, ..., P; J = Q, Q-1, ..., 1.
			for i := 1; i <= p; i++ {
				is := iwork[i-1]
				ie := abs(iwork[i]) - 1
				mbi := ie - abs(is) + 1
				if is < 0 {
					is = -is
				}
				is-- // 0-based

				for j := p + q; j >= p+1; j-- {
					js := iwork[j]
					je := abs(iwork[j+1]) - 1
					nbj := je - abs(js) + 1
					if js < 0 {
						js = -js
					}
					js-- // 0-based

					// Solve the (I, J)-subsystem using Dtgsy2.
					var ppqq int
					scaloc, dsum, dscale, ppqq, lok = impl.Dtgsy2(trans, ifunc, mbi, nbj,
						a[is*lda+is:], lda, b[js*ldb+js:], ldb, c[is*ldc+js:], ldc,
						d[is*ldd+is:], ldd, e[js*lde+js:], lde, f[is*ldf+js:], ldf,
						dsum, dscale, iwork[p+q+2:])
					pq += ppqq
					if !lok {
						ok = false
					}

					if scaloc != 1 {
						for k := 0; k < js; k++ {
							bi.Dscal(m, scaloc, c[k:], ldc)
							bi.Dscal(m, scaloc, f[k:], ldf)
						}
						for k := js; k < js+nbj; k++ {
							bi.Dscal(is, scaloc, c[k:], ldc)
							bi.Dscal(is, scaloc, f[k:], ldf)
							bi.Dscal(m-is-mbi, scaloc, c[(is+mbi)*ldc+k:], ldc)
							bi.Dscal(m-is-mbi, scaloc, f[(is+mbi)*ldf+k:], ldf)
						}
						for k := js + nbj; k < n; k++ {
							bi.Dscal(m, scaloc, c[k:], ldc)
							bi.Dscal(m, scaloc, f[k:], ldf)
						}
						scale *= scaloc
					}

					// Substitute R[I,J] and L[I,J] into remaining equation.
					if j > p+1 {
						bi.Dgemm(blas.NoTrans, blas.Trans, mbi, js, nbj, 1,
							c[is*ldc+js:], ldc, b[js:], ldb, 1, f[is*ldf:], ldf)
						bi.Dgemm(blas.NoTrans, blas.Trans, mbi, js, nbj, 1,
							f[is*ldf+js:], ldf, e[js:], lde, 1, f[is*ldf:], ldf)
					}
					if i < p {
						bi.Dgemm(blas.Trans, blas.NoTrans, m-is-mbi, nbj, mbi, -1,
							a[is*lda+is+mbi:], lda, c[is*ldc+js:], ldc, 1, c[(is+mbi)*ldc+js:], ldc)
						bi.Dgemm(blas.Trans, blas.NoTrans, m-is-mbi, nbj, mbi, -1,
							d[is*ldd+is+mbi:], ldd, f[is*ldf+js:], ldf, 1, c[(is+mbi)*ldc+js:], ldc)
					}
				}
			}
		}

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

	// Compute Dif.
	if notran && dscale != 0 {
		switch ijob {
		case 1, 3:
			dif = math.Sqrt(float64(2*m*n)) / (dscale * math.Sqrt(dsum))
		case 2, 4:
			dif = math.Sqrt(float64(pq)) / (dscale * math.Sqrt(dsum))
		}
	}

	work[0] = float64(lwmin)
	return scale, dif, ok
}
