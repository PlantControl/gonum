// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/lapack"
)

// Dtgsy2 solves the generalized Sylvester equation using Level 1 and 2 BLAS.
//
// When trans is blas.NoTrans, Dtgsy2 solves the coupled system:
//
//	A*R - L*B = scale*C
//	D*R - L*E = scale*F
//
// When trans is blas.Trans, Dtgsy2 solves the transposed system:
//
//	Aᵀ*R + Dᵀ*L = scale*C
//	R*Bᵀ + L*Eᵀ = scale*(-F)
//
// A and D are m×m upper quasi-triangular matrices in Schur canonical form,
// B and E are n×n upper triangular matrices. C, F, R, L are m×n matrices,
// with R and L unknown. On input, C and F contain the right-hand sides.
// On output, C and F are overwritten by the solutions R and L.
//
// The ijob parameter specifies what is computed:
//   - ijob=0: solve only
//   - ijob=1: look-ahead strategy for Dif estimation (only when trans=NoTrans)
//   - ijob=2: DGECON-based Dif estimation (only when trans=NoTrans)
//
// The pq return value gives the number of subsystems solved (of size 2×2, 4×4, 8×8).
// rdsum and rdscal are used for Dif estimation and are updated accordingly.
//
// Dtgsy2 returns ok=false if the matrix pair (A,D) and (B,E) share eigenvalues,
// which indicates an ill-conditioned problem.
//
// Dtgsy2 is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dtgsy2(trans blas.Transpose, ijob, m, n int, a []float64, lda int,
	b []float64, ldb int, c []float64, ldc int, d []float64, ldd int,
	e []float64, lde int, f []float64, ldf int,
	rdsum, rdscal float64, iwork []int) (scale, rdsum2, rdscal2 float64, pq int, ok bool) {
	return impl.dtgsy2(trans, ijob, m, n, a, lda, b, ldb, c, ldc, d, ldd, e, lde, f, ldf, rdsum, rdscal, iwork, nil)
}

type dtgsy2Scratch struct {
	z   [64]float64
	rhs [8]float64
}

func (impl Implementation) dtgsy2(trans blas.Transpose, ijob, m, n int, a []float64, lda int,
	b []float64, ldb int, c []float64, ldc int, d []float64, ldd int,
	e []float64, lde int, f []float64, ldf int,
	rdsum, rdscal float64, iwork []int, scratch *dtgsy2Scratch) (scale, rdsum2, rdscal2 float64, pq int, ok bool) {

	switch trans {
	case blas.NoTrans, blas.Trans:
	default:
		panic(badTrans)
	}
	switch {
	case trans == blas.NoTrans && (ijob < 0 || ijob > 2):
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
	}

	scale = 1
	rdsum2, rdscal2 = rdsum, rdscal
	ok = true

	if m == 0 || n == 0 {
		return scale, rdsum2, rdscal2, 0, ok
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
	case len(iwork) < m+n+2:
		panic(shortIWork)
	}

	notran := trans == blas.NoTrans

	bi := blas64.Implementation()

	const ldz = 8
	if scratch == nil {
		scratch = new(dtgsy2Scratch)
	}
	*scratch = dtgsy2Scratch{}
	z, rhs := scratch.z[:], scratch.rhs[:]
	var ipiv, jpiv [ldz]int

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
	q -= p + 1

	pq = 0

	if notran {
		// Solve (I,J) - subsystem:
		// A[I,I]*R[I,J] - L[I,J]*B[J,J] = C[I,J]
		// D[I,I]*R[I,J] - L[I,J]*E[J,J] = F[I,J]
		// for I = P, P-1,...,1; J = 1,2,...,Q.

		for j := p + 1; j <= p+q; j++ {
			js := iwork[j]
			je := abs(iwork[j+1]) - 1
			nb := je - abs(js) + 1
			if js < 0 {
				js = -js
			}
			js--                        // Convert to 0-based.
			je = abs(iwork[j]) + nb - 2 // Convert to 0-based end index.

			for i := p; i >= 1; i-- {
				is := iwork[i-1]
				ie := abs(iwork[i]) - 1
				mb := ie - abs(is) + 1
				if is < 0 {
					is = -is
				}
				is-- // Convert to 0-based.

				// Solve the (I,J)-subsystem.
				if mb == 1 && nb == 1 {
					// Build a 2×2 system:
					// [A[I,I]  -B[J,J]] [R] = [C[I,J]]
					// [D[I,I]  -E[J,J]] [L]   [F[I,J]]
					z[0] = a[is*lda+is]
					z[1] = -b[js*ldb+js]
					z[ldz] = d[is*ldd+is]
					z[ldz+1] = -e[js*lde+js]

					rhs[0] = c[is*ldc+js]
					rhs[1] = f[is*ldf+js]

					k := impl.Dgetc2(2, z[:], ldz, ipiv[:2], jpiv[:2])
					if k >= 0 {
						ok = false
					}
					if ijob == 0 {
						scaloc := impl.Dgesc2(2, z[:], ldz, rhs[:], ipiv[:2], jpiv[:2])
						if scaloc != 1 {
							for kk := 0; kk < n; kk++ {
								bi.Dscal(m, scaloc, c[kk:], ldc)
								bi.Dscal(m, scaloc, f[kk:], ldf)
							}
							scale *= scaloc
						}
					} else {
						rdscal2, rdsum2 = impl.Dlatdf(lapack.MaximizeNormXJob(ijob&2), 2, z[:], ldz, rhs[:], rdsum2, rdscal2, ipiv[:2], jpiv[:2])
					}
					c[is*ldc+js] = rhs[0]
					f[is*ldf+js] = rhs[1]

				} else if mb == 1 && nb == 2 {
					// Build a 4×4 system.
					z[0] = a[is*lda+is]
					z[1] = 0
					z[2] = -b[js*ldb+js]
					z[3] = -b[(js+1)*ldb+js]

					z[ldz] = 0
					z[ldz+1] = a[is*lda+is]
					z[ldz+2] = -b[js*ldb+js+1]
					z[ldz+3] = -b[(js+1)*ldb+js+1]

					z[2*ldz] = d[is*ldd+is]
					z[2*ldz+1] = 0
					z[2*ldz+2] = -e[js*lde+js]
					z[2*ldz+3] = 0

					z[3*ldz] = 0
					z[3*ldz+1] = d[is*ldd+is]
					z[3*ldz+2] = -e[js*lde+js+1]
					z[3*ldz+3] = -e[(js+1)*lde+js+1]

					rhs[0] = c[is*ldc+js]
					rhs[1] = c[is*ldc+js+1]
					rhs[2] = f[is*ldf+js]
					rhs[3] = f[is*ldf+js+1]

					k := impl.Dgetc2(4, z[:], ldz, ipiv[:4], jpiv[:4])
					if k >= 0 {
						ok = false
					}
					if ijob == 0 {
						scaloc := impl.Dgesc2(4, z[:], ldz, rhs[:], ipiv[:4], jpiv[:4])
						if scaloc != 1 {
							for kk := 0; kk < n; kk++ {
								bi.Dscal(m, scaloc, c[kk:], ldc)
								bi.Dscal(m, scaloc, f[kk:], ldf)
							}
							scale *= scaloc
						}
					} else {
						rdscal2, rdsum2 = impl.Dlatdf(lapack.MaximizeNormXJob(ijob&2), 4, z[:], ldz, rhs[:], rdsum2, rdscal2, ipiv[:4], jpiv[:4])
					}
					c[is*ldc+js] = rhs[0]
					c[is*ldc+js+1] = rhs[1]
					f[is*ldf+js] = rhs[2]
					f[is*ldf+js+1] = rhs[3]

				} else if mb == 2 && nb == 1 {
					// Build a 4×4 system.
					// Row 0: A[is,is]*R[is] + A[is,is+1]*R[is+1] - B[js,js]*L[is] = C[is,js]
					z[0] = a[is*lda+is]
					z[1] = a[is*lda+is+1]
					z[2] = -b[js*ldb+js]
					z[3] = 0

					// Row 1: A[is+1,is]*R[is] + A[is+1,is+1]*R[is+1] - B[js,js]*L[is+1] = C[is+1,js]
					z[ldz] = a[(is+1)*lda+is]
					z[ldz+1] = a[(is+1)*lda+is+1]
					z[ldz+2] = 0
					z[ldz+3] = -b[js*ldb+js]

					// Row 2: D[is,is]*R[is] + D[is,is+1]*R[is+1] - E[js,js]*L[is] = F[is,js]
					z[2*ldz] = d[is*ldd+is]
					z[2*ldz+1] = d[is*ldd+is+1]
					z[2*ldz+2] = -e[js*lde+js]
					z[2*ldz+3] = 0

					// Row 3: D[is+1,is+1]*R[is+1] - E[js,js]*L[is+1] = F[is+1,js] (D[is+1,is]=0 for upper triangular)
					z[3*ldz] = d[(is+1)*ldd+is]
					z[3*ldz+1] = d[(is+1)*ldd+is+1]
					z[3*ldz+2] = 0
					z[3*ldz+3] = -e[js*lde+js]

					rhs[0] = c[is*ldc+js]
					rhs[1] = c[(is+1)*ldc+js]
					rhs[2] = f[is*ldf+js]
					rhs[3] = f[(is+1)*ldf+js]

					k := impl.Dgetc2(4, z[:], ldz, ipiv[:4], jpiv[:4])
					if k >= 0 {
						ok = false
					}
					if ijob == 0 {
						scaloc := impl.Dgesc2(4, z[:], ldz, rhs[:], ipiv[:4], jpiv[:4])
						if scaloc != 1 {
							for kk := 0; kk < n; kk++ {
								bi.Dscal(m, scaloc, c[kk:], ldc)
								bi.Dscal(m, scaloc, f[kk:], ldf)
							}
							scale *= scaloc
						}
					} else {
						rdscal2, rdsum2 = impl.Dlatdf(lapack.MaximizeNormXJob(ijob&2), 4, z[:], ldz, rhs[:], rdsum2, rdscal2, ipiv[:4], jpiv[:4])
					}
					c[is*ldc+js] = rhs[0]
					c[(is+1)*ldc+js] = rhs[1]
					f[is*ldf+js] = rhs[2]
					f[(is+1)*ldf+js] = rhs[3]

				} else {
					// Build an 8×8 system.
					// mb==2, nb==2
					// Unknowns: [R[is,js], R[is+1,js], R[is,js+1], R[is+1,js+1], L[is,js], L[is+1,js], L[is,js+1], L[is+1,js+1]]
					impl.Dlaset(blas.All, ldz, ldz, 0, 0, z[:], ldz)

					// Row 0: A[is,is]*R[is,js] + A[is,is+1]*R[is+1,js] - L[is,js]*B[js,js] - L[is,js+1]*B[js+1,js] = C[is,js]
					z[0] = a[is*lda+is]
					z[1] = a[is*lda+is+1]
					z[4] = -b[js*ldb+js]
					z[6] = -b[(js+1)*ldb+js]

					// Row 1: A[is+1,is]*R[is,js] + A[is+1,is+1]*R[is+1,js] - L[is+1,js]*B[js,js] - L[is+1,js+1]*B[js+1,js] = C[is+1,js]
					z[ldz] = a[(is+1)*lda+is]
					z[ldz+1] = a[(is+1)*lda+is+1]
					z[ldz+5] = -b[js*ldb+js]
					z[ldz+7] = -b[(js+1)*ldb+js]

					// Row 2: A[is,is]*R[is,js+1] + A[is,is+1]*R[is+1,js+1] - L[is,js]*B[js,js+1] - L[is,js+1]*B[js+1,js+1] = C[is,js+1]
					z[2*ldz+2] = a[is*lda+is]
					z[2*ldz+3] = a[is*lda+is+1]
					z[2*ldz+4] = -b[js*ldb+js+1]
					z[2*ldz+6] = -b[(js+1)*ldb+js+1]

					// Row 3: A[is+1,is]*R[is,js+1] + A[is+1,is+1]*R[is+1,js+1] - L[is+1,js]*B[js,js+1] - L[is+1,js+1]*B[js+1,js+1] = C[is+1,js+1]
					z[3*ldz+2] = a[(is+1)*lda+is]
					z[3*ldz+3] = a[(is+1)*lda+is+1]
					z[3*ldz+5] = -b[js*ldb+js+1]
					z[3*ldz+7] = -b[(js+1)*ldb+js+1]

					// Row 4: D[is,is]*R[is,js] + D[is,is+1]*R[is+1,js] - L[is,js]*E[js,js] = F[is,js]
					z[4*ldz] = d[is*ldd+is]
					z[4*ldz+1] = d[is*ldd+is+1]
					z[4*ldz+4] = -e[js*lde+js]

					// Row 5: D[is+1,is]*R[is,js] + D[is+1,is+1]*R[is+1,js] - L[is+1,js]*E[js,js] = F[is+1,js]
					z[5*ldz] = d[(is+1)*ldd+is]
					z[5*ldz+1] = d[(is+1)*ldd+is+1]
					z[5*ldz+5] = -e[js*lde+js]

					// Row 6: D[is,is]*R[is,js+1] + D[is,is+1]*R[is+1,js+1] - L[is,js]*E[js,js+1] - L[is,js+1]*E[js+1,js+1] = F[is,js+1]
					z[6*ldz+2] = d[is*ldd+is]
					z[6*ldz+3] = d[is*ldd+is+1]
					z[6*ldz+4] = -e[js*lde+js+1]
					z[6*ldz+6] = -e[(js+1)*lde+js+1]

					// Row 7: D[is+1,is]*R[is,js+1] + D[is+1,is+1]*R[is+1,js+1] - L[is+1,js]*E[js,js+1] - L[is+1,js+1]*E[js+1,js+1] = F[is+1,js+1]
					z[7*ldz+2] = d[(is+1)*ldd+is]
					z[7*ldz+3] = d[(is+1)*ldd+is+1]
					z[7*ldz+5] = -e[js*lde+js+1]
					z[7*ldz+7] = -e[(js+1)*lde+js+1]

					rhs[0] = c[is*ldc+js]
					rhs[1] = c[(is+1)*ldc+js]
					rhs[2] = c[is*ldc+js+1]
					rhs[3] = c[(is+1)*ldc+js+1]
					rhs[4] = f[is*ldf+js]
					rhs[5] = f[(is+1)*ldf+js]
					rhs[6] = f[is*ldf+js+1]
					rhs[7] = f[(is+1)*ldf+js+1]

					k := impl.Dgetc2(8, z[:], ldz, ipiv[:8], jpiv[:8])
					if k >= 0 {
						ok = false
					}
					if ijob == 0 {
						scaloc := impl.Dgesc2(8, z[:], ldz, rhs[:], ipiv[:8], jpiv[:8])
						if scaloc != 1 {
							for kk := 0; kk < n; kk++ {
								bi.Dscal(m, scaloc, c[kk:], ldc)
								bi.Dscal(m, scaloc, f[kk:], ldf)
							}
							scale *= scaloc
						}
					} else {
						rdscal2, rdsum2 = impl.Dlatdf(lapack.MaximizeNormXJob(ijob&2), 8, z[:], ldz, rhs[:], rdsum2, rdscal2, ipiv[:8], jpiv[:8])
					}
					c[is*ldc+js] = rhs[0]
					c[(is+1)*ldc+js] = rhs[1]
					c[is*ldc+js+1] = rhs[2]
					c[(is+1)*ldc+js+1] = rhs[3]
					f[is*ldf+js] = rhs[4]
					f[(is+1)*ldf+js] = rhs[5]
					f[is*ldf+js+1] = rhs[6]
					f[(is+1)*ldf+js+1] = rhs[7]
				}

				pq++

				// Substitute R[I,J] and L[I,J] into remaining equation.
				if i > 1 {
					alpha := -1.0
					bi.Dgemm(blas.NoTrans, blas.NoTrans, is, nb, mb, alpha, a[is:], lda, c[is*ldc+js:], ldc, 1, c[js:], ldc)
					bi.Dgemm(blas.NoTrans, blas.NoTrans, is, nb, mb, alpha, d[is:], ldd, c[is*ldc+js:], ldc, 1, f[js:], ldf)
				}
				if j < p+q {
					bi.Dgemm(blas.NoTrans, blas.NoTrans, mb, n-je-1, nb, 1, f[is*ldf+js:], ldf, b[js*ldb+je+1:], ldb, 1, c[is*ldc+je+1:], ldc)
					bi.Dgemm(blas.NoTrans, blas.NoTrans, mb, n-je-1, nb, 1, f[is*ldf+js:], ldf, e[js*lde+je+1:], lde, 1, f[is*ldf+je+1:], ldf)
				}
			}
		}
	} else {
		// Solve transposed system:
		// Aᵀ*R[I,J] + Dᵀ*L[I,J] = C[I,J]
		// R[I,J]*Bᵀ + L[I,J]*Eᵀ = -F[I,J]
		// for I = 1,2,...,P; J = Q, Q-1,...,1.

		for i := 1; i <= p; i++ {
			is := iwork[i-1]
			ie := abs(iwork[i]) - 1
			mb := ie - abs(is) + 1
			if is < 0 {
				is = -is
			}
			is--             // Convert to 0-based.
			ie = is + mb - 1 // 0-based end index.

			for j := p + q; j >= p+1; j-- {
				js := iwork[j]
				je := abs(iwork[j+1]) - 1
				nb := je - abs(js) + 1
				if js < 0 {
					js = -js
				}
				js-- // Convert to 0-based.

				// Solve the (I,J)-subsystem.
				if mb == 1 && nb == 1 {
					// Build a 2×2 system:
					// [Aᵀ[I,I]  Dᵀ[I,I]] [R] = [C[I,J]]
					// [Bᵀ[J,J]  Eᵀ[J,J]] [L]   [-F[I,J]]
					z[0] = a[is*lda+is]
					z[1] = d[is*ldd+is]
					z[ldz] = b[js*ldb+js]
					z[ldz+1] = e[js*lde+js]

					rhs[0] = c[is*ldc+js]
					rhs[1] = -f[is*ldf+js]

					k := impl.Dgetc2(2, z[:], ldz, ipiv[:2], jpiv[:2])
					if k >= 0 {
						ok = false
					}
					scaloc := impl.Dgesc2(2, z[:], ldz, rhs[:], ipiv[:2], jpiv[:2])
					if scaloc != 1 {
						for kk := 0; kk < n; kk++ {
							bi.Dscal(m, scaloc, c[kk:], ldc)
							bi.Dscal(m, scaloc, f[kk:], ldf)
						}
						scale *= scaloc
					}
					c[is*ldc+js] = rhs[0]
					f[is*ldf+js] = rhs[1]

				} else if mb == 1 && nb == 2 {
					// Build a 4×4 system.
					// For trans: Aᵀ*R + Dᵀ*L = C, R*Bᵀ + L*Eᵀ = -F
					// Unknowns: [R[is,js], R[is,js+1], L[is,js], L[is,js+1]]
					// Rows 0,1: C[is,js] and C[is,js+1]
					z[0] = a[is*lda+is]
					z[1] = 0
					z[2] = d[is*ldd+is]
					z[3] = 0

					z[ldz] = 0
					z[ldz+1] = a[is*lda+is]
					z[ldz+2] = 0
					z[ldz+3] = d[is*ldd+is]

					// Rows 2,3: -F[is,js] and -F[is,js+1] from R*Bᵀ + L*Eᵀ
					z[2*ldz] = b[js*ldb+js]
					z[2*ldz+1] = b[js*ldb+js+1]
					z[2*ldz+2] = e[js*lde+js]
					z[2*ldz+3] = e[js*lde+js+1]

					z[3*ldz] = b[(js+1)*ldb+js]
					z[3*ldz+1] = b[(js+1)*ldb+js+1]
					z[3*ldz+2] = e[(js+1)*lde+js]
					z[3*ldz+3] = e[(js+1)*lde+js+1]

					rhs[0] = c[is*ldc+js]
					rhs[1] = c[is*ldc+js+1]
					rhs[2] = -f[is*ldf+js]
					rhs[3] = -f[is*ldf+js+1]

					k := impl.Dgetc2(4, z[:], ldz, ipiv[:4], jpiv[:4])
					if k >= 0 {
						ok = false
					}
					scaloc := impl.Dgesc2(4, z[:], ldz, rhs[:], ipiv[:4], jpiv[:4])
					if scaloc != 1 {
						for kk := 0; kk < n; kk++ {
							bi.Dscal(m, scaloc, c[kk:], ldc)
							bi.Dscal(m, scaloc, f[kk:], ldf)
						}
						scale *= scaloc
					}
					c[is*ldc+js] = rhs[0]
					c[is*ldc+js+1] = rhs[1]
					f[is*ldf+js] = rhs[2]
					f[is*ldf+js+1] = rhs[3]

				} else if mb == 2 && nb == 1 {
					// Build a 4×4 system.
					// For trans case: Aᵀ*R + Dᵀ*L = C, R*Bᵀ + L*Eᵀ = -F
					// Unknowns: [R[is,js], R[is+1,js], L[is,js], L[is+1,js]]
					// Row 0: Aᵀ[is,is]*R[is] + Aᵀ[is,is+1]*R[is+1] + Dᵀ[is,is]*L[is] + Dᵀ[is,is+1]*L[is+1] = C[is]
					// Note: Aᵀ[i,j] = A[j,i]
					z[0] = a[is*lda+is]
					z[1] = a[(is+1)*lda+is]
					z[2] = d[is*ldd+is]
					z[3] = d[(is+1)*ldd+is]

					// Row 1: Aᵀ[is+1,is]*R[is] + Aᵀ[is+1,is+1]*R[is+1] + Dᵀ[is+1,is]*L[is] + Dᵀ[is+1,is+1]*L[is+1] = C[is+1]
					z[ldz] = a[is*lda+is+1]
					z[ldz+1] = a[(is+1)*lda+is+1]
					z[ldz+2] = d[is*ldd+is+1]
					z[ldz+3] = d[(is+1)*ldd+is+1]

					// Row 2: R[is]*Bᵀ[js,js] + L[is]*Eᵀ[js,js] = -F[is] -> same as B[js,js] and E[js,js]
					z[2*ldz] = b[js*ldb+js]
					z[2*ldz+1] = 0
					z[2*ldz+2] = e[js*lde+js]
					z[2*ldz+3] = 0

					// Row 3: R[is+1]*Bᵀ[js,js] + L[is+1]*Eᵀ[js,js] = -F[is+1]
					z[3*ldz] = 0
					z[3*ldz+1] = b[js*ldb+js]
					z[3*ldz+2] = 0
					z[3*ldz+3] = e[js*lde+js]

					rhs[0] = c[is*ldc+js]
					rhs[1] = c[(is+1)*ldc+js]
					rhs[2] = -f[is*ldf+js]
					rhs[3] = -f[(is+1)*ldf+js]

					k := impl.Dgetc2(4, z[:], ldz, ipiv[:4], jpiv[:4])
					if k >= 0 {
						ok = false
					}
					scaloc := impl.Dgesc2(4, z[:], ldz, rhs[:], ipiv[:4], jpiv[:4])
					if scaloc != 1 {
						for kk := 0; kk < n; kk++ {
							bi.Dscal(m, scaloc, c[kk:], ldc)
							bi.Dscal(m, scaloc, f[kk:], ldf)
						}
						scale *= scaloc
					}
					c[is*ldc+js] = rhs[0]
					c[(is+1)*ldc+js] = rhs[1]
					f[is*ldf+js] = rhs[2]
					f[(is+1)*ldf+js] = rhs[3]

				} else {
					// Build an 8×8 system.
					// mb==2, nb==2
					// For trans: Aᵀ*R + Dᵀ*L = C, R*Bᵀ + L*Eᵀ = -F
					// Unknowns: [R[is,js], R[is+1,js], R[is,js+1], R[is+1,js+1], L[is,js], L[is+1,js], L[is,js+1], L[is+1,js+1]]
					// Note: Aᵀ[i,j] = A[j,i], Bᵀ[i,j] = B[j,i]
					impl.Dlaset(blas.All, ldz, ldz, 0, 0, z[:], ldz)

					// Row 0: C[is,js] from Aᵀ*R + Dᵀ*L
					z[0] = a[is*lda+is]
					z[1] = a[(is+1)*lda+is]
					z[4] = d[is*ldd+is]
					z[5] = d[(is+1)*ldd+is]

					// Row 1: C[is+1,js]
					z[ldz] = a[is*lda+is+1]
					z[ldz+1] = a[(is+1)*lda+is+1]
					z[ldz+4] = d[is*ldd+is+1]
					z[ldz+5] = d[(is+1)*ldd+is+1]

					// Row 2: C[is,js+1]
					z[2*ldz+2] = a[is*lda+is]
					z[2*ldz+3] = a[(is+1)*lda+is]
					z[2*ldz+6] = d[is*ldd+is]
					z[2*ldz+7] = d[(is+1)*ldd+is]

					// Row 3: C[is+1,js+1]
					z[3*ldz+2] = a[is*lda+is+1]
					z[3*ldz+3] = a[(is+1)*lda+is+1]
					z[3*ldz+6] = d[is*ldd+is+1]
					z[3*ldz+7] = d[(is+1)*ldd+is+1]

					// Row 4: -F[is,js] from R*Bᵀ + L*Eᵀ
					z[4*ldz] = b[js*ldb+js]
					z[4*ldz+2] = b[js*ldb+js+1]
					z[4*ldz+4] = e[js*lde+js]
					z[4*ldz+6] = e[js*lde+js+1]

					// Row 5: -F[is+1,js]
					z[5*ldz+1] = b[js*ldb+js]
					z[5*ldz+3] = b[js*ldb+js+1]
					z[5*ldz+5] = e[js*lde+js]
					z[5*ldz+7] = e[js*lde+js+1]

					// Row 6: -F[is,js+1]
					z[6*ldz] = b[(js+1)*ldb+js]
					z[6*ldz+2] = b[(js+1)*ldb+js+1]
					z[6*ldz+4] = e[(js+1)*lde+js]
					z[6*ldz+6] = e[(js+1)*lde+js+1]

					// Row 7: -F[is+1,js+1]
					z[7*ldz+1] = b[(js+1)*ldb+js]
					z[7*ldz+3] = b[(js+1)*ldb+js+1]
					z[7*ldz+5] = e[(js+1)*lde+js]
					z[7*ldz+7] = e[(js+1)*lde+js+1]

					rhs[0] = c[is*ldc+js]
					rhs[1] = c[(is+1)*ldc+js]
					rhs[2] = c[is*ldc+js+1]
					rhs[3] = c[(is+1)*ldc+js+1]
					rhs[4] = -f[is*ldf+js]
					rhs[5] = -f[(is+1)*ldf+js]
					rhs[6] = -f[is*ldf+js+1]
					rhs[7] = -f[(is+1)*ldf+js+1]

					k := impl.Dgetc2(8, z[:], ldz, ipiv[:8], jpiv[:8])
					if k >= 0 {
						ok = false
					}
					scaloc := impl.Dgesc2(8, z[:], ldz, rhs[:], ipiv[:8], jpiv[:8])
					if scaloc != 1 {
						for kk := 0; kk < n; kk++ {
							bi.Dscal(m, scaloc, c[kk:], ldc)
							bi.Dscal(m, scaloc, f[kk:], ldf)
						}
						scale *= scaloc
					}
					c[is*ldc+js] = rhs[0]
					c[(is+1)*ldc+js] = rhs[1]
					c[is*ldc+js+1] = rhs[2]
					c[(is+1)*ldc+js+1] = rhs[3]
					f[is*ldf+js] = rhs[4]
					f[(is+1)*ldf+js] = rhs[5]
					f[is*ldf+js+1] = rhs[6]
					f[(is+1)*ldf+js+1] = rhs[7]
				}

				pq++

				// Substitute R[I,J] and L[I,J] into remaining equation.
				// For trans case: R*Bᵀ + L*Eᵀ = -F, so both R*B and L*E update F.
				if j > p+1 {
					bi.Dgemm(blas.NoTrans, blas.Trans, mb, js, nb, 1, c[is*ldc+js:], ldc, b[js:], ldb, 1, f[is*ldf:], ldf)
					bi.Dgemm(blas.NoTrans, blas.Trans, mb, js, nb, 1, f[is*ldf+js:], ldf, e[js:], lde, 1, f[is*ldf:], ldf)
				}
				if i < p {
					alpha := -1.0
					bi.Dgemm(blas.Trans, blas.NoTrans, m-ie-1, nb, mb, alpha, a[is*lda+ie+1:], lda, c[is*ldc+js:], ldc, 1, c[(ie+1)*ldc+js:], ldc)
					bi.Dgemm(blas.Trans, blas.NoTrans, m-ie-1, nb, mb, alpha, d[is*ldd+ie+1:], ldd, f[is*ldf+js:], ldf, 1, c[(ie+1)*ldc+js:], ldc)
				}
			}
		}
	}

	return scale, rdsum2, rdscal2, pq, ok
}
