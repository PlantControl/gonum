// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/lapack"
)

// Dtgsen reorders the generalized real Schur decomposition of a real matrix
// pair (A, B) so that a selected cluster of eigenvalues appears in the leading
// diagonal blocks of the upper quasi-triangular matrix A and the upper
// triangular B.
//
// The reordering is performed using orthogonal transformations:
//
//	A := Q^T * A * Z
//	B := Q^T * B * Z
//
// The input matrices (A, B) must be in generalized real Schur canonical form
// as returned by Dhgeqz: A is block upper triangular with 1×1 and 2×2 diagonal
// blocks, and B is upper triangular.
//
// The ijob parameter specifies what is computed:
//
//	ijob=0: Reorder only, no condition estimates
//	ijob=1: Compute PL and PR.
//	ijob=2: Compute Frobenius norm-based estimates of Difu and Difl.
//	ijob=3: Compute 1-norm-based estimates of Difu and Difl.
//	ijob=4: Compute PL and PR and Frobenius norm-based Difu and Difl.
//	ijob=5: Compute PL and PR and 1-norm-based Difu and Difl.
//
// If wantq is true, the left transformation Q is updated. If wantz is true,
// the right transformation Z is updated.
//
// selected specifies which eigenvalues in the cluster to reorder to the leading
// diagonal blocks. For a complex conjugate pair of eigenvalues, both must be
// selected when either corresponding element of selected is true.
//
// On return, m is the dimension of the specified eigenspace, and the first m
// columns of Q and Z span the corresponding left and right deflating subspaces.
//
// pl and pr are lower bounds on the reciprocal of the norm of "weights" used
// to compute the average of the selected eigenvalue group. They are valid for
// ijob=1, 4, or 5.
//
// dif is a 2-element slice where dif[0] and dif[1] are estimates of Difu and
// Difl, the separation between the matrix pairs (A11, B11) and (A22, B22).
// These are valid only when ijob >= 2.
//
// work must have length at least max(1, lwork). If lwork is -1, a workspace
// query is performed.
//
// iwork must have length at least max(1, liwork). If liwork is -1, a workspace
// query is performed.
//
// Dtgsen returns ok=false if the reordering failed because the problem is
// very ill-conditioned.
func (impl Implementation) Dtgsen(ijob int, wantq, wantz bool, selected []bool, n int,
	a []float64, lda int, b []float64, ldb int,
	alphar, alphai, beta []float64,
	q []float64, ldq int, z []float64, ldz int,
	work []float64, lwork int, iwork []int, liwork int) (m int, pl, pr float64, dif [2]float64, ok bool) {

	switch {
	case ijob < 0 || ijob > 5:
		panic(badIJob)
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	case ldb < max(1, n):
		panic(badLdB)
	case ldq < 1, wantq && ldq < n:
		panic(badLdQ)
	case ldz < 1, wantz && ldz < n:
		panic(badLdZ)
	}

	lquery := lwork == -1 || liwork == -1

	if !lquery || ijob != 0 {
		switch {
		case len(selected) < n:
			panic(badLenSelected)
		case len(a) < (n-1)*lda+n:
			panic(shortA)
		}
		pair := false
		for k := 0; k < n; k++ {
			if pair {
				pair = false
				continue
			}
			if k < n-1 && a[(k+1)*lda+k] != 0 {
				if selected[k] || selected[k+1] {
					m += 2
				}
				pair = true
			} else if selected[k] {
				m++
			}
		}
	}

	// Compute workspace requirements.
	var lwmin, liwmin int
	if ijob == 1 || ijob == 2 || ijob == 4 {
		lwmin = max(1, 4*n+16, 2*m*(n-m))
		liwmin = max(1, n+6)
	} else if ijob == 3 || ijob == 5 {
		lwmin = max(1, 4*n+16, 4*m*(n-m))
		liwmin = max(1, 2*m*(n-m), n+6)
	} else {
		lwmin = max(1, 4*n+16)
		liwmin = 1
	}

	switch {
	case !lquery && lwork < lwmin:
		panic(badLWork)
	case !lquery && liwork < liwmin:
		panic(badLIWork)
	case len(work) < 1:
		panic(shortWork)
	case len(iwork) < 1:
		panic(shortIWork)
	}

	work[0] = float64(lwmin)
	iwork[0] = liwmin
	if lquery {
		return m, 0, 0, dif, true
	}

	switch {
	case len(work) < lwork:
		panic(shortWork)
	case len(iwork) < liwork:
		panic(shortIWork)
	}

	switch {
	case len(a) < (n-1)*lda+n:
		panic(shortA)
	case len(b) < (n-1)*ldb+n:
		panic(shortB)
	case wantq && len(q) < (n-1)*ldq+n:
		panic(shortQ)
	case wantz && len(z) < (n-1)*ldz+n:
		panic(shortZ)
	case len(selected) < n:
		panic(badLenSelected)
	case len(alphar) < n:
		panic(badLenAlphaR)
	case len(alphai) < n:
		panic(badLenAlphaI)
	case len(beta) < n:
		panic(badLenBeta)
	}

	ok = true

	n1 := m
	n2 := n - m
	if m == 0 || m == n {
		if ijob == 1 || ijob >= 4 {
			pl = 1
			pr = 1
		}
		if ijob >= 2 {
			dscale := 0.0
			dsum := 1.0
			for i := 0; i < n; i++ {
				dscale, dsum = impl.Dlassq(n, a[i*lda:], 1, dscale, dsum)
				dscale, dsum = impl.Dlassq(n, b[i*ldb:], 1, dscale, dsum)
			}
			dif[0] = dscale * math.Sqrt(dsum)
			dif[1] = dif[0]
		}
		goto extractEigenvalues
	}

	// Reorder blocks to collect selected eigenvalues at top-left.
	// Use a bubble-sort like approach with Dtgexc.
	{
		var scratch *dtgex2Scratch
		ks := 0 // Target position (where to move selected eigenvalues)
		pair := false

		for k := 0; k < n; k++ {
			if pair {
				pair = false
				continue
			}

			// Determine block size at position k.
			blockSize := 1
			if k < n-1 && a[(k+1)*lda+k] != 0 {
				blockSize = 2
				pair = true
			}

			// Check if this block is selected.
			isSelected := selected[k]
			if blockSize == 2 {
				isSelected = selected[k] || selected[k+1]
			}

			if isSelected {
				// Move this block to position ks.
				if k != ks {
					if scratch == nil {
						scratch = new(dtgex2Scratch)
					}
					_, _, swapOk := impl.dtgexc(wantq, wantz, n, a, lda, b, ldb,
						q, ldq, z, ldz, k, ks, work, lwork, scratch)
					if !swapOk {
						ok = false
						// Continue to extract eigenvalues even if reordering failed.
						goto extractEigenvalues
					}
				}
				ks += blockSize
			}
		}
	}
	if ijob == 0 {
		goto extractEigenvalues
	}

	// Compute condition estimates.
	// Compute PL and PR (reciprocal norms for eigenvalue averaging).
	if ijob == 1 || ijob >= 4 {
		// Copy A12 and B12 to work arrays.
		c := work[:n1*n2]
		f := work[n1*n2 : 2*n1*n2]
		for i := 0; i < n1; i++ {
			copy(c[i*n2:(i+1)*n2], a[i*lda+n1:i*lda+n1+n2])
			copy(f[i*n2:(i+1)*n2], b[i*ldb+n1:i*ldb+n1+n2])
		}

		var workTgsyl [1]float64

		scale, _, _ := impl.Dtgsyl(blas.NoTrans, 0, n1, n2,
			a, lda, a[n1*lda+n1:], lda, c, n2,
			b, ldb, b[n1*ldb+n1:], ldb, f, n2,
			workTgsyl[:], 1, iwork)

		rnorm := impl.Dlange(lapack.Frobenius, n1, n2, c, n2, nil)
		lnorm := impl.Dlange(lapack.Frobenius, n1, n2, f, n2, nil)
		pl = scale / math.Hypot(scale, rnorm)
		pr = scale / math.Hypot(scale, lnorm)
	}

	// Compute DIF (separation estimates).
	if ijob >= 2 {
		if ijob == 2 || ijob == 4 {
			c := work[:n1*n2]
			f := work[n1*n2 : 2*n1*n2]
			var workTgsyl [1]float64

			_, dif[0], _ = impl.Dtgsyl(blas.NoTrans, 3, n1, n2,
				a, lda, a[n1*lda+n1:], lda, c, n2,
				b, ldb, b[n1*ldb+n1:], ldb, f, n2,
				workTgsyl[:], 1, iwork)

			_, dif[1], _ = impl.Dtgsyl(blas.NoTrans, 3, n2, n1,
				a[n1*lda+n1:], lda, a, lda, c, n1,
				b[n1*ldb+n1:], ldb, b, ldb, f, n1,
				workTgsyl[:], 1, iwork)
		} else {
			dif[0] = impl.dtgsenDif(n1, n2,
				a, lda, a[n1*lda+n1:], lda,
				b, ldb, b[n1*ldb+n1:], ldb,
				work, iwork)
			dif[1] = impl.dtgsenDif(n2, n1,
				a[n1*lda+n1:], lda, a, lda,
				b[n1*ldb+n1:], ldb, b, ldb,
				work, iwork)
		}
	}

extractEigenvalues:
	// Extract eigenvalues from the reordered (A, B).
	// Real eigenvalues: alphar[k] = A[k,k], alphai[k] = 0, beta[k] = B[k,k]
	// Complex pairs: use Dlag2 to compute.
	{
		pair := false
		for k := 0; k < n; k++ {
			if pair {
				pair = false
				continue
			}

			if k < n-1 && a[(k+1)*lda+k] != 0 {
				// 2x2 block - complex eigenvalue pair.
				var scale1, scale2, wr1, wr2, wi float64
				scale1, scale2, wr1, wr2, wi = impl.Dlag2(
					a[k*lda+k:], lda,
					b[k*ldb+k:], ldb,
				)
				alphar[k] = wr1
				alphar[k+1] = wr2
				alphai[k] = wi
				alphai[k+1] = -wi
				beta[k] = scale1
				beta[k+1] = scale2
				pair = true
			} else {
				// 1x1 block - real eigenvalue.
				if math.Signbit(b[k*ldb+k]) {
					for j := range n {
						a[k*lda+j] = -a[k*lda+j]
						b[k*ldb+j] = -b[k*ldb+j]
						if wantq {
							q[j*ldq+k] = -q[j*ldq+k]
						}
					}
				}
				alphar[k] = a[k*lda+k]
				alphai[k] = 0
				beta[k] = b[k*ldb+k]
			}
		}
	}

	work[0] = float64(lwmin)
	iwork[0] = liwmin
	return m, pl, pr, dif, ok
}

func (impl Implementation) dtgsenDif(m, n int, a []float64, lda int, b []float64, ldb int, d []float64, ldd int, e []float64, lde int, work []float64, iwork []int) float64 {
	mn := m * n
	nn := 2 * mn
	x := work[:nn]
	v := work[nn : 2*nn]
	var workTgsyl [1]float64

	var isave [3]int
	var est, scale float64
	var kase int
	for {
		est, kase = impl.Dlacn2(nn, v, x, iwork, est, kase, &isave)
		if kase == 0 {
			break
		}
		trans := blas.NoTrans
		if kase != 1 {
			trans = blas.Trans
		}
		scale, _, _ = impl.Dtgsyl(trans, 0, m, n,
			a, lda, b, ldb, x[:mn], n,
			d, ldd, e, lde, x[mn:], n,
			workTgsyl[:], 1, iwork)
	}
	if est == 0 {
		return 0
	}
	return scale / est
}
