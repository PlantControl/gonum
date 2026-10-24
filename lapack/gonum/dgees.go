// Copyright ©2024 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/lapack"
)

// Dgees computes for an n×n real nonsymmetric matrix A, the eigenvalues, the
// real Schur form T, and, optionally, the matrix of Schur vectors Z. This
// gives the Schur factorization
//
//	A = Z * T * Zᵀ
//
// where T is upper quasi-triangular (the Schur form). A matrix is in real Schur
// form if it is upper quasi-triangular with 1×1 and 2×2 blocks. 2×2 blocks will
// be standardized in the form
//
//	[ a  b ]
//	[ c  a ]
//
// where b*c < 0. The eigenvalues of such a block are a ± sqrt(bc).
//
// Optionally, the routine can order the eigenvalues on the diagonal of the real
// Schur form so that selected eigenvalues are at the top left. The leading
// columns of Z then form an orthonormal basis for the invariant subspace
// corresponding to the selected eigenvalues.
//
// A matrix is in Schur canonical form if it is quasi-triangular and every 2×2
// diagonal block has the form shown above.
//
// If jobvs is lapack.SchurNone, no Schur vectors are computed.
// If jobvs is lapack.SchurHess, the Schur vectors are computed and returned in
// vs.
// For other values of jobvs, Dgees will panic.
//
// If sort is lapack.SortNone, eigenvalues are not ordered.
// If sort is lapack.SortSelected, eigenvalues are reordered so that selected
// eigenvalues appear in the leading diagonal blocks of the Schur form.
//
// selctg is a function that selects eigenvalues. It is only used when sort is
// lapack.SortSelected. An eigenvalue wr[j]+i*wi[j] is selected if
// selctg(wr[j], wi[j]) is true. Note that a selected complex eigenvalue may no
// longer satisfy selctg after ordering, since ordering may change the value of
// complex eigenvalues (especially if the eigenvalue is ill-conditioned).
//
// On entry, a contains the n×n matrix A. On return, a has been overwritten by
// its real Schur form T.
//
// wr and wi will contain the real and imaginary parts, respectively, of the
// computed eigenvalues in the same order that they appear on the diagonal of
// the output Schur form T. Complex conjugate pairs of eigenvalues will appear
// consecutively with the eigenvalue having positive imaginary part first.
// wr and wi must have length n, otherwise Dgees will panic.
//
// If jobvs is lapack.SchurHess, vs must have leading dimension ldvs >= n and
// have length at least (n-1)*ldvs+n, and on return will contain the orthogonal
// matrix Z of Schur vectors.
// If jobvs is lapack.SchurNone, vs is not referenced.
//
// work must have length at least lwork and lwork must be at least max(1,3*n).
// For good performance, lwork should generally be larger. On return, work[0]
// will contain the optimal value of lwork.
//
// If lwork is -1, instead of performing Dgees, only the optimal value of lwork
// is computed and stored into work[0].
//
// If sort is lapack.SortSelected, bwork must have length n; otherwise bwork is
// not referenced and can be nil.
//
// On return, sdim contains the number of eigenvalues (after sorting) for which
// selctg is true. If sort is lapack.SortNone, sdim will be 0.
//
// On return, ok reports whether all eigenvalues have been computed and, if
// sort is lapack.SortSelected, whether the reordering succeeded and the
// selection still holds after it. If the QR iteration failed, a has been
// partially reduced and the eigenvalues in wr and wi are correct only for
// indices where they were computed. If sorting failed because two eigenvalues
// were too close to swap, or because rounding after reordering changed which
// eigenvalues satisfy selctg (LAPACK INFO = N+1 or N+2), ok is false but the
// Schur form and vectors are valid.
//
// Dgees does not check A for NaN or Inf. Like reference LAPACK 3.12 DGEES,
// which returns INFO >= 0 for such input, Dgees then returns with ok false, or
// with ok true and NaN in a, wr or wi; it does not panic.
//
// Dgees is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dgees(jobvs lapack.SchurComp, sort lapack.SchurSort, selctg func(wr, wi float64) bool, n int, a []float64, lda int, wr, wi []float64, vs []float64, ldvs int, work []float64, lwork int, bwork []bool) (sdim int, ok bool) {
	wantvs := jobvs == lapack.SchurHess
	wantst := sort == lapack.SortSelected

	minwrk := max(1, 3*n)

	switch {
	case jobvs != lapack.SchurHess && jobvs != lapack.SchurNone:
		panic(badSchurComp)
	case sort != lapack.SortNone && sort != lapack.SortSelected:
		panic(badSchurSort)
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	case ldvs < 1 || (wantvs && ldvs < n):
		panic(badLdZ)
	case lwork < minwrk && lwork != -1:
		panic(badLWork)
	case len(work) < max(1, lwork):
		panic(shortWork)
	}

	// Compute workspace.
	var maxwrk int
	if n == 0 {
		maxwrk = 1
	} else {
		maxwrk = 2*n + n*impl.Ilaenv(1, "DGEHRD", " ", n, 1, n, 0)
		if wantvs {
			maxwrk = max(maxwrk, 2*n+(n-1)*impl.Ilaenv(1, "DORGHR", " ", n, 1, n, -1))
			impl.Dhseqr(lapack.EigenvaluesAndSchur, lapack.SchurHess, n, 0, n-1,
				nil, n, nil, nil, nil, max(1, ldvs), work, -1)
		} else {
			impl.Dhseqr(lapack.EigenvaluesAndSchur, lapack.SchurNone, n, 0, n-1,
				nil, n, nil, nil, nil, 1, work, -1)
		}
		hswork := int(work[0])

		trsenwork := 0
		if wantst {
			var iwork [1]int
			impl.Dtrsen(0, wantvs, nil, n, nil, n, nil, n, nil, nil, work, -1, iwork[:], -1)
			trsenwork = int(work[0])
		}
		maxwrk = max(maxwrk, max(1, 2*n+hswork), 2*n+trsenwork)
	}

	if lwork == -1 {
		work[0] = float64(maxwrk)
		return 0, true
	}

	// Quick return if possible.
	if n == 0 {
		work[0] = 1
		return 0, true
	}

	switch {
	case len(a) < (n-1)*lda+n:
		panic(shortA)
	case len(wr) != n:
		panic(badLenWr)
	case len(wi) != n:
		panic(badLenWi)
	case wantvs && len(vs) < (n-1)*ldvs+n:
		panic(shortZ)
	case wantst && len(bwork) < n:
		panic("lapack: insufficient length of bwork")
	case wantst && selctg == nil:
		panic("lapack: selctg is nil but sort is SortSelected")
	}

	// Get machine constants.
	eps := dlamchP
	smlnum := math.Sqrt(dlamchS) / eps
	bignum := 1 / smlnum

	// Scale A if max element outside range [smlnum,bignum].
	anrm := impl.Dlange(lapack.MaxAbs, n, n, a, lda, nil)
	var scalea bool
	var cscale float64
	if anrm > 0 && anrm < smlnum {
		scalea = true
		cscale = smlnum
	} else if anrm > bignum {
		scalea = true
		cscale = bignum
	}
	if scalea {
		impl.Dlascl(lapack.General, 0, 0, anrm, cscale, n, n, a, lda)
	}

	// Balance the matrix.
	// Use permutation-only balancing (not scaling) because scaling would
	// affect the Schur vectors. This matches LAPACK's DGEES.
	workbal := work[:n]
	ilo, ihi := impl.Dgebal(lapack.Permute, n, a, lda, workbal)

	// Reduce to upper Hessenberg form.
	iwrk := 2 * n
	tau := work[n : iwrk-1]
	impl.Dgehrd(n, ilo, ihi, a, lda, tau, work[iwrk:], lwork-iwrk)

	if wantvs {
		// Copy Householder vectors to vs.
		impl.Dlacpy(blas.Lower, n, n, a, lda, vs, ldvs)
		// Generate orthogonal matrix in vs.
		impl.Dorghr(n, ilo, ihi, vs, ldvs, tau, work[iwrk:], lwork-iwrk)
	}

	// Perform QR iteration, accumulating Schur vectors in vs if desired.
	iwrk = n
	var compz lapack.SchurComp
	if wantvs {
		compz = lapack.SchurOrig
	} else {
		compz = lapack.SchurNone
	}
	ieval := impl.Dhseqr(lapack.EigenvaluesAndSchur, compz, n, ilo, ihi,
		a, lda, wr, wi, vs, ldvs, work[iwrk:], lwork-iwrk)
	ok = ieval == 0

	// Sort eigenvalues if desired.
	sdim = 0
	if wantst && ok {
		if scalea {
			impl.Dlascl(lapack.General, 0, 0, cscale, anrm, n, 1, wr, 1)
			impl.Dlascl(lapack.General, 0, 0, cscale, anrm, n, 1, wi, 1)
		}
		for i := range n {
			bwork[i] = selctg(wr[i], wi[i])
		}
		var iwork [1]int
		sdim, _, _, ok = impl.Dtrsen(0, wantvs, bwork, n,
			a, lda, vs, ldvs, wr, wi, work[iwrk:], lwork-iwrk, iwork[:], 1)
	}

	if wantvs {
		// Undo balancing (permutation only).
		impl.Dgebak(lapack.Permute, lapack.EVRight, n, ilo, ihi, workbal, n, vs, ldvs)
	}

	if scalea {
		// Undo scaling for the Schur form of A.
		impl.Dlascl(lapack.UpperHessenberg, 0, 0, cscale, anrm, n, n, a, lda)
		for i := range n {
			wr[i] = a[i*lda+i]
		}
		if cscale == smlnum {
			// If scaling back towards underflow, adjust wi if an off-diagonal
			// element of a 2×2 block in the Schur form underflows.
			var i1, i2 int
			switch {
			case ieval > 0:
				i1, i2 = ieval, ihi-1
				impl.Dlascl(lapack.General, 0, 0, cscale, anrm, ilo, 1, wi, 1)
			case wantst:
				i1, i2 = 0, n-2
			default:
				i1, i2 = ilo, ihi-1
			}
			bi := blas64.Implementation()
			inxt := i1 - 1
			for i := i1; i <= i2; i++ {
				if i < inxt {
					continue
				}
				if wi[i] == 0 {
					inxt = i + 1
					continue
				}
				if a[(i+1)*lda+i] == 0 {
					wi[i], wi[i+1] = 0, 0
				} else if a[i*lda+i+1] == 0 {
					wi[i], wi[i+1] = 0, 0
					if i > 0 {
						bi.Dswap(i, a[i:], lda, a[i+1:], lda)
					}
					if n > i+2 {
						bi.Dswap(n-i-2, a[i*lda+i+2:], 1, a[(i+1)*lda+i+2:], 1)
					}
					if wantvs {
						bi.Dswap(n, vs[i:], ldvs, vs[i+1:], ldvs)
					}
					a[i*lda+i+1] = a[(i+1)*lda+i]
					a[(i+1)*lda+i] = 0
				}
				inxt = i + 2
			}
		}
		// Undo scaling for the imaginary part of the eigenvalues.
		impl.Dlascl(lapack.General, 0, 0, cscale, anrm, n-ieval, 1, wi[ieval:], 1)
	}

	if wantst && ok {
		// Check if reordering was successful.
		lastsl, lst2sl := true, true
		sdim = 0
		ip := 0
		for i := range n {
			cursl := selctg(wr[i], wi[i])
			if wi[i] == 0 {
				if cursl {
					sdim++
				}
				ip = 0
				if cursl && !lastsl {
					ok = false
				}
			} else if ip == 1 {
				// Last eigenvalue of conjugate pair.
				cursl = cursl || lastsl
				lastsl = cursl
				if cursl {
					sdim += 2
				}
				ip = -1
				if cursl && !lst2sl {
					ok = false
				}
			} else {
				// First eigenvalue of conjugate pair.
				ip = 1
			}
			lst2sl = lastsl
			lastsl = cursl
		}
	}

	work[0] = float64(maxwrk)
	return sdim, ok
}
