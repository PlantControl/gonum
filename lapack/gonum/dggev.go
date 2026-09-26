// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/lapack"
)

// Dggev computes for a pair of n×n real nonsymmetric matrices (A,B) the
// generalized eigenvalues and, optionally, the left and/or right generalized
// eigenvectors.
//
// A generalized eigenvalue for a pair of matrices (A,B) is a scalar λ or a
// ratio alpha/beta = λ, such that A - λ*B is singular. It is usually
// represented as the pair (alpha, beta), as there is a reasonable
// interpretation for beta=0, and even for both being zero.
//
// The right generalized eigenvector v_j corresponding to the generalized
// eigenvalue λ_j of (A,B) satisfies
//
//	A * v_j = λ_j * B * v_j
//
// The left generalized eigenvector u_j corresponding to the generalized
// eigenvalue λ_j of (A,B) satisfies
//
//	u_jᴴ * A = λ_j * u_jᴴ * B
//
// where u_jᴴ is the conjugate-transpose of u_j.
//
// On return, A will be overwritten and the left and right eigenvectors will be
// stored, respectively, in the columns of the n×n matrices VL and VR.
// If the j-th eigenvalue is real, then
//
//	u_j = VL[:,j],
//	v_j = VR[:,j],
//
// and if it is not real, then j and j+1 form a complex conjugate pair and the
// eigenvectors can be recovered as
//
//	u_j     = VL[:,j] + i*VL[:,j+1],
//	u_{j+1} = VL[:,j] - i*VL[:,j+1],
//	v_j     = VR[:,j] + i*VR[:,j+1],
//	v_{j+1} = VR[:,j] - i*VR[:,j+1],
//
// where i is the imaginary unit. The eigenvectors are normalized so that the
// largest component has abs(real part) + abs(imag part) = 1.
//
// Left eigenvectors will be computed only if jobvl == lapack.LeftEVCompute,
// otherwise jobvl must be lapack.LeftEVNone.
// Right eigenvectors will be computed only if jobvr == lapack.RightEVCompute,
// otherwise jobvr must be lapack.RightEVNone.
// For other values of jobvl and jobvr Dggev will panic.
//
// alphar, alphai, and beta contain the real and imaginary parts of the
// generalized eigenvalues. The generalized eigenvalue is
// (alphar[j] + i*alphai[j]) / beta[j]. Complex conjugate pairs of eigenvalues
// appear consecutively. alphar, alphai, and beta must have length n, and Dggev
// will panic otherwise.
//
// work must have length at least lwork and lwork must be at least max(1,8*n).
// For good performance, lwork should generally be larger.
// On return, optimal value of lwork will be stored in work[0].
//
// If lwork == -1, instead of performing Dggev, the function only calculates
// the optimal value of lwork and stores it into work[0].
//
// On return, ok reports whether the QZ iteration converged. If ok is false,
// some eigenvalues may not have been computed, and the eigenvectors are not
// meaningful.
//
// Dggev is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dggev(jobvl lapack.LeftEVJob, jobvr lapack.RightEVJob, n int,
	a []float64, lda int, b []float64, ldb int,
	alphar, alphai, beta []float64,
	vl []float64, ldvl int, vr []float64, ldvr int,
	work []float64, lwork int) (ok bool) {

	wantvl := jobvl == lapack.LeftEVCompute
	wantvr := jobvr == lapack.RightEVCompute

	minwrk := max(1, 8*n)

	switch {
	case jobvl != lapack.LeftEVCompute && jobvl != lapack.LeftEVNone:
		panic(badLeftEVJob)
	case jobvr != lapack.RightEVCompute && jobvr != lapack.RightEVNone:
		panic(badRightEVJob)
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	case ldb < max(1, n):
		panic(badLdB)
	case ldvl < 1 || (wantvl && ldvl < n):
		panic(badLdVL)
	case ldvr < 1 || (wantvr && ldvr < n):
		panic(badLdVR)
	case lwork < minwrk && lwork != -1:
		panic(badLWork)
	case len(work) < max(1, lwork):
		panic(shortWork)
	}

	// Quick return if possible.
	if n == 0 {
		work[0] = 1
		return true
	}

	// Compute workspace.
	// Layout:
	//   work[0:n]     - lscale (Dggbal output)
	//   work[n:2n]    - rscale (Dggbal output)
	//   work[2n:3n]   - tau (Dgeqrf output)
	//   work[3n:]     - subroutine workspace
	var maxwrk int

	// Query Dgeqrf.
	impl.Dgeqrf(n, n, nil, max(1, n), nil, work, -1)
	qrwork := int(work[0])

	// Query Dormqr.
	impl.Dormqr(blas.Left, blas.Trans, n, n, n, nil, max(1, n), nil, nil, max(1, n), work, -1)
	mqrwork := int(work[0])

	// Query Dorgqr.
	gqrwork := 0
	if wantvl {
		impl.Dorgqr(n, n, n, nil, max(1, n), nil, work, -1)
		gqrwork = int(work[0])
	}

	// Query Dhgeqz.
	impl.Dhgeqz(lapack.EigenvaluesAndSchur, lapack.SchurOrig, lapack.SchurOrig, n, 0, n-1,
		nil, max(1, n), nil, max(1, n), nil, nil, nil,
		nil, max(1, n), nil, max(1, n), work, -1)
	hqzwork := int(work[0])

	// Dtgevc needs 6*n workspace.
	tgvcwork := 6 * n

	maxwrk = max(3*n+max(qrwork, mqrwork, gqrwork), 2*n+max(hqzwork, tgvcwork))
	maxwrk = max(maxwrk, minwrk)

	if lwork == -1 {
		work[0] = float64(maxwrk)
		return true
	}

	switch {
	case len(a) < (n-1)*lda+n:
		panic(shortA)
	case len(b) < (n-1)*ldb+n:
		panic(shortB)
	case len(alphar) != n:
		panic(badLenAlphaR)
	case len(alphai) != n:
		panic(badLenAlphaI)
	case len(beta) != n:
		panic(badLenBeta)
	case wantvl && len(vl) < (n-1)*ldvl+n:
		panic(shortVL)
	case wantvr && len(vr) < (n-1)*ldvr+n:
		panic(shortVR)
	}

	// Get machine constants.
	eps := dlamchP
	smlnum := math.Sqrt(dlamchS) / eps
	bignum := 1 / smlnum

	// Scale A if max element outside range [smlnum,bignum].
	anrm := impl.Dlange(lapack.MaxAbs, n, n, a, lda, nil)
	var scalea bool
	var cscalea float64
	if anrm > 0 && anrm < smlnum {
		scalea = true
		cscalea = smlnum
	} else if anrm > bignum {
		scalea = true
		cscalea = bignum
	}
	if scalea {
		impl.Dlascl(lapack.General, 0, 0, anrm, cscalea, n, n, a, lda)
	}

	// Scale B if max element outside range [smlnum,bignum].
	bnrm := impl.Dlange(lapack.MaxAbs, n, n, b, ldb, nil)
	var scaleb bool
	var cscaleb float64
	if bnrm > 0 && bnrm < smlnum {
		scaleb = true
		cscaleb = smlnum
	} else if bnrm > bignum {
		scaleb = true
		cscaleb = bignum
	}
	if scaleb {
		impl.Dlascl(lapack.General, 0, 0, bnrm, cscaleb, n, n, b, ldb)
	}

	// Workspace layout.
	lscale := work[:n]
	rscale := work[n : 2*n]
	tau := work[2*n : 3*n]
	iwrk := 3 * n

	// Balance the matrix pair (A,B).
	ilo, ihi := impl.Dggbal(lapack.Permute, n, a, lda, b, ldb, lscale, rscale, work[2*n:])

	// Compute dimensions of the active submatrix.
	irows := ihi - ilo + 1
	icols := irows
	if wantvl || wantvr {
		icols = n - ilo
	}

	// QR factorization of B[ilo:ihi+1, ilo:n].
	impl.Dgeqrf(irows, icols, b[ilo*ldb+ilo:], ldb, tau[:irows], work[iwrk:], lwork-iwrk)

	// Apply Q^T to A[ilo:ihi+1, ilo:n].
	impl.Dormqr(blas.Left, blas.Trans, irows, icols, irows, b[ilo*ldb+ilo:], ldb, tau[:irows], a[ilo*lda+ilo:], lda, work[iwrk:], lwork-iwrk)

	// Initialize VL to identity and copy Householder vectors.
	if wantvl {
		impl.Dlaset(blas.All, n, n, 0, 1, vl, ldvl)
		if irows > 1 {
			impl.Dlacpy(blas.Lower, irows-1, irows-1, b[(ilo+1)*ldb+ilo:], ldb, vl[(ilo+1)*ldvl+ilo:], ldvl)
		}
		impl.Dorgqr(irows, irows, irows, vl[ilo*ldvl+ilo:], ldvl, tau[:irows], work[iwrk:], lwork-iwrk)
	}

	// Initialize VR to identity.
	if wantvr {
		impl.Dlaset(blas.All, n, n, 0, 1, vr, ldvr)
	}

	// Zero lower triangle of B.
	for i := 1; i < n; i++ {
		for j := 0; j < i; j++ {
			b[i*ldb+j] = 0
		}
	}

	// Reduce to generalized Hessenberg form.
	var compq, compz lapack.OrthoComp
	if wantvl {
		compq = lapack.OrthoPostmul
	} else {
		compq = lapack.OrthoNone
	}
	if wantvr {
		compz = lapack.OrthoPostmul
	} else {
		compz = lapack.OrthoNone
	}
	if wantvl || wantvr {
		impl.Dgghrd(compq, compz, n, ilo, ihi, a, lda, b, ldb, vl, ldvl, vr, ldvr)
	} else {
		impl.Dgghrd(lapack.OrthoNone, lapack.OrthoNone, irows, 0, irows-1,
			a[ilo*lda+ilo:], lda, b[ilo*ldb+ilo:], ldb, nil, 1, nil, 1)
	}

	// Perform QZ algorithm to get Schur form.
	iwrk = 2 * n
	var compqz, compzz lapack.SchurComp
	if wantvl {
		compqz = lapack.SchurOrig
	} else {
		compqz = lapack.SchurNone
	}
	if wantvr {
		compzz = lapack.SchurOrig
	} else {
		compzz = lapack.SchurNone
	}
	job := lapack.EigenvaluesOnly
	if wantvl || wantvr {
		job = lapack.EigenvaluesAndSchur
	}
	ok = impl.Dhgeqz(job, compqz, compzz, n, ilo, ihi,
		a, lda, b, ldb, alphar, alphai, beta,
		vl, ldvl, vr, ldvr, work[iwrk:], lwork-iwrk)

	if !ok {
		work[0] = float64(maxwrk)
		return false
	}

	// Compute eigenvectors.
	if wantvl || wantvr {
		var side lapack.EVSide
		if wantvl && wantvr {
			side = lapack.EVBoth
		} else if wantvl {
			side = lapack.EVLeft
		} else {
			side = lapack.EVRight
		}

		// Dtgevc with howmny=EVAllMulQ multiplies eigenvectors by Q and Z.
		_, ok = impl.Dtgevc(side, lapack.EVAllMulQ, nil, n,
			a, lda, b, ldb, vl, ldvl, vr, ldvr, n, work[iwrk:])
		if !ok {
			work[0] = float64(maxwrk)
			return false
		}
	}

	bi := blas64.Implementation()

	// Back-transform eigenvectors.
	if wantvl {
		impl.Dggbak(lapack.Permute, blas.Left, n, ilo, ihi, lscale, rscale, n, vl, ldvl)
		normalizeDggevEigenvectors(bi, n, alphai, vl, ldvl, smlnum)
	}

	if wantvr {
		impl.Dggbak(lapack.Permute, blas.Right, n, ilo, ihi, lscale, rscale, n, vr, ldvr)
		normalizeDggevEigenvectors(bi, n, alphai, vr, ldvr, smlnum)
	}

	// Undo scaling.
	logAlphaScale := 0.0
	logBetaScale := 0.0
	if scalea {
		logAlphaScale = math.Log(anrm) - math.Log(cscalea)
	}
	if scaleb {
		logBetaScale = math.Log(bnrm) - math.Log(cscaleb)
	}
	if scalea || scaleb {
		rescaleGeneralizedEigenvalues(alphar, alphai, beta, logAlphaScale, logBetaScale, math.Max(anrm, bnrm))
	}

	work[0] = float64(maxwrk)
	return true
}

func normalizeDggevEigenvectors(bi blas.Float64, n int, alphai, v []float64, ldv int, smlnum float64) {
	for j := 0; j < n; j++ {
		var vmax float64
		switch {
		case alphai[j] == 0:
			for i := 0; i < n; i++ {
				vmax = math.Max(vmax, math.Abs(v[i*ldv+j]))
			}
		case alphai[j] > 0:
			for i := 0; i < n; i++ {
				vmax = math.Max(vmax, math.Abs(v[i*ldv+j])+math.Abs(v[i*ldv+j+1]))
			}
		default:
			continue
		}
		if vmax < smlnum {
			continue
		}
		bi.Dscal(n, 1/vmax, v[j:], ldv)
		if alphai[j] > 0 {
			bi.Dscal(n, 1/vmax, v[j+1:], ldv)
		}
	}
}
