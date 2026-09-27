// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/lapack"
)

// This implementation follows DGGES in Reference LAPACK 3.12.1 at commit
// 6ec7f2bc4ecf4c4a93496aa2fa519575bc0e39ca, adapted to Gonum's row-major
// storage and panic/bool error contracts. See THIRD_PARTY_LICENSES/LAPACK-LICENSE.
//
// Dgges computes for a pair of n×n real nonsymmetric matrices (A,B), the
// generalized eigenvalues, the generalized real Schur form (S,T), and,
// optionally, the left and/or right matrices of Schur vectors (VSL and VSR).
// This gives the generalized Schur factorization
//
//	A = VSL * S * VSRᵀ
//	B = VSL * T * VSRᵀ
//
// where S is upper quasi-triangular (the real Schur form) and T is upper
// triangular. A matrix is in real Schur form if it is upper quasi-triangular
// with 1×1 and 2×2 blocks on the diagonal.
//
// Optionally, the routine can order the eigenvalues so that selected eigenvalues
// are at the top left of the diagonal blocks. The leading columns of VSL and VSR
// then form an orthonormal basis for the corresponding left and right eigenspaces
// (deflating subspaces).
//
// The generalized eigenvalues are stored as (alphar[j] + i*alphai[j]) / beta[j]
// for j = 0, ..., n-1. When beta[j] is zero, the eigenvalue is infinite.
// Complex eigenvalues occur in complex conjugate pairs.
//
// If jobvsl is lapack.SchurNone, no left Schur vectors are computed.
// If jobvsl is lapack.SchurHess, the left Schur vectors are computed and
// returned in vsl.
//
// If jobvsr is lapack.SchurNone, no right Schur vectors are computed.
// If jobvsr is lapack.SchurHess, the right Schur vectors are computed and
// returned in vsr.
//
// If sort is lapack.SortNone, eigenvalues are not ordered.
// If sort is lapack.SortSelected, eigenvalues are reordered so that selected
// eigenvalues appear in the leading diagonal blocks of the Schur form.
//
// selctg is a function that selects eigenvalues. It is only used when sort is
// lapack.SortSelected. An eigenvalue (alphar[j] + i*alphai[j]) / beta[j] is
// selected if selctg(alphar[j], alphai[j], beta[j]) returns true. Note that
// when beta[j] is zero, the eigenvalue is infinite; the caller must handle this
// case. For a complex conjugate pair, selecting either member selects both.
//
// On entry, a and b contain the n×n matrices A and B. On return, a has been
// overwritten by its real Schur form S, and b has been overwritten by the upper
// triangular matrix T.
//
// alphar, alphai, and beta must have length n. On return they contain the
// generalized eigenvalues.
//
// If jobvsl is lapack.SchurHess, vsl must have leading dimension ldvsl >= n and
// have length at least (n-1)*ldvsl+n.
// If jobvsl is lapack.SchurNone, vsl is not referenced.
//
// If jobvsr is lapack.SchurHess, vsr must have leading dimension ldvsr >= n and
// have length at least (n-1)*ldvsr+n.
// If jobvsr is lapack.SchurNone, vsr is not referenced.
//
// work must have length at least lwork. For good performance, lwork should
// generally be larger. On return, work[0] will contain the optimal value of lwork.
//
// If lwork is -1, instead of performing Dgges, only the optimal value of lwork
// is computed and stored into work[0].
//
// If sort is lapack.SortSelected, bwork must have length n; otherwise bwork is
// not referenced and can be nil.
//
// On return, sdim contains the number of eigenvalues (after sorting) for which
// selctg is true. If sort is lapack.SortNone, sdim will be 0. For a complex
// conjugate pair, both eigenvalues are counted when either or both are selected.
//
// On return, ok reports whether the QZ iteration converged, all requested block
// swaps were accepted, and the selected eigenvalues occupy the leading Schur
// blocks. If QZ iteration fails, the partial outputs are returned without
// back-permutation or unscaling, matching LAPACK. A rejected reorder is returned
// after back-permutation and unscaling.
//
// Dgges is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dgges(jobvsl, jobvsr lapack.SchurComp, sort lapack.SchurSort, selctg lapack.SchurSelect, n int, a []float64, lda int, b []float64, ldb int, alphar, alphai, beta []float64, vsl []float64, ldvsl int, vsr []float64, ldvsr int, work []float64, lwork int, bwork []bool) (sdim int, ok bool) {
	wantvsl := jobvsl == lapack.SchurHess
	wantvsr := jobvsr == lapack.SchurHess
	wantst := sort == lapack.SortSelected

	minwrk := 1
	if n > 0 {
		minwrk = max(8*n, 6*n+16)
	}

	switch {
	case jobvsl != lapack.SchurHess && jobvsl != lapack.SchurNone:
		panic(badSchurComp)
	case jobvsr != lapack.SchurHess && jobvsr != lapack.SchurNone:
		panic(badSchurComp)
	case sort != lapack.SortNone && sort != lapack.SortSelected:
		panic(badSchurSort)
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	case ldb < max(1, n):
		panic(badLdB)
	case ldvsl < 1 || (wantvsl && ldvsl < n):
		panic(badLdVSL)
	case ldvsr < 1 || (wantvsr && ldvsr < n):
		panic(badLdVSR)
	case lwork < minwrk && lwork != -1:
		panic(badLWork)
	case len(work) < max(1, lwork):
		panic(shortWork)
	}

	// Compute workspace.
	// Workspace layout:
	//   work[0:n]     - lscale for Dggbal
	//   work[n:2n]    - rscale for Dggbal
	//   work[2n:3n]   - tau for Dgeqrf
	//   work[3n:]     - workspace for subroutines
	var maxwrk int
	if n == 0 {
		maxwrk = 1
	} else {
		// Query Dgeqrf.
		impl.Dgeqrf(n, n, nil, max(1, n), nil, work, -1)
		qrwork := int(work[0])

		// Query Dormqr.
		impl.Dormqr(blas.Left, blas.Trans, n, n, n, nil, max(1, n), nil, nil, max(1, n), work, -1)
		mqrwork := int(work[0])

		// Query Dorgqr when left Schur vectors are requested.
		gqrwork := 0
		if wantvsl {
			impl.Dorgqr(n, n, n, nil, max(1, n), nil, work, -1)
			gqrwork = int(work[0])
		}

		// Query Dhgeqz.
		var compq, compz lapack.SchurComp
		if wantvsl {
			compq = lapack.SchurOrig
		} else {
			compq = lapack.SchurNone
		}
		if wantvsr {
			compz = lapack.SchurOrig
		} else {
			compz = lapack.SchurNone
		}
		impl.Dhgeqz(lapack.EigenvaluesAndSchur, compq, compz, n, 0, n-1,
			nil, max(1, n), nil, max(1, n), nil, nil, nil,
			nil, max(1, ldvsl), nil, max(1, ldvsr), work, -1)
		hqzwork := int(work[0])

		// Query Dtgsen (if sorting requested).
		tgsenwork := 0
		if wantst {
			var iwork [1]int
			impl.Dtgsen(0, wantvsl, wantvsr, nil, n,
				nil, max(1, n), nil, max(1, n), nil, nil, nil,
				nil, max(1, ldvsl), nil, max(1, ldvsr), work, -1, iwork[:], -1)
			tgsenwork = int(work[0])
		}

		// QR routines use the three n-element prefix arrays. Once the
		// reduction is complete, the tau region can be reused.
		maxwrk = max(3*n+max(qrwork, mqrwork, gqrwork), 2*n+max(hqzwork, tgsenwork))
		maxwrk = max(maxwrk, minwrk)
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
	case len(b) < (n-1)*ldb+n:
		panic(shortB)
	case len(alphar) != n:
		panic(badLenAlphaR)
	case len(alphai) != n:
		panic(badLenAlphaI)
	case len(beta) != n:
		panic(badLenBeta)
	case wantvsl && len(vsl) < (n-1)*ldvsl+n:
		panic(shortVSL)
	case wantvsr && len(vsr) < (n-1)*ldvsr+n:
		panic(shortVSR)
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

	// Workspace layout:
	//   work[0:n]     - lscale for Dggbal
	//   work[n:2n]    - rscale for Dggbal
	//   work[2n:2n+irows] - tau for Dgeqrf
	//   remaining work    - workspace for QR subroutines
	lscale := work[:n]
	rscale := work[n : 2*n]

	// Balance the matrix pair (A,B).
	ilo, ihi := impl.Dggbal(lapack.Permute, n, a, lda, b, ldb, lscale, rscale, work[2*n:])

	// Compute dimensions of the active submatrix.
	irows := ihi - ilo + 1
	icols := n - ilo
	tau := work[2*n : 2*n+irows]
	iwrk := 2*n + irows

	// Compute QR factorization of B[ilo:ihi+1, ilo:n].
	impl.Dgeqrf(irows, icols, b[ilo*ldb+ilo:], ldb, tau[:irows], work[iwrk:], lwork-iwrk)

	// Apply Qᵀ to A[ilo:ihi+1, ilo:n].
	impl.Dormqr(blas.Left, blas.Trans, irows, icols, irows, b[ilo*ldb+ilo:], ldb, tau[:irows], a[ilo*lda+ilo:], lda, work[iwrk:], lwork-iwrk)

	// Initialize VSL: copy Householder vectors from B, then generate Q.
	if wantvsl {
		impl.Dlaset(blas.All, n, n, 0, 1, vsl, ldvsl)
		if irows > 1 {
			impl.Dlacpy(blas.Lower, irows-1, irows-1, b[(ilo+1)*ldb+ilo:], ldb, vsl[(ilo+1)*ldvsl+ilo:], ldvsl)
		}
		impl.Dorgqr(irows, irows, irows, vsl[ilo*ldvsl+ilo:], ldvsl, tau[:irows], work[iwrk:], lwork-iwrk)
	}

	// Initialize VSR to identity.
	if wantvsr {
		impl.Dlaset(blas.All, n, n, 0, 1, vsr, ldvsr)
	}

	// Reduce to generalized Hessenberg form.
	var compq, compz lapack.OrthoComp
	if wantvsl {
		compq = lapack.OrthoPostmul
	} else {
		compq = lapack.OrthoNone
	}
	if wantvsr {
		compz = lapack.OrthoPostmul
	} else {
		compz = lapack.OrthoNone
	}
	impl.Dgghrd(compq, compz, n, ilo, ihi, a, lda, b, ldb, vsl, ldvsl, vsr, ldvsr)

	// Perform QZ algorithm.
	iwrk = 2 * n
	var compqz, compzz lapack.SchurComp
	if wantvsl {
		compqz = lapack.SchurOrig
	} else {
		compqz = lapack.SchurNone
	}
	if wantvsr {
		compzz = lapack.SchurOrig
	} else {
		compzz = lapack.SchurNone
	}
	ok = impl.Dhgeqz(lapack.EigenvaluesAndSchur, compqz, compzz, n, ilo, ihi,
		a, lda, b, ldb, alphar, alphai, beta,
		vsl, ldvsl, vsr, ldvsr, work[iwrk:], lwork-iwrk)
	if !ok {
		work[0] = float64(maxwrk)
		return 0, false
	}

	// Sort eigenvalues if desired.
	sdim = 0
	if ok && wantst {
		if scalea {
			impl.Dlascl(lapack.General, 0, 0, cscalea, anrm, 1, n, alphar, n)
			impl.Dlascl(lapack.General, 0, 0, cscalea, anrm, 1, n, alphai, n)
		}
		if scaleb {
			impl.Dlascl(lapack.General, 0, 0, cscaleb, bnrm, 1, n, beta, n)
		}

		// Populate bwork[] using selctg callback.
		for i := range n {
			bwork[i] = selctg(alphar[i], alphai[i], beta[i])
		}

		// Call Dtgsen to reorder eigenvalues.
		var iwork [1]int
		sdim, _, _, _, ok = impl.Dtgsen(0, wantvsl, wantvsr, bwork, n,
			a, lda, b, ldb, alphar, alphai, beta,
			vsl, ldvsl, vsr, ldvsr, work[iwrk:], lwork-iwrk, iwork[:], len(iwork))

		if !ok {
			// Continue through back-permutation and unscaling. LAPACK
			// returns the partially reordered Schur form on this path.
		}
	}

	// Back-transform eigenvectors.
	if wantvsl {
		impl.Dggbak(lapack.Permute, blas.Left, n, ilo, ihi, lscale, rscale, n, vsl, ldvsl)
	}
	if wantvsr {
		impl.Dggbak(lapack.Permute, blas.Right, n, ilo, ihi, lscale, rscale, n, vsr, ldvsr)
	}

	// Protect complex eigenvalue representations from overflow and underflow.
	safmin := dlamchS
	safmax := 1 / safmin
	if scalea {
		for i := range n {
			if alphai[i] == 0 {
				continue
			}
			var scale float64
			if alphar[i]/safmax > cscalea/anrm || safmin/alphar[i] > anrm/cscalea {
				scale = math.Abs(a[i*lda+i] / alphar[i])
			} else if alphai[i]/safmax > cscalea/anrm || safmin/alphai[i] > anrm/cscalea {
				scale = math.Abs(a[i*lda+i+1] / alphai[i])
			} else {
				continue
			}
			beta[i] *= scale
			alphar[i] *= scale
			alphai[i] *= scale
		}
	}
	if scaleb {
		for i := range n {
			if alphai[i] == 0 || !(beta[i]/safmax > cscaleb/bnrm || safmin/beta[i] > bnrm/cscaleb) {
				continue
			}
			scale := math.Abs(b[i*ldb+i] / beta[i])
			beta[i] *= scale
			alphar[i] *= scale
			alphai[i] *= scale
		}
	}

	// Undo scaling.
	if scalea {
		impl.Dlascl(lapack.UpperHessenberg, 0, 0, cscalea, anrm, n, n, a, lda)
		impl.Dlascl(lapack.General, 0, 0, cscalea, anrm, 1, n, alphar, n)
		impl.Dlascl(lapack.General, 0, 0, cscalea, anrm, 1, n, alphai, n)
	}
	if scaleb {
		impl.Dlascl(lapack.UpperTri, 0, 0, cscaleb, bnrm, n, n, b, ldb)
		impl.Dlascl(lapack.General, 0, 0, cscaleb, bnrm, 1, n, beta, n)
	}

	if wantst {
		sdim, ok = recheckDggesSelection(ok, selctg, alphar, alphai, beta)
	}

	work[0] = float64(maxwrk)
	return sdim, ok
}

func recheckDggesSelection(ok bool, selctg lapack.SchurSelect, alphar, alphai, beta []float64) (sdim int, reordered bool) {
	reordered = ok
	lastSelected := true
	lastPairSelected := true
	pair := 0
	for i := range alphar {
		selected := selctg(alphar[i], alphai[i], beta[i])
		if alphai[i] == 0 {
			if selected {
				sdim++
			}
			pair = 0
			if selected && !lastSelected {
				reordered = false
			}
		} else if pair == 1 {
			selected = selected || lastSelected
			lastSelected = selected
			if selected {
				sdim += 2
			}
			pair = -1
			if selected && !lastPairSelected {
				reordered = false
			}
		} else {
			pair = 1
		}
		lastPairSelected = lastSelected
		lastSelected = selected
	}
	return sdim, reordered
}

// rescaleGeneralizedEigenvalues undoes the independent pencil scaling while
// keeping each homogeneous eigenvalue triplet within the floating-point range.
func rescaleGeneralizedEigenvalues(alphar, alphai, beta []float64, logAlphaScale, logBetaScale, target float64) {
	if target == 0 {
		target = 1
	}
	logTarget := math.Log(target)
	for i := range alphar {
		logR := scaledLogAbs(alphar[i], logAlphaScale)
		logI := scaledLogAbs(alphai[i], logAlphaScale)
		logB := scaledLogAbs(beta[i], logBetaScale)
		logMax := math.Max(logR, math.Max(logI, logB))
		if math.IsInf(logMax, -1) {
			continue
		}
		alphar[i] = math.Copysign(math.Exp(logTarget+logR-logMax), alphar[i])
		alphai[i] = math.Copysign(math.Exp(logTarget+logI-logMax), alphai[i])
		beta[i] = math.Copysign(math.Exp(logTarget+logB-logMax), beta[i])
	}
}

func scaledLogAbs(value, logScale float64) float64 {
	if value == 0 {
		return math.Inf(-1)
	}
	return math.Log(math.Abs(value)) + logScale
}
