// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/lapack"
)

// Dggbak back-transforms eigenvectors of a balanced matrix pair (A,B).
//
// Given eigenvectors V of the balanced matrix pair from Dggbal, Dggbak
// computes the eigenvectors of the original matrix pair by applying
// the inverse transformations.
//
// For right eigenvectors (side == blas.Right):
//
//	V = diag(rscale[ilo:ihi+1]) * P * V
//
// For left eigenvectors (side == blas.Left):
//
//	V = diag(lscale[ilo:ihi+1]) * P * V
//
// where P is the permutation matrix determined by ilo, ihi, and the
// scale arrays from Dggbal.
//
// job specifies what transformations to apply:
//   - lapack.BalanceNone: none
//   - lapack.Permute: permutation only
//   - lapack.Scale: scaling only
//   - lapack.PermuteScale: both permutation and scaling
//
// n is the order of the matrix pair. ilo and ihi are as returned by Dggbal.
// lscale and rscale contain the scaling factors and permutation information
// from Dggbal. m is the number of columns of V.
//
// On entry, V contains the eigenvectors to be back-transformed.
// On return, V is overwritten with the transformed eigenvectors.
//
// Dggbak is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dggbak(job lapack.BalanceJob, side blas.Side, n, ilo, ihi int, lscale, rscale []float64, m int, v []float64, ldv int) {
	switch {
	case job != lapack.BalanceNone && job != lapack.Permute && job != lapack.Scale && job != lapack.PermuteScale:
		panic(badBalanceJob)
	case side != blas.Left && side != blas.Right:
		panic(badSide)
	case n < 0:
		panic(nLT0)
	case ilo < 0 || max(0, n-1) < ilo:
		panic(badIlo)
	case ihi < min(ilo, n-1) || n <= ihi:
		panic(badIhi)
	case m < 0:
		panic(mLT0)
	case ldv < max(1, m):
		panic(badLdV)
	}

	// Quick return if possible.
	if n == 0 || m == 0 {
		return
	}
	doScale := (job == lapack.Scale || job == lapack.PermuteScale) && ilo != ihi
	doPermute := (job == lapack.Permute || job == lapack.PermuteScale) && (ilo != 0 || ihi != n-1)
	if !doScale && !doPermute {
		return
	}

	var scale []float64
	if side == blas.Right {
		if len(rscale) < n {
			panic(shortRscale)
		}
		scale = rscale
	} else {
		if len(lscale) < n {
			panic(shortLscale)
		}
		scale = lscale
	}
	if len(v) < (n-1)*ldv+m {
		panic(shortV)
	}

	bi := blas64.Implementation()

	// Backward scaling.
	if doScale {
		for i := ilo; i <= ihi; i++ {
			bi.Dscal(m, scale[i], v[i*ldv:], 1)
		}
	}

	if !doPermute {
		return
	}

	// Backward permutation.
	// Undo permutations in reverse order.
	for i := ilo - 1; i >= 0; i-- {
		k := int(scale[i])
		if k == i {
			continue
		}
		bi.Dswap(m, v[i*ldv:], 1, v[k*ldv:], 1)
	}
	for i := ihi + 1; i < n; i++ {
		k := int(scale[i])
		if k == i {
			continue
		}
		bi.Dswap(m, v[i*ldv:], 1, v[k*ldv:], 1)
	}
}
