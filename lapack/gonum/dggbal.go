// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/lapack"
)

// Dggbal balances a pair of general real matrices (A,B). Balancing involves:
//  1. Permuting rows and columns to isolate eigenvalues
//  2. Applying diagonal equivalence transformations to make rows and columns
//     as close in norm as possible
//
// This may improve the accuracy of computed eigenvalues and eigenvectors
// in the generalized eigenvalue problem A*x = lambda*B*x.
//
// job specifies the operations to perform:
//   - lapack.BalanceNone: set lscale[i]=rscale[i]=1 for all i, return ilo=0, ihi=n-1
//   - lapack.Permute: permute only
//   - lapack.Scale: scale only
//   - lapack.PermuteScale: both permute and scale
//
// On return, A and B are overwritten by the balanced matrices.
//
// ilo and ihi mark the starting and ending columns of the balanced submatrix.
// For permuting, A[i,j]=0 and B[i,j]=0 for i>j and j in {0,...,ilo-1,ihi+1,...,n-1}.
//
// lscale and rscale contain the left and right scaling factors:
//   - For j in {0,...,ilo-1,ihi+1,...,n-1}: permutation indices
//   - For j in {ilo,...,ihi}: scaling factors
//
// lscale, rscale must have length n. work must have length at least max(1,6*n).
//
// Dggbal is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dggbal(job lapack.BalanceJob, n int, a []float64, lda int, b []float64, ldb int, lscale, rscale, work []float64) (ilo, ihi int) {
	switch {
	case job != lapack.BalanceNone && job != lapack.Permute && job != lapack.Scale && job != lapack.PermuteScale:
		panic(badBalanceJob)
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	case ldb < max(1, n):
		panic(badLdB)
	}

	ilo = 0
	ihi = n - 1

	if n == 0 {
		return ilo, ihi
	}

	switch {
	case len(lscale) < n:
		panic(shortLscale)
	case len(rscale) < n:
		panic(shortRscale)
	}

	if job == lapack.BalanceNone {
		for i := 0; i < n; i++ {
			lscale[i] = 1
			rscale[i] = 1
		}
		return ilo, ihi
	}
	if n == 1 {
		lscale[0] = 1
		rscale[0] = 1
		return ilo, ihi
	}

	switch {
	case len(a) < (n-1)*lda+n:
		panic(shortA)
	case len(b) < (n-1)*ldb+n:
		panic(shortB)
	case job != lapack.Permute && len(work) < 6*n:
		panic(shortWork)
	}

	bi := blas64.Implementation()

	if job != lapack.Scale {
		// Permute rows/columns to isolate eigenvalues.
		k, l := 0, n-1
		permute := func(i, j, m int) {
			lscale[m] = float64(i)
			rscale[m] = float64(j)
			if i != m {
				bi.Dswap(n-k, a[i*lda+k:], 1, a[m*lda+k:], 1)
				bi.Dswap(n-k, b[i*ldb+k:], 1, b[m*ldb+k:], 1)
			}
			if j != m {
				bi.Dswap(l+1, a[j:], lda, a[m:], lda)
				bi.Dswap(l+1, b[j:], ldb, b[m:], ldb)
			}
		}

		for {
			if l == 0 {
				lscale[0] = 0
				rscale[0] = 0
				break
			}
			found := false
			for i := l; i >= 0; i-- {
				jFound := -1
				for j := 0; j <= l; j++ {
					if a[i*lda+j] == 0 && b[i*ldb+j] == 0 {
						continue
					}
					if jFound != -1 {
						jFound = -2
						break
					}
					jFound = j
				}
				if jFound == -1 {
					jFound = l
				}
				if jFound >= 0 {
					permute(i, jFound, l)
					l--
					found = true
					break
				}
			}
			if !found {
				break
			}
		}

		if l > 0 {
			for k <= l {
				found := false
				for j := k; j <= l; j++ {
					iFound := -1
					for i := k; i <= l; i++ {
						if a[i*lda+j] == 0 && b[i*ldb+j] == 0 {
							continue
						}
						if iFound != -1 {
							iFound = -2
							break
						}
						iFound = i
					}
					if iFound == -1 {
						iFound = l
					}
					if iFound >= 0 {
						permute(iFound, j, k)
						k++
						found = true
						break
					}
				}
				if !found {
					break
				}
			}
		}
		ilo, ihi = k, l
	}

	if job == lapack.Permute {
		for i := ilo; i <= ihi; i++ {
			lscale[i] = 1
			rscale[i] = 1
		}
		return ilo, ihi
	}
	if ilo == ihi {
		lscale[ilo] = 1
		rscale[ilo] = 1
		return ilo, ihi
	}

	// Balance the submatrix in rows ilo to ihi using the generalized
	// conjugate-gradient iteration from reference LAPACK.
	const sclfac = 10.0
	nr := ihi - ilo + 1
	for i := ilo; i <= ihi; i++ {
		lscale[i] = 0
		rscale[i] = 0
		for k := 0; k < 6; k++ {
			work[k*n+i] = 0
		}
	}

	basl := math.Log10(sclfac)
	for i := ilo; i <= ihi; i++ {
		for j := ilo; j <= ihi; j++ {
			var ta, tb float64
			if a[i*lda+j] != 0 {
				ta = math.Log10(math.Abs(a[i*lda+j])) / basl
			}
			if b[i*ldb+j] != 0 {
				tb = math.Log10(math.Abs(b[i*ldb+j])) / basl
			}
			work[4*n+i] -= ta + tb
			work[5*n+j] -= ta + tb
		}
	}

	coef := 1 / float64(2*nr)
	coef2 := coef * coef
	coef5 := 0.5 * coef2
	var beta, pgamma float64
	for iter := 0; iter < nr+2; iter++ {
		gamma := coef * (bi.Ddot(nr, work[4*n+ilo:], 1, work[4*n+ilo:], 1) + bi.Ddot(nr, work[5*n+ilo:], 1, work[5*n+ilo:], 1))
		var ew, ewc float64
		for i := ilo; i <= ihi; i++ {
			ew += work[4*n+i]
			ewc += work[5*n+i]
		}
		gamma -= coef2*(ew*ew+ewc*ewc) + coef5*(ew-ewc)*(ew-ewc)
		if gamma == 0 {
			break
		}
		if iter != 0 {
			beta = gamma / pgamma
		}
		t := coef5 * (ewc - 3*ew)
		tc := coef5 * (ew - 3*ewc)
		bi.Dscal(nr, beta, work[ilo:], 1)
		bi.Dscal(nr, beta, work[n+ilo:], 1)
		bi.Daxpy(nr, coef, work[4*n+ilo:], 1, work[n+ilo:], 1)
		bi.Daxpy(nr, coef, work[5*n+ilo:], 1, work[ilo:], 1)
		for i := ilo; i <= ihi; i++ {
			work[i] += tc
			work[n+i] += t
		}

		for i := ilo; i <= ihi; i++ {
			var count int
			var sum float64
			for j := ilo; j <= ihi; j++ {
				if a[i*lda+j] != 0 {
					count++
					sum += work[j]
				}
				if b[i*ldb+j] != 0 {
					count++
					sum += work[j]
				}
			}
			work[2*n+i] = float64(count)*work[n+i] + sum
		}
		for j := ilo; j <= ihi; j++ {
			var count int
			var sum float64
			for i := ilo; i <= ihi; i++ {
				if a[i*lda+j] != 0 {
					count++
					sum += work[n+i]
				}
				if b[i*ldb+j] != 0 {
					count++
					sum += work[n+i]
				}
			}
			work[3*n+j] = float64(count)*work[j] + sum
		}

		sum := bi.Ddot(nr, work[n+ilo:], 1, work[2*n+ilo:], 1) + bi.Ddot(nr, work[ilo:], 1, work[3*n+ilo:], 1)
		alpha := gamma / sum
		var cmax float64
		for i := ilo; i <= ihi; i++ {
			cor := alpha * work[n+i]
			cmax = math.Max(cmax, math.Abs(cor))
			lscale[i] += cor
			cor = alpha * work[i]
			cmax = math.Max(cmax, math.Abs(cor))
			rscale[i] += cor
		}
		if cmax < 0.5 {
			break
		}
		bi.Daxpy(nr, -alpha, work[2*n+ilo:], 1, work[4*n+ilo:], 1)
		bi.Daxpy(nr, -alpha, work[3*n+ilo:], 1, work[5*n+ilo:], 1)
		pgamma = gamma
	}

	sfmin := dlamchS
	sfmax := 1 / sfmin
	lsfmin := int(math.Log10(sfmin)/basl + 1)
	lsfmax := int(math.Log10(sfmax) / basl)
	for i := ilo; i <= ihi; i++ {
		var rab float64
		for j := ilo; j < n; j++ {
			rab = math.Max(rab, math.Abs(a[i*lda+j]))
			rab = math.Max(rab, math.Abs(b[i*ldb+j]))
		}
		lrab := int(math.Log10(rab+sfmin)/basl + 1)
		ir := int(lscale[i] + math.Copysign(0.5, lscale[i]))
		ir = min(max(ir, lsfmin), lsfmax, lsfmax-lrab)
		lscale[i] = math.Pow(sclfac, float64(ir))

		var cab float64
		for j := 0; j <= ihi; j++ {
			cab = math.Max(cab, math.Abs(a[j*lda+i]))
			cab = math.Max(cab, math.Abs(b[j*ldb+i]))
		}
		lcab := int(math.Log10(cab+sfmin)/basl + 1)
		jc := int(rscale[i] + math.Copysign(0.5, rscale[i]))
		jc = min(max(jc, lsfmin), lsfmax, lsfmax-lcab)
		rscale[i] = math.Pow(sclfac, float64(jc))
	}

	for i := ilo; i <= ihi; i++ {
		bi.Dscal(n-ilo, lscale[i], a[i*lda+ilo:], 1)
		bi.Dscal(n-ilo, lscale[i], b[i*ldb+ilo:], 1)
	}
	for j := ilo; j <= ihi; j++ {
		bi.Dscal(ihi+1, rscale[j], a[j:], lda)
		bi.Dscal(ihi+1, rscale[j], b[j:], ldb)
	}

	return ilo, ihi
}

const (
	shortLscale = "lapack: insufficient length of lscale"
	shortRscale = "lapack: insufficient length of rscale"
)
