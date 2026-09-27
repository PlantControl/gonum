// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

// Dtgexc reorders the generalized Schur decomposition of a real matrix pair
// (A,B), so that the diagonal block of (A,B) at row ifst is moved to row ilst.
//
// The matrix pair (A,B) is in generalized Schur form: A is quasi-upper
// triangular (block upper triangular with 1x1 and 2x2 blocks) and B is upper
// triangular. After reordering, the matrices are updated:
//
//	A := Q^T * A * Z
//	B := Q^T * B * Z
//
// where Q and Z are orthogonal matrices.
//
// If wantq is true, Q is updated. If wantz is true, Z is updated.
//
// ifst and ilst are 0-indexed positions of the diagonal blocks.
//
// work must have length at least lwork. If lwork is -1, a workspace query is
// performed and the optimal size is returned in work[0].
//
// The return values ifstOut and ilstOut give the final positions.
// ok is false if the swap failed (ill-conditioned problem).
//
// Dtgexc is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dtgexc(wantq, wantz bool, n int, a []float64, lda int, b []float64, ldb int,
	q []float64, ldq int, z []float64, ldz int, ifst, ilst int, work []float64, lwork int) (ifstOut, ilstOut int, ok bool) {
	return impl.dtgexc(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, ifst, ilst, work, lwork, nil)
}

func (impl Implementation) dtgexc(wantq, wantz bool, n int, a []float64, lda int, b []float64, ldb int,
	q []float64, ldq int, z []float64, ldz int, ifst, ilst int, work []float64, lwork int, scratch *dtgex2Scratch) (ifstOut, ilstOut int, ok bool) {
	minWork := 1
	if n > 1 {
		minWork = 4*n + 16
	}

	switch {
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
	case ifst < 0 || ifst >= n:
		panic(badIfst)
	case ilst < 0 || ilst >= n:
		panic(badIlst)
	case lwork < minWork && lwork != -1:
		panic(badLWork)
	case len(work) < 1:
		panic(shortWork)
	}
	work[0] = float64(minWork)

	// Workspace query.
	if lwork == -1 {
		return ifst, ilst, true
	}

	if n <= 1 {
		return ifst, ilst, true
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
	case len(work) < lwork:
		panic(shortWork)
	}
	if ifst > 0 && a[ifst*lda+ifst-1] != 0 {
		ifst--
	}
	if ilst > 0 && a[ilst*lda+ilst-1] != 0 {
		ilst--
	}
	nbf := 1
	if ifst < n-1 && a[(ifst+1)*lda+ifst] != 0 {
		nbf = 2
	}
	nbl := 1
	if ilst < n-1 && a[(ilst+1)*lda+ilst] != 0 {
		nbl = 2
	}
	if ifst == ilst {
		return ifst, ilst, true
	}
	if ifst < ilst {
		if nbf == 2 && nbl == 1 {
			ilst--
		}
		if nbf == 1 && nbl == 2 {
			ilst++
		}
	}

	if scratch == nil {
		scratch = new(dtgex2Scratch)
	}
	here := ifst

	if ilst > ifst {
		for here < ilst {
			if nbf == 1 || nbf == 2 {
				nbl = 1
				if here+nbf < n-1 && a[(here+nbf+1)*lda+here+nbf] != 0 {
					nbl = 2
				}
				if !impl.dtgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, here, nbf, nbl, work, lwork, scratch) {
					return ifst, here, false
				}
				here += nbl
				if nbf == 2 && here+1 < n && a[(here+1)*lda+here] == 0 {
					nbf = 3
				}
				continue
			}

			nbl = 1
			if here+3 < n && a[(here+3)*lda+here+2] != 0 {
				nbl = 2
			}
			if !impl.dtgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, here+1, 1, nbl, work, lwork, scratch) {
				return ifst, here, false
			}
			if nbl == 1 {
				if !impl.dtgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, here, 1, 1, work, lwork, scratch) {
					return ifst, here, false
				}
				here++
				continue
			}
			if a[(here+2)*lda+here+1] == 0 {
				nbl = 1
			}
			if nbl == 2 {
				if !impl.dtgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, here, 1, 2, work, lwork, scratch) {
					return ifst, here, false
				}
				here += 2
				continue
			}
			if !impl.dtgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, here, 1, 1, work, lwork, scratch) {
				return ifst, here, false
			}
			here++
			if !impl.dtgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, here, 1, 1, work, lwork, scratch) {
				return ifst, here, false
			}
			here++
		}
	} else {
		for here > ilst {
			if nbf == 1 || nbf == 2 {
				nbl = 1
				if here >= 2 && a[(here-1)*lda+here-2] != 0 {
					nbl = 2
				}
				if !impl.dtgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, here-nbl, nbl, nbf, work, lwork, scratch) {
					return ifst, here, false
				}
				here -= nbl
				if nbf == 2 && here+1 < n && a[(here+1)*lda+here] == 0 {
					nbf = 3
				}
				continue
			}

			nbl = 1
			if here >= 2 && a[(here-1)*lda+here-2] != 0 {
				nbl = 2
			}
			if !impl.dtgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, here-nbl, nbl, 1, work, lwork, scratch) {
				return ifst, here, false
			}
			if nbl == 1 {
				if !impl.dtgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, here, 1, 1, work, lwork, scratch) {
					return ifst, here, false
				}
				here--
				continue
			}
			if a[here*lda+here-1] == 0 {
				nbl = 1
			}
			if nbl == 2 {
				if !impl.dtgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, here-1, 2, 1, work, lwork, scratch) {
					return ifst, here, false
				}
				here -= 2
				continue
			}
			if !impl.dtgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, here, 1, 1, work, lwork, scratch) {
				return ifst, here, false
			}
			here--
			if !impl.dtgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, here, 1, 1, work, lwork, scratch) {
				return ifst, here, false
			}
			here--
		}
	}

	work[0] = float64(minWork)
	return ifst, here, true
}
