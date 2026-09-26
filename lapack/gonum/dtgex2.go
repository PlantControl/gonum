// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
)

// Dtgex2 swaps adjacent diagonal 1×1 or 2×2 blocks in a generalized real
// Schur form matrix pair (A,B) by an orthogonal equivalence transformation.
// If wantq or wantz is true, the corresponding Schur vectors are updated.
//
// work must have length at least lwork and lwork must be at least
// max(1,n*(n1+n2),2*(n1+n2)*(n1+n2)). If lwork is -1, Dtgex2 performs a
// workspace query and stores the required size in work[0].
//
// Dtgex2 returns false without modifying A, B, Q, or Z when the tentative swap
// fails the weak or strong stability test.
//
// Dtgex2 is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dtgex2(wantq, wantz bool, n int, a []float64, lda int, b []float64, ldb int,
	q []float64, ldq int, z []float64, ldz int, j1, n1, n2 int, work []float64, lwork int) bool {
	return impl.dtgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, j1, n1, n2, work, lwork, nil)
}

// dtgex2Scratch is reused across a sequence of swaps. Its slices are passed to
// the configurable BLAS backend, so they may escape to the heap.
type dtgex2Scratch struct {
	s, t, li, ir       [16]float64
	taul, taur         [4]float64
	scpy, tcpy, licopy [16]float64
	sylvester          dtgsy2Scratch
}

func (impl Implementation) dtgex2(wantq, wantz bool, n int, a []float64, lda int, b []float64, ldb int,
	q []float64, ldq int, z []float64, ldz int, j1, n1, n2 int, work []float64, lwork int, scratch *dtgex2Scratch) bool {

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
	case j1 < 0:
		panic(badJ1)
	case n1 < 1 || n1 > 2:
		panic("lapack: invalid n1")
	case n2 < 1 || n2 > 2:
		panic("lapack: invalid n2")
	case j1+n1+n2 > n:
		panic("lapack: blocks extend beyond matrix")
	}

	m := n1 + n2
	minWork := max(1, n*m, 2*m*m)
	if lwork == -1 {
		if len(work) < 1 {
			panic(shortWork)
		}
		work[0] = float64(minWork)
		return true
	}
	if lwork < minWork {
		panic(badLWork)
	}
	if n <= 1 {
		work[0] = float64(minWork)
		return true
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
	work[0] = float64(minWork)

	const ld = 4
	if scratch == nil {
		scratch = new(dtgex2Scratch)
	}
	*scratch = dtgex2Scratch{}
	s, tt, li, ir := scratch.s[:], scratch.t[:], scratch.li[:], scratch.ir[:]
	copyLocalBlock(m, a[j1*lda+j1:], lda, s[:], ld)
	copyLocalBlock(m, b[j1*ldb+j1:], ldb, tt[:], ld)

	eps := dlamchP
	smlnum := dlamchS / eps
	threshA := math.Max(20*eps*localNorm(m, s[:], ld), smlnum)
	threshB := math.Max(20*eps*localNorm(m, tt[:], ld), smlnum)

	if m == 2 {
		if !impl.dtgex2Swap11(s[:], tt[:], li[:], ir[:], ld, threshA, threshB,
			a[j1*lda+j1:], lda, b[j1*ldb+j1:], ldb) {
			return false
		}
		impl.dtgex2Apply11(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, j1, s[:], tt[:], li[:], ir[:])
		return true
	}

	if !impl.dtgex2SwapLarge(m, n1, n2, s[:], tt[:], li[:], ir[:], ld, threshA, threshB,
		a[j1*lda+j1:], lda, b[j1*ldb+j1:], ldb, work, scratch) {
		return false
	}

	for i := n2; i < m; i++ {
		for j := 0; j < n2; j++ {
			s[i*ld+j] = 0
		}
	}
	copyLocalBlock(m, s[:], ld, a[j1*lda+j1:], lda)
	copyLocalBlock(m, tt[:], ld, b[j1*ldb+j1:], ldb)

	impl.dtgex2Canonicalize(n1, n2, a[j1*lda+j1:], lda, b[j1*ldb+j1:], ldb, li[:], ir[:], ld)
	applyDtgex2Transforms(wantq, wantz, n, m, a, lda, b, ldb, q, ldq, z, ldz, j1, li[:], ir[:], ld, work)
	return true
}

func (impl Implementation) dtgex2Swap11(s, t, li, ir []float64, ld int, threshA, threshB float64,
	a []float64, lda int, b []float64, ldb int) bool {
	sa := math.Abs(s[ld+1]) * math.Abs(t[0])
	sb := math.Abs(s[0]) * math.Abs(t[ld+1])
	f := s[ld+1]*t[0] - t[ld+1]*s[0]
	g := s[ld+1]*t[1] - t[ld+1]*s[1]
	c, sn, _ := impl.Dlartg(f, g)
	ir[0], ir[1], ir[ld], ir[ld+1] = sn, c, -c, sn

	bi := blas64.Implementation()
	bi.Drot(2, s, ld, s[1:], ld, ir[0], ir[ld])
	bi.Drot(2, t, ld, t[1:], ld, ir[0], ir[ld])
	if sa >= sb {
		li[0], li[ld], _ = impl.Dlartg(s[0], s[ld])
	} else {
		li[0], li[ld], _ = impl.Dlartg(t[0], t[ld])
	}
	li[1], li[ld+1] = -li[ld], li[0]
	bi.Drot(2, s, 1, s[ld:], 1, li[0], li[ld])
	bi.Drot(2, t, 1, t[ld:], 1, li[0], li[ld])
	if math.Abs(s[ld]) > threshA || math.Abs(t[ld]) > threshB {
		return false
	}
	residA, residB := localStrongResidual(2, a, lda, b, ldb, li, s, ir, t, ld, true)
	if residA > threshA || residB > threshB {
		return false
	}
	s[ld], t[ld] = 0, 0
	return true
}

func (impl Implementation) dtgex2Apply11(wantq, wantz bool, n int, a []float64, lda int, b []float64, ldb int,
	q []float64, ldq int, z []float64, ldz int, j1 int, s, t, li, ir []float64) {
	bi := blas64.Implementation()
	bi.Drot(j1+2, a[j1:], lda, a[j1+1:], lda, ir[0], ir[4])
	bi.Drot(j1+2, b[j1:], ldb, b[j1+1:], ldb, ir[0], ir[4])
	bi.Drot(n-j1, a[j1*lda+j1:], 1, a[(j1+1)*lda+j1:], 1, li[0], li[4])
	bi.Drot(n-j1, b[j1*ldb+j1:], 1, b[(j1+1)*ldb+j1:], 1, li[0], li[4])
	a[(j1+1)*lda+j1], b[(j1+1)*ldb+j1] = 0, 0
	if wantz {
		bi.Drot(n, z[j1:], ldz, z[j1+1:], ldz, ir[0], ir[4])
	}
	if wantq {
		bi.Drot(n, q[j1:], ldq, q[j1+1:], ldq, li[0], li[4])
	}
}

func (impl Implementation) dtgex2SwapLarge(m, n1, n2 int, s, t, li, ir []float64, ld int, threshA, threshB float64,
	a []float64, lda int, b []float64, ldb int, work []float64, scratch *dtgex2Scratch) bool {
	for i := 0; i < n1; i++ {
		for j := 0; j < n2; j++ {
			li[i*ld+j] = t[i*ld+n1+j]
			ir[(n2+i)*ld+n1+j] = s[i*ld+n1+j]
		}
	}
	var iwork [8]int
	scale, _, _, _, ok := impl.dtgsy2(blas.NoTrans, 0, n1, n2,
		s, ld, s[n1*ld+n1:], ld, ir[n2*ld+n1:], ld,
		t, ld, t[n1*ld+n1:], ld, li, ld, 0, 1, iwork[:], &scratch.sylvester)
	if !ok {
		return false
	}
	for j := 0; j < n2; j++ {
		for i := 0; i < n1; i++ {
			li[i*ld+j] = -li[i*ld+j]
		}
		li[(n1+j)*ld+j] = scale
	}
	for i := 0; i < n1; i++ {
		ir[(n2+i)*ld+i] = scale
	}

	taul, taur := scratch.taul[:], scratch.taur[:]
	impl.Dgeqr2(m, n2, li, ld, taul[:n2], work)
	impl.Dorg2r(m, m, n2, li, ld, taul[:n2], work)
	impl.Dgerq2(n1, m, ir[n2*ld:], ld, taur[:n1], work)
	impl.Dorgr2(m, m, n1, ir, ld, taur[:n1], work)

	var tmp, ircopy [16]float64
	scpy, tcpy, licopy := scratch.scpy[:], scratch.tcpy[:], scratch.licopy[:]
	localMul(m, li, true, s, false, tmp[:], ld)
	localMul(m, tmp[:], false, ir, true, s, ld)
	localMul(m, li, true, t, false, tmp[:], ld)
	localMul(m, tmp[:], false, ir, true, t, ld)
	copy(scpy[:], s)
	copy(tcpy[:], t)
	copy(licopy[:], li)
	copy(ircopy[:], ir)

	impl.Dgerq2(m, m, t, ld, taur[:m], work)
	impl.Dormr2(blas.Right, blas.Trans, m, m, m, t, ld, taur[:m], s, ld, work)
	impl.Dormr2(blas.Left, blas.NoTrans, m, m, m, t, ld, taur[:m], ir, ld, work)
	brqa21 := lowerLeftNorm(m, n1, n2, s, ld)

	impl.Dgeqr2(m, m, tcpy[:], ld, taul[:m], work)
	impl.Dorm2r(blas.Left, blas.Trans, m, m, m, tcpy[:], ld, taul[:m], scpy[:], ld, work)
	impl.Dorm2r(blas.Right, blas.NoTrans, m, m, m, tcpy[:], ld, taul[:m], licopy[:], ld, work)
	bqra21 := lowerLeftNorm(m, n1, n2, scpy[:], ld)

	if bqra21 <= brqa21 && bqra21 <= threshA {
		copy(s, scpy[:])
		copy(t, tcpy[:])
		copy(li, licopy[:])
		copy(ir, ircopy[:])
	} else if brqa21 >= threshA {
		return false
	}
	for i := 1; i < m; i++ {
		for j := 0; j < i; j++ {
			t[i*ld+j] = 0
		}
	}
	residA, residB := localStrongResidual(m, a, lda, b, ldb, li, s, ir, t, ld, false)
	return residA <= threshA && residB <= threshB
}

func (impl Implementation) dtgex2Canonicalize(n1, n2 int, a []float64, lda int, b []float64, ldb int,
	li, ir []float64, ld int) {
	m := n1 + n2
	var ql, zr [16]float64
	for i := 0; i < m; i++ {
		ql[i*ld+i], zr[i*ld+i] = 1, 1
	}
	if n2 == 2 {
		cq, sq, _, _, cz, sz, _, _, _, _, _, _ := impl.Dlagv2(a, lda, b, ldb)
		setRotation(ql[:], ld, 0, cq, sq)
		setRotation(zr[:], ld, 0, cz, sz)
	}
	if n1 == 2 {
		off := n2
		cq, sq, _, _, cz, sz, _, _, _, _, _, _ := impl.Dlagv2(a[off*lda+off:], lda, b[off*ldb+off:], ldb)
		setRotation(ql[:], ld, off, cq, sq)
		setRotation(zr[:], ld, off, cz, sz)
	}

	var tmp [16]float64
	for i := 0; i < n2; i++ {
		for j := n2; j < m; j++ {
			v := 0.0
			for k := 0; k < n2; k++ {
				for l := n2; l < m; l++ {
					v += ql[k*ld+i] * a[k*lda+l] * zr[l*ld+j]
				}
			}
			tmp[i*ld+j] = v
			v = 0
			for k := 0; k < n2; k++ {
				for l := n2; l < m; l++ {
					v += ql[k*ld+i] * b[k*ldb+l] * zr[l*ld+j]
				}
			}
			tmp[(i+n2)*ld+(j-n2)] = v
		}
	}
	for i := 0; i < n2; i++ {
		for j := n2; j < m; j++ {
			a[i*lda+j] = tmp[i*ld+j]
			b[i*ldb+j] = tmp[(i+n2)*ld+(j-n2)]
		}
	}
	localMul(m, li, false, ql[:], false, tmp[:], ld)
	copy(li, tmp[:])
	localMul(m, ir, true, zr[:], false, tmp[:], ld)
	copy(ir, tmp[:])
}

func applyDtgex2Transforms(wantq, wantz bool, n, m int, a []float64, lda int, b []float64, ldb int,
	q []float64, ldq int, z []float64, ldz int, j1 int, li, ir []float64, ld int, work []float64) {
	bi := blas64.Implementation()
	if j1+m < n {
		cols := n - j1 - m
		bi.Dgemm(blas.Trans, blas.NoTrans, m, cols, m, 1, li, ld, a[j1*lda+j1+m:], lda, 0, work, cols)
		copyRect(m, cols, work, cols, a[j1*lda+j1+m:], lda)
		bi.Dgemm(blas.Trans, blas.NoTrans, m, cols, m, 1, li, ld, b[j1*ldb+j1+m:], ldb, 0, work, cols)
		copyRect(m, cols, work, cols, b[j1*ldb+j1+m:], ldb)
	}
	if j1 > 0 {
		bi.Dgemm(blas.NoTrans, blas.NoTrans, j1, m, m, 1, a[j1:], lda, ir, ld, 0, work, m)
		copyRect(j1, m, work, m, a[j1:], lda)
		bi.Dgemm(blas.NoTrans, blas.NoTrans, j1, m, m, 1, b[j1:], ldb, ir, ld, 0, work, m)
		copyRect(j1, m, work, m, b[j1:], ldb)
	}
	if wantq {
		bi.Dgemm(blas.NoTrans, blas.NoTrans, n, m, m, 1, q[j1:], ldq, li, ld, 0, work, m)
		copyRect(n, m, work, m, q[j1:], ldq)
	}
	if wantz {
		bi.Dgemm(blas.NoTrans, blas.NoTrans, n, m, m, 1, z[j1:], ldz, ir, ld, 0, work, m)
		copyRect(n, m, work, m, z[j1:], ldz)
	}
}

func localStrongResidual(m int, a []float64, lda int, b []float64, ldb int,
	li, s, ir, t []float64, ld int, transposeIR bool) (residA, residB float64) {
	var tmp, got [16]float64
	localMul(m, li, false, s, false, tmp[:], ld)
	localMul(m, tmp[:], false, ir, transposeIR, got[:], ld)
	scaleA, sumsqA := 0.0, 1.0
	for i := 0; i < m; i++ {
		for j := 0; j < m; j++ {
			d := a[i*lda+j] - got[i*ld+j]
			scaleA, sumsqA = updateScaledSquare(scaleA, sumsqA, d)
		}
	}
	localMul(m, li, false, t, false, tmp[:], ld)
	localMul(m, tmp[:], false, ir, transposeIR, got[:], ld)
	scaleB, sumsqB := 0.0, 1.0
	for i := 0; i < m; i++ {
		for j := 0; j < m; j++ {
			d := b[i*ldb+j] - got[i*ld+j]
			scaleB, sumsqB = updateScaledSquare(scaleB, sumsqB, d)
		}
	}
	return scaleA * math.Sqrt(sumsqA), scaleB * math.Sqrt(sumsqB)
}

func localNorm(m int, a []float64, ld int) float64 {
	scale, sumsq := 0.0, 1.0
	for i := 0; i < m; i++ {
		for j := 0; j < m; j++ {
			scale, sumsq = updateScaledSquare(scale, sumsq, a[i*ld+j])
		}
	}
	return scale * math.Sqrt(sumsq)
}

func updateScaledSquare(scale, sumsq, x float64) (float64, float64) {
	ax := math.Abs(x)
	if ax == 0 {
		return scale, sumsq
	}
	if scale < ax {
		return ax, 1 + sumsq*(scale/ax)*(scale/ax)
	}
	return scale, sumsq + (ax/scale)*(ax/scale)
}

func lowerLeftNorm(m, n1, n2 int, a []float64, ld int) float64 {
	scale, sumsq := 0.0, 1.0
	for i := n2; i < n2+n1; i++ {
		for j := 0; j < n2; j++ {
			scale, sumsq = updateScaledSquare(scale, sumsq, a[i*ld+j])
		}
	}
	return scale * math.Sqrt(sumsq)
}

func localMul(n int, a []float64, transA bool, b []float64, transB bool, dst []float64, ld int) {
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			v := 0.0
			for k := 0; k < n; k++ {
				ai := a[i*ld+k]
				if transA {
					ai = a[k*ld+i]
				}
				bj := b[k*ld+j]
				if transB {
					bj = b[j*ld+k]
				}
				v += ai * bj
			}
			dst[i*ld+j] = v
		}
	}
}

func setRotation(a []float64, ld, off int, c, s float64) {
	a[off*ld+off] = c
	a[off*ld+off+1] = -s
	a[(off+1)*ld+off] = s
	a[(off+1)*ld+off+1] = c
}

func copyLocalBlock(n int, src []float64, ldsrc int, dst []float64, lddst int) {
	copyRect(n, n, src, ldsrc, dst, lddst)
}

func copyRect(rows, cols int, src []float64, ldsrc int, dst []float64, lddst int) {
	for i := 0; i < rows; i++ {
		copy(dst[i*lddst:i*lddst+cols], src[i*ldsrc:i*ldsrc+cols])
	}
}
