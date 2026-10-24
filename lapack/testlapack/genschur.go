// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlapack

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/lapack"
)

// genSchurBlocks returns the start and size of each diagonal block of the
// quasi-upper triangular s.
func genSchurBlocks(s blas64.General) (starts, sizes []int) {
	n := s.Rows
	for j := 0; j < n; {
		size := 1
		if j < n-1 && s.Data[(j+1)*s.Stride+j] != 0 {
			size = 2
		}
		starts = append(starts, j)
		sizes = append(sizes, size)
		j += size
	}
	return starts, sizes
}

// pencilBlockEigenvalues returns the eigenvalues of the 1×1 or 2×2 diagonal
// block of (s,p) at j, with an infinite eigenvalue as cmplx.Inf().
func pencilBlockEigenvalues(s, p blas64.General, j, size int) []complex128 {
	if size == 1 {
		if p.Data[j*p.Stride+j] == 0 {
			return []complex128{cmplx.Inf()}
		}
		return []complex128{complex(s.Data[j*s.Stride+j]/p.Data[j*p.Stride+j], 0)}
	}
	s11, s12 := s.Data[j*s.Stride+j], s.Data[j*s.Stride+j+1]
	s21, s22 := s.Data[(j+1)*s.Stride+j], s.Data[(j+1)*s.Stride+j+1]
	p11, p12, p22 := p.Data[j*p.Stride+j], p.Data[j*p.Stride+j+1], p.Data[(j+1)*p.Stride+j+1]
	qa := p11 * p22
	qb := -(s11*p22 + s22*p11 - p12*s21)
	qc := s11*s22 - s12*s21
	if qa == 0 {
		return []complex128{cmplx.Inf(), cmplx.Inf()}
	}
	d := cmplx.Sqrt(complex(qb*qb-4*qa*qc, 0))
	return []complex128{(complex(-qb, 0) + d) / complex(2*qa, 0), (complex(-qb, 0) - d) / complex(2*qa, 0)}
}

// pencilEigenvalues returns the eigenvalues of the generalized Schur pencil
// (s,p) in diagonal order.
func pencilEigenvalues(s, p blas64.General) []complex128 {
	var ev []complex128
	starts, sizes := genSchurBlocks(s)
	for k, j := range starts {
		ev = append(ev, pencilBlockEigenvalues(s, p, j, sizes[k])...)
	}
	return ev
}

// homogeneousEigenvalues returns (alphar+i*alphai)/beta, with an infinite
// eigenvalue as cmplx.Inf().
func homogeneousEigenvalues(alphar, alphai, beta []float64) []complex128 {
	ev := make([]complex128, len(alphar))
	for i := range ev {
		if beta[i] == 0 {
			ev[i] = cmplx.Inf()
			continue
		}
		ev[i] = complex(alphar[i], alphai[i]) / complex(beta[i], 0)
	}
	return ev
}

func eigenvalueClose(a, b complex128, tol float64) bool {
	return chordalDistance(a, b) <= tol
}

// chordalDistance returns the chordal distance between a and b on the
// Riemann sphere, with cmplx.IsInf values at the pole.
func chordalDistance(a, b complex128) float64 {
	switch {
	case cmplx.IsInf(a) && cmplx.IsInf(b):
		return 0
	case cmplx.IsInf(a):
		return 1 / math.Hypot(1, cmplx.Abs(b))
	case cmplx.IsInf(b):
		return 1 / math.Hypot(1, cmplx.Abs(a))
	}
	return cmplx.Abs(a-b) / (math.Hypot(1, cmplx.Abs(a)) * math.Hypot(1, cmplx.Abs(b)))
}

// sameEigenvalues reports whether got and want are equal as multisets, to
// within the relative tolerance tol.
func sameEigenvalues(got, want []complex128, tol float64) bool {
	if len(got) != len(want) {
		return false
	}
	used := make([]bool, len(want))
	for _, g := range got {
		best := -1
		for k, w := range want {
			if used[k] || !eigenvalueClose(g, w, tol) {
				continue
			}
			if best < 0 || chordalDistance(g, w) < chordalDistance(g, want[best]) {
				best = k
			}
		}
		if best < 0 {
			return false
		}
		used[best] = true
	}
	return true
}

// checkPaddingNaN checks that the elements of m outside its Rows×Cols view
// that lie inside m.Data are still NaN.
func checkPaddingNaN(t *testing.T, name string, m blas64.General) {
	t.Helper()
	for i := 0; i < m.Rows; i++ {
		for j := m.Cols; j < m.Stride && i*m.Stride+j < len(m.Data); j++ {
			if !math.IsNaN(m.Data[i*m.Stride+j]) {
				t.Errorf("%s: padding [%d,%d] modified to %v", name, i, j, m.Data[i*m.Stride+j])
				return
			}
		}
	}
}

// checkGeneralizedSchurForm checks that (s,p) is in real generalized Schur
// form: s quasi-upper triangular with 2×2 blocks having complex conjugate
// eigenvalues, p upper triangular with the 2×2 diagonal blocks diagonal, and
// positive if positive is true. DHGEQZ makes them positive; DTGEX2 does not.
func checkGeneralizedSchurForm(t *testing.T, name string, s, p blas64.General, positive bool) {
	t.Helper()
	n := s.Rows
	for i := 0; i < n; i++ {
		for j := 0; j < i; j++ {
			if p.Data[i*p.Stride+j] != 0 {
				t.Errorf("%s: P[%d,%d]=%v, want 0", name, i, j, p.Data[i*p.Stride+j])
				return
			}
			if j < i-1 && s.Data[i*s.Stride+j] != 0 {
				t.Errorf("%s: S[%d,%d]=%v, want 0", name, i, j, s.Data[i*s.Stride+j])
				return
			}
		}
	}
	for i := 0; i < n-2; i++ {
		if s.Data[(i+1)*s.Stride+i] != 0 && s.Data[(i+2)*s.Stride+i+1] != 0 {
			t.Errorf("%s: S has consecutive nonzero subdiagonals at %d", name, i)
			return
		}
	}
	starts, sizes := genSchurBlocks(s)
	for k, j := range starts {
		if sizes[k] != 2 {
			continue
		}
		p11, p22 := p.Data[j*p.Stride+j], p.Data[(j+1)*p.Stride+j+1]
		if p.Data[j*p.Stride+j+1] != 0 || p11 == 0 || p22 == 0 || (positive && (p11 < 0 || p22 < 0)) {
			t.Errorf("%s: P 2×2 block at %d is not a valid diagonal: [%v %v; 0 %v]", name, j,
				p.Data[j*p.Stride+j], p.Data[j*p.Stride+j+1], p.Data[(j+1)*p.Stride+j+1])
		}
		ev := pencilBlockEigenvalues(s, p, j, 2)
		if imag(ev[0]) == 0 {
			t.Errorf("%s: 2×2 block at %d has real eigenvalues %v", name, j, ev)
		}
	}
}

// checkGeneralizedSchurFactors checks that a0 = q*s*zᵀ and b0 = q*p*zᵀ and
// that q and z are orthogonal.
func checkGeneralizedSchurFactors(t *testing.T, name string, a0, b0, s, p, q, z blas64.General) {
	t.Helper()
	n := a0.Rows
	if n == 0 {
		return
	}
	tol := 100 * float64(n) * dlamchE
	for _, m := range []struct {
		label  string
		orig   blas64.General
		factor blas64.General
	}{{"A", a0, s}, {"B", b0, p}} {
		norm := math.Max(1, dlange(lapack.MaxColumnSum, n, n, m.orig.Data, m.orig.Stride))
		if r := residualGeneralizedSchur(m.orig, m.factor, q, z) / norm; r > tol {
			t.Errorf("%s: |%s - Q*S*Zᵀ|/|%s|=%v, want <= %v", name, m.label, m.label, r, tol)
		}
	}
	if r := residualOrthogonal(q, false); r > tol {
		t.Errorf("%s: Q not orthogonal, resid=%v", name, r)
	}
	if r := residualOrthogonal(z, false); r > tol {
		t.Errorf("%s: Z not orthogonal, resid=%v", name, r)
	}
}

// randomOrthogonalStrided returns a random n×n orthogonal matrix stored with
// stride ld and NaN padding.
func randomOrthogonalStrided(n, ld int, rnd *rand.Rand) blas64.General {
	q := nanGeneral(n, n, ld)
	if n == 0 {
		return q
	}
	r := randomOrthogonal(n, rnd)
	for i := 0; i < n; i++ {
		copy(q.Data[i*ld:i*ld+n], r.Data[i*r.Stride:i*r.Stride+n])
	}
	return q
}

// pencilProduct returns q*s*zᵀ with stride n.
func pencilProduct(q, s, z blas64.General) blas64.General {
	n := s.Rows
	w := zeros(n, n, n)
	r := zeros(n, n, n)
	if n == 0 {
		return r
	}
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, q, s, 0, w)
	blas64.Gemm(blas.NoTrans, blas.Trans, 1, w, z, 0, r)
	return r
}

// testDtgex2Random checks Dtgex2 on random generalized Schur pencils: an
// accepted swap must keep the factorization, the Schur form and exchange the
// block eigenvalues; a rejected swap must leave all outputs unchanged.
func testDtgex2Random(t *testing.T, impl Dtgex2er) {
	rnd := rand.New(rand.NewPCG(11, 12))
	for n := 2; n <= 8; n++ {
		for trial := 0; trial < 40; trial++ {
			ld := n + 3
			s, p, _, _, _ := randomGeneralizedSchurPair(n, ld, trial%4 == 0, rnd)
			starts, sizes := genSchurBlocks(s)
			if len(starts) < 2 {
				continue
			}
			k := rnd.IntN(len(starts) - 1)
			j1, n1, n2 := starts[k], sizes[k], sizes[k+1]
			q := randomOrthogonalStrided(n, ld, rnd)
			z := randomOrthogonalStrided(n, ld, rnd)
			a0, b0 := pencilProduct(q, s, z), pencilProduct(q, p, z)
			ev1 := pencilBlockEigenvalues(s, p, j1, n1)
			ev2 := pencilBlockEigenvalues(s, p, j1+n1, n2)
			sOrig, pOrig, qOrig, zOrig := cloneGeneral(s), cloneGeneral(p), cloneGeneral(q), cloneGeneral(z)
			name := fmt.Sprintf("n=%d trial=%d j1=%d n1=%d n2=%d", n, trial, j1, n1, n2)

			var query [1]float64
			impl.Dtgex2(true, true, n, nil, ld, nil, ld, nil, ld, nil, ld, j1, n1, n2, query[:], -1)
			work := nanSlice(int(query[0]))
			ok := impl.Dtgex2(true, true, n, s.Data, ld, p.Data, ld, q.Data, ld, z.Data, ld, j1, n1, n2, work, len(work))
			for _, m := range []struct {
				label string
				m     blas64.General
			}{{"S", s}, {"P", p}, {"Q", q}, {"Z", z}} {
				checkPaddingNaN(t, name+" "+m.label, m.m)
			}

			s2, p2 := cloneGeneral(sOrig), cloneGeneral(pOrig)
			ok2 := impl.Dtgex2(false, false, n, s2.Data, ld, p2.Data, ld, nil, 1, nil, 1, j1, n1, n2, work, len(work))
			if ok2 != ok || !equalGeneral(s2, s) || !equalGeneral(p2, p) {
				t.Errorf("%s: result depends on wantq/wantz", name)
			}

			if !ok {
				if !equalGeneral(s, sOrig) || !equalGeneral(p, pOrig) || !equalGeneral(q, qOrig) || !equalGeneral(z, zOrig) {
					t.Errorf("%s: rejected swap modified its outputs", name)
				}
				continue
			}
			checkGeneralizedSchurFactors(t, name, a0, b0, s, p, q, z)
			checkGeneralizedSchurForm(t, name, s, p, false)
			gotStarts, gotSizes := genSchurBlocks(s)
			nb := 0
			for kk, j := range gotStarts {
				if j == j1 {
					nb = gotSizes[kk]
				}
			}
			if nb != n2 {
				t.Errorf("%s: leading block after swap has size %d, want %d", name, nb, n2)
				continue
			}
			const tol = 1e-8
			if got := pencilBlockEigenvalues(s, p, j1, n2); !sameEigenvalues(got, ev2, tol) {
				t.Errorf("%s: leading block eigenvalues %v, want %v", name, got, ev2)
			}
			if got := pencilBlockEigenvalues(s, p, j1+n2, n1); !sameEigenvalues(got, ev1, tol) {
				t.Errorf("%s: trailing block eigenvalues %v, want %v", name, got, ev1)
			}
		}
	}
}

// testDtgexcRandom checks that Dtgexc moves the block at ifst to ilst while
// keeping the factorization, the Schur form and the spectrum.
func testDtgexcRandom(t *testing.T, impl Dtgexcer) {
	rnd := rand.New(rand.NewPCG(13, 14))
	for n := 1; n <= 8; n++ {
		for trial := 0; trial < 30; trial++ {
			ld := n + 2
			s, p, _, _, _ := randomGeneralizedSchurPair(n, ld, trial%3 == 0, rnd)
			q := randomOrthogonalStrided(n, ld, rnd)
			z := randomOrthogonalStrided(n, ld, rnd)
			a0, b0 := pencilProduct(q, s, z), pencilProduct(q, p, z)
			evOrig := pencilEigenvalues(s, p)
			ifst, ilst := rnd.IntN(n), rnd.IntN(n)
			starts, sizes := genSchurBlocks(s)
			var evMoved []complex128
			for k, j := range starts {
				if ifst == j || (sizes[k] == 2 && ifst == j+1) {
					evMoved = pencilBlockEigenvalues(s, p, j, sizes[k])
				}
			}
			name := fmt.Sprintf("n=%d trial=%d ifst=%d ilst=%d", n, trial, ifst, ilst)

			var query [1]float64
			impl.Dtgexc(true, true, n, nil, ld, nil, ld, nil, ld, nil, ld, ifst, ilst, query[:], -1)
			work := nanSlice(max(1, int(query[0])))
			_, ilstOut, ok := impl.Dtgexc(true, true, n, s.Data, ld, p.Data, ld, q.Data, ld, z.Data, ld, ifst, ilst, work, len(work))
			for _, m := range []struct {
				label string
				m     blas64.General
			}{{"S", s}, {"P", p}, {"Q", q}, {"Z", z}} {
				checkPaddingNaN(t, name+" "+m.label, m.m)
			}
			checkGeneralizedSchurFactors(t, name, a0, b0, s, p, q, z)
			checkGeneralizedSchurForm(t, name, s, p, false)
			const tol = 1e-8
			if got := pencilEigenvalues(s, p); !sameEigenvalues(got, evOrig, tol) {
				t.Errorf("%s: spectrum changed from %v to %v", name, evOrig, got)
			}
			if !ok {
				continue
			}
			gotStarts, gotSizes := genSchurBlocks(s)
			found := false
			for k, j := range gotStarts {
				if j == ilstOut {
					found = true
					if got := pencilBlockEigenvalues(s, p, j, gotSizes[k]); !sameEigenvalues(got, evMoved, tol) {
						t.Errorf("%s: block at ilst=%d has eigenvalues %v, want %v", name, ilstOut, got, evMoved)
					}
				}
			}
			if !found {
				t.Errorf("%s: returned ilst=%d is not a block start", name, ilstOut)
			}
		}
	}
}

// testDtgsenRandom checks that Dtgsen moves the selected eigenvalues to the
// leading m positions while keeping the factorization, the Schur form, the
// spectrum and consistent alphar, alphai and beta.
func testDtgsenRandom(t *testing.T, impl Dtgsener) {
	rnd := rand.New(rand.NewPCG(15, 16))
	for n := 1; n <= 8; n++ {
		for trial := 0; trial < 30; trial++ {
			ld := n + 3
			s, p, _, _, _ := randomGeneralizedSchurPair(n, ld, trial%3 == 0, rnd)
			q := randomOrthogonalStrided(n, ld, rnd)
			z := randomOrthogonalStrided(n, ld, rnd)
			a0, b0 := pencilProduct(q, s, z), pencilProduct(q, p, z)
			sOrig, pOrig := cloneGeneral(s), cloneGeneral(p)
			selected := make([]bool, n)
			for i := range selected {
				selected[i] = rnd.IntN(2) == 0
			}
			var evSel, evRest []complex128
			starts, sizes := genSchurBlocks(s)
			for k, j := range starts {
				ev := pencilBlockEigenvalues(s, p, j, sizes[k])
				if selected[j] || (sizes[k] == 2 && selected[j+1]) {
					evSel = append(evSel, ev...)
				} else {
					evRest = append(evRest, ev...)
				}
			}
			name := fmt.Sprintf("n=%d trial=%d", n, trial)

			alphar, alphai, beta := nanSlice(n), nanSlice(n), nanSlice(n)
			work := make([]float64, 1)
			iwork := make([]int, 1)
			impl.Dtgsen(0, true, true, selected, n, nil, ld, nil, ld, nil, nil, nil, nil, ld, nil, ld, work, -1, iwork, -1)
			work = nanSlice(max(1, int(work[0])))
			iwork = make([]int, max(1, iwork[0]))
			m, _, _, _, ok := impl.Dtgsen(0, true, true, selected, n, s.Data, ld, p.Data, ld,
				alphar, alphai, beta, q.Data, ld, z.Data, ld, work, len(work), iwork, len(iwork))
			for _, mm := range []struct {
				label string
				m     blas64.General
			}{{"S", s}, {"P", p}, {"Q", q}, {"Z", z}} {
				checkPaddingNaN(t, name+" "+mm.label, mm.m)
			}

			s2, p2 := cloneGeneral(sOrig), cloneGeneral(pOrig)
			ar2, ai2, beta2 := nanSlice(n), nanSlice(n), nanSlice(n)
			m2, _, _, _, ok2 := impl.Dtgsen(0, false, false, selected, n, s2.Data, ld, p2.Data, ld,
				ar2, ai2, beta2, nil, 1, nil, 1, work, len(work), iwork, len(iwork))
			if ok2 != ok || m2 != m || !equalGeneral(s2, s) || !equalGeneral(p2, p) {
				t.Errorf("%s: result depends on wantq/wantz", name)
			}

			checkGeneralizedSchurFactors(t, name, a0, b0, s, p, q, z)
			checkGeneralizedSchurForm(t, name, s, p, false)
			gotStarts, gotSizes := genSchurBlocks(s)
			for k, j := range gotStarts {
				if gotSizes[k] == 1 && math.Signbit(p.Data[j*ld+j]) {
					t.Errorf("%s: P[%d,%d]=%v, want nonnegative", name, j, j, p.Data[j*ld+j])
				}
			}
			for j := 0; j < n; j++ {
				if beta[j] < 0 {
					t.Errorf("%s: beta[%d]=%v, want nonnegative", name, j, beta[j])
				}
			}
			const tol = 1e-8
			ev := homogeneousEigenvalues(alphar, alphai, beta)
			if !sameEigenvalues(ev, pencilEigenvalues(s, p), tol) {
				t.Errorf("%s: alpha/beta %v disagree with blocks %v", name, ev, pencilEigenvalues(s, p))
			}
			if m != len(evSel) {
				t.Errorf("%s: m=%d, want %d", name, m, len(evSel))
				continue
			}
			if !ok {
				continue
			}
			if !sameEigenvalues(ev[:m], evSel, tol) || !sameEigenvalues(ev[m:], evRest, tol) {
				t.Errorf("%s: leading eigenvalues %v, want selected %v", name, ev[:m], evSel)
			}
		}
	}
}

// testDggesSortedRandom checks Dgges with sorting on random nonsymmetric
// pencils against the original pencil and an unsorted run.
func testDggesSortedRandom(t *testing.T, impl Dggeser) {
	rnd := rand.New(rand.NewPCG(17, 18))
	selctg := func(ar, ai, b float64) bool { return b != 0 && ar < 0 }
	for n := 1; n <= 8; n++ {
		for trial := 0; trial < 15; trial++ {
			ld := n + 3
			a0 := randomGeneral(n, n, ld, rnd)
			b0 := randomGeneral(n, n, ld, rnd)
			if trial%3 == 0 && n > 1 {
				for i := 0; i < n; i++ {
					b0.Data[i*ld+n-1] = 0
				}
			}
			name := fmt.Sprintf("n=%d trial=%d", n, trial)
			run := func(sort lapack.SchurSort) (s, p, vsl, vsr blas64.General, ev []complex128, sdim int, ok bool) {
				s, p = cloneGeneral(a0), cloneGeneral(b0)
				vsl, vsr = nanGeneral(n, n, ld), nanGeneral(n, n, ld)
				alphar, alphai, beta := nanSlice(n), nanSlice(n), nanSlice(n)
				work := make([]float64, 1)
				impl.Dgges(lapack.SchurHess, lapack.SchurHess, sort, selctg, n, nil, ld, nil, ld,
					nil, nil, nil, nil, ld, nil, ld, work, -1, nil)
				work = nanSlice(int(work[0]))
				sdim, ok = impl.Dgges(lapack.SchurHess, lapack.SchurHess, sort, selctg, n, s.Data, ld, p.Data, ld,
					alphar, alphai, beta, vsl.Data, ld, vsr.Data, ld, work, len(work), make([]bool, n))
				for i := range beta {
					if beta[i] < 0 {
						t.Errorf("%s: beta[%d]=%v, want nonnegative", name, i, beta[i])
					}
				}
				return s, p, vsl, vsr, homogeneousEigenvalues(alphar, alphai, beta), sdim, ok
			}
			_, _, _, _, evAll, _, ok := run(lapack.SortNone)
			if !ok {
				t.Errorf("%s: unsorted Dgges failed", name)
				continue
			}
			s, p, vsl, vsr, ev, sdim, ok := run(lapack.SortSelected)
			for _, m := range []struct {
				label string
				m     blas64.General
			}{{"S", s}, {"T", p}, {"VSL", vsl}, {"VSR", vsr}} {
				checkPaddingNaN(t, name+" "+m.label, m.m)
			}
			if !ok {
				t.Errorf("%s: sorted Dgges failed", name)
				continue
			}
			checkGeneralizedSchurFactors(t, name, a0, b0, s, p, vsl, vsr)
			checkGeneralizedSchurForm(t, name, s, p, false)
			const tol = 1e-8
			if !sameEigenvalues(ev, evAll, tol) {
				t.Errorf("%s: sorted spectrum %v, want %v", name, ev, evAll)
			}
			if !sameEigenvalues(ev, pencilEigenvalues(s, p), tol) {
				t.Errorf("%s: alpha/beta %v disagree with blocks %v", name, ev, pencilEigenvalues(s, p))
			}
			want := 0
			for _, e := range evAll {
				if !cmplx.IsInf(e) && real(e) < 0 {
					want++
				}
			}
			if sdim != want {
				t.Errorf("%s: sdim=%d, want %d", name, sdim, want)
			}
			for i, e := range ev {
				if (i < sdim) != (!cmplx.IsInf(e) && real(e) < 0) {
					t.Errorf("%s: eigenvalue %d=%v misplaced for sdim=%d", name, i, e, sdim)
				}
			}
		}
	}
}

// testDhgeqzRandomProperties checks Dhgeqz on random Hessenberg-triangular
// pencils including zero diagonal elements of T (infinite eigenvalues) and
// active subranges.
func testDhgeqzRandomProperties(t *testing.T, impl Dhgeqzer) {
	rnd := rand.New(rand.NewPCG(19, 20))
	for n := 1; n <= 10; n++ {
		for trial := 0; trial < 30; trial++ {
			ld := n + 3
			h := nanGeneral(n, n, ld)
			tt := nanGeneral(n, n, ld)
			for i := 0; i < n; i++ {
				for j := 0; j < n; j++ {
					h.Data[i*ld+j], tt.Data[i*ld+j] = 0, 0
					if j >= i-1 {
						h.Data[i*ld+j] = rnd.NormFloat64()
					}
					if j >= i {
						tt.Data[i*ld+j] = rnd.NormFloat64()
					}
				}
			}
			ilo, ihi := 0, n-1
			if trial%3 == 1 && n > 2 {
				ilo = rnd.IntN(n / 2)
				ihi = ilo + rnd.IntN(n-ilo)
			}
			if ilo > 0 {
				h.Data[ilo*ld+ilo-1] = 0
				for i := 0; i < ilo; i++ {
					if i > 0 {
						h.Data[i*ld+i-1] = 0
					}
				}
			}
			for i := ihi + 1; i < n; i++ {
				h.Data[i*ld+i-1] = 0
			}
			var nzero int
			switch trial % 5 {
			case 1:
				tt.Data[ilo*ld+ilo] = 0
			case 2:
				tt.Data[ihi*ld+ihi] = 0
			case 3:
				j := ilo + rnd.IntN(ihi-ilo+1)
				tt.Data[j*ld+j] = 0
				if j+1 <= ihi {
					tt.Data[(j+1)*ld+j+1] = 0
				}
			case 4:
				if ihi > ilo+1 {
					h.Data[(ilo+2)*ld+ilo+1] = 1e-17
					tt.Data[(ilo+1)*ld+ilo+1] = 0
				}
			}
			for j := ilo; j <= ihi; j++ {
				if tt.Data[j*ld+j] == 0 {
					nzero++
				}
			}
			name := fmt.Sprintf("n=%d trial=%d ilo=%d ihi=%d", n, trial, ilo, ihi)
			q := randomOrthogonalStrided(n, ld, rnd)
			z := randomOrthogonalStrided(n, ld, rnd)
			a0, b0 := pencilProduct(q, h, z), pencilProduct(q, tt, z)
			hOrig, tOrig := cloneGeneral(h), cloneGeneral(tt)

			alphar, alphai, beta := nanSlice(n), nanSlice(n), nanSlice(n)
			work := nanSlice(max(1, n))
			ok := impl.Dhgeqz(lapack.EigenvaluesAndSchur, lapack.SchurOrig, lapack.SchurOrig, n, ilo, ihi,
				h.Data, ld, tt.Data, ld, alphar, alphai, beta, q.Data, ld, z.Data, ld, work, len(work))
			if !ok {
				t.Errorf("%s: Dhgeqz failed", name)
				continue
			}
			for _, m := range []struct {
				label string
				m     blas64.General
			}{{"S", h}, {"P", tt}, {"Q", q}, {"Z", z}} {
				checkPaddingNaN(t, name+" "+m.label, m.m)
			}
			checkGeneralizedSchurFactors(t, name, a0, b0, h, tt, q, z)
			checkGeneralizedSchurForm(t, name, h, tt, true)
			starts, sizes := genSchurBlocks(h)
			for k, j := range starts {
				if sizes[k] == 1 {
					if alphar[j] != h.Data[j*ld+j] || alphai[j] != 0 || beta[j] != tt.Data[j*ld+j] {
						t.Errorf("%s: eigenvalue %d alpha=(%v,%v) beta=%v, want (%v,0)/%v", name, j,
							alphar[j], alphai[j], beta[j], h.Data[j*ld+j], tt.Data[j*ld+j])
					}
					if beta[j] < 0 {
						t.Errorf("%s: beta[%d]=%v, want nonnegative", name, j, beta[j])
					}
					continue
				}
				if !(alphai[j] > 0) || !(alphai[j+1] < 0) {
					t.Errorf("%s: pair %d alphai=%v,%v, want positive then negative", name, j, alphai[j], alphai[j+1])
				}
				if !(beta[j] > 0 && beta[j+1] > 0) {
					t.Errorf("%s: pair %d beta=%v,%v, want positive", name, j, beta[j], beta[j+1])
				}
				ev := homogeneousEigenvalues(alphar[j:j+2], alphai[j:j+2], beta[j:j+2])
				if !sameEigenvalues(ev, pencilBlockEigenvalues(h, tt, j, 2), 1e-8) {
					t.Errorf("%s: pair %d eigenvalues %v disagree with block %v", name, j, ev, pencilBlockEigenvalues(h, tt, j, 2))
				}
			}
			var ninf int
			for j := ilo; j <= ihi; j++ {
				if beta[j] == 0 {
					ninf++
				}
			}
			if nzero > 0 && ninf == 0 {
				t.Errorf("%s: no infinite eigenvalue with %d zero diagonal elements of T", name, nzero)
			}

			he, te := cloneGeneral(hOrig), cloneGeneral(tOrig)
			are, aie, betae := nanSlice(n), nanSlice(n), nanSlice(n)
			if !impl.Dhgeqz(lapack.EigenvaluesOnly, lapack.SchurNone, lapack.SchurNone, n, ilo, ihi,
				he.Data, ld, te.Data, ld, are, aie, betae, nil, 1, nil, 1, work, len(work)) {
				t.Errorf("%s: eigenvalue-only Dhgeqz failed", name)
				continue
			}
			if !sameEigenvalues(homogeneousEigenvalues(are, aie, betae), homogeneousEigenvalues(alphar, alphai, beta), 1e-8) {
				t.Errorf("%s: eigenvalue-only spectrum %v, want %v", name,
					homogeneousEigenvalues(are, aie, betae), homogeneousEigenvalues(alphar, alphai, beta))
			}
		}
	}
}

// testDggbalRoundTrip checks that the balanced pencil equals Lᵀ*(A,B)*R where
// L and R are obtained by applying Dggbak to the identity, and that the rows
// and columns isolated by permutation are triangular.
func testDggbalRoundTrip(t *testing.T, bal Dggbaler, bak Dggbaker) {
	rnd := rand.New(rand.NewPCG(21, 22))
	for n := 1; n <= 8; n++ {
		for trial := 0; trial < 20; trial++ {
			ld := n + 2
			a0 := randomGeneral(n, n, ld, rnd)
			b0 := randomGeneral(n, n, ld, rnd)
			ntop, nbot := 0, 0
			if n > 2 {
				ntop, nbot = rnd.IntN(2), rnd.IntN(2)
			}
			for i := 0; i < n; i++ {
				for j := 0; j < n; j++ {
					if (j < ntop && i > j) || (i >= n-nbot && j < i) {
						a0.Data[i*ld+j], b0.Data[i*ld+j] = 0, 0
					}
				}
			}
			perm := rnd.Perm(n)
			pa, pb := cloneGeneral(a0), cloneGeneral(b0)
			for i := 0; i < n; i++ {
				for j := 0; j < n; j++ {
					scale := math.Pow(2, float64(rnd.IntN(41)-20))
					if trial%2 == 0 {
						scale = 1
					}
					pa.Data[perm[i]*ld+perm[j]] = scale * a0.Data[i*ld+j]
					pb.Data[perm[i]*ld+perm[j]] = scale * b0.Data[i*ld+j]
				}
			}
			for _, job := range []lapack.BalanceJob{lapack.BalanceNone, lapack.Permute, lapack.Scale, lapack.PermuteScale} {
				name := fmt.Sprintf("n=%d trial=%d job=%c", n, trial, job)
				a, b := cloneGeneral(pa), cloneGeneral(pb)
				lscale, rscale := nanSlice(n), nanSlice(n)
				ilo, ihi := bal.Dggbal(job, n, a.Data, ld, b.Data, ld, lscale, rscale, nanSlice(6*n))
				checkPaddingNaN(t, name+" A", a)
				checkPaddingNaN(t, name+" B", b)
				l, r := eye(n, n), eye(n, n)
				bak.Dggbak(job, blas.Left, n, ilo, ihi, lscale, rscale, n, l.Data, n)
				bak.Dggbak(job, blas.Right, n, ilo, ihi, lscale, rscale, n, r.Data, n)
				for _, m := range []struct {
					label     string
					orig, got blas64.General
				}{{"A", pa, a}, {"B", pb, b}} {
					w := zeros(n, n, n)
					x := zeros(n, n, n)
					blas64.Gemm(blas.Trans, blas.NoTrans, 1, l, m.orig, 0, w)
					blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, w, r, 0, x)
					var diff, norm float64
					for i := 0; i < n; i++ {
						for j := 0; j < n; j++ {
							diff = math.Max(diff, math.Abs(x.Data[i*n+j]-m.got.Data[i*ld+j]))
							norm = math.Max(norm, math.Abs(m.got.Data[i*ld+j]))
						}
					}
					if diff > 1e-13*math.Max(norm, 1e-300) {
						t.Errorf("%s: balanced %s differs from Lᵀ*%s*R by %v (norm %v)", name, m.label, m.label, diff, norm)
					}
					for i := 0; i < n; i++ {
						for j := 0; j < i; j++ {
							if (i > ihi || j < ilo) && m.got.Data[i*ld+j] != 0 {
								t.Errorf("%s: balanced %s[%d,%d]=%v outside ilo=%d ihi=%d, want 0", name, m.label, i, j, m.got.Data[i*ld+j], ilo, ihi)
							}
						}
					}
				}
				if (job == lapack.BalanceNone || job == lapack.Scale) && (ilo != 0 || ihi != n-1) {
					t.Errorf("%s: ilo=%d ihi=%d, want 0, %d", name, ilo, ihi, n-1)
				}
				if (job == lapack.Permute || job == lapack.PermuteScale) && ihi-ilo+1 > n-ntop-nbot {
					t.Errorf("%s: active block %d..%d larger than %d", name, ilo, ihi, n-ntop-nbot)
				}
			}
		}
	}
}
