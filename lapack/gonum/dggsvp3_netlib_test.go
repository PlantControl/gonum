// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package gonum

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/lapack/gonum/internal/netlib"
)

// Includes rank-deficient pairs from https://github.com/gonum/gonum/issues/2127.
func TestDggsvp3NetlibDifferential(t *testing.T) {
	const tol = 1e-12
	impl := Implementation{}
	type gsvpCase struct {
		m, p, n        int
		dupA, dupB     bool    // Set column 1 equal to column 0.
		zeroA, zeroB   bool    // Set the operand to zero.
		scaleA, scaleB float64 // Operand scale; 0 means 1.
	}
	cases := []gsvpCase{
		{m: 5, p: 5, n: 5},
		{m: 3, p: 3, n: 5},
		{m: 10, p: 5, n: 5},
		{m: 1, p: 1, n: 1},
		{m: 1, p: 3, n: 4},
		{m: 4, p: 1, n: 4},
		{m: 4, p: 2, n: 5, dupA: true, dupB: true},
		{m: 4, p: 3, n: 3, dupB: true},
		{m: 6, p: 4, n: 6, dupA: true, dupB: true},
		{m: 3, p: 2, n: 7},
		{m: 3, p: 2, n: 7, dupA: true, dupB: true},
		{m: 4, p: 3, n: 5, zeroA: true},
		{m: 4, p: 3, n: 5, zeroB: true},
		{m: 4, p: 3, n: 5, zeroB: true, dupA: true},
		{m: 5, p: 4, n: 6, scaleA: 1e-150, dupB: true},
		{m: 5, p: 4, n: 6, scaleB: 1e-150, dupA: true},
		{m: 5, p: 4, n: 6, scaleA: 1e150, dupA: true, dupB: true},
		{m: 5, p: 4, n: 6, scaleB: 1e150},
		// Blocked Dgeqp3 (Dlaqps) path: min(p, n) above the crossover.
		{m: 150, p: 140, n: 150},
		{m: 150, p: 140, n: 150, dupA: true, dupB: true},
	}
	jobs := []struct {
		u, v, q    lapack.GSVDJob
		nu, nv, nq byte
	}{
		{lapack.GSVDU, lapack.GSVDV, lapack.GSVDQ, 'U', 'V', 'Q'},
		{lapack.GSVDNone, lapack.GSVDNone, lapack.GSVDNone, 'N', 'N', 'N'},
		{lapack.GSVDU, lapack.GSVDNone, lapack.GSVDQ, 'U', 'N', 'Q'},
		{lapack.GSVDNone, lapack.GSVDV, lapack.GSVDNone, 'N', 'V', 'N'},
	}
	for _, test := range cases {
		m, p, n := test.m, test.p, test.n
		rnd := rand.New(rand.NewPCG(1, 1))
		a := make([]float64, m*n)
		b := make([]float64, p*n)
		for i := range a {
			a[i] = rnd.NormFloat64()
		}
		for i := range b {
			b[i] = rnd.NormFloat64()
		}
		if test.dupA {
			for i := 0; i < m; i++ {
				a[i*n+1] = a[i*n]
			}
		}
		if test.dupB {
			for i := 0; i < p; i++ {
				b[i*n+1] = b[i*n]
			}
		}
		scaleA, scaleB := 1.0, 1.0
		if test.scaleA != 0 {
			scaleA = test.scaleA
		}
		if test.scaleB != 0 {
			scaleB = test.scaleB
		}
		if test.zeroA {
			scaleA = 0
		}
		if test.zeroB {
			scaleB = 0
		}
		for i := range a {
			a[i] *= scaleA
		}
		for i := range b {
			b[i] *= scaleB
		}
		tola := float64(max(m, n)) * impl.Dlange(lapack.Frobenius, m, n, a, n, nil) * dlamchE
		tolb := float64(max(p, n)) * impl.Dlange(lapack.Frobenius, p, n, b, n, nil) * dlamchE

		for _, job := range jobs {
			name := fmt.Sprintf("%+v/job=%c%c%c", test, job.nu, job.nv, job.nq)

			ga, gb := append([]float64(nil), a...), append([]float64(nil), b...)
			gu, gv, gq := make([]float64, m*m), make([]float64, p*p), make([]float64, n*n)
			iwork := make([]int, n)
			tau := make([]float64, n)
			work := []float64{0}
			impl.Dggsvp3(job.u, job.v, job.q, m, p, n, ga, n, gb, n, tola, tolb,
				gu, m, gv, p, gq, n, iwork, tau, work, -1)
			work = make([]float64, int(work[0]))
			k, l := impl.Dggsvp3(job.u, job.v, job.q, m, p, n, ga, n, gb, n, tola, tolb,
				gu, m, gv, p, gq, n, iwork, tau, work, len(work))

			na, nb := append([]float64(nil), a...), append([]float64(nil), b...)
			nu, nv, nq := make([]float64, m*m), make([]float64, p*p), make([]float64, n*n)
			wantK, wantL, info := netlib.Dggsvp3(job.nu, job.nv, job.nq, m, p, n, na, nb, tola, tolb, nu, nv, nq)
			if info != 0 {
				t.Fatalf("%s: netlib info=%d", name, info)
			}
			if k != wantK || l != wantL {
				t.Errorf("%s: (k,l)=(%d,%d), want (%d,%d)", name, k, l, wantK, wantL)
				continue
			}
			// With exactly duplicated columns, the pivot order and reflector signs in the
			// null direction are decided by roundoff, so compare invariants instead: the
			// singular values of A12, B13 and the nonzero rows of Uᵀ*A*Q.
			if test.dupA || test.dupB {
				for _, blk := range []struct {
					name           string
					r0, r1, c0, c1 int
					x, y           []float64
					scale          float64
				}{
					{"A12", 0, k, n - k - l, n - l, ga, na, scaleA},
					{"B13", 0, l, n - l, n, gb, nb, scaleB},
					{"A[0:k+l]", 0, min(m, k+l), n - k - l, n, ga, na, scaleA},
				} {
					got := gsvp3BlockSV(blk.x, n, blk.r0, blk.r1, blk.c0, blk.c1, blk.scale)
					want := gsvp3BlockSV(blk.y, n, blk.r0, blk.r1, blk.c0, blk.c1, blk.scale)
					for i := range got {
						if math.Abs(got[i]-want[i]) > tol*float64(n)*max(1, want[0]) {
							t.Errorf("%s: σ_%d(%s)=%v, want %v", name, i, blk.name, got[i], want[i])
							break
						}
					}
				}
				continue
			}
			type output struct {
				name      string
				got, want []float64
				scale     float64
				compare   bool
			}
			for _, c := range []output{
				{"A", ga, na, scaleA, true},
				{"B", gb, nb, scaleB, true},
				{"U", gu, nu, 1, job.nu == 'U'},
				{"V", gv, nv, 1, job.nv == 'V'},
				{"Q", gq, nq, 1, job.nq == 'Q'},
			} {
				if !c.compare {
					continue
				}
				for i := range c.got {
					if math.Abs(c.got[i]-c.want[i]) > tol*max(1, float64(n))*max(c.scale, math.SmallestNonzeroFloat64) {
						t.Errorf("%s: %s[%d]=%v, want %v", name, c.name, i, c.got[i], c.want[i])
						break
					}
				}
			}
		}
	}
}

// gsvp3BlockSV returns the singular values of a[r0:r1, c0:c1]/scale.
func gsvp3BlockSV(a []float64, lda, r0, r1, c0, c1 int, scale float64) []float64 {
	rows, cols := r1-r0, c1-c0
	if rows == 0 || cols == 0 {
		return nil
	}
	if scale == 0 {
		scale = 1
	}
	blk := make([]float64, rows*cols)
	for i := range rows {
		for j := range cols {
			blk[i*cols+j] = a[(r0+i)*lda+c0+j] / scale
		}
	}
	s := make([]float64, min(rows, cols))
	impl := Implementation{}
	work := []float64{0}
	impl.Dgesvd(lapack.SVDNone, lapack.SVDNone, rows, cols, blk, cols, s, nil, 1, nil, 1, work, -1)
	work = make([]float64, int(work[0]))
	if !impl.Dgesvd(lapack.SVDNone, lapack.SVDNone, rows, cols, blk, cols, s, nil, 1, nil, 1, work, len(work)) {
		panic("Dgesvd did not converge")
	}
	return s
}
