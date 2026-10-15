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
	for _, test := range []struct {
		m, p, n    int
		dupA, dupB bool // Set column 1 equal to column 0.
	}{
		{m: 5, p: 5, n: 5},
		{m: 3, p: 3, n: 5},
		{m: 10, p: 5, n: 5},
		{m: 4, p: 2, n: 5, dupA: true, dupB: true},
		{m: 4, p: 3, n: 3, dupB: true},
		{m: 6, p: 4, n: 6, dupA: true, dupB: true},
		{m: 3, p: 2, n: 7},
		{m: 3, p: 2, n: 7, dupA: true, dupB: true},
	} {
		name := fmt.Sprintf("m=%d,p=%d,n=%d,dupA=%t,dupB=%t", test.m, test.p, test.n, test.dupA, test.dupB)
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
		eps := dlamchE
		tola := float64(max(m, n)) * impl.Dlange(lapack.Frobenius, m, n, a, n, nil) * eps
		tolb := float64(max(p, n)) * impl.Dlange(lapack.Frobenius, p, n, b, n, nil) * eps

		ga, gb := append([]float64(nil), a...), append([]float64(nil), b...)
		gu, gv, gq := make([]float64, m*m), make([]float64, p*p), make([]float64, n*n)
		iwork := make([]int, n)
		tau := make([]float64, n)
		work := []float64{0}
		impl.Dggsvp3(lapack.GSVDU, lapack.GSVDV, lapack.GSVDQ, m, p, n, ga, n, gb, n, tola, tolb,
			gu, m, gv, p, gq, n, iwork, tau, work, -1)
		work = make([]float64, int(work[0]))
		k, l := impl.Dggsvp3(lapack.GSVDU, lapack.GSVDV, lapack.GSVDQ, m, p, n, ga, n, gb, n, tola, tolb,
			gu, m, gv, p, gq, n, iwork, tau, work, len(work))

		na, nb := append([]float64(nil), a...), append([]float64(nil), b...)
		nu, nv, nq := make([]float64, m*m), make([]float64, p*p), make([]float64, n*n)
		wantK, wantL, info := netlib.Dggsvp3(m, p, n, na, nb, tola, tolb, nu, nv, nq)
		if info != 0 {
			t.Fatalf("%s: netlib info=%d", name, info)
		}
		if k != wantK || l != wantL {
			t.Errorf("%s: (k,l)=(%d,%d), want (%d,%d)", name, k, l, wantK, wantL)
			continue
		}
		for _, c := range []struct {
			name      string
			got, want []float64
		}{
			{"A", ga, na}, {"B", gb, nb}, {"U", gu, nu}, {"V", gv, nv}, {"Q", gq, nq},
		} {
			for i := range c.got {
				if math.Abs(c.got[i]-c.want[i]) > tol {
					t.Errorf("%s: %s[%d]=%v, want %v", name, c.name, i, c.got[i], c.want[i])
					break
				}
			}
		}
	}
}
