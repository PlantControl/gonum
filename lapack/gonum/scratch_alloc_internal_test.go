// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/blas"
)

func TestDlahqrNoAllocs(t *testing.T) {
	const n = 12
	rnd := rand.New(rand.NewPCG(1, 1))
	h0 := make([]float64, n*n)
	for i := range n {
		for j := max(0, i-1); j < n; j++ {
			h0[i*n+j] = rnd.NormFloat64()
		}
	}
	h := make([]float64, n*n)
	z := make([]float64, n*n)
	wr := make([]float64, n)
	wi := make([]float64, n)
	allocs := testing.AllocsPerRun(5, func() {
		copy(h, h0)
		clear(z)
		for i := range n {
			z[i*n+i] = 1
		}
		if unconverged := (Implementation{}).Dlahqr(true, true, n, 0, n-1, h, n, wr, wi, 0, n-1, z, n); unconverged != 0 {
			t.Fatalf("unconverged = %d", unconverged)
		}
	})
	if allocs != 0 {
		t.Errorf("Dlahqr allocs = %v, want 0", allocs)
	}
}

func TestDlaexcNoAllocs(t *testing.T) {
	for _, tc := range []struct {
		name   string
		n1, n2 int
		t      []float64
	}{
		{"1x1", 1, 1, []float64{
			1, 2,
			0, 3,
		}},
		{"1x2", 1, 2, []float64{
			1, 0.5, 0.7,
			0, 2, 3,
			0, -4, 2,
		}},
		{"2x1", 2, 1, []float64{
			2, 3, 0.5,
			-4, 2, 0.7,
			0, 0, 1,
		}},
		{"2x2", 2, 2, []float64{
			1, 2, 0.3, 0.4,
			-3, 1, 0.5, 0.6,
			0, 0, 4, 5,
			0, 0, -6, 4,
		}},
	} {
		n := tc.n1 + tc.n2
		tt := make([]float64, n*n)
		q := make([]float64, n*n)
		work := make([]float64, n)
		allocs := testing.AllocsPerRun(5, func() {
			copy(tt, tc.t)
			clear(q)
			for i := range n {
				q[i*n+i] = 1
			}
			if !(Implementation{}).Dlaexc(true, n, tt, n, q, n, 0, tc.n1, tc.n2, work) {
				t.Fatalf("%s: swap rejected", tc.name)
			}
		})
		if allocs != 0 {
			t.Errorf("%s: Dlaexc allocs = %v, want 0", tc.name, allocs)
		}
	}
}

func TestDlasy2NoAllocs(t *testing.T) {
	for _, tc := range []struct{ n1, n2 int }{{1, 1}, {1, 2}, {2, 1}, {2, 2}} {
		tl := []float64{1, 2, -3, 1}
		tr := []float64{4, 5, -6, 4}
		b := []float64{1, 2, 3, 4}
		x := make([]float64, 4)
		allocs := testing.AllocsPerRun(5, func() {
			(Implementation{}).Dlasy2(false, false, 1, tc.n1, tc.n2, tl, 2, tr, 2, b, 2, x, 2)
		})
		if allocs != 0 {
			t.Errorf("n1=%d n2=%d: Dlasy2 allocs = %v, want 0", tc.n1, tc.n2, allocs)
		}
	}
}

func TestDlarfgNativeMatchesDlarfg(t *testing.T) {
	rnd := rand.New(rand.NewPCG(2, 2))
	for _, scale := range []float64{1, 1e-300, 1e300} {
		for n := 0; n <= 4; n++ {
			for range 20 {
				alpha := scale * rnd.NormFloat64()
				x := make([]float64, max(n-1, 0))
				for i := range x {
					x[i] = scale * rnd.NormFloat64()
				}
				xn := append([]float64(nil), x...)
				beta, tau := Implementation{}.Dlarfg(n, alpha, x, 1)
				betaN, tauN := Implementation{}.dlarfgNative(n, alpha, xn, 1)
				if !sameFloat(beta, betaN) || !sameFloat(tau, tauN) {
					t.Fatalf("n=%d scale=%g: (beta,tau) = (%v,%v), want (%v,%v)", n, scale, betaN, tauN, beta, tau)
				}
				for i := range x {
					if !sameFloat(x[i], xn[i]) {
						t.Fatalf("n=%d scale=%g: v[%d] = %v, want %v", n, scale, i, xn[i], x[i])
					}
				}
			}
		}
	}
}

func TestDlarfxSmallMatchesDlarf(t *testing.T) {
	rnd := rand.New(rand.NewPCG(3, 3))
	for _, side := range []blas.Side{blas.Left, blas.Right} {
		for nh := 1; nh <= 10; nh++ {
			m, n := nh, 3
			if side == blas.Right {
				m, n = 4, nh
			}
			ldc := n + 2
			v := make([]float64, nh)
			for i := range v {
				v[i] = rnd.NormFloat64()
			}
			tau := rnd.Float64()
			c := make([]float64, m*ldc)
			for i := range c {
				c[i] = rnd.NormFloat64()
			}
			want := append([]float64(nil), c...)
			Implementation{}.Dlarf(side, m, n, v, 1, tau, want, ldc, make([]float64, max(m, n)))
			dlarfxSmall(side, m, n, v, tau, c, ldc)
			for i := range c {
				if math.Abs(c[i]-want[i]) > 1e-13*(1+math.Abs(want[i])) {
					t.Fatalf("side=%c nh=%d: c[%d] = %v, want %v", side, nh, i, c[i], want[i])
				}
			}
		}
	}
}

func sameFloat(a, b float64) bool {
	return a == b || (math.IsNaN(a) && math.IsNaN(b))
}
