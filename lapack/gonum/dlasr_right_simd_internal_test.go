// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && arm64 && !safe && !noasm && !gccgo && !race

package gonum

import (
	"math"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/lapack"
)

func TestDlasrRightVariableCarry4(t *testing.T) {
	for _, direct := range []lapack.Direct{lapack.Forward, lapack.Backward} {
		for _, m := range []int{64, 65, 66, 67, 79, 80, 81, 256} {
			for _, n := range []int{64, 65, 129} {
				for _, extra := range []int{0, 7} {
					lda := n + extra
					const offset = 3
					a := make([]float64, offset+(m-1)*lda+n+5)
					for i := range a {
						a[i] = math.Float64frombits(0x3ff0000000000000 + uint64(i%1024))
					}
					matrix := a[offset:]
					c, s := dlasrRightRotations(n)
					want := slices.Clone(a)
					dlasrRightSequential(direct, m, n, c, s, want[offset:], lda)

					if !dlasrRightVariableCarry4(direct, m, n, c, s, matrix, lda) {
						t.Fatalf("candidate rejected direct=%v m=%d n=%d lda=%d", direct, m, n, lda)
					}
					// The carry changes compiler FMA opportunities, so allow error proportional to the recurrence length.
					tol := 8 * float64(n) * (math.Nextafter(1, 2) - 1)
					for i, got := range a {
						row, col := -1, -1
						if i >= offset {
							row = (i - offset) / lda
							col = (i - offset) % lda
						}
						if row >= 0 && row < m && col < n {
							if math.IsNaN(got) || math.IsInf(got, 0) || math.IsNaN(want[i]) || math.IsInf(want[i], 0) {
								t.Fatalf("non-finite direct=%v m=%d n=%d lda=%d at (%d,%d): got %g want %g", direct, m, n, lda, row, col, got, want[i])
							}
							scale := math.Max(1, math.Abs(want[i]))
							if math.Abs(got-want[i]) > tol*scale {
								t.Fatalf("mismatch direct=%v m=%d n=%d lda=%d at (%d,%d): got %g want %g", direct, m, n, lda, row, col, got, want[i])
							}
							continue
						}
						if math.Float64bits(got) != math.Float64bits(want[i]) {
							t.Fatalf("outside matrix modified direct=%v m=%d n=%d lda=%d index=%d", direct, m, n, lda, i)
						}
					}
				}
			}
		}
	}
}

func TestDlasrRightVariableCarry4Reject(t *testing.T) {
	const (
		m   = 64
		n   = 64
		lda = 71
	)
	c, s := dlasrRightRotations(n)
	tests := []struct {
		name string
		m    int
		n    int
		edit func(c, s []float64)
	}{
		{name: "short_m", m: 63, n: n},
		{name: "short_n", m: m, n: 63},
		{name: "identity", m: m, n: n, edit: func(c, s []float64) { c[17], s[17] = 1, 0 }},
		{name: "sparse", m: m, n: n, edit: func(c, s []float64) {
			for i := 0; i < len(c); i += 8 {
				c[i], s[i] = 1, 0
			}
		}},
		{name: "c_nan", m: m, n: n, edit: func(c, _ []float64) { c[3] = math.NaN() }},
		{name: "s_nan", m: m, n: n, edit: func(_, s []float64) { s[3] = math.NaN() }},
		{name: "c_inf", m: m, n: n, edit: func(c, _ []float64) { c[3] = math.Inf(1) }},
		{name: "s_inf", m: m, n: n, edit: func(_, s []float64) { s[3] = math.Inf(-1) }},
		{name: "c_out_of_range", m: m, n: n, edit: func(c, _ []float64) { c[3] = 1.01 }},
		{name: "s_out_of_range", m: m, n: n, edit: func(_, s []float64) { s[3] = -1.01 }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cc, ss := slices.Clone(c), slices.Clone(s)
			if test.edit != nil {
				test.edit(cc, ss)
			}
			a := make([]float64, (m-1)*lda+n)
			for i := range a {
				a[i] = float64(i%31-15) / 17
			}
			before := slices.Clone(a)
			if dlasrRightVariableCarry4(lapack.Forward, test.m, test.n, cc, ss, a, lda) {
				t.Fatal("candidate accepted rejected input")
			}
			if !slices.Equal(a, before) {
				t.Fatal("candidate mutated matrix before rejection")
			}
		})
	}
}

func TestDlasrRightVariableCarry4RejectAlias(t *testing.T) {
	const (
		m   = 64
		n   = 64
		lda = 80
	)
	for _, test := range []struct {
		name string
		cOff int
		sOff int
	}{
		{name: "c_active", cOff: 1, sOff: -1},
		{name: "s_active", cOff: -1, sOff: lda + 2},
		{name: "c_padding_gap", cOff: n, sOff: -1},
	} {
		t.Run(test.name, func(t *testing.T) {
			shared := make([]float64, (m-1)*lda+n+2*n)
			for i := range shared {
				shared[i] = float64(i%29-14) / 16
			}
			c, s := dlasrRightRotations(n)
			if test.cOff >= 0 {
				c = shared[test.cOff : test.cOff+n-1]
			}
			if test.sOff >= 0 {
				s = shared[test.sOff : test.sOff+n-1]
			}
			before := slices.Clone(shared)
			if dlasrRightVariableCarry4(lapack.Forward, m, n, c, s, shared, lda) {
				t.Fatal("candidate accepted overlapping rotation storage")
			}
			if !slices.Equal(shared, before) {
				t.Fatal("candidate mutated storage before overlap rejection")
			}
		})
	}
}

func dlasrRightRotations(n int) (c, s []float64) {
	c = make([]float64, n-1)
	s = make([]float64, n-1)
	for i := range c {
		angle := 0.013 * float64(i+1)
		c[i], s[i] = math.Cos(angle), math.Sin(angle)
	}
	return c, s
}

func dlasrRightSequential(direct lapack.Direct, m, n int, c, s, a []float64, lda int) {
	if direct == lapack.Forward {
		for j := 0; j < n-1; j++ {
			for i, k := 0, j; i < m; i, k = i+1, k+lda {
				right, left := a[k+1], a[k]
				a[k+1] = c[j]*right - s[j]*left
				a[k] = s[j]*right + c[j]*left
			}
		}
		return
	}
	for j := n - 2; j >= 0; j-- {
		for i, k := 0, j; i < m; i, k = i+1, k+lda {
			right, left := a[k+1], a[k]
			a[k+1] = c[j]*right - s[j]*left
			a[k] = s[j]*right + c[j]*left
		}
	}
}
