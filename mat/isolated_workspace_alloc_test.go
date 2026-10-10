// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !race && !safe

package mat

import "testing"

// Race mode drops sync.Pool items at random, and the safe build's reflect-based
// overlap checks allocate, so allocation counts are only meaningful without
// either.
func TestIsolatedWorkspaceNoAllocs(t *testing.T) {
	eye := NewDense(3, 3, []float64{1, 0, 0, 0, 1, 0, 0, 0, 1})
	a := NewDense(3, 3, []float64{1, 2, 3, 4, 5, 6, 7, 8, 10})
	aT := a.T()
	v := NewVecDense(3, []float64{1, 2, 3})
	s := NewSymDense(3, []float64{4, 1, 2, 1, 5, 3, 2, 3, 6})
	set := []int{2, 0, 1}
	tri := NewTriDense(3, Upper, []float64{1, 2, 3, 0, 4, 5, 0, 0, 6})
	triEye := NewTriDense(3, Upper, []float64{1, 0, 0, 0, 1, 0, 0, 0, 1})
	c := NewCDense(2, 2, []complex128{1 + 2i, 3 - 1i, -2i, 4})
	for _, tc := range []struct {
		name string
		op   func()
	}{
		{"Dense.Mul", func() { a.Mul(a, eye) }},
		{"Dense.Add", func() { a.Add(aT, eye) }},
		{"Dense.Scale", func() { a.Scale(1, aT) }},
		{"VecDense.MulVec", func() { v.MulVec(eye, v) }},
		{"SymDense.SubsetSym", func() { s.SubsetSym(s, set) }},
		{"TriDense.MulTri", func() { tri.MulTri(tri, triEye) }},
		{"CDense.Conj", func() { c.Conj(c) }},
	} {
		tc.op()
		if allocs := testing.AllocsPerRun(20, tc.op); allocs != 0 {
			t.Errorf("%s: allocs = %v, want 0", tc.name, allocs)
		}
	}
}

func TestEigenFactorizeAllocs(t *testing.T) {
	a := NewDense(4, 4, []float64{
		1, 2, 0, 3,
		-2, 1, 4, 0,
		0, 1, 3, -1,
		2, 0, 1, 5,
	})
	var e Eigen
	for _, tc := range []struct {
		kind EigenKind
		want float64
	}{
		// values slice only.
		{EigenNone, 1},
		// values plus the retained complex right vectors.
		{EigenRight, 3},
	} {
		op := func() {
			if !e.Factorize(a, tc.kind) {
				t.Fatal("Factorize failed")
			}
		}
		op()
		if allocs := testing.AllocsPerRun(20, op); allocs > tc.want {
			t.Errorf("kind %d: allocs = %v, want <= %v", tc.kind, allocs, tc.want)
		}
	}
}
