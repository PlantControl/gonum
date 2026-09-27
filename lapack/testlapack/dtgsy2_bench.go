// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlapack

import (
	"testing"

	"plantcontrol.org/v1/gonum/blas"
)

func Dtgsy2ScratchBenchmark(b *testing.B, impl Dtgsy2er) {
	a := []float64{2, 1, 0, 3}
	bm := []float64{4, 0.25, 0, 5}
	d := []float64{1, 0.5, 0, 2}
	e := []float64{1, 0.2, 0, 1.5}
	cOrig := []float64{1, -2, 3, -4}
	fOrig := []float64{-1, 2, -3, 4}
	c := make([]float64, 4)
	f := make([]float64, 4)
	iwork := make([]int, 6)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		copy(c, cOrig)
		copy(f, fOrig)
		_, _, _, _, ok := impl.Dtgsy2(blas.NoTrans, 0, 2, 2,
			a, 2, bm, 2, c, 2, d, 2, e, 2, f, 2, 0, 1, iwork)
		if !ok {
			b.Fatal("Dtgsy2 solve failed")
		}
	}
}
