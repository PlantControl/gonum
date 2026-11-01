// Copyright ©2014 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"testing"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/testblas"
)

func TestDgemm(t *testing.T) {
	testblas.TestDgemm(t, impl)
}

func TestDsymm(t *testing.T) {
	testblas.DsymmTest(t, impl)
}

func TestDtrsm(t *testing.T) {
	testblas.DtrsmTest(t, impl)
}

func TestDsyrk(t *testing.T) {
	testblas.DsyrkTest(t, impl)
}

func TestDsyr2k(t *testing.T) {
	testblas.Dsyr2kTest(t, impl)
}

func TestDtrmm(t *testing.T) {
	testblas.DtrmmTest(t, impl)
}

// The left-side solve divides by the diagonal, as reference DTRSM does, rather
// than scaling by its reciprocal: 64764752532480000*(1/64764752532480000) rounds
// to 0.99999999999999989, which moved the exact row of a matrix exponential.
func TestDtrsmLeftDividesByDiagonal(t *testing.T) {
	const d = 64764752532480000
	for _, ul := range []blas.Uplo{blas.Upper, blas.Lower} {
		for _, tA := range []blas.Transpose{blas.NoTrans, blas.Trans} {
			a, b := []float64{d}, []float64{d, 2 * d}
			impl.Dtrsm(blas.Left, ul, tA, blas.NonUnit, 1, 2, 1, a, 1, b, 2)
			if b[0] != 1 || b[1] != 2 {
				t.Errorf("ul=%v tA=%v: b = %v, want [1 2]", ul, tA, b)
			}
		}
	}
}
