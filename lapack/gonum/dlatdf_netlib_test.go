// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package gonum

import (
	"fmt"
	"testing"

	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/lapack/gonum/internal/netlib"
)

func TestDlatdfNetlibEightByEight(t *testing.T) {
	z := []float64{
		1, 2, 0, 0, -3, 0, 0.5, 0,
		-1, 1, 0, 0, 0, -3, 0, 0.5,
		0, 0, 1, 2, -1, 0, -3, 0,
		0, 0, -1, 1, 0, -1, 0, -3,
		2, 0.2, 0, 0, -1.5, 0, 0.1, 0,
		0, 3, 0, 0, 0, -1.5, 0, 0.1,
		0, 0, 2, 0.2, 0, 0, -2.5, 0,
		0, 0, 0, 3, 0, 0, 0, -2.5,
	}
	gz := append([]float64(nil), z...)
	nz := append([]float64(nil), z...)
	grhs := make([]float64, 8)
	nrhs := make([]float64, 8)
	ipiv := make([]int, 8)
	jpiv := make([]int, 8)
	Implementation{}.Dgetc2(8, gz, 8, ipiv, jpiv)
	gscale, gsum := Implementation{}.Dlatdf(lapack.LocalLookAhead, 8, gz, 8,
		grhs, 1, 0, ipiv, jpiv)
	nsum, nscale, _ := netlib.Dlatdf(lapack.LocalLookAhead, 8, nz, nrhs, 1, 0)
	checkCloseNetlib(t, "scale", gscale, nscale)
	checkCloseNetlib(t, "sum", gsum, nsum)
	for i := range grhs {
		checkCloseNetlib(t, fmt.Sprintf("rhs[%d]", i), grhs[i], nrhs[i])
	}
}
