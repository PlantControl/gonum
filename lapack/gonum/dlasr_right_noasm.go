// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !arm64 || !go1.27 || !goexperiment.simd || safe || noasm || gccgo || race

package gonum

import "plantcontrol.org/v1/gonum/lapack"

// Race instrumentation changes the floating-point fusion of the scalar fallback.
func dlasrRightVariableCarry4(direct lapack.Direct, m, n int, c, s, a []float64, lda int) bool {
	return false
}

func dlasrRightVariableSequential(direct lapack.Direct, m, n int, c, s, a []float64, lda int) bool {
	return false
}
