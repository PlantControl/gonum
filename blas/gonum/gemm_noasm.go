// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !go1.27 || !goexperiment.simd || safe || noasm || gccgo

package gonum

import "plantcontrol.org/v1/gonum/blas"

const useGEMMSIMD = false

func dsyrkBlocked(ul blas.Uplo, n, k int, alpha float64, a []float64, lda int, beta float64, c []float64, ldc int) bool {
	return false
}
func ssyrkBlocked(ul blas.Uplo, n, k int, alpha float32, a []float32, lda int, beta float32, c []float32, ldc int) bool {
	return false
}

func gemmSIMDHardware() bool {
	return false
}

func dgemmSerialSIMD(aTrans, bTrans bool, m, n, k int, a []float64, lda int, b []float64, ldb int, c []float64, ldc int, alpha float64) bool {
	return false
}

func sgemmSerialSIMD(aTrans, bTrans bool, m, n, k int, a []float32, lda int, b []float32, ldb int, c []float32, ldc int, alpha float32) bool {
	return false
}
