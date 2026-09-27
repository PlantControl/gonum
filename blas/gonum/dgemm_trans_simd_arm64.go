// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && arm64 && !safe && !noasm && !gccgo

package gonum

import (
	"math"
	"simd/archsimd"

	"plantcontrol.org/v1/gonum/internal/asm/f64"
)

func dgemmSerialNotTransSIMD(m, n, k int, a []float64, lda int, b []float64, ldb int, c []float64, ldc int, alpha float64) bool {
	if m < 1 || n < 2 || k < 16 {
		return false
	}
	if !gemmSIMDDisjoint(c, m, n, ldc, a, m, k, lda) || !gemmSIMDDisjoint(c, m, n, ldc, b, n, k, ldb) {
		return false
	}

	for row := 0; row < m; row++ {
		atmp := a[row*lda : row*lda+k]
		ctmp := c[row*ldc : row*ldc+n]
		j := 0
		for ; j+1 < n; j += 2 {
			b0 := b[j*ldb : j*ldb+k]
			b1 := b[(j+1)*ldb : (j+1)*ldb+k]
			var sum00, sum01, sum02, sum03 archsimd.Float64x2
			var sum10, sum11, sum12, sum13 archsimd.Float64x2
			i := 0
			for ; i+7 < k; i += 8 {
				ab := atmp[i : i+8 : i+8]
				b0b := b0[i : i+8 : i+8]
				b1b := b1[i : i+8 : i+8]
				a0 := archsimd.LoadFloat64x2(ab[0:2])
				a1 := archsimd.LoadFloat64x2(ab[2:4])
				a2 := archsimd.LoadFloat64x2(ab[4:6])
				a3 := archsimd.LoadFloat64x2(ab[6:8])
				sum00 = a0.MulAdd(archsimd.LoadFloat64x2(b0b[0:2]), sum00)
				sum01 = a1.MulAdd(archsimd.LoadFloat64x2(b0b[2:4]), sum01)
				sum02 = a2.MulAdd(archsimd.LoadFloat64x2(b0b[4:6]), sum02)
				sum03 = a3.MulAdd(archsimd.LoadFloat64x2(b0b[6:8]), sum03)
				sum10 = a0.MulAdd(archsimd.LoadFloat64x2(b1b[0:2]), sum10)
				sum11 = a1.MulAdd(archsimd.LoadFloat64x2(b1b[2:4]), sum11)
				sum12 = a2.MulAdd(archsimd.LoadFloat64x2(b1b[4:6]), sum12)
				sum13 = a3.MulAdd(archsimd.LoadFloat64x2(b1b[6:8]), sum13)
			}
			sum00 = sum00.Add(sum01).Add(sum02.Add(sum03))
			sum10 = sum10.Add(sum11).Add(sum12.Add(sum13))
			for ; i+1 < k; i += 2 {
				av := archsimd.LoadFloat64x2(atmp[i : i+2 : i+2])
				sum00 = av.MulAdd(archsimd.LoadFloat64x2(b0[i:i+2:i+2]), sum00)
				sum10 = av.MulAdd(archsimd.LoadFloat64x2(b1[i:i+2:i+2]), sum10)
			}
			dot0 := sum00.ConcatAddPairs(sum00).GetElem(0)
			dot1 := sum10.ConcatAddPairs(sum10).GetElem(0)
			for ; i < k; i++ {
				dot0 = math.FMA(atmp[i], b0[i], dot0)
				dot1 = math.FMA(atmp[i], b1[i], dot1)
			}
			ctmp[j] += alpha * dot0
			ctmp[j+1] += alpha * dot1
		}
		if j < n {
			ctmp[j] += alpha * f64.DotUnitary(atmp, b[j*ldb:j*ldb+k])
		}
	}
	return true
}
