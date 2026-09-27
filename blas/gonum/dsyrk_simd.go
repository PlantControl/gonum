// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package gonum

import (
	"simd"

	"plantcontrol.org/v1/gonum/blas"
)

func dsyrkBlocked(ul blas.Uplo, n, k int, alpha float64, a []float64, lda int, beta float64, c []float64, ldc int) bool {
	if !gemmSIMDDisjoint(c, n, n, ldc, a, k, n, lda) {
		return false
	}
	width := simd.BroadcastFloat64s(0).Len()
	for start := 0; start < max(1, k); start += 64 {
		end := min(k, start+64)
		for i := 0; i < n; i++ {
			lo, hi := 0, i+1
			if ul == blas.Upper {
				lo, hi = i, n
			}
			out := c[i*ldc+lo : i*ldc+hi]
			if start == 0 {
				if beta == 0 {
					clear(out)
				} else if beta != 1 {
					for j := range out {
						out[j] *= beta
					}
				}
			}
			j := 0
			for ; j+4*width <= len(out); j += 4 * width {
				v0 := simd.LoadFloat64s(out[j : j+width])
				v1 := simd.LoadFloat64s(out[j+width : j+2*width])
				v2 := simd.LoadFloat64s(out[j+2*width : j+3*width])
				v3 := simd.LoadFloat64s(out[j+3*width : j+4*width])
				for l := start; l < end; l++ {
					if scale := alpha * a[l*lda+i]; scale != 0 {
						av := simd.BroadcastFloat64s(scale)
						x := a[l*lda+lo+j : l*lda+lo+j+4*width]
						v0 = av.MulAdd(simd.LoadFloat64s(x[:width]), v0)
						v1 = av.MulAdd(simd.LoadFloat64s(x[width:2*width]), v1)
						v2 = av.MulAdd(simd.LoadFloat64s(x[2*width:3*width]), v2)
						v3 = av.MulAdd(simd.LoadFloat64s(x[3*width:]), v3)
					}
				}
				v0.Store(out[j : j+width])
				v1.Store(out[j+width : j+2*width])
				v2.Store(out[j+2*width : j+3*width])
				v3.Store(out[j+3*width : j+4*width])
			}
			for ; j+width <= len(out); j += width {
				v := simd.LoadFloat64s(out[j : j+width])
				for l := start; l < end; l++ {
					if scale := alpha * a[l*lda+i]; scale != 0 {
						v = simd.BroadcastFloat64s(scale).MulAdd(simd.LoadFloat64s(a[l*lda+lo+j:l*lda+lo+j+width]), v)
					}
				}
				v.Store(out[j : j+width])
			}
			for ; j < len(out); j++ {
				v := out[j]
				for l := start; l < end; l++ {
					if scale := alpha * a[l*lda+i]; scale != 0 {
						v += scale * a[l*lda+lo+j]
					}
				}
				out[j] = v
			}
		}
	}
	return true
}
