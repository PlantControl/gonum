// Copyright ©2019 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package f32

import (
	"math"

	"plantcontrol.org/v1/gonum/internal/math32"
)

// L2NormUnitary is the level 2 norm of x.
func L2NormUnitary(x []float32) (sum float32) {
	// Float64 holds every finite float32 square exactly, without scaling.
	var s0, s1, s2, s3 float64
	for len(x) >= 4 {
		v0, v1 := float64(x[0]), float64(x[1])
		v2, v3 := float64(x[2]), float64(x[3])
		s0 += v0 * v0
		s1 += v1 * v1
		s2 += v2 * v2
		s3 += v3 * v3
		x = x[4:]
	}
	for _, value := range x {
		v := float64(value)
		s0 += v * v
	}
	return float32(math.Sqrt((s0 + s1) + (s2 + s3)))
}

// L2NormInc is the level 2 norm of x.
func L2NormInc(x []float32, n, incX uintptr) (sum float32) {
	if incX == 0 {
		return 0
	}
	var s0, s1, s2, s3 float64
	i, ix := uintptr(0), uintptr(0)
	for ; i+4 <= n; i += 4 {
		v0, v1 := float64(x[ix]), float64(x[ix+incX])
		v2, v3 := float64(x[ix+2*incX]), float64(x[ix+3*incX])
		s0 += v0 * v0
		s1 += v1 * v1
		s2 += v2 * v2
		s3 += v3 * v3
		ix += 4 * incX
	}
	for ; i < n; i++ {
		v := float64(x[ix])
		s0 += v * v
		ix += incX
	}
	return float32(math.Sqrt((s0 + s1) + (s2 + s3)))
}

// L2DistanceUnitary is the L2 norm of x-y.
func L2DistanceUnitary(x, y []float32) (sum float32) {
	var scale float32
	var sumSquares float32 = 1
	for i, v := range x {
		v -= y[i]
		if v == 0 {
			continue
		}
		absxi := math32.Abs(v)
		if math32.IsNaN(absxi) {
			return math32.NaN()
		}
		if scale < absxi {
			s := scale / absxi
			sumSquares = 1 + sumSquares*s*s
			scale = absxi
		} else {
			s := absxi / scale
			sumSquares += s * s
		}
	}
	if math32.IsInf(scale, 1) {
		return math32.Inf(1)
	}
	return scale * math32.Sqrt(sumSquares)
}
