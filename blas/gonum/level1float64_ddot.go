// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"plantcontrol.org/v1/gonum/internal/asm/f64"
)

// Ddot computes the dot product of the two vectors
//
//	\sum_i x[i]*y[i]
func (Implementation) Ddot(n int, x []float64, incX int, y []float64, incY int) float64 {
	if incX == 0 {
		panic(zeroIncX)
	}
	if incY == 0 {
		panic(zeroIncY)
	}
	if n <= 0 {
		if n == 0 {
			return 0
		}
		panic(nLT0)
	}
	if incX == 1 && incY == 1 {
		if len(x) < n {
			panic(shortX)
		}
		if len(y) < n {
			panic(shortY)
		}
		return f64.DotUnitary(x[:n], y[:n])
	}
	ix, ok := checkedDotStart(n, incX, len(x))
	if !ok {
		panic(shortX)
	}
	iy, ok := checkedDotStart(n, incY, len(y))
	if !ok {
		panic(shortY)
	}
	return f64.DotInc(x, y, uintptr(n), uintptr(incX), uintptr(incY), uintptr(ix), uintptr(iy))
}
