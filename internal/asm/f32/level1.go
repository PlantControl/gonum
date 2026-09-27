// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !arm64 || !go1.27 || !goexperiment.simd || noasm || gccgo || safe

package f32

import "plantcontrol.org/v1/gonum/internal/math32"

// SasumUnitary returns the sum of the absolute values of x.
func SasumUnitary(x []float32) (sum float32) {
	for _, v := range x {
		sum += math32.Abs(v)
	}
	return sum
}

// SwapUnitary exchanges corresponding elements of x and y.
func SwapUnitary(x, y []float32) {
	for i, v := range x {
		x[i], y[i] = y[i], v
	}
}

// RotUnitary applies a plane rotation to x and y.
func RotUnitary(x, y []float32, c, s float32) {
	for i, vx := range x {
		vy := y[i]
		x[i], y[i] = c*vx+s*vy, c*vy-s*vx
	}
}

// RotmUnitaryRescaling applies a full modified plane rotation to x and y.
func RotmUnitaryRescaling(x, y []float32, h11, h12, h21, h22 float32) {
	for i, vx := range x {
		vy := y[i]
		x[i], y[i] = vx*h11+vy*h12, vx*h21+vy*h22
	}
}

// RotmUnitaryOffDiagonal applies an off-diagonal modified plane rotation.
func RotmUnitaryOffDiagonal(x, y []float32, h12, h21 float32) {
	for i, vx := range x {
		vy := y[i]
		x[i], y[i] = vx+vy*h12, vx*h21+vy
	}
}

// RotmUnitaryDiagonal applies a diagonal modified plane rotation.
func RotmUnitaryDiagonal(x, y []float32, h11, h22 float32) {
	for i, vx := range x {
		vy := y[i]
		x[i], y[i] = vx*h11+vy, -vx+vy*h22
	}
}
