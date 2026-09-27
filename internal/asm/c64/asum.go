// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !arm64 || !go1.27 || !goexperiment.simd || noasm || gccgo || safe

package c64

import "plantcontrol.org/v1/gonum/internal/math32"

// AsumUnitary returns the sum of the absolute real and imaginary components.
func AsumUnitary(x []complex64) float32 {
	var sum float32
	for _, v := range x {
		sum += math32.Abs(real(v)) + math32.Abs(imag(v))
	}
	return sum
}
