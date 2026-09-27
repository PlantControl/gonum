// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && arm64 && !safe && !noasm && !gccgo

package c64

import (
	"unsafe"

	"plantcontrol.org/v1/gonum/internal/asm/f32"
)

// AsumUnitary returns the sum of the absolute real and imaginary components.
func AsumUnitary(x []complex64) float32 {
	if len(x) == 0 {
		return 0
	}
	components := unsafe.Slice((*float32)(unsafe.Pointer(unsafe.SliceData(x))), 2*len(x))
	return f32.SasumUnitary(components)
}
