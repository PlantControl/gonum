// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && arm64 && !safe && !noasm && !gccgo

package gonum

import (
	"math"
	"unsafe"

	"plantcontrol.org/v1/gonum/internal/asm/f64"
)

func dznrm2Unitary(x []complex128) (norm float64, ok bool) {
	if len(x) < 32 {
		return 0, false
	}
	first := math.Abs(real(x[0]))
	if first < 0x1p-485 || first >= 0x1p512 {
		return 0, false
	}
	components := unsafe.Slice((*float64)(unsafe.Pointer(unsafe.SliceData(x))), 2*len(x))
	norm = f64.L2NormUnitarySIMD(components)
	// The complex BLAS contract gives Inf priority over NaN, unlike the real
	// helper. Retry the original complex recurrence for non-finite results.
	return norm, !math.IsNaN(norm) && !math.IsInf(norm, 0)
}
