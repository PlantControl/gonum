// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && arm64 && !safe && !noasm && !gccgo

package gonum

import (
	"unsafe"

	"plantcontrol.org/v1/gonum/internal/asm/f64"
	"plantcontrol.org/v1/gonum/lapack"
)

func dlasrLeftVariableSIMD(direct lapack.Direct, m, n int, c, s, a []float64, lda int) bool {
	activeC := c[:m-1]
	activeS := s[:m-1]
	active := false
	for j, cv := range activeC {
		if cv != 1 || activeS[j] != 0 {
			active = true
			break
		}
	}
	if !active {
		return true
	}
	activeA := a[:(m-1)*lda+n]
	if !dlasrDisjoint(activeA, activeC) || !dlasrDisjoint(activeA, activeS) {
		return false
	}
	for j, cv := range activeC {
		sv := activeS[j]
		if !(cv >= -1 && cv <= 1) || !(sv >= -1 && sv <= 1) {
			return false
		}
	}
	if direct == lapack.Forward {
		for j := 0; j < m-1; j++ {
			ctmp := c[j]
			stmp := s[j]
			if ctmp != 1 || stmp != 0 {
				f64.RotUnitary(a[j*lda:j*lda+n], a[(j+1)*lda:(j+1)*lda+n], ctmp, stmp)
			}
		}
		return true
	}
	for j := m - 2; j >= 0; j-- {
		ctmp := c[j]
		stmp := s[j]
		if ctmp != 1 || stmp != 0 {
			f64.RotUnitary(a[j*lda:j*lda+n], a[(j+1)*lda:(j+1)*lda+n], ctmp, stmp)
		}
	}
	return true
}

func dlasrDisjoint(a, b []float64) bool {
	aStart := uintptr(unsafe.Pointer(unsafe.SliceData(a)))
	bStart := uintptr(unsafe.Pointer(unsafe.SliceData(b)))
	return aStart+uintptr(len(a))*unsafe.Sizeof(float64(0)) <= bStart ||
		bStart+uintptr(len(b))*unsafe.Sizeof(float64(0)) <= aStart
}
