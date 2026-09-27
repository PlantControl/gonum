// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && darwin && arm64 && !safe && !noasm && !gccgo

package f64_test

import (
	"fmt"
	"math"
	"syscall"
	"testing"
	"unsafe"

	. "plantcontrol.org/v1/gonum/internal/asm/f64"
)

func TestDotUnitarySIMDGuardPage(t *testing.T) {
	for _, n := range []int{16, 17, 23, 24, 25, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257} {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			x, unmapX := dotGuardedEnd(t, n)
			defer unmapX()
			y, unmapY := dotGuardedEnd(t, n)
			defer unmapY()
			for i := range x {
				x[i] = float64(i%19-9) / 16
				y[i] = float64(i%13-6) / 8
			}
			want := dotUnitaryARM64Reference(x, y)
			got := DotUnitary(x, y)
			if math.Float64bits(got) != math.Float64bits(want) {
				t.Fatalf("got %g (%#x) want %g (%#x)", got, math.Float64bits(got), want, math.Float64bits(want))
			}
		})
	}
}

func dotGuardedEnd(t *testing.T, n int) ([]float64, func()) {
	t.Helper()
	page := syscall.Getpagesize()
	if n*8 > page {
		t.Fatal("guard fixture exceeds one page")
	}
	mapping, err := syscall.Mmap(-1, 0, 2*page, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Fatal(err)
	}
	if err := syscall.Mprotect(mapping[page:], syscall.PROT_NONE); err != nil {
		syscall.Munmap(mapping)
		t.Fatal(err)
	}
	start := page - n*8
	x := unsafe.Slice((*float64)(unsafe.Pointer(&mapping[start])), n)
	return x, func() {
		if err := syscall.Munmap(mapping); err != nil {
			t.Errorf("munmap: %v", err)
		}
	}
}
