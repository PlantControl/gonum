// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && amd64 && !safe && !noasm && !gccgo

package f64_test

import (
	"fmt"
	"math"
	"syscall"
	"testing"
	"unsafe"

	"plantcontrol.org/v1/gonum/internal/asm/f64"
)

func TestL1NormReadBoundsAMD64(t *testing.T) {
	page := syscall.Getpagesize()
	for _, n := range []int{1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 127, 128, 129, 4095, 4096, 4097} {
		for _, side := range []string{"front", "back"} {
			for _, special := range []bool{false, true} {
				t.Run(fmt.Sprintf("n%d/%s/infinity%t", n, side, special), func(t *testing.T) {
					accessible := (8*n + page - 1) / page * page
					memory, err := syscall.Mmap(-1, 0, accessible+2*page, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_PRIVATE|syscall.MAP_ANON)
					if err != nil {
						t.Fatal(err)
					}
					defer func() {
						if err := syscall.Munmap(memory); err != nil {
							t.Error(err)
						}
					}()
					if err := syscall.Mprotect(memory[:page], syscall.PROT_NONE); err != nil {
						t.Fatal(err)
					}
					if err := syscall.Mprotect(memory[page+accessible:], syscall.PROT_NONE); err != nil {
						t.Fatal(err)
					}
					start := page
					if side == "back" {
						start += accessible - 8*n
					}
					x := unsafe.Slice((*float64)(unsafe.Pointer(&memory[start])), n)
					for i := range x {
						x[i] = -0.25
					}
					want := float64(n) / 4
					if special {
						x[0], x[n-1] = math.Inf(1), math.Inf(1)
						want = math.Inf(1)
					}
					if err := syscall.Mprotect(memory[page:page+accessible], syscall.PROT_READ); err != nil {
						t.Fatal(err)
					}
					if got := f64.L1Norm(x); got != want {
						t.Fatalf("got %g, want %g", got, want)
					}
				})
			}
		}
	}
}
