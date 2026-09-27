// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package simdbench

import (
	"fmt"
	"simd/archsimd"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/internal/asm/c64"
)

type dotEntryFixture struct {
	name  string
	run   func()
	check func(testing.TB)
}

func dotEntryFixtures() []dotEntryFixture {
	var fs []dotEntryFixture
	for _, conjugate := range []bool{false, true} {
		name := "DotuInc"
		if conjugate {
			name = "DotcInc"
		}
		for _, n := range []int{0, 1, 2, 3, 4, 7, 31, 32, 33, 63, 64, 65, 4096} {
			for _, stride := range []int{1, 2, 3, 7} {
				size := 0
				if n != 0 {
					size = (n-1)*stride + 1
				}
				x, y := make([]complex64, size), make([]complex64, size)
				for i := range x {
					x[i] = complex(float32(i%11-5)/16, float32(i%7-3)/8)
					y[i] = complex(float32(i%13-6)/8, float32(i%5-2)/16)
				}
				beforeX, beforeY := slices.Clone(x), slices.Clone(y)
				var want complex128
				for i := 0; i < n; i++ {
					xv := complex128(x[i*stride])
					if conjugate {
						xv = complex(real(xv), -imag(xv))
					}
					want += xv * complex128(y[i*stride])
				}
				for _, impl := range []string{"simd", "current"} {
					fn := c64.DotuIncSIMD
					if conjugate {
						fn = c64.DotcIncSIMD
					}
					if impl == "current" {
						fn = c64.DotuInc
						if conjugate {
							fn = c64.DotcInc
						}
					}
					var got complex64
					fs = append(fs, dotEntryFixture{
						name: fmt.Sprintf("c64/%s/n=%d/stride=%d/implementation=%s", name, n, stride, impl),
						run: func() {
							if archsimd.X86.AVX() {
								archsimd.ClearAVXUpperBits()
							}
							got = fn(x, y, uintptr(n), uintptr(stride), uintptr(stride), 0, 0)
						},
						check: func(t testing.TB) {
							if complex128(got) != want {
								t.Fatalf("got=%v want=%v", got, want)
							}
							if !slices.Equal(x, beforeX) || !slices.Equal(y, beforeY) {
								t.Fatal("input changed")
							}
						},
					})
				}
			}
		}
	}
	return fs
}
func TestSIMDDotEntryFixtures(t *testing.T) {
	for _, f := range dotEntryFixtures() {
		t.Run(f.name, func(t *testing.T) { f.run(); f.check(t) })
	}
}
func TestSIMDDotEntryAllocations(t *testing.T) {
	for _, f := range dotEntryFixtures() {
		t.Run(f.name, func(t *testing.T) {
			if got := testing.AllocsPerRun(100, f.run); got != 0 {
				t.Fatalf("allocs=%v", got)
			}
			f.check(t)
		})
	}
}
func BenchmarkSIMDDotEntry(b *testing.B) {
	for _, f := range dotEntryFixtures() {
		b.Run(f.name, func(b *testing.B) {
			f.run()
			f.check(b)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				f.run()
			}
			b.StopTimer()
			f.check(b)
		})
	}
}
