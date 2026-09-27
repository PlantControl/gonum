// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package f64_test

import (
	"fmt"
	"testing"

	. "plantcontrol.org/v1/gonum/internal/asm/f64"
)

var dotUnitaryLengthsSink float64

func BenchmarkDotUnitaryLengths(b *testing.B) {
	lengths := []int{0, 1, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 128, 256, 480, 1024, 4096}
	for _, n := range lengths {
		b.Run(fmt.Sprintf("n=%d/offset=0", n), func(b *testing.B) {
			benchmarkDotUnitaryLength(b, n, 0)
		})
	}
	for _, n := range []int{7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 256, 480, 4096} {
		b.Run(fmt.Sprintf("n=%d/offset=1", n), func(b *testing.B) {
			benchmarkDotUnitaryLength(b, n, 1)
		})
	}
}

func benchmarkDotUnitaryLength(b *testing.B, n, offset int) {
	x := make([]float64, offset+n+1)
	y := make([]float64, offset+n+1)
	for i := range x {
		x[i] = 1 + float64(i%17)/32
		y[i] = 1 + float64(i%13)/16
	}
	x = x[offset : offset+n]
	y = y[offset : offset+n]
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dotUnitaryLengthsSink = DotUnitary(x, y)
	}
}
