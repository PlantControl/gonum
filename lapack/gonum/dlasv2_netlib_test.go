// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package gonum

import (
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/lapack/gonum/internal/netlib"
)

func TestDlasv2NetlibDifferential(t *testing.T) {
	for _, tc := range []struct {
		name    string
		f, g, h float64
	}{
		{"Zero", 0, 0, 0},
		{"Diagonal", -3, 0, 2},
		{"Ordinary", -3, 4, 2},
		{"Tiny", 1e-300, -2e-300, 3e-300},
		{"Huge", 1e300, -2e300, 3e300},
		{"InfiniteF", math.Inf(1), 2, 3},
		{"InfiniteG", 2, math.Inf(-1), 3},
		{"InfiniteH", 2, 3, math.Inf(1)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g0, g1, g2, g3, g4, g5 := Implementation{}.Dlasv2(tc.f, tc.g, tc.h)
			n0, n1, n2, n3, n4, n5 := netlib.Dlasv2(tc.f, tc.g, tc.h)
			for i, pair := range [][2]float64{{g0, n0}, {g1, n1}, {g2, n2}, {g3, n3}, {g4, n4}, {g5, n5}} {
				comparePrimitiveFloat(t, string(rune('0'+i)), pair[0], pair[1], 5e-14)
			}
		})
	}
}
