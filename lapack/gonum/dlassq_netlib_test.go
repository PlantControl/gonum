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

func TestDlassqNetlibDifferential(t *testing.T) {
	for _, tc := range []struct {
		name         string
		x            []float64
		n, inc       int
		scale, sumsq float64
	}{
		{"Zeros", []float64{0, 0, 0}, 3, 1, 0, 1},
		{"Ordinary", []float64{3, -4, 12}, 3, 1, 0, 1},
		{"Stride", []float64{3, 99, -4, 99, 12}, 3, 2, 0, 1},
		{"MixedScale", []float64{1e-300, 2, -1e300}, 3, 1, 0, 1},
		{"Initial", []float64{3, -4}, 2, 1, 5, 2},
		{"Infinity", []float64{2, math.Inf(1), 3}, 3, 1, 0, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			gs, gq := Implementation{}.Dlassq(tc.n, tc.x, tc.inc, tc.scale, tc.sumsq)
			ns, nq := netlib.Dlassq(tc.n, tc.x, tc.inc, tc.scale, tc.sumsq)
			gn, nn := gs*math.Sqrt(gq), ns*math.Sqrt(nq)
			comparePrimitiveFloat(t, "norm", gn, nn, 5e-15)
		})
	}
}
