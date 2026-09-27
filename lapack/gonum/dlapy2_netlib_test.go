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

func TestDlapy2NetlibExceptional(t *testing.T) {
	for _, tc := range []struct {
		name string
		x, y float64
	}{
		{"NaNInf", math.NaN(), math.Inf(1)},
		{"InfNaN", math.Inf(1), math.NaN()},
		{"NaNNegInf", math.NaN(), math.Inf(-1)},
		{"FiniteInf", 1, math.Inf(1)},
		{"NaNFinite", math.NaN(), 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := Implementation{}.Dlapy2(tc.x, tc.y)
			want := netlib.Dlapy2(tc.x, tc.y)
			if math.IsNaN(want) {
				if !math.IsNaN(got) {
					t.Fatalf("classification mismatch: got %v, Netlib NaN", got)
				}
				return
			}
			if got != want {
				t.Fatalf("got %v, Netlib %v", got, want)
			}
		})
	}
}
