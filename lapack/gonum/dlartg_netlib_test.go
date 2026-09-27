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

func TestDlartgNetlibDifferential(t *testing.T) {
	for _, tc := range []struct {
		name string
		f, g float64
	}{
		{"BothZero", 0, 0},
		{"GZero", -3, 0},
		{"FZero", 0, -4},
		{"Ordinary", -3, 4},
		{"Tiny", 1e-300, -2e-300},
		{"Huge", 1e300, 2e300},
		{"InfiniteF", math.Inf(1), 2},
		{"InfiniteG", 2, math.Inf(-1)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			gcs, gsn, gr := Implementation{}.Dlartg(tc.f, tc.g)
			ncs, nsn, nr := netlib.Dlartg(tc.f, tc.g)
			comparePrimitiveFloat(t, "cs", gcs, ncs, 2e-15)
			comparePrimitiveFloat(t, "sn", gsn, nsn, 2e-15)
			comparePrimitiveFloat(t, "r", gr, nr, 2e-15)
		})
	}
}

func comparePrimitiveFloat(t *testing.T, name string, got, want, tol float64) {
	t.Helper()
	if math.IsNaN(want) {
		if !math.IsNaN(got) {
			t.Fatalf("%s classification mismatch: got %g, Netlib NaN", name, got)
		}
		return
	}
	if math.IsInf(want, 0) {
		if !math.IsInf(got, int(math.Copysign(1, want))) {
			t.Fatalf("%s classification mismatch: got %g, Netlib %g", name, got, want)
		}
		return
	}
	if math.IsNaN(got) || math.IsInf(got, 0) {
		t.Fatalf("%s unexpectedly nonfinite: got %g, Netlib %g", name, got, want)
	}
	scale := math.Max(1, math.Max(math.Abs(got), math.Abs(want)))
	if math.Abs(got-want) > tol*scale {
		t.Fatalf("%s mismatch: got %g, Netlib %g", name, got, want)
	}
}
