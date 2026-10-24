// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package gonum

import (
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/lapack/gonum/internal/netlib"
)

// TestZlartgNetlib compares Zlartg with Netlib ZLARTG across all of its
// scaling branches.
func TestZlartgNetlib(t *testing.T) {
	mags := []float64{0, 1e-310, 1e-300, 1e-200, 1e-160, 1e-30, 0.5, 1, 3, 1e30, 1e150, 1e160, 1e200, 1e300, math.MaxFloat64 / 4}
	var vals []complex128
	for _, re := range mags {
		for _, im := range []float64{0, 1, 1e-200, 1e200} {
			vals = append(vals, complex(re, im*re), complex(-re, im), complex(im, -re))
		}
	}
	for _, f := range vals {
		for _, g := range vals {
			gc, gs, gr := Implementation{}.Zlartg(f, g)
			nc, ns, nr := netlib.Zlartg(f, g)
			const tol = 1e-14
			if math.Abs(gc-nc) > tol*math.Max(nc, 1e-300) && math.Abs(gc-nc) > 1e-300 {
				t.Errorf("f=%v g=%v: c=%v, want Netlib %v", f, g, gc, nc)
			}
			if cmplx.Abs(gs-ns) > tol*math.Max(cmplx.Abs(ns), 1e-300) {
				t.Errorf("f=%v g=%v: s=%v, want Netlib %v", f, g, gs, ns)
			}
			if cmplx.Abs(gr-nr) > tol*cmplx.Abs(nr) {
				t.Errorf("f=%v g=%v: r=%v, want Netlib %v", f, g, gr, nr)
			}
		}
	}
}
