// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package blas32_test

import (
	"testing"

	"plantcontrol.org/v1/gonum/blas/blas32"
)

func TestSDDotWidenedBias(t *testing.T) {
	x := blas32.Vector{N: 2, Inc: 1, Data: []float32{-0x1p24, 0.5}}
	y := blas32.Vector{N: 2, Inc: 1, Data: []float32{1, 1}}
	if got := blas32.SDDot(0x1p24, x, y); got != 0.5 {
		t.Errorf("widened bias: got %g, want 0.5", got)
	}
	empty := blas32.Vector{Inc: 1}
	if got := blas32.SDDot(3, empty, empty); got != 3 {
		t.Errorf("empty vectors: got %g, want 3", got)
	}
}
