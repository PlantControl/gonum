// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package gonum

import (
	"testing"

	"plantcontrol.org/v1/gonum/blas/gonum/internal/netlib"
)

func TestDgemmZeroAlphaNetlib(t *testing.T) {
	testGemmZeroAlpha[float64](t, netlib.Implementation{}.Dgemm)
}
func TestSgemmZeroAlphaNetlib(t *testing.T) {
	testGemmZeroAlpha[float32](t, netlib.Implementation{}.Sgemm)
}
