// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package gonum

import (
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/lapack/gonum/internal/netlib"
)

func TestDlangeNetlibExceptional(t *testing.T) {
	const n = 2
	for _, norm := range []lapack.MatrixNorm{
		lapack.MaxAbs,
		lapack.MaxColumnSum,
		lapack.MaxRowSum,
		lapack.Frobenius,
	} {
		for _, tc := range []struct {
			name string
			a    []float64
		}{
			{"NaNThenInf", []float64{math.NaN(), math.Inf(1), 2, 3}},
			{"InfThenNaN", []float64{math.Inf(1), math.NaN(), 2, 3}},
		} {
			t.Run(string(norm)+"/"+tc.name, func(t *testing.T) {
				work := make([]float64, n)
				got := Implementation{}.Dlange(norm, n, n, tc.a, n, work)
				colMajor := []float64{tc.a[0], tc.a[2], tc.a[1], tc.a[3]}
				want := netlib.Dlange(byte(norm), n, n, colMajor, n, make([]float64, n))
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
}
