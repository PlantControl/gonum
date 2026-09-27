// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package gonum

import (
	"fmt"
	"testing"

	"plantcontrol.org/v1/gonum/lapack"
)

func TestDhgeqzNetlibZeroDiagonalChase(t *testing.T) {
	const n = 5
	h := []float64{
		2, 3, -1, 4, 2,
		1, 5, 2, -3, 1,
		0, 2, 3, 1, 2,
		0, 0, 1, 4, -2,
		0, 0, 0, 2, 6,
	}
	for _, zero := range []int{0, 1, 2, 3, 4} {
		for _, tiny := range []float64{0, 1e-310} {
			t.Run(fmt.Sprintf("Diagonal=%d/Value=%g", zero, tiny), func(t *testing.T) {
				tt := []float64{
					2, 1, -1, 2, 3,
					0, 3, 2, -1, 1,
					0, 0, 4, 1, -2,
					0, 0, 0, 5, 2,
					0, 0, 0, 0, 6,
				}
				tt[zero*n+zero] = tiny
				compareDhgeqzWithNetlib(t, lapack.EigenvaluesAndSchur, lapack.SchurHess, lapack.SchurHess, n, 0, n-1, h, tt, true)
				compareDhgeqzWithNetlib(t, lapack.EigenvaluesOnly, lapack.SchurNone, lapack.SchurNone, n, 0, n-1, h, tt, false)
			})
		}
	}
}

func TestDhgeqzNetlibSingularSplits(t *testing.T) {
	const n = 5
	for _, tc := range []struct {
		name          string
		sub, diagonal float64
		zeros         []int
	}{
		{"SmallPivot", 0.125, 0.375, []int{2}},
		{"TwoSmallSubdiagonals", 1e-14, 3, []int{2}},
		{"ConsecutiveZeros", 1, 3, []int{1, 2, 3}},
		{"AllZeros", 1, 3, []int{0, 1, 2, 3, 4}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := []float64{
				2, 3, -1, 4, 2,
				1, 5, 2, -3, 1,
				0, tc.sub, tc.diagonal, 1, 2,
				0, 0, 0.001, 4, -2,
				0, 0, 0, 2, 6,
			}
			tt := []float64{
				2, 1, -1, 2, 3,
				0, 3, 2, -1, 1,
				0, 0, 4, 1, -2,
				0, 0, 0, 5, 2,
				0, 0, 0, 0, 6,
			}
			for _, j := range tc.zeros {
				tt[j*n+j] = 0
			}
			compareDhgeqzWithNetlib(t, lapack.EigenvaluesAndSchur, lapack.SchurHess, lapack.SchurHess, n, 0, n-1, h, tt, true)
			compareDhgeqzWithNetlib(t, lapack.EigenvaluesOnly, lapack.SchurNone, lapack.SchurNone, n, 0, n-1, h, tt, false)
		})
	}
}
