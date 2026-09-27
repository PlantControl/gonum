// Copyright ©2014 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/blas/testblas"
)

var impl Implementation

func TestDasum(t *testing.T) {
	testblas.DasumTest(t, impl)
}

func TestDaxpy(t *testing.T) {
	testblas.DaxpyTest(t, impl)
}

func TestDdot(t *testing.T) {
	testblas.DdotTest(t, impl)
}

func TestDnrm2(t *testing.T) {
	testblas.Dnrm2Test(t, impl)
}

func TestIdamax(t *testing.T) {
	testblas.IdamaxTest(t, impl)
}

func TestIdamaxUnitaryOrder(t *testing.T) {
	for _, test := range []struct {
		name string
		x    []float64
		want int
	}{
		{name: "FirstNaN", x: []float64{math.NaN(), 2, 3, 4, 5, 6}, want: 0},
		{name: "LaterNaN", x: []float64{1, math.NaN(), 3, 4, 5, 6}, want: 5},
		{name: "FirstTie", x: []float64{1, -7, 3, 7, 5, 6}, want: 1},
		{name: "ChunkBoundary", x: []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}, want: 8},
	} {
		t.Run(test.name, func(t *testing.T) {
			if got := impl.Idamax(len(test.x), test.x, 1); got != test.want {
				t.Fatalf("unexpected index: got %d want %d", got, test.want)
			}
		})
	}
}

func TestDswap(t *testing.T) {
	testblas.DswapTest(t, impl)
}

func TestDcopy(t *testing.T) {
	testblas.DcopyTest(t, impl)
}

func TestDrotg(t *testing.T) {
	testblas.DrotgTest(t, impl, false)
}

func TestDrotmg(t *testing.T) {
	testblas.DrotmgTest(t, impl)
}

func TestDrot(t *testing.T) {
	testblas.DrotTest(t, impl)
}

func TestDrotm(t *testing.T) {
	testblas.DrotmTest(t, impl)
}

func TestDscal(t *testing.T) {
	testblas.DscalTest(t, impl)
}
