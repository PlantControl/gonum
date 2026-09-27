// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlapack

import (
	"math"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/floats/scalar"
)

type Dlapy2er interface {
	Dlapy2(float64, float64) float64
}

func Dlapy2Test(t *testing.T, impl Dlapy2er) {
	rnd := rand.New(rand.NewPCG(1, 1))
	for i := 0; i < 10; i++ {
		x := math.Abs(1e200 * rnd.NormFloat64())
		y := math.Abs(1e200 * rnd.NormFloat64())
		got := impl.Dlapy2(x, y)
		want := math.Hypot(x, y)
		if !scalar.EqualWithinRel(got, want, 1e-15) {
			t.Errorf("Dlapy2(%g, %g) = %g, want %g", x, y, got, want)
		}
	}
	for _, tc := range []struct {
		x, y float64
	}{
		{math.NaN(), math.Inf(1)},
		{math.Inf(1), math.NaN()},
		{math.NaN(), math.Inf(-1)},
	} {
		if got := impl.Dlapy2(tc.x, tc.y); !math.IsNaN(got) {
			t.Errorf("Dlapy2(%g, %g) = %g, want NaN", tc.x, tc.y, got)
		}
	}
}
