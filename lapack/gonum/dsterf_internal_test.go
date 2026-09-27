// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"fmt"
	"math"
	"testing"
)

func TestDsterfScaling(t *testing.T) {
	for _, scale := range []float64{1, 1e-200, 1e200} {
		t.Run(fmt.Sprintf("scale=%g", scale), func(t *testing.T) {
			defer func() {
				if p := recover(); p != nil {
					t.Errorf("Dsterf panicked: %v", p)
				}
			}()
			d := []float64{2 * scale, 2 * scale, 2 * scale}
			e := []float64{scale, scale}
			if !(Implementation{}).Dsterf(3, d, e) {
				t.Fatal("no convergence")
			}
			for i, want := range []float64{2 - math.Sqrt(2), 2, 2 + math.Sqrt(2)} {
				if !(math.Abs(d[i]/scale-want) <= 1e-13) {
					t.Errorf("eigenvalue %d: got %g want %g times scale", i, d[i], want)
				}
			}
		})
	}
}
