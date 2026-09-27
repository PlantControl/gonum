// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"fmt"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/lapack"
)

func TestDgeesScaling(t *testing.T) {
	for _, scale := range []float64{1, 1e-200, 1e200} {
		for _, complexPair := range []bool{false, true} {
			for _, job := range []lapack.SchurComp{lapack.SchurNone, lapack.SchurHess} {
				t.Run(fmt.Sprintf("scale=%g/complex=%v/job=%c", scale, complexPair, job), func(t *testing.T) {
					defer func() {
						if p := recover(); p != nil {
							t.Errorf("Dgees panicked: %v", p)
						}
					}()
					base := []float64{1, 2, 0, 3}
					if complexPair {
						base = []float64{1, 2, -2, 1}
					}
					a := make([]float64, 4)
					for i, v := range base {
						a[i] = v * scale
					}
					wr, wi, z := make([]float64, 2), make([]float64, 2), make([]float64, 4)
					work := make([]float64, 6)
					_, ok := (Implementation{}).Dgees(job, lapack.SortNone, nil, 2, a, 2, wr, wi, z, 2, work, len(work), nil)
					if !ok {
						t.Fatal("no convergence")
					}
					wantR, wantI := []float64{1, 3}, []float64{0, 0}
					if complexPair {
						wantR = []float64{1, 1}
						wantI = []float64{2, -2}
					}
					for i := 0; i < 2; i++ {
						if !(math.Abs(wr[i]/scale-wantR[i]) <= 1e-13) || !(math.Abs(wi[i]/scale-wantI[i]) <= 1e-13) {
							t.Fatalf("wrong eigenvalues: %v %v", wr, wi)
						}
					}
					if job == lapack.SchurHess {
						for i := 0; i < 2; i++ {
							for j := 0; j < 2; j++ {
								var got float64
								for k := 0; k < 2; k++ {
									for l := 0; l < 2; l++ {
										got += z[i*2+k] * (a[k*2+l] / scale) * z[j*2+l]
									}
								}
								if math.IsNaN(got) || math.Abs(got-base[i*2+j]) > 1e-13 {
									t.Errorf("reconstruction [%d,%d]: %g want %g", i, j, got, base[i*2+j])
								}
							}
						}
					}
				})
			}
		}
	}
}
