// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import "testing"

func TestDgemmParallelBlockedAllocs(t *testing.T) {
	const n, workers = 200, 4
	a := make([]float64, n*n)
	b := make([]float64, n*n)
	c := make([]float64, n*n)
	for i := range a {
		a[i], b[i] = float64(i%7)-3, float64(i%5)-2
	}
	op := func() {
		dgemmParallelBlockedWorkers(false, false, n, n, n, a, n, b, n, c, n, 1, workers)
	}
	op()
	// One shared job plus one closure per worker, independent of the 16 blocks.
	if allocs := testing.AllocsPerRun(20, op); allocs > workers+1 {
		t.Errorf("allocs = %v, want <= %d", allocs, workers+1)
	}
}
