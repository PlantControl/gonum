// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !race

package mat

import "testing"

// Race mode drops sync.Pool items at random, so allocation counts are only
// meaningful without it.
func TestPoolSlicesNoAllocs(t *testing.T) {
	putFloat64s(getFloat64s(100, false))
	putInts(getInts(100, false))
	allocs := testing.AllocsPerRun(100, func() {
		w := getFloat64s(100, true)
		if len(w) != 100 {
			t.Fatalf("len = %d, want 100", len(w))
		}
		putFloat64s(w)
		iw := getInts(100, true)
		if len(iw) != 100 {
			t.Fatalf("len = %d, want 100", len(iw))
		}
		putInts(iw)
	})
	if allocs != 0 {
		t.Errorf("get/put allocs = %v, want 0", allocs)
	}
}
