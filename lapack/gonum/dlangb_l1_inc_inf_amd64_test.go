// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64

package gonum

import (
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/lapack"
)

// Valid2x1 band storage has active coordinates(0,0),(1,0), at indices1,3
// with ldab3. Production MaxColumnSum calls L1NormInc(ab[1:],2,2).
func TestDlangbL1IncPositiveInfWitness(t *testing.T) {
	storage := make([]float64, 8)
	for i := range storage {
		storage[i] = math.Float64frombits(0x7ff800000000c000 + uint64(i))
	}
	ab := storage[1:7:7]
	ab[1], ab[3] = math.Inf(1), math.Inf(1)
	before := make([]uint64, len(storage))
	for i, v := range storage {
		before[i] = math.Float64bits(v)
	}
	var want float64
	for i := 0; i < 2; i++ {
		want += math.Abs(ab[i*3+0-i+1])
	}
	got := Implementation{}.Dlangb(lapack.MaxColumnSum, 2, 1, 1, 0, ab, 3)
	t.Logf("DLANGB_L1_INC_WITNESS m=2 n=1 kl=1 ku=0 ldab=3 got=%g got_bits=%016x want=%g want_bits=%016x", got, math.Float64bits(got), want, math.Float64bits(want))
	for i, v := range storage {
		if math.Float64bits(v) != before[i] {
			t.Fatalf("input/padding/sentinel changed at %d", i)
		}
	}
	if !math.IsInf(want, 1) {
		t.Fatal("coordinate oracle is not +Inf")
	}
	if !math.IsInf(got, 1) {
		t.Fatalf("MaxColumnSum must return +Inf for two positive infinities: got %g", got)
	}
}
