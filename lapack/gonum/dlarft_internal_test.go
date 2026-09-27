// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/lapack"
)

func TestDlarftZeroReflectors(t *testing.T) {
	for _, store := range []lapack.StoreV{lapack.ColumnWise, lapack.RowWise} {
		v := []float64{1, 2, 3}
		tm := []float64{4, 5, 6}
		wantV := slices.Clone(v)
		wantT := slices.Clone(tm)
		Implementation{}.Dlarft(lapack.Forward, store, 3, 0, v, 3, nil, tm, 1)
		if !slices.Equal(v, wantV) || !slices.Equal(tm, wantT) {
			t.Fatalf("store=%v: zero-reflector call modified data", store)
		}
	}
}
