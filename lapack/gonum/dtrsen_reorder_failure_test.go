// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"fmt"
	"testing"
)

// dtrsenRejectedSwapFixture is a 4×4 real Schur form, two 2×2 blocks in
// standard form, whose swap Dlaexc rejects: the weak stability ratio is
// about 44·eps·‖D‖ against the threshold 10·eps·‖D‖. It was found by hill
// climbing on that ratio; Reference LAPACK 3.12 DTRSEN also returns INFO=1
// on it (TestDtrsenNetlibReorderFailure).
var dtrsenRejectedSwapFixture = []float64{
	-3.333419506092188, 0.0004896915795339604, -2.428346849649743, -3.445704900621385,
	-7.184010700302862, -3.333419506092188, 0.5915379976921887, 3.1958988665291583,
	0, 0, -3.0738820901054704, 0.027424367165052442,
	0, 0, -0.002678551556826866, -3.0738820901054704,
}

// TestDtrsenReorderFailure checks that when the reordering fails Dtrsen
// returns ok=false with s=0 for ijob 1 and 3 and sep=0, as DTRSEN does.
func TestDtrsenReorderFailure(t *testing.T) {
	const n = 4
	sel := []bool{false, false, true, true}
	for ijob := 0; ijob <= 3; ijob++ {
		t.Run(fmt.Sprintf("ijob=%d", ijob), func(t *testing.T) {
			tt := append([]float64(nil), dtrsenRejectedSwapFixture...)
			q := make([]float64, n*n)
			for i := range n {
				q[i*n+i] = 1
			}
			work, iwork := make([]float64, 1), make([]int, 1)
			impl := Implementation{}
			impl.Dtrsen(ijob, true, sel, n, tt, n, q, n, make([]float64, n), make([]float64, n), work, -1, iwork, -1)
			work, iwork = make([]float64, int(work[0])), make([]int, iwork[0])
			m, s, sep, ok := impl.Dtrsen(ijob, true, sel, n, tt, n, q, n, make([]float64, n), make([]float64, n), work, len(work), iwork, len(iwork))
			if ok || m != 2 {
				t.Fatalf("ok=%v m=%d, want ok=false m=2", ok, m)
			}
			if (ijob == 1 || ijob == 3) && s != 0 {
				t.Errorf("s=%g, want 0", s)
			}
			if sep != 0 {
				t.Errorf("sep=%g, want 0", sep)
			}
		})
	}
}
