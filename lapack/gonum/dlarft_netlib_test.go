// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package gonum

import (
	"fmt"
	"testing"

	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/lapack/gonum/internal/netlib"
)

func TestDlarftNetlib(t *testing.T) {
	const n, k = 9, 4
	tau := []float64{0.25, 0, -0.125, 0.375}
	for _, direct := range []lapack.Direct{lapack.Forward, lapack.Backward} {
		for _, store := range []lapack.StoreV{lapack.ColumnWise, lapack.RowWise} {
			t.Run(fmt.Sprintf("direct=%c/store=%c", direct, store), func(t *testing.T) {
				vr, vld := reflectorData(n, k, direct, store)
				vc, cvld := reflectorColMajor(n, k, store, vr, vld)
				const ldt = k + 3
				gt := make([]float64, (k-1)*ldt+k)
				nt := make([]float64, (k-1)*ldt+k)
				setRowPadding(gt, k, k, ldt, 12345)
				setColPadding(nt, k, k, ldt, 12345)
				gt0 := append([]float64(nil), gt...)
				nt0 := append([]float64(nil), nt...)
				Implementation{}.Dlarft(direct, store, n, k, vr, vld, tau, gt, ldt)
				netlib.Dlarft(byte(direct), byte(store), n, k, vc, cvld, tau, nt, ldt)
				checkRowPadding(t, gt, gt0, k, k, ldt)
				checkColPadding(t, nt, nt0, k, k, ldt)
				for i := 0; i < k; i++ {
					for j := 0; j < k; j++ {
						if direct == lapack.Forward && i > j || direct == lapack.Backward && i < j {
							continue
						}
						checkClose(t, fmt.Sprintf("T[%d,%d]", i, j), gt[i*ldt+j], nt[j*ldt+i], 2e-13)
					}
				}
			})
		}
	}
}
