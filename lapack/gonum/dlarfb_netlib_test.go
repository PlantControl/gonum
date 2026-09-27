// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package gonum

import (
	"fmt"
	"testing"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/lapack/gonum/internal/netlib"
)

func TestDlarfbNetlib(t *testing.T) {
	const m, n, k = 7, 6, 4
	tau := []float64{0.25, 0, -0.125, 0.375}
	for _, direct := range []lapack.Direct{lapack.Forward, lapack.Backward} {
		for _, store := range []lapack.StoreV{lapack.ColumnWise, lapack.RowWise} {
			for _, side := range []blas.Side{blas.Left, blas.Right} {
				for _, trans := range []blas.Transpose{blas.NoTrans, blas.Trans} {
					name := fmt.Sprintf("direct=%c/store=%c/side=%c/trans=%c", direct, store, side, trans)
					t.Run(name, func(t *testing.T) {
						nv := n
						if side == blas.Left {
							nv = m
						}
						vr, vld := reflectorData(nv, k, direct, store)
						vc, cvld := reflectorColMajor(nv, k, store, vr, vld)
						const ldt = k + 2
						gt := make([]float64, (k-1)*ldt+k)
						nt := make([]float64, (k-1)*ldt+k)
						Implementation{}.Dlarft(direct, store, nv, k, vr, vld, tau, gt, ldt)
						netlib.Dlarft(byte(direct), byte(store), nv, k, vc, cvld, tau, nt, ldt)

						const ldc = n + 3
						c0 := make([]float64, (m-1)*ldc+n)
						for i := 0; i < m; i++ {
							for j := 0; j < n; j++ {
								c0[i*ldc+j] = float64((i+2)*(j+3)%17-8) / 16
							}
						}
						setRowPadding(c0, m, n, ldc, 12345)
						gc := append([]float64(nil), c0...)
						const cldc = m + 2
						nc := rowToCol(m, n, c0, ldc, cldc)
						setColPadding(nc, m, n, cldc, 12345)
						nc0 := append([]float64(nil), nc...)
						nw := n
						if side == blas.Right {
							nw = m
						}
						Implementation{}.Dlarfb(side, trans, direct, store, m, n, k, vr, vld,
							gt, ldt, gc, ldc, make([]float64, nw*k), k)
						netlib.Dlarfb(byte(side), byte(trans), byte(direct), byte(store), m, n, k,
							vc, cvld, nt, ldt, nc, cldc, make([]float64, nw*k), nw)
						checkRowPadding(t, gc, c0, m, n, ldc)
						checkColPadding(t, nc, nc0, m, n, cldc)
						for i := 0; i < m; i++ {
							for j := 0; j < n; j++ {
								checkClose(t, fmt.Sprintf("C[%d,%d]", i, j), gc[i*ldc+j], nc[j*cldc+i], 8e-13)
							}
						}
					})
				}
			}
		}
	}
}
