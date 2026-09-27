// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package gonum

import (
	"fmt"
	"testing"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/lapack/gonum/internal/netlib"
)

func TestDormqrNetlib(t *testing.T) {
	for _, side := range []blas.Side{blas.Left, blas.Right} {
		for _, trans := range []blas.Transpose{blas.NoTrans, blas.Trans} {
			t.Run(fmt.Sprintf("side=%c/trans=%c", side, trans), func(t *testing.T) {
				const m, n, k = 11, 8, 5
				nq := n
				if side == blas.Left {
					nq = m
				}
				const lda = k + 2
				a := make([]float64, (nq-1)*lda+k)
				for i := 0; i < nq; i++ {
					for j := 0; j < min(i+1, k); j++ {
						a[i*lda+j] = float64((i+3)*(j+1)%13-6) / 20
					}
				}
				tau := []float64{0.25, -0.125, 0, 0.375, 0.2}
				calda := nq + 2
				ca := rowToCol(nq, k, a, lda, calda)
				const ldc = n + 2
				c0 := make([]float64, (m-1)*ldc+n)
				for i := 0; i < m; i++ {
					for j := 0; j < n; j++ {
						c0[i*ldc+j] = float64((i+1)*(j+4)%19-9) / 16
					}
				}
				setRowPadding(c0, m, n, ldc, 12345)
				const cldc = m + 3
				nc0 := rowToCol(m, n, c0, ldc, cldc)
				setColPadding(nc0, m, n, cldc, 12345)
				gq, nqwork := []float64{0}, []float64{0}
				Implementation{}.Dormqr(side, trans, m, n, k, a, lda, tau,
					append([]float64(nil), c0...), ldc, gq, -1)
				if info := netlib.Dormqr(byte(side), byte(trans), m, n, k, ca, calda, tau,
					append([]float64(nil), nc0...), cldc, nqwork, -1); info != 0 {
					t.Fatalf("query info=%d", info)
				}
				nw := n
				if side == blas.Right {
					nw = m
				}
				if gq[0] < float64(nw) || nqwork[0] < float64(nw) {
					t.Fatalf("query below minimum: Gonum=%g Netlib=%g minimum=%d", gq[0], nqwork[0], nw)
				}
				for _, workKind := range []string{"minimum", "optimal"} {
					t.Run(workKind, func(t *testing.T) {
						glwork, nlwork := nw, nw
						if workKind == "optimal" {
							glwork, nlwork = int(gq[0]), int(nqwork[0])
						}
						gc := append([]float64(nil), c0...)
						nc := append([]float64(nil), nc0...)
						gwork := make([]float64, glwork)
						nwork := make([]float64, nlwork)
						Implementation{}.Dormqr(side, trans, m, n, k, a, lda, tau,
							gc, ldc, gwork, glwork)
						if info := netlib.Dormqr(byte(side), byte(trans), m, n, k, ca, calda, tau,
							nc, cldc, nwork, nlwork); info != 0 {
							t.Fatalf("info=%d", info)
						}
						if gwork[0] != gq[0] || nwork[0] != nqwork[0] {
							t.Errorf("work[0] not restored: Gonum=%g want %g Netlib=%g want %g", gwork[0], gq[0], nwork[0], nqwork[0])
						}
						checkRowPadding(t, gc, c0, m, n, ldc)
						checkColPadding(t, nc, nc0, m, n, cldc)
						for i := 0; i < m; i++ {
							for j := 0; j < n; j++ {
								checkClose(t, fmt.Sprintf("C[%d,%d]", i, j), gc[i*ldc+j], nc[j*cldc+i], 8e-13)
							}
						}
					})
				}
			})
		}
	}
}

func TestDormqrBlockedNetlib(t *testing.T) {
	for _, side := range []blas.Side{blas.Left, blas.Right} {
		for _, trans := range []blas.Transpose{blas.NoTrans, blas.Trans} {
			t.Run(fmt.Sprintf("side=%c/trans=%c", side, trans), func(t *testing.T) {
				m, n, k := 73, 9, 70
				if side == blas.Right {
					m, n = n, m
				}
				nq := n
				if side == blas.Left {
					nq = m
				}
				lda := k + 1
				a := make([]float64, (nq-1)*lda+k)
				for i := 0; i < nq; i++ {
					for j := 0; j < min(i+1, k); j++ {
						a[i*lda+j] = float64((i+3)*(j+1)%17-8) / 64
					}
				}
				tau := make([]float64, k)
				for i := range tau {
					tau[i] = float64(i%5-2) / 16
				}
				calda := nq + 1
				ca := rowToCol(nq, k, a, lda, calda)
				ldc := n + 1
				c0 := make([]float64, (m-1)*ldc+n)
				for i := 0; i < m; i++ {
					for j := 0; j < n; j++ {
						c0[i*ldc+j] = float64((i+2)*(j+5)%19-9) / 32
					}
				}
				setRowPadding(c0, m, n, ldc, 12345)
				cldc := m + 1
				nc := rowToCol(m, n, c0, ldc, cldc)
				setColPadding(nc, m, n, cldc, 12345)
				nc0 := append([]float64(nil), nc...)
				gq, nqwork := []float64{0}, []float64{0}
				Implementation{}.Dormqr(side, trans, m, n, k, a, lda, tau,
					append([]float64(nil), c0...), ldc, gq, -1)
				if info := netlib.Dormqr(byte(side), byte(trans), m, n, k, ca, calda, tau,
					append([]float64(nil), nc...), cldc, nqwork, -1); info != 0 {
					t.Fatalf("query info=%d", info)
				}
				nw := n
				if side == blas.Right {
					nw = m
				}
				if gq[0] < float64(nw) || nqwork[0] < float64(nw) {
					t.Fatalf("query below minimum: Gonum=%g Netlib=%g minimum=%d", gq[0], nqwork[0], nw)
				}
				gc := append([]float64(nil), c0...)
				gwork := make([]float64, int(gq[0]))
				nwork := make([]float64, int(nqwork[0]))
				Implementation{}.Dormqr(side, trans, m, n, k, a, lda, tau,
					gc, ldc, gwork, len(gwork))
				if info := netlib.Dormqr(byte(side), byte(trans), m, n, k, ca, calda, tau,
					nc, cldc, nwork, len(nwork)); info != 0 {
					t.Fatalf("info=%d", info)
				}
				if gwork[0] != gq[0] || nwork[0] != nqwork[0] {
					t.Errorf("work[0] not restored: Gonum=%g want %g Netlib=%g want %g", gwork[0], gq[0], nwork[0], nqwork[0])
				}
				checkRowPadding(t, gc, c0, m, n, ldc)
				checkColPadding(t, nc, nc0, m, n, cldc)
				for i := 0; i < m; i++ {
					for j := 0; j < n; j++ {
						checkClose(t, fmt.Sprintf("C[%d,%d]", i, j), gc[i*ldc+j], nc[j*cldc+i], 2e-12)
					}
				}
			})
		}
	}
}
