// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package gonum

import (
	"fmt"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/lapack/gonum/internal/netlib"
)

func TestDbdsqrNetlibDifferential(t *testing.T) {
	for _, uplo := range []blas.Uplo{blas.Upper, blas.Lower} {
		t.Run(fmt.Sprintf("uplo=%c", uplo), func(t *testing.T) {
			const n, ncvt, nru, ncc = 6, 4, 5, 3
			const ldvt, ldu, ldc = 7, 8, 6
			d0 := []float64{4.1, -3.2, 2.4, -1.7, 0.9, -0.3}
			e0 := []float64{0.8, -0.6, 0.45, -0.2, 0.07}
			vt0, u0, c0 := make([]float64, n*ldvt), make([]float64, nru*ldu), make([]float64, n*ldc)
			for i := range vt0 {
				vt0[i] = math.Sin(float64(i + 1))
			}
			for i := range u0 {
				u0[i] = math.Cos(float64(2*i + 1))
			}
			for i := range c0 {
				c0[i] = math.Sin(float64(3*i + 2))
			}

			gd, ge := append([]float64(nil), d0...), append([]float64(nil), e0...)
			gvt, gu, gc := append([]float64(nil), vt0...), append([]float64(nil), u0...), append([]float64(nil), c0...)
			gok := Implementation{}.Dbdsqr(uplo, n, ncvt, nru, ncc, gd, ge, gvt, ldvt, gu, ldu, gc, ldc, make([]float64, 4*(n-1)))

			nd, ne := append([]float64(nil), d0...), append([]float64(nil), e0...)
			nvt := netlibColMajor(n, ncvt, vt0, ldvt, n+2)
			nu := netlibColMajor(nru, n, u0, ldu, nru+2)
			nc := netlibColMajor(n, ncc, c0, ldc, n+2)
			info := netlib.Dbdsqr(byte(uplo), n, ncvt, nru, ncc, nd, ne, nvt, n+2, nu, nru+2, nc, n+2, make([]float64, 4*(n-1)))
			if !gok || info != 0 {
				t.Fatalf("Gonum ok=%v, Netlib info=%d", gok, info)
			}
			for i := range gd {
				if math.IsNaN(gd[i]) || math.IsInf(gd[i], 0) || math.IsNaN(nd[i]) || math.IsInf(nd[i], 0) {
					t.Fatalf("non-finite d[%d]: Gonum=%g Netlib=%g", i, gd[i], nd[i])
				}
				if math.Abs(gd[i]-nd[i]) > 2e-13*math.Max(1, nd[i]) {
					t.Fatalf("d[%d]=%g, Netlib=%g", i, gd[i], nd[i])
				}
			}
			netlibCheckMatrix(t, "VT", n, ncvt, gvt, ldvt, netlibRowMajor(n, ncvt, nvt, n+2, ldvt), ldvt, 2e-12)
			netlibCheckMatrix(t, "U", nru, n, gu, ldu, netlibRowMajor(nru, n, nu, nru+2, ldu), ldu, 2e-12)
			netlibCheckMatrix(t, "C", n, ncc, gc, ldc, netlibRowMajor(n, ncc, nc, n+2, ldc), ldc, 2e-12)
		})
	}
}

func TestDbdsqrNetlibValuesOnly(t *testing.T) {
	for _, tc := range []struct {
		name string
		d    []float64
	}{
		{name: "decreasing", d: []float64{9, 7, 5, 3, 2, 1}},
		{name: "increasing", d: []float64{1, 2, 3, 5, 7, 9}},
	} {
		for _, uplo := range []blas.Uplo{blas.Upper, blas.Lower} {
			t.Run(fmt.Sprintf("%s/uplo=%c", tc.name, uplo), func(t *testing.T) {
				gd, nd := append([]float64(nil), tc.d...), append([]float64(nil), tc.d...)
				ge := []float64{0.35, -0.2, 0.12, -0.08, 0.03}
				ne := append([]float64(nil), ge...)
				gok := Implementation{}.Dbdsqr(uplo, len(gd), 0, 0, 0, gd, ge, nil, 1, nil, 1, nil, 1, make([]float64, 4*len(gd)))
				info := netlib.Dbdsqr(byte(uplo), len(nd), 0, 0, 0, nd, ne, nil, 1, nil, 1, nil, 1, make([]float64, 4*len(nd)))
				if !gok || info != 0 {
					t.Fatalf("convergence: Gonum ok=%v, Netlib info=%d", gok, info)
				}
				for i := range gd {
					if math.IsNaN(gd[i]) || math.IsInf(gd[i], 0) || math.IsNaN(nd[i]) || math.IsInf(nd[i], 0) || math.Abs(gd[i]-nd[i]) > 2e-13*math.Max(1, nd[i]) {
						t.Fatalf("d[%d]=%g, Netlib=%g", i, gd[i], nd[i])
					}
				}
			})
		}
	}
}
