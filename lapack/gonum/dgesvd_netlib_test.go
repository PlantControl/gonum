// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package gonum

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/lapack/gonum/internal/netlib"
)

func TestNetlibRuntimeVersion(t *testing.T) {
	major, minor, patch := netlib.Version()
	t.Logf("runtime ILAVER=%d.%d.%d; source review pin is Reference-LAPACK v3.12.1", major, minor, patch)
}

func TestDgesvdNetlibDifferential(t *testing.T) {
	jobs := []struct{ u, vt lapack.SVDJob }{
		{lapack.SVDAll, lapack.SVDAll},
		{lapack.SVDStore, lapack.SVDStore},
		{lapack.SVDAll, lapack.SVDNone},
		{lapack.SVDNone, lapack.SVDAll},
		{lapack.SVDNone, lapack.SVDNone},
	}
	for _, shape := range [][2]int{{7, 4}, {4, 7}, {5, 5}} {
		m, n := shape[0], shape[1]
		for _, scale := range []float64{1, 1e-200, 1e200} {
			for _, job := range jobs {
				name := fmt.Sprintf("m=%d/n=%d/scale=%g/jobs=%c%c", m, n, scale, job.u, job.vt)
				t.Run(name, func(t *testing.T) {
					k, lda := min(m, n), n+3
					a0 := make([]float64, m*lda)
					rnd := rand.New(rand.NewPCG(uint64(m), uint64(n)))
					for i := 0; i < m; i++ {
						for j := 0; j < n; j++ {
							a0[i*lda+j] = scale * rnd.NormFloat64()
						}
					}
					guCols, nvtRows := svdVectorDims(job.u, m, k), svdVectorDims(job.vt, n, k)
					ldu, ldvt := max(1, guCols+2), n+2
					ga := append([]float64(nil), a0...)
					gs, gu, gvt := make([]float64, k), make([]float64, max(1, m*ldu)), make([]float64, max(1, nvtRows*ldvt))
					gquery := make([]float64, 1)
					Implementation{}.Dgesvd(job.u, job.vt, m, n, ga, lda, gs, gu, ldu, gvt, ldvt, gquery, -1)
					gok := Implementation{}.Dgesvd(job.u, job.vt, m, n, ga, lda, gs, gu, ldu, gvt, ldvt, make([]float64, int(gquery[0])), int(gquery[0]))

					na, nlda := netlibColMajor(m, n, a0, lda, m+2), m+2
					ns := make([]float64, k)
					nldu, nldvt := max(1, m+2), max(1, nvtRows+2)
					nu := make([]float64, max(1, nldu*guCols))
					nvt := make([]float64, max(1, nldvt*n))
					nquery := make([]float64, 1)
					info := netlib.Dgesvd(byte(job.u), byte(job.vt), m, n, na, nlda, ns, nu, nldu, nvt, nldvt, nquery, -1)
					if info != 0 {
						t.Fatalf("Netlib workspace query info=%d", info)
					}
					info = netlib.Dgesvd(byte(job.u), byte(job.vt), m, n, na, nlda, ns, nu, nldu, nvt, nldvt, make([]float64, int(nquery[0])), int(nquery[0]))
					if !gok || info != 0 {
						t.Fatalf("convergence: Gonum ok=%v, Netlib info=%d", gok, info)
					}
					for i := range gs {
						if math.IsNaN(gs[i]) || math.IsInf(gs[i], 0) || math.IsNaN(ns[i]) || math.IsInf(ns[i], 0) {
							t.Fatalf("non-finite singular value at %d: Gonum=%g Netlib=%g", i, gs[i], ns[i])
						}
						if math.Abs(gs[i]-ns[i])/scale > 5e-13 {
							t.Fatalf("s[%d]=%g, Netlib=%g", i, gs[i], ns[i])
						}
					}
					if job.u != lapack.SVDNone {
						nuRow := netlibRowMajor(m, guCols, nu, nldu, ldu)
						if r := netlibOrthoResidual(m, guCols, gu, ldu, true); math.IsNaN(r) || r > 2e-12 {
							t.Errorf("Gonum U orthogonality=%g", r)
						}
						if r := netlibOrthoResidual(m, guCols, nuRow, ldu, true); math.IsNaN(r) || r > 2e-12 {
							t.Errorf("Netlib U orthogonality=%g", r)
						}
					}
					if job.vt != lapack.SVDNone {
						nvtRow := netlibRowMajor(nvtRows, n, nvt, nldvt, ldvt)
						if r := netlibOrthoResidual(nvtRows, n, gvt, ldvt, false); math.IsNaN(r) || r > 2e-12 {
							t.Errorf("Gonum VT orthogonality=%g", r)
						}
						if r := netlibOrthoResidual(nvtRows, n, nvtRow, ldvt, false); math.IsNaN(r) || r > 2e-12 {
							t.Errorf("Netlib VT orthogonality=%g", r)
						}
					}
					if job.u != lapack.SVDNone && job.vt != lapack.SVDNone {
						nuRow := netlibRowMajor(m, guCols, nu, nldu, ldu)
						nvtRow := netlibRowMajor(nvtRows, n, nvt, nldvt, ldvt)
						if r := netlibSVDResidual(m, n, a0, lda, gs, gu, ldu, gvt, ldvt); math.IsNaN(r) || r > 2e-12 {
							t.Errorf("Gonum residual=%g", r)
						}
						if r := netlibSVDResidual(m, n, a0, lda, ns, nuRow, ldu, nvtRow, ldvt); math.IsNaN(r) || r > 2e-12 {
							t.Errorf("Netlib residual=%g", r)
						}
					}
				})
			}
		}
	}
}

func TestDgesvdNetlibRankDeficient(t *testing.T) {
	const m, n, lda = 6, 4, 7
	a0 := make([]float64, m*lda)
	for i, value := range []float64{5, 2, 0, 0} {
		a0[i*lda+i] = value
	}
	const k, ldu, ldvt = 4, 6, 4
	ga := append([]float64(nil), a0...)
	gs, gu, gvt := make([]float64, k), make([]float64, m*ldu), make([]float64, k*ldvt)
	gquery := make([]float64, 1)
	Implementation{}.Dgesvd(lapack.SVDStore, lapack.SVDStore, m, n, ga, lda, gs, gu, ldu, gvt, ldvt, gquery, -1)
	gok := Implementation{}.Dgesvd(lapack.SVDStore, lapack.SVDStore, m, n, ga, lda, gs, gu, ldu, gvt, ldvt, make([]float64, int(gquery[0])), int(gquery[0]))

	na := netlibColMajor(m, n, a0, lda, m)
	ns, nu, nvt := make([]float64, k), make([]float64, m*k), make([]float64, k*n)
	nquery := make([]float64, 1)
	if info := netlib.Dgesvd('S', 'S', m, n, na, m, ns, nu, m, nvt, k, nquery, -1); info != 0 {
		t.Fatalf("query info=%d", info)
	}
	info := netlib.Dgesvd('S', 'S', m, n, na, m, ns, nu, m, nvt, k, make([]float64, int(nquery[0])), int(nquery[0]))
	if !gok || info != 0 {
		t.Fatalf("convergence: Gonum ok=%v, Netlib info=%d", gok, info)
	}
	for i := range gs {
		if math.IsNaN(gs[i]) || math.IsInf(gs[i], 0) || math.IsNaN(ns[i]) || math.IsInf(ns[i], 0) || math.Abs(gs[i]-ns[i]) > 5e-14 {
			t.Fatalf("s[%d]=%g, Netlib=%g", i, gs[i], ns[i])
		}
	}
	nuRow := netlibRowMajor(m, k, nu, m, ldu)
	nvtRow := netlibRowMajor(k, n, nvt, k, ldvt)
	for name, residual := range map[string]float64{
		"Gonum reconstruction":    netlibSVDResidual(m, n, a0, lda, gs, gu, ldu, gvt, ldvt),
		"Netlib reconstruction":   netlibSVDResidual(m, n, a0, lda, ns, nuRow, ldu, nvtRow, ldvt),
		"Gonum U orthogonality":   netlibOrthoResidual(m, k, gu, ldu, true),
		"Netlib U orthogonality":  netlibOrthoResidual(m, k, nuRow, ldu, true),
		"Gonum VT orthogonality":  netlibOrthoResidual(k, n, gvt, ldvt, false),
		"Netlib VT orthogonality": netlibOrthoResidual(k, n, nvtRow, ldvt, false),
	} {
		if math.IsNaN(residual) || residual > 2e-12 {
			t.Errorf("%s=%g", name, residual)
		}
	}
}

func svdVectorDims(job lapack.SVDJob, full, compact int) int {
	switch job {
	case lapack.SVDAll:
		return full
	case lapack.SVDStore:
		return compact
	case lapack.SVDNone:
		return 0
	default:
		panic("unsupported benchmark/test job")
	}
}

func BenchmarkDgesvdNetlibKernels(b *testing.B) {
	for _, shape := range [][2]int{{32, 24}, {24, 32}, {96, 64}, {64, 96}, {256, 256}} {
		m, n := shape[0], shape[1]
		name := fmt.Sprintf("m=%d/n=%d", m, n)
		inputRow := make([]float64, m*n)
		rnd := rand.New(rand.NewPCG(uint64(m), uint64(n)))
		for i := range inputRow {
			inputRow[i] = rnd.NormFloat64()
		}
		inputCol := netlibColMajor(m, n, inputRow, n, m)
		k := min(m, n)
		b.Run("implementation=Gonum/"+name, func(b *testing.B) {
			a, s, u, vt := make([]float64, len(inputRow)), make([]float64, k), make([]float64, m*k), make([]float64, k*n)
			query := make([]float64, 1)
			Implementation{}.Dgesvd(lapack.SVDStore, lapack.SVDStore, m, n, a, n, s, u, k, vt, n, query, -1)
			work := make([]float64, int(query[0]))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				copy(a, inputRow)
				if !(Implementation{}).Dgesvd(lapack.SVDStore, lapack.SVDStore, m, n, a, n, s, u, k, vt, n, work, len(work)) {
					b.Fatal("Dgesvd did not converge")
				}
			}
		})
		b.Run("implementation=Netlib/"+name, func(b *testing.B) {
			a, s, u, vt := make([]float64, len(inputCol)), make([]float64, k), make([]float64, m*k), make([]float64, k*n)
			query := make([]float64, 1)
			if info := netlib.Dgesvd('S', 'S', m, n, a, m, s, u, m, vt, k, query, -1); info != 0 {
				b.Fatalf("query info=%d", info)
			}
			work := make([]float64, int(query[0]))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				copy(a, inputCol)
				if info := netlib.Dgesvd('S', 'S', m, n, a, m, s, u, m, vt, k, work, len(work)); info != 0 {
					b.Fatalf("Dgesvd info=%d", info)
				}
			}
		})
	}
}
