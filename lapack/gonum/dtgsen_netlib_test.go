// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package gonum

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/blas"

	"plantcontrol.org/v1/gonum/lapack/gonum/internal/netlib"
)

func TestDtgsenNetlibWorkspaceQueries(t *testing.T) {
	for _, tc := range []struct {
		name                   string
		ijob, n, lwork, liwork int
	}{
		{name: "EmptyReorder", ijob: 0, n: 0, lwork: -1, liwork: -1},
		{name: "EmptyCondition", ijob: 1, n: 0, lwork: -1, liwork: -1},
		{name: "FloatOnly", ijob: 1, n: 2, lwork: -1, liwork: 1},
		{name: "IntegerOnly", ijob: 1, n: 2, lwork: 1, liwork: -1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			work := []float64{-1}
			iwork := []int{-1}
			var selected []bool
			var a []float64
			if tc.n != 0 {
				selected = []bool{true, false}
				a = []float64{1, 0, 0, 2}
			}
			Implementation{}.Dtgsen(tc.ijob, false, false, selected, tc.n,
				a, max(1, tc.n), nil, max(1, tc.n), nil, nil, nil,
				nil, 1, nil, 1, work, tc.lwork, iwork, tc.liwork)
			nwork, niwork, info := netlib.DtgsenWorkspace(tc.ijob, tc.n, tc.lwork, tc.liwork)
			if info != 0 {
				t.Fatalf("Netlib workspace query failed with info=%d", info)
			}
			if work[0] != nwork || iwork[0] != niwork {
				t.Fatalf("Gonum workspace=(%v,%d), Netlib=(%v,%d)", work[0], iwork[0], nwork, niwork)
			}
		})
	}
}

func TestDtgsenNetlibSingularConditionEstimate(t *testing.T) {
	const n = 2
	work := make([]float64, 4*n+16)
	iwork := make([]int, n+6)
	_, _, _, _, ok := Implementation{}.Dtgsen(1, false, false, []bool{true, false}, n,
		[]float64{1, 0, 0, 1}, n, []float64{1, 0, 0, 1}, n,
		make([]float64, n), make([]float64, n), make([]float64, n),
		nil, 1, nil, 1, work, len(work), iwork, len(iwork))
	info := netlib.DtgsenSingular()
	if ok != (info == 0) {
		t.Fatalf("success mismatch: Gonum=%v Netlib info=%d", ok, info)
	}
}

func TestDtgsenNetlibDifferential(t *testing.T) {
	const n = 5
	aOrig, bOrig, selected := dtgsenOraclePencil()

	for ijob := 0; ijob <= 5; ijob++ {
		for _, wantq := range []bool{false, true} {
			for _, wantz := range []bool{false, true} {
				name := fmt.Sprintf("ijob=%d/Q=%v/Z=%v", ijob, wantq, wantz)
				t.Run(name, func(t *testing.T) {
					ga, gb := append([]float64(nil), aOrig...), append([]float64(nil), bOrig...)
					na, nb := append([]float64(nil), aOrig...), append([]float64(nil), bOrig...)
					gar, gai, gbet := make([]float64, n), make([]float64, n), make([]float64, n)
					nar, nai, nbet := make([]float64, n), make([]float64, n), make([]float64, n)
					gq, gz := identityData(n), identityData(n)
					nq, nz := identityData(n), identityData(n)

					workQuery := make([]float64, 1)
					iworkQuery := make([]int, 1)
					Implementation{}.Dtgsen(ijob, wantq, wantz, selected, n,
						ga, n, gb, n, gar, gai, gbet, gq, n, gz, n,
						workQuery, -1, iworkQuery, -1)
					work := make([]float64, int(workQuery[0]))
					iwork := make([]int, iworkQuery[0])
					gm, gpl, gpr, gdif, gok := Implementation{}.Dtgsen(ijob, wantq, wantz, selected, n,
						ga, n, gb, n, gar, gai, gbet, gq, n, gz, n,
						work, len(work), iwork, len(iwork))
					nm, npl, npr, ndif, info := netlib.Dtgsen(ijob, wantq, wantz, selected, n,
						na, nb, nar, nai, nbet, nq, nz)
					if gok != (info == 0) || gm != nm {
						t.Fatalf("Gonum=(m=%d,ok=%v), Netlib=(m=%d,info=%d)", gm, gok, nm, info)
					}
					if !gok {
						return
					}
					compareGeneralizedEigenvalues(t, gar, gai, gbet, nar, nai, nbet)
					checkGeneralizedSchurStructure(t, "Gonum DTGSEN", ga, gb, n)
					checkGeneralizedSchurStructure(t, "Netlib DTGSEN", na, nb, n)
					if wantq && wantz {
						checkGeneralizedSchurResult(t, "Gonum DTGSEN", aOrig, bOrig, ga, gb, gq, gz, n)
						checkGeneralizedSchurResult(t, "Netlib DTGSEN", aOrig, bOrig, na, nb, nq, nz, n)
					}
					if ijob == 1 || ijob >= 4 {
						checkCloseNetlib(t, "pl", gpl, npl)
						checkCloseNetlib(t, "pr", gpr, npr)
					}
					if ijob >= 2 {
						if ijob == 2 || ijob == 4 {
							checkEstimateNetlib(t, "dif[0]", gdif[0], ndif[0])
							checkEstimateNetlib(t, "dif[1]", gdif[1], ndif[1])
						} else {
							checkCloseNetlib(t, "dif[0]", gdif[0], ndif[0])
							checkCloseNetlib(t, "dif[1]", gdif[1], ndif[1])
						}
					}
				})
			}
		}
	}
}

func TestDtgsenNetlibConditionBeforeNormalization(t *testing.T) {
	const n = 4
	a := []float64{
		-3, 0.2, 0.1, 0.3,
		0, 1, 0.4, -0.2,
		0, 0, 4, 0.7,
		0, 0, 0, -2,
	}
	b := []float64{
		-1, 0.1, -0.2, 0.05,
		0, 2, 0.2, -0.1,
		0, 0, -1.5, -0.2,
		0, 0, 0, 0.8,
	}
	selected := []bool{true, true, false, false}
	for _, ijob := range []int{2, 4} {
		ga, gb := append([]float64(nil), a...), append([]float64(nil), b...)
		na, nb := append([]float64(nil), a...), append([]float64(nil), b...)
		work := make([]float64, max(4*n+16, 2*2*(n-2)))
		iwork := make([]int, n+6)
		gm, gpl, gpr, gdif, gok := Implementation{}.Dtgsen(ijob, false, false, selected, n,
			ga, n, gb, n, make([]float64, n), make([]float64, n), make([]float64, n),
			nil, 1, nil, 1, work, len(work), iwork, len(iwork))
		nm, npl, npr, ndif, info := netlib.Dtgsen(ijob, false, false, selected, n,
			na, nb, make([]float64, n), make([]float64, n), make([]float64, n),
			identityData(n), identityData(n))
		if gok != (info == 0) || gm != nm {
			t.Fatalf("ijob=%d: Gonum=(m=%d,ok=%v), Netlib=(m=%d,info=%d)", ijob, gm, gok, nm, info)
		}
		checkCloseNetlib(t, "dif[0]", gdif[0], ndif[0])
		checkCloseNetlib(t, "dif[1]", gdif[1], ndif[1])
		if ijob == 4 {
			checkCloseNetlib(t, "pl", gpl, npl)
			checkCloseNetlib(t, "pr", gpr, npr)
		}
	}
}

func TestDtgsenNetlibFrobeniusEstimate(t *testing.T) {
	const n = 5
	rnd := rand.New(rand.NewPCG(11, 11))
	selected := []bool{false, false, false, true, true}
	accepted := 0
	for k := 0; k < 100; k++ {
		a := make([]float64, n*n)
		b := make([]float64, n*n)
		a[0], a[1], a[n], a[n+1] = -3, 0.5+rnd.Float64(), -0.5-rnd.Float64(), -3
		b[0], b[1], b[n+1] = 1, 0.1*rnd.NormFloat64(), 1.2
		a[2*n+2], b[2*n+2] = 4, 0.8
		a[3*n+3], a[3*n+4], a[4*n+3], a[4*n+4] = 1, 0.5+rnd.Float64(), -0.5-rnd.Float64(), 1
		b[3*n+3], b[3*n+4], b[4*n+4] = 1.5, 0.1*rnd.NormFloat64(), 1.1
		for i := 0; i < n; i++ {
			for j := i + 1; j < n; j++ {
				if a[i*n+j] == 0 {
					a[i*n+j] = 0.2 * rnd.NormFloat64()
				}
				if b[i*n+j] == 0 {
					b[i*n+j] = 0.2 * rnd.NormFloat64()
				}
			}
		}
		ga, gb := append([]float64(nil), a...), append([]float64(nil), b...)
		na, nb := append([]float64(nil), a...), append([]float64(nil), b...)
		work := make([]float64, 4*n+16)
		iwork := make([]int, n+6)
		gm, _, _, gdif, gok := Implementation{}.Dtgsen(2, false, false, selected, n,
			ga, n, gb, n, make([]float64, n), make([]float64, n), make([]float64, n),
			nil, 1, nil, 1, work, len(work), iwork, len(iwork))
		nm, _, _, ndif, info := netlib.Dtgsen(2, false, false, selected, n,
			na, nb, make([]float64, n), make([]float64, n), make([]float64, n),
			identityData(n), identityData(n))
		if gok != (info == 0) || gm != nm {
			t.Fatalf("case %d: Gonum=(m=%d,ok=%v), Netlib=(m=%d,info=%d)", k, gm, gok, nm, info)
		}
		if !gok {
			continue
		}
		accepted++
		for i := range gdif {
			checkEstimateNetlib(t, fmt.Sprintf("case %d dif[%d]", k, i), gdif[i], ndif[i])
		}
	}
	if accepted < 90 {
		t.Fatalf("only %d of 100 oracle cases were accepted", accepted)
	}
}

func TestDtgsenNetlibReverseDifKernel(t *testing.T) {
	const n = 5
	a, b, selected := dtgsenOraclePencil()
	work := make([]float64, 4*n+16)
	iwork := make([]int, n+6)
	m, _, _, _, ok := Implementation{}.Dtgsen(0, false, false, selected, n,
		a, n, b, n, make([]float64, n), make([]float64, n), make([]float64, n),
		nil, 1, nil, 1, work, len(work), iwork, 1)
	if !ok || m != 2 {
		t.Fatalf("reorder=(m=%d,ok=%v), want (2,true)", m, ok)
	}

	m1, n1 := n-m, m
	a22, a11 := make([]float64, m1*m1), make([]float64, n1*n1)
	b22, b11 := make([]float64, m1*m1), make([]float64, n1*n1)
	copyLocalBlock(m1, a[m*n+m:], n, a22, m1)
	copyLocalBlock(n1, a, n, a11, n1)
	copyLocalBlock(m1, b[m*n+m:], n, b22, m1)
	copyLocalBlock(n1, b, n, b11, n1)
	gc, gf := make([]float64, m1*n1), make([]float64, m1*n1)
	nc, nf := make([]float64, m1*n1), make([]float64, m1*n1)
	_, gdif, gok := Implementation{}.Dtgsyl(blas.NoTrans, 3, m1, n1,
		a22, m1, a11, n1, gc, n1,
		b22, m1, b11, n1, gf, n1,
		[]float64{0}, 1, make([]int, n+6))
	_, ndif, info := netlib.Dtgsyl(byte(blas.NoTrans), 3, m1, n1,
		a22, a11, nc, b22, b11, nf)
	if gok != (info == 0) {
		t.Fatalf("Gonum ok=%v, Netlib info=%d", gok, info)
	}
	checkCloseNetlib(t, "reverse dif", gdif, ndif)
	gc, gf = make([]float64, m1*n1), make([]float64, m1*n1)
	_, gdif, gok = Implementation{}.Dtgsyl(blas.NoTrans, 3, m1, n1,
		a[m*n+m:], n, a, n, gc, n1,
		b[m*n+m:], n, b, n, gf, n1,
		[]float64{0}, 1, make([]int, n+6))
	if !gok {
		t.Fatal("strided reverse kernel reported coincident eigenvalues")
	}
	checkCloseNetlib(t, "strided reverse dif", gdif, ndif)
}

func dtgsenOraclePencil() (a, b []float64, selected []bool) {
	a = []float64{
		-3, 0.2, 0.1, 0.3, 0.4,
		0, 1, 2, 0.2, -0.1,
		0, -1, 1, 0.5, 0.25,
		0, 0, 0, 4, 0.7,
		0, 0, 0, 0, -2,
	}
	b = []float64{
		1, 0.1, -0.2, 0.05, 0.3,
		0, 2, 0, 0.2, -0.1,
		0, 0, 3, 0.15, 0.25,
		0, 0, 0, 1.5, -0.2,
		0, 0, 0, 0, 0.8,
	}
	return a, b, []bool{false, false, false, true, true}
}
