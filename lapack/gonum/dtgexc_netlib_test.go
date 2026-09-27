// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package gonum

import (
	"testing"

	"plantcontrol.org/v1/gonum/lapack/gonum/internal/netlib"
)

func TestDtgexcNetlibPartialRejection(t *testing.T) {
	const n = 8
	a, b := dtgexcPartialRejectionPencil()
	checkGeneralizedSchurStructure(t, "partial-rejection input", a, b, n)
	for _, tc := range []struct {
		name       string
		ifst, ilst int
	}{{"Down", 0, 6}, {"Up", 6, 0}} {
		t.Run(tc.name, func(t *testing.T) {
			ga, gb := append([]float64(nil), a...), append([]float64(nil), b...)
			na, nb := append([]float64(nil), a...), append([]float64(nil), b...)
			gq, gz := identityData(n), identityData(n)
			nq, nz := identityData(n), identityData(n)
			work := make([]float64, 4*n+16)
			gifst, gilst, gok := Implementation{}.Dtgexc(true, true, n, ga, n, gb, n,
				gq, n, gz, n, tc.ifst, tc.ilst, work, len(work))
			nifst, nilst, info, _ := netlib.DtgexcOutputs(true, true, n, na, nb, nq, nz, tc.ifst, tc.ilst)
			if gok || info == 0 {
				t.Fatalf("fixture did not reject: Gonum ok=%v Netlib info=%d", gok, info)
			}
			if gifst != nifst || gilst != nilst || gilst != 2 {
				t.Fatalf("positions: Gonum=(%d,%d), Netlib=(%d,%d), want partial=2", gifst, gilst, nifst, nilst)
			}
			checkGeneralizedSchurResult(t, "Gonum partial rejection", a, b, ga, gb, gq, gz, n)
			checkGeneralizedSchurResult(t, "Netlib partial rejection", a, b, na, nb, nq, nz, n)
			gar, gai, gbeta := generalizedSchurEigenvalues(ga, gb, n)
			nar, nai, nbeta := generalizedSchurEigenvalues(na, nb, n)
			compareGeneralizedEigenvalues(t, gar, gai, gbeta, nar, nai, nbeta)
		})
	}
}

func generalizedSchurEigenvalues(a, b []float64, n int) (ar, ai, beta []float64) {
	for i := 0; i < n; {
		size := 1
		if i+1 < n && a[(i+1)*n+i] != 0 {
			size = 2
		}
		br, bi, bb := schurBlockEigenvalues(a, b, n, i, size)
		ar, ai, beta = append(ar, br...), append(ai, bi...), append(beta, bb...)
		i += size
	}
	return ar, ai, beta
}

func TestDtgexcNetlibSplitBlock(t *testing.T) {
	const n = 8
	aOrig := []float64{
		1, 1, 1.1, 1.3, 2, 3, -4.7, 3.3,
		1, 1, 3.7, 7.9, 4, 5.3, 3.3, -0.9,
		0, 0, 2, -3, 3.4, 6.5, 5.2, 1.8,
		0, 0, 4, 2, -5.3, -8.9, -0.2, -0.5,
		0, 0, 0, 0, 4.2, 2, 3.3, 2.3,
		0, 0, 0, 0, 3.7, 4.2, 9.9, 8.8,
		0, 0, 0, 0, 0, 0, 9.9, 8.8,
		0, 0, 0, 0, 0, 0, -9.9, 9.9,
	}
	for _, move := range []struct {
		name       string
		ifst, ilst int
	}{{"Down", 0, 4}, {"Up", 4, 0}} {
		t.Run(move.name, func(t *testing.T) {
			bOrig := identityData(n)
			ga, gb := append([]float64(nil), aOrig...), append([]float64(nil), bOrig...)
			na, nb := append([]float64(nil), aOrig...), append([]float64(nil), bOrig...)
			q, z := identityData(n), identityData(n)
			work := make([]float64, 4*n+16)
			gifst, gilst, gok := Implementation{}.Dtgexc(true, true, n, ga, n, gb, n, q, n, z, n, move.ifst, move.ilst, work, len(work))
			nifst, nilst, info := netlib.Dtgexc(n, na, nb, move.ifst, move.ilst)
			if gok != (info == 0) || gifst != nifst || gilst != nilst {
				t.Fatalf("Gonum=(%d,%d,%v), Netlib=(%d,%d,info=%d)", gifst, gilst, gok, nifst, nilst, info)
			}
			for i := 0; i < n-1; i++ {
				if (ga[(i+1)*n+i] == 0) != (na[(i+1)*n+i] == 0) {
					t.Fatalf("block split differs at %d: Gonum subdiagonal=%g, Netlib subdiagonal=%g", i, ga[(i+1)*n+i], na[(i+1)*n+i])
				}
			}
		})
	}
}

func dtgexcPartialRejectionPencil() (a, b []float64) {
	a = []float64{
		0.8837381174848745, 1.0278086418238366, 1.242287046190262, -2.0245599621365535, 0.13193391371050078, 5.243817447973301, -0.1675012866314296, 3.20907340493771,
		-1.0278086418238366, 0.8837381174848745, 0.8672985136207249, -0.8300287088045253, -2.23414672507098, 2.4661439378168004, 0.020898547068929574, -1.5011359637902215,
		0, 0, 0.8837381174848745, 1.0278086418238366, -2.481314736327007, -0.3006933195679389, 1.692879622856612, -2.5679324290885357,
		0, 0, -1.0278086418238366, 0.8837381174848745, -0.4424311915886636, 0.899490410035128, -0.749035432848699, 0.4344087724419372,
		0, 0, 0, 0, 0.8837381174848745, 1.0278086418238366, -0.6103735031670565, 1.833160887141268,
		0, 0, 0, 0, -1.0278086418238366, 0.8837381174848745, 1.9085972975964831, -0.3804221526095448,
		0, 0, 0, 0, 0, 0, 0.8837381174848745, 1.0278086418238366,
		0, 0, 0, 0, 0, 0, -1.0278086418238366, 0.8837381174848745,
	}
	b = []float64{
		1, 0, -1.843115030649015, 1.9735513678899907, -2.195256814134712, -2.9788430358724156, 0.6417391888956059, -2.4767329822088406,
		0, 1, -2.2704751878407095, 0.5132913903067711, -2.0877947460586688, -1.123721364485692, 1.4573368931600816, -4.53983032845278,
		0, 0, 1, 0, -2.562175795319728, 2.015223421910731, 2.078714475364924, 4.510341717226485,
		0, 0, 0, 1, 1.9956769148497866, 2.7048980317840643, -1.9266679533872089, -0.7733398919029104,
		0, 0, 0, 0, 1, 0, -1.0544496662886582, -0.13878207378196655,
		0, 0, 0, 0, 0, 1, -1.5318937334869955, -0.2925099807256252,
		0, 0, 0, 0, 0, 0, 1, 0,
		0, 0, 0, 0, 0, 0, 0, 1,
	}
	return a, b
}
