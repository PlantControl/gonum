// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"
	"math/cmplx"
	"testing"
)

// TestZlarfgBoundaries pins Zlarfg for n == 0, where beta drops imag(alpha)
// instead of returning alpha unchanged as ZLARFG does, and for n == 1 with a
// complex alpha, where ZLARFG builds a reflector that makes beta real.
func TestZlarfgBoundaries(t *testing.T) {
	impl := Implementation{}
	for _, alpha := range []complex128{0, 2, -3, 1 + 2i, -1 - 2i} {
		beta, tau := impl.Zlarfg(0, alpha, nil, 1)
		if beta != real(alpha) || tau != 0 {
			t.Errorf("n=0, alpha=%v: beta=%v tau=%v, want %v, 0", alpha, beta, tau, real(alpha))
		}
	}
	for _, alpha := range []complex128{2, -3, 1 + 2i, -1 - 2i, 2i} {
		beta, tau := impl.Zlarfg(1, alpha, nil, 1)
		if imag(alpha) == 0 {
			if beta != real(alpha) || tau != 0 {
				t.Errorf("n=1, alpha=%v: beta=%v tau=%v, want %v, 0", alpha, beta, tau, real(alpha))
			}
			continue
		}
		// ZLARFG: beta = -sign(real(alpha))·|alpha|, tau = (beta-alpha)/beta.
		wantBeta := -math.Copysign(cmplx.Abs(alpha), real(alpha))
		wantTau := (complex(wantBeta, 0) - alpha) / complex(wantBeta, 0)
		if math.Abs(beta-wantBeta) > 1e-15*math.Abs(wantBeta) || cmplx.Abs(tau-wantTau) > 1e-15 {
			t.Errorf("n=1, alpha=%v: beta=%v tau=%v, want %v, %v", alpha, beta, tau, wantBeta, wantTau)
		}
		// H^H·(alpha) = (beta) with H = I - tau.
		if got := (1 - cmplx.Conj(tau)) * alpha; cmplx.Abs(got-complex(beta, 0)) > 1e-14*cmplx.Abs(alpha) {
			t.Errorf("n=1, alpha=%v: H^H*alpha=%v, want beta=%v", alpha, got, beta)
		}
	}
}
