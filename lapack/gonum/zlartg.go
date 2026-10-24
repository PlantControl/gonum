// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"
	"math/cmplx"
)

// Zlartg generates a plane rotation so that
//
//	[  cs        sn] * [f] = [r]
//	[-conj(sn)   cs]   [g] = [0]
//
// where cs is real non-negative, cs*cs + conj(sn)*sn = 1, and |r| = |(f,g)|.
//
// This is a more accurate complex version of BLAS Zrotg that uses scaling to
// avoid overflow or underflow. Properties:
//   - cs >= 0
//   - if g = 0, then cs = 1 and sn = 0
//   - if f = 0 and g != 0, then cs = 0 and r is real and positive
//   - if f != 0, then r has the phase of f
//
// Zlartg is a translation of the LAPACK 3.12 ZLARTG, based on Algorithm 978
// of Anderson, Safe Scaling in the Level 1 BLAS, ACM Trans. Math. Softw. 44
// (2017).
//
// Zlartg is an internal routine. It is exported for testing purposes.
func (impl Implementation) Zlartg(f, g complex128) (cs float64, sn, r complex128) {
	const safmin = dlamchS
	const safmax = 1 / safmin
	rtmin := math.Sqrt(safmin)

	if g == 0 {
		return 1, 0, f
	}

	if f == 0 {
		var d float64
		switch {
		case real(g) == 0:
			d = math.Abs(imag(g))
			sn = zdivr(cmplx.Conj(g), d)
		case imag(g) == 0:
			d = math.Abs(real(g))
			sn = zdivr(cmplx.Conj(g), d)
		default:
			g1 := math.Max(math.Abs(real(g)), math.Abs(imag(g)))
			rtmax := math.Sqrt(safmax / 2)
			if g1 > rtmin && g1 < rtmax {
				d = math.Sqrt(zabssq(g))
				sn = zdivr(cmplx.Conj(g), d)
			} else {
				u := math.Min(safmax, math.Max(safmin, g1))
				gs := zdivr(g, u)
				d = math.Sqrt(zabssq(gs))
				sn = zdivr(cmplx.Conj(gs), d)
				d *= u
			}
		}
		return 0, sn, complex(d, 0)
	}

	f1 := math.Max(math.Abs(real(f)), math.Abs(imag(f)))
	g1 := math.Max(math.Abs(real(g)), math.Abs(imag(g)))
	rtmax := math.Sqrt(safmax / 4)
	if f1 > rtmin && f1 < rtmax && g1 > rtmin && g1 < rtmax {
		f2 := zabssq(f)
		g2 := zabssq(g)
		h2 := f2 + g2
		if f2 >= h2*safmin {
			cs = math.Sqrt(f2 / h2)
			r = zdivr(f, cs)
			rtmax *= 2
			if f2 > rtmin && h2 < rtmax {
				sn = cmplx.Conj(g) * zdivr(f, math.Sqrt(f2*h2))
			} else {
				sn = cmplx.Conj(g) * zdivr(r, h2)
			}
		} else {
			d := math.Sqrt(f2 * h2)
			cs = f2 / d
			if cs >= safmin {
				r = zdivr(f, cs)
			} else {
				r = zmulr(f, h2/d)
			}
			sn = cmplx.Conj(g) * zdivr(f, d)
		}
		return cs, sn, r
	}

	u := math.Min(safmax, math.Max(safmin, math.Max(f1, g1)))
	gs := zdivr(g, u)
	g2 := zabssq(gs)
	var w, f2, h2 float64
	var fs complex128
	if f1/u < rtmin {
		v := math.Min(safmax, math.Max(safmin, f1))
		w = v / u
		fs = zdivr(f, v)
		f2 = zabssq(fs)
		h2 = f2*w*w + g2
	} else {
		w = 1
		fs = zdivr(f, u)
		f2 = zabssq(fs)
		h2 = f2 + g2
	}
	if f2 >= h2*safmin {
		cs = math.Sqrt(f2 / h2)
		r = zdivr(fs, cs)
		rtmax *= 2
		if f2 > rtmin && h2 < rtmax {
			sn = cmplx.Conj(gs) * zdivr(fs, math.Sqrt(f2*h2))
		} else {
			sn = cmplx.Conj(gs) * zdivr(r, h2)
		}
	} else {
		d := math.Sqrt(f2 * h2)
		cs = f2 / d
		if cs >= safmin {
			r = zdivr(fs, cs)
		} else {
			r = zmulr(fs, h2/d)
		}
		sn = cmplx.Conj(gs) * zdivr(fs, d)
	}
	cs *= w
	r = zmulr(r, u)
	return cs, sn, r
}

// zabssq returns real(z)^2 + imag(z)^2.
func zabssq(z complex128) float64 {
	re, im := real(z), imag(z)
	return re*re + im*im
}

// zdivr returns z/d for real d, dividing each part.
func zdivr(z complex128, d float64) complex128 {
	return complex(real(z)/d, imag(z)/d)
}

// zmulr returns z*d for real d, multiplying each part.
func zmulr(z complex128, d float64) complex128 {
	return complex(real(z)*d, imag(z)*d)
}
