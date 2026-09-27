// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && arm64 && !safe && !noasm && !gccgo && !race

package gonum

import (
	"math"

	"plantcontrol.org/v1/gonum/lapack"
)

//go:noinline
func dlasrRightVariableSequential(direct lapack.Direct, m, n int, c, s, a []float64, lda int) bool {
	if direct == lapack.Forward {
		for j := 0; j < n-1; j++ {
			ctmp := c[j]
			stmp := s[j]
			if ctmp != 1 || stmp != 0 {
				for i := 0; i < m; i++ {
					tmp := a[i*lda+j+1]
					tmp2 := a[i*lda+j]
					a[i*lda+j+1] = ctmp*tmp - stmp*tmp2
					a[i*lda+j] = stmp*tmp + ctmp*tmp2
				}
			}
		}
		return true
	}
	for j := n - 2; j >= 0; j-- {
		ctmp := c[j]
		stmp := s[j]
		if ctmp != 1 || stmp != 0 {
			for i := 0; i < m; i++ {
				tmp := a[i*lda+j+1]
				tmp2 := a[i*lda+j]
				a[i*lda+j+1] = ctmp*tmp - stmp*tmp2
				a[i*lda+j] = stmp*tmp + ctmp*tmp2
			}
		}
	}
	return true
}

func dlasrRightVariableCarry4(direct lapack.Direct, m, n int, c, s, a []float64, lda int) bool {
	if m < 64 || n < 64 {
		return false
	}
	c = c[:n-1]
	s = s[:n-1]
	activeA := a[:(m-1)*lda+n]
	if !dlasrDisjoint(activeA, c) || !dlasrDisjoint(activeA, s) {
		return false
	}
	for j, ctmp := range c {
		if !(ctmp >= -1 && ctmp <= 1 && s[j] >= -1 && s[j] <= 1) {
			return false
		}
		if ctmp == 1 && s[j] == 0 {
			return false
		}
	}

	// Rows are independent; interleaving them preserves the rotation order of each row.
	i := 0
	for ; i+4 <= m; i += 4 {
		k0 := (i + 0) * lda
		k1 := (i + 1) * lda
		k2 := (i + 2) * lda
		k3 := (i + 3) * lda
		if direct == lapack.Forward {
			x0, x1, x2, x3 := a[k0], a[k1], a[k2], a[k3]
			for j, ctmp := range c {
				stmp := s[j]
				y0, y1, y2, y3 := a[k0+j+1], a[k1+j+1], a[k2+j+1], a[k3+j+1]
				a[k0+j] = dlasrRightLeft(ctmp, stmp, y0, x0)
				a[k1+j] = dlasrRightLeft(ctmp, stmp, y1, x1)
				a[k2+j] = dlasrRightLeft(ctmp, stmp, y2, x2)
				a[k3+j] = dlasrRightLeft(ctmp, stmp, y3, x3)
				x0 = dlasrRightRight(ctmp, stmp, y0, x0)
				x1 = dlasrRightRight(ctmp, stmp, y1, x1)
				x2 = dlasrRightRight(ctmp, stmp, y2, x2)
				x3 = dlasrRightRight(ctmp, stmp, y3, x3)
			}
			a[k0+n-1], a[k1+n-1], a[k2+n-1], a[k3+n-1] = x0, x1, x2, x3
			continue
		}
		x0, x1, x2, x3 := a[k0+n-1], a[k1+n-1], a[k2+n-1], a[k3+n-1]
		for j := n - 2; j >= 0; j-- {
			ctmp, stmp := c[j], s[j]
			y0, y1, y2, y3 := a[k0+j], a[k1+j], a[k2+j], a[k3+j]
			a[k0+j+1] = dlasrRightRight(ctmp, stmp, x0, y0)
			a[k1+j+1] = dlasrRightRight(ctmp, stmp, x1, y1)
			a[k2+j+1] = dlasrRightRight(ctmp, stmp, x2, y2)
			a[k3+j+1] = dlasrRightRight(ctmp, stmp, x3, y3)
			x0 = dlasrRightLeftBackward(ctmp, stmp, x0, y0)
			x1 = dlasrRightLeftBackward(ctmp, stmp, x1, y1)
			x2 = dlasrRightLeftBackward(ctmp, stmp, x2, y2)
			x3 = dlasrRightLeftBackward(ctmp, stmp, x3, y3)
		}
		a[k0], a[k1], a[k2], a[k3] = x0, x1, x2, x3
	}
	for ; i+2 <= m; i += 2 {
		k0 := (i + 0) * lda
		k1 := (i + 1) * lda
		if direct == lapack.Forward {
			x0, x1 := a[k0], a[k1]
			for j, ctmp := range c {
				stmp := s[j]
				y0, y1 := a[k0+j+1], a[k1+j+1]
				a[k0+j] = dlasrRightLeft(ctmp, stmp, y0, x0)
				a[k1+j] = dlasrRightLeft(ctmp, stmp, y1, x1)
				x0 = dlasrRightRight(ctmp, stmp, y0, x0)
				x1 = dlasrRightRight(ctmp, stmp, y1, x1)
			}
			a[k0+n-1], a[k1+n-1] = x0, x1
			continue
		}
		x0, x1 := a[k0+n-1], a[k1+n-1]
		for j := n - 2; j >= 0; j-- {
			ctmp, stmp := c[j], s[j]
			y0, y1 := a[k0+j], a[k1+j]
			a[k0+j+1] = dlasrRightRight(ctmp, stmp, x0, y0)
			a[k1+j+1] = dlasrRightRight(ctmp, stmp, x1, y1)
			x0 = dlasrRightLeftBackward(ctmp, stmp, x0, y0)
			x1 = dlasrRightLeftBackward(ctmp, stmp, x1, y1)
		}
		a[k0], a[k1] = x0, x1
	}
	for ; i < m; i++ {
		k := i * lda
		if direct == lapack.Forward {
			x := a[k]
			for j, ctmp := range c {
				stmp := s[j]
				y := a[k+j+1]
				a[k+j] = dlasrRightLeft(ctmp, stmp, y, x)
				x = dlasrRightRight(ctmp, stmp, y, x)
			}
			a[k+n-1] = x
			continue
		}
		x := a[k+n-1]
		for j := n - 2; j >= 0; j-- {
			ctmp, stmp := c[j], s[j]
			y := a[k+j]
			a[k+j+1] = dlasrRightRight(ctmp, stmp, x, y)
			x = dlasrRightLeftBackward(ctmp, stmp, x, y)
		}
		a[k] = x
	}
	return true
}

// Keep the fused term used by the original ARM64 loops; it differs by direction.
func dlasrRightLeft(c, s, right, left float64) float64 {
	return math.FMA(s, right, c*left)
}

func dlasrRightLeftBackward(c, s, right, left float64) float64 {
	return math.FMA(c, left, s*right)
}

func dlasrRightRight(c, s, right, left float64) float64 {
	return math.FMA(-s, left, c*right)
}
