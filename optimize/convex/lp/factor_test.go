// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lp

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
)

// randomBasis returns m columns of an m×2m matrix. Each column has one
// entry on a permuted diagonal and up to three others; singular repeats a
// column.
func randomBasis(rnd *rand.Rand, m int, singular bool) (*columns, []int) {
	a := &columns{start: []int{0}}
	perm := rnd.Perm(m)
	for j := range 2 * m {
		d := perm[j%m]
		a.index = append(a.index, d)
		a.value = append(a.value, (1+rnd.Float64())*float64(1-2*rnd.IntN(2)))
		used := map[int]bool{d: true}
		for range rnd.IntN(4) {
			i := rnd.IntN(m)
			if used[i] {
				continue
			}
			used[i] = true
			a.index = append(a.index, i)
			a.value = append(a.value, rnd.NormFloat64())
		}
		a.start = append(a.start, len(a.index))
	}
	head := rnd.Perm(2 * m)[:m]
	for i := range head {
		head[i] = i
		if rnd.IntN(2) == 0 {
			head[i] += m
		}
	}
	rnd.Shuffle(m, func(i, j int) { head[i], head[j] = head[j], head[i] })
	if singular && m > 1 {
		head[1] = head[0]
	}
	return a, head
}

// residual returns max |B x - z| / (1 + max |z|) for basis B.
func residual(a *columns, head []int, x, z []float64, trans bool) float64 {
	m := len(head)
	r := make([]float64, m)
	if trans {
		// (Bᵀx)_i = column head[i] · x.
		for i, j := range head {
			for k := a.start[j]; k < a.start[j+1]; k++ {
				r[i] += a.value[k] * x[a.index[k]]
			}
		}
	} else {
		for i, j := range head {
			for k := a.start[j]; k < a.start[j+1]; k++ {
				r[a.index[k]] += a.value[k] * x[i]
			}
		}
	}
	worst, scale := 0.0, 1.0
	for i := range r {
		worst = math.Max(worst, math.Abs(r[i]-z[i]))
		scale = math.Max(scale, math.Abs(z[i]))
	}
	return worst / scale
}

func TestSparseLU(t *testing.T) {
	t.Parallel()
	rnd := rand.New(rand.NewPCG(11, 11))
	solved := 0
	for range 2000 {
		m := 1 + rnd.IntN(60)
		singular := rnd.IntN(10) == 0
		a, head := randomBasis(rnd, m, singular)
		dense, sparse := newDenseLU(m), newSparseLU(m)
		errDense := dense.factorize(a, head)
		errSparse := sparse.factorize(a, head)
		if singular && m > 1 && errSparse == nil {
			t.Fatalf("m=%d: repeated column factorized", m)
		}
		if !singular && errDense == nil && errSparse != nil {
			t.Fatalf("m=%d: sparse %v, dense factorized", m, errSparse)
		}
		if errDense != nil || errSparse != nil {
			continue
		}
		// Replace a few basis columns through updates, as the simplex does.
		for range rnd.IntN(5) {
			q := rnd.IntN(len(a.start) - 1)
			alpha := make([]float64, m)
			for k := a.start[q]; k < a.start[q+1]; k++ {
				alpha[a.index[k]] = a.value[k]
			}
			sparse.ftran(alpha)
			row := rnd.IntN(m)
			if math.Abs(alpha[row]) < 1e-3 {
				continue
			}
			sparse.update(row, alpha)
			head = append([]int(nil), head...)
			head[row] = q
		}
		z := make([]float64, m)
		for i := range z {
			z[i] = rnd.NormFloat64()
		}
		x := append([]float64(nil), z...)
		sparse.ftran(x)
		y := append([]float64(nil), z...)
		sparse.btran(y)
		if r := residual(a, head, x, z, false); r > 1e-8 {
			t.Fatalf("m=%d: ftran residual %g", m, r)
		}
		if r := residual(a, head, y, z, true); r > 1e-8 {
			t.Fatalf("m=%d: btran residual %g", m, r)
		}
		solved++
	}
	if solved < 1500 {
		t.Fatalf("only %d of 2000 bases solved", solved)
	}
}

// masterBasis returns a basis of a Dantzig–Wolfe master: 2·periods linking
// rows that nearly every column enters, and one convexity row per block
// with a single entry per column. Each block has one column in the basis
// and 2·periods blocks have a second.
func masterBasis(rnd *rand.Rand, periods, blocks int) (*columns, []int) {
	link := 2 * periods
	a := &columns{start: []int{0}}
	var head []int
	column := func(block int) {
		for i := range link {
			if rnd.IntN(8) == 0 {
				continue
			}
			a.index = append(a.index, i)
			a.value = append(a.value, 5+400*rnd.Float64())
		}
		a.index = append(a.index, link+block)
		a.value = append(a.value, 1)
		a.start = append(a.start, len(a.index))
		head = append(head, len(a.start)-2)
	}
	for b := range blocks {
		column(b)
	}
	for _, b := range rnd.Perm(blocks)[:min(link, blocks)] {
		column(b)
	}
	// Slack columns fill the basis when there are fewer blocks than
	// linking rows.
	for i := len(head); i < link+blocks; i++ {
		a.index = append(a.index, i-blocks)
		a.value = append(a.value, 1)
		a.start = append(a.start, len(a.index))
		head = append(head, len(a.start)-2)
	}
	rnd.Shuffle(len(head), func(i, j int) { head[i], head[j] = head[j], head[i] })
	return a, head
}

// TestSparseLUColumnMax checks the cached column maxima of the pivot
// search against a scan of the column at every use, and the factors of a
// basis with dense rows against the basis.
func TestSparseLUColumnMax(t *testing.T) {
	t.Parallel()
	rnd := rand.New(rand.NewPCG(13, 13))
	uses := 0
	for k := range 400 {
		var a *columns
		var head []int
		if k%4 == 0 {
			a, head = masterBasis(rnd, 1+rnd.IntN(12), 1+rnd.IntN(40))
		} else {
			a, head = randomBasis(rnd, 1+rnd.IntN(60), false)
		}
		m := len(head)
		f := newSparseLU(m)
		f.audit = func(j int) {
			uses++
			if want := f.colScan(j); f.colBig[j] != want {
				t.Fatalf("case %d: column %d cached maximum %v, scan %v", k, j, f.colBig[j], want)
			}
		}
		if f.factorize(a, head) != nil {
			continue
		}
		z := make([]float64, m)
		for i := range z {
			z[i] = rnd.NormFloat64()
		}
		x := append([]float64(nil), z...)
		f.ftran(x)
		y := append([]float64(nil), z...)
		f.btran(y)
		if r := residual(a, head, x, z, false); r > 1e-8 {
			t.Fatalf("case %d: ftran residual %g", k, r)
		}
		if r := residual(a, head, y, z, true); r > 1e-8 {
			t.Fatalf("case %d: btran residual %g", k, r)
		}
	}
	if uses < 10000 {
		t.Fatalf("only %d uses of the cache", uses)
	}
}

func BenchmarkSparseLUMasterBasis(b *testing.B) {
	for _, size := range [][2]int{{48, 73}, {24, 48}, {24, 200}, {24, 610}} {
		a, head := masterBasis(rand.New(rand.NewPCG(17, 17)), size[0], size[1])
		f := newSparseLU(len(head))
		b.Run(fmt.Sprintf("periods=%d/blocks=%d", size[0], size[1]), func(b *testing.B) {
			for b.Loop() {
				if err := f.factorize(a, head); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
