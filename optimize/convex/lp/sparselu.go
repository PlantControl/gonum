// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lp

import "math"

const (
	// markowitzThreshold is the threshold partial pivoting factor: a pivot
	// must be at least this fraction of the largest magnitude in its column
	// of the active submatrix, which bounds the multipliers by its inverse.
	markowitzThreshold = 0.1
	// markowitzSearch is how many rows or columns are examined for a pivot
	// once one is available (Suhl and Suhl, ORSA J. Computing 2(4), 1990).
	markowitzSearch = 4
	// dropTol drops an updated entry that has cancelled to this fraction of
	// the magnitudes that formed it.
	dropTol = 1e-14
)

// sparseLU factorizes the basis by Gaussian elimination with Markowitz
// pivot selection, so the cost of factorizing and solving follows the
// number of nonzeros in the factors rather than m². Updates are product
// form etas.
//
// Pivot k is matrix row prow[k] and basis position pcol[k]. L is stored as
// one column of multipliers per pivot, and U by rows: row prow[k] holds the
// diagonal and entries in the positions pivoted after k.
type sparseLU struct {
	m          int
	prow, pcol []int
	lStart     []int
	lIndex     []int // matrix rows
	lValue     []float64
	uDiag      []float64
	uStart     []int
	uIndex     []int // basis positions
	uValue     []float64
	work       []float64
	etas       etaFile

	// Active submatrix during factorization.
	rowIdx  [][]int // basis positions
	rowVal  [][]float64
	colIdx  [][]int // matrix rows
	rows    countList
	cols    countList
	scatter []float64
	stamp   []int
	seen    []int
}

func newSparseLU(m int) *sparseLU {
	return &sparseLU{
		m:       m,
		prow:    make([]int, m),
		pcol:    make([]int, m),
		uDiag:   make([]float64, m),
		work:    make([]float64, m),
		rowIdx:  make([][]int, m),
		rowVal:  make([][]float64, m),
		colIdx:  make([][]int, m),
		rows:    newCountList(m),
		cols:    newCountList(m),
		scatter: make([]float64, m),
		stamp:   make([]int, m),
		seen:    make([]int, m),
	}
}

func (f *sparseLU) factorize(a *columns, head []int) error {
	f.etas.reset()
	m := f.m
	if m == 0 {
		return nil
	}
	for i := range m {
		f.rowIdx[i], f.rowVal[i], f.colIdx[i] = f.rowIdx[i][:0], f.rowVal[i][:0], f.colIdx[i][:0]
		f.stamp[i], f.seen[i] = -1, -1
	}
	for pos, j := range head {
		for k := a.start[j]; k < a.start[j+1]; k++ {
			r := a.index[k]
			f.rowIdx[r] = append(f.rowIdx[r], pos)
			f.rowVal[r] = append(f.rowVal[r], a.value[k])
			f.colIdx[pos] = append(f.colIdx[pos], r)
		}
	}
	f.rows.reset()
	f.cols.reset()
	for i := range m {
		if len(f.rowIdx[i]) == 0 || len(f.colIdx[i]) == 0 {
			return ErrSingular
		}
		f.rows.insert(i, len(f.rowIdx[i]))
		f.cols.insert(i, len(f.colIdx[i]))
	}
	f.lStart = append(f.lStart[:0], 0)
	f.lIndex, f.lValue = f.lIndex[:0], f.lValue[:0]
	f.uStart = append(f.uStart[:0], 0)
	f.uIndex, f.uValue = f.uIndex[:0], f.uValue[:0]
	pass := 0
	for k := range m {
		p, q := f.choose()
		if p < 0 {
			return ErrSingular
		}
		f.prow[k], f.pcol[k] = p, q
		f.rows.remove(p)
		f.cols.remove(q)
		// Record the pivot row in U and scatter it.
		for t, j := range f.rowIdx[p] {
			v := f.rowVal[p][t]
			if j == q {
				f.uDiag[k] = v
				continue
			}
			f.uIndex = append(f.uIndex, j)
			f.uValue = append(f.uValue, v)
			f.scatter[j] = v
			f.stamp[j] = k
			f.colIdx[j] = removeInt(f.colIdx[j], p)
		}
		f.uStart = append(f.uStart, len(f.uIndex))
		diag := f.uDiag[k]
		// Eliminate the pivot column from the other active rows.
		for _, i := range f.colIdx[q] {
			if i == p {
				continue
			}
			pass++
			idx, val := f.rowIdx[i], f.rowVal[i]
			l := 0.0
			for t := 0; t < len(idx); {
				j := idx[t]
				switch {
				case j == q:
					l = val[t] / diag
					idx[t], val[t] = idx[len(idx)-1], val[len(val)-1]
					idx, val = idx[:len(idx)-1], val[:len(val)-1]
					continue
				case f.stamp[j] == k:
					f.seen[j] = pass
				}
				t++
			}
			f.lIndex = append(f.lIndex, i)
			f.lValue = append(f.lValue, l)
			for t := 0; t < len(idx); {
				j := idx[t]
				if f.stamp[j] == k {
					d := l * f.scatter[j]
					v := val[t] - d
					if math.Abs(v) <= dropTol*math.Max(math.Abs(val[t]), math.Abs(d)) {
						idx[t], val[t] = idx[len(idx)-1], val[len(val)-1]
						idx, val = idx[:len(idx)-1], val[:len(val)-1]
						f.colIdx[j] = removeInt(f.colIdx[j], i)
						continue
					}
					val[t] = v
				}
				t++
			}
			for t := f.uStart[k]; t < f.uStart[k+1]; t++ {
				j := f.uIndex[t]
				if f.seen[j] == pass {
					continue
				}
				idx = append(idx, j)
				val = append(val, -l*f.scatter[j])
				f.colIdx[j] = append(f.colIdx[j], i)
			}
			f.rowIdx[i], f.rowVal[i] = idx, val
			f.rows.move(i, len(idx))
		}
		f.lStart = append(f.lStart, len(f.lIndex))
		f.colIdx[q] = f.colIdx[q][:0]
		for t := f.uStart[k]; t < f.uStart[k+1]; t++ {
			j := f.uIndex[t]
			f.cols.move(j, len(f.colIdx[j]))
		}
	}
	return nil
}

// choose returns a pivot of the active submatrix that passes the threshold
// test with the smallest Markowitz count found, or -1 if none exists.
func (f *sparseLU) choose() (p, q int) {
	p, q = -1, -1
	best, searched := math.MaxInt, 0
	consider := func(i, j int, v, colMax float64) {
		if v == 0 || math.Abs(v) < markowitzThreshold*colMax {
			return
		}
		if mk := (len(f.rowIdx[i]) - 1) * (len(f.colIdx[j]) - 1); mk < best {
			best, p, q = mk, i, j
		}
	}
	for c := 1; c <= f.m; c++ {
		for j := f.cols.first(c); j >= 0; j = f.cols.next[j] {
			colMax := f.colMax(j)
			for _, i := range f.colIdx[j] {
				consider(i, j, f.lookup(i, j), colMax)
			}
			if searched++; p >= 0 && (best == 0 || searched >= markowitzSearch) {
				return p, q
			}
		}
		for i := f.rows.first(c); i >= 0; i = f.rows.next[i] {
			for t, j := range f.rowIdx[i] {
				consider(i, j, f.rowVal[i][t], f.colMax(j))
			}
			if searched++; p >= 0 && (best == 0 || searched >= markowitzSearch) {
				return p, q
			}
		}
	}
	return p, q
}

func (f *sparseLU) lookup(i, j int) float64 {
	for t, jj := range f.rowIdx[i] {
		if jj == j {
			return f.rowVal[i][t]
		}
	}
	return 0
}

func (f *sparseLU) colMax(j int) float64 {
	big := 0.0
	for _, i := range f.colIdx[j] {
		big = math.Max(big, math.Abs(f.lookup(i, j)))
	}
	return big
}

func removeInt(s []int, v int) []int {
	for t, x := range s {
		if x == v {
			s[t] = s[len(s)-1]
			return s[:len(s)-1]
		}
	}
	return s
}

func (f *sparseLU) ftran(z []float64) {
	if f.m == 0 {
		return
	}
	w := f.work
	copy(w, z)
	for k, p := range f.prow {
		if wp := w[p]; wp != 0 {
			for t := f.lStart[k]; t < f.lStart[k+1]; t++ {
				w[f.lIndex[t]] -= f.lValue[t] * wp
			}
		}
	}
	for k := f.m - 1; k >= 0; k-- {
		sum := w[f.prow[k]]
		for t := f.uStart[k]; t < f.uStart[k+1]; t++ {
			sum -= f.uValue[t] * z[f.uIndex[t]]
		}
		z[f.pcol[k]] = sum / f.uDiag[k]
	}
	f.etas.ftran(z)
}

func (f *sparseLU) btran(v []float64) {
	if f.m == 0 {
		return
	}
	f.etas.btran(v)
	w := f.work
	for k, q := range f.pcol {
		zk := v[q] / f.uDiag[k]
		w[f.prow[k]] = zk
		if zk != 0 {
			for t := f.uStart[k]; t < f.uStart[k+1]; t++ {
				v[f.uIndex[t]] -= f.uValue[t] * zk
			}
		}
	}
	for k := f.m - 1; k >= 0; k-- {
		p := f.prow[k]
		sum := w[p]
		for t := f.lStart[k]; t < f.lStart[k+1]; t++ {
			sum -= f.lValue[t] * w[f.lIndex[t]]
		}
		w[p] = sum
	}
	copy(v, w)
}

func (f *sparseLU) update(row int, alpha []float64) bool {
	f.etas.add(row, alpha)
	return f.etas.len() >= refactorEvery
}

// countList keeps items in doubly linked lists by count, for finding rows
// and columns with few nonzeros in constant time.
type countList struct {
	head, next, prev, count []int
}

func newCountList(m int) countList {
	l := countList{head: make([]int, m+1), next: make([]int, m), prev: make([]int, m), count: make([]int, m)}
	l.reset()
	return l
}

func (l *countList) reset() {
	for c := range l.head {
		l.head[c] = -1
	}
	for i := range l.count {
		l.count[i] = -1
	}
}

func (l *countList) first(c int) int { return l.head[c] }

func (l *countList) insert(i, c int) {
	l.count[i] = c
	l.prev[i] = -1
	l.next[i] = l.head[c]
	if l.head[c] >= 0 {
		l.prev[l.head[c]] = i
	}
	l.head[c] = i
}

func (l *countList) remove(i int) {
	c := l.count[i]
	if c < 0 {
		return
	}
	if l.prev[i] >= 0 {
		l.next[l.prev[i]] = l.next[i]
	} else {
		l.head[c] = l.next[i]
	}
	if l.next[i] >= 0 {
		l.prev[l.next[i]] = l.prev[i]
	}
	l.count[i] = -1
}

// move changes the count of an item still in the lists.
func (l *countList) move(i, c int) {
	if l.count[i] < 0 || l.count[i] == c {
		return
	}
	l.remove(i)
	l.insert(i, c)
}
