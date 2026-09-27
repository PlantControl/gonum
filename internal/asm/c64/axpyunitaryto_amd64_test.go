// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !noasm && !gccgo && !safe

package c64_test

import (
	"fmt"
	"slices"
	"testing"
	"unsafe"

	"plantcontrol.org/v1/gonum/internal/asm/c64"
)

func TestAxpyUnitaryToAlignment(t *testing.T) {
	testAxpyAlignment(t, true)
}

func TestAxpyUnitaryAlignment(t *testing.T) {
	testAxpyAlignment(t, false)
}

func testAxpyAlignment(t *testing.T, to bool) {
	aliases := []string{"y"}
	if to {
		aliases = []string{"none", "x", "y"}
	}
	// A single element at an address congruent to 8 modulo 16 is fully
	// consumed by the alignment trim. It must not enter the scalar tail with
	// zero elements remaining, which would underflow the assembly loop count.
	for _, n := range []int{0, 1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33} {
		for _, alignment := range []uintptr{0, 4, 8, 12} {
			for _, alias := range aliases {
				t.Run(fmt.Sprintf("n=%d/alignment=%d/alias=%s", n, alignment, alias), func(t *testing.T) {
					// Include guards on both sides, and explicitly select y's
					// alignment instead of relying on the allocation history.
					ybuf := make([]complex64, n+3)
					if alignment&7 != 0 {
						// complex64 requires only 4-byte alignment. A preceding
						// float32 field produces it without an unsafe conversion.
						storage := new(struct {
							pad  float32
							data [36]complex64
						})
						ybuf = storage.data[:n+3]
					}
					start := 1
					if uintptr(unsafe.Pointer(&ybuf[start]))&15 != alignment {
						start++
					}
					if got := uintptr(unsafe.Pointer(&ybuf[start])) & 15; got != alignment {
						t.Fatalf("unexpected y alignment: got %d, want %d", got, alignment)
					}
					y := ybuf[start : start+n]
					xbuf := make([]complex64, n+2)
					x := xbuf[1 : n+1]
					dstbuf := make([]complex64, n+2)
					dst := dstbuf[1 : n+1]
					for i := range ybuf {
						ybuf[i] = complex(float32(i+3), float32(2*i-5))
					}
					for i := range xbuf {
						xbuf[i] = complex(float32(2*i-7), float32(i+1))
						dstbuf[i] = 123 - 456i
					}
					switch alias {
					case "x":
						dst = x
					case "y":
						dst = y
					}
					xwant, ywant, dstwant := slices.Clone(xbuf), slices.Clone(ybuf), slices.Clone(dstbuf)
					const alpha = complex64(2 - 3i)
					for i := range n {
						want := alpha*x[i] + y[i]
						switch alias {
						case "x":
							xwant[i+1] = want
						case "y":
							ywant[i+start] = want
						default:
							dstwant[i+1] = want
						}
					}
					if to {
						c64.AxpyUnitaryTo(dst, alpha, x, y)
					} else {
						c64.AxpyUnitary(alpha, x, y)
					}
					for _, check := range []struct {
						name      string
						got, want []complex64
					}{{"x", xbuf, xwant}, {"y", ybuf, ywant}, {"dst", dstbuf, dstwant}} {
						if !slices.Equal(check.got, check.want) {
							t.Errorf("unexpected %s buffer:\ngot  %v\nwant %v", check.name, check.got, check.want)
						}
					}
				})
			}
		}
	}
}
