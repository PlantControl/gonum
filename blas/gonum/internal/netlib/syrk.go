// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package netlib

/*
#cgo CFLAGS: -I/opt/homebrew/opt/lapack/include
#cgo LDFLAGS: -L/opt/homebrew/opt/lapack/lib -Wl,-rpath,/opt/homebrew/opt/lapack/lib -lblas
#include <lapack.h>
#include <stdint.h>
#include <stddef.h>
_Static_assert(sizeof(lapack_int) == sizeof(int32_t), "SYRK bridge requires LP64");
extern void dsyrk_(char*, char*, lapack_int*, lapack_int*, double*, double*,
 lapack_int*, double*, double*, lapack_int*, size_t, size_t);
extern void ssyrk_(char*, char*, lapack_int*, lapack_int*, float*, float*,
 lapack_int*, float*, float*, lapack_int*, size_t, size_t);
static void run_dsyrk(char ul, char trans, lapack_int n, lapack_int k,
 double alpha, double *a, lapack_int lda, double beta, double *c, lapack_int ldc) {
 dsyrk_(&ul,&trans,&n,&k,&alpha,a,&lda,&beta,c,&ldc,1,1);
}
static void run_ssyrk(char ul, char trans, lapack_int n, lapack_int k,
 float alpha, float *a, lapack_int lda, float beta, float *c, lapack_int ldc) {
 ssyrk_(&ul,&trans,&n,&k,&alpha,a,&lda,&beta,c,&ldc,1,1);
}
*/
import "C"

import (
	"unsafe"

	"plantcontrol.org/v1/gonum/blas"
)

func syrkLayout(ul blas.Uplo, trans blas.Transpose) (byte, byte) {
	if ul == blas.Upper {
		ul = blas.Lower
	} else {
		ul = blas.Upper
	}
	if trans == blas.NoTrans {
		trans = blas.Trans
	} else {
		trans = blas.NoTrans
	}
	return byte(ul), byte(trans)
}

// Dsyrk computes a row-major symmetric rank-k update using the reference BLAS.
func (Implementation) Dsyrk(ul blas.Uplo, trans blas.Transpose, n, k int, alpha float64, a []float64, lda int, beta float64, c []float64, ldc int) {
	cul, ctrans := syrkLayout(ul, trans)
	C.run_dsyrk(C.char(cul), C.char(ctrans), C.lapack_int(n), C.lapack_int(k), C.double(alpha),
		(*C.double)(unsafe.Pointer(unsafe.SliceData(a))), C.lapack_int(lda), C.double(beta),
		(*C.double)(unsafe.Pointer(unsafe.SliceData(c))), C.lapack_int(ldc))
}

// Ssyrk computes a row-major symmetric rank-k update using the reference BLAS.
func (Implementation) Ssyrk(ul blas.Uplo, trans blas.Transpose, n, k int, alpha float32, a []float32, lda int, beta float32, c []float32, ldc int) {
	cul, ctrans := syrkLayout(ul, trans)
	C.run_ssyrk(C.char(cul), C.char(ctrans), C.lapack_int(n), C.lapack_int(k), C.float(alpha),
		(*C.float)(unsafe.Pointer(unsafe.SliceData(a))), C.lapack_int(lda), C.float(beta),
		(*C.float)(unsafe.Pointer(unsafe.SliceData(c))), C.lapack_int(ldc))
}
