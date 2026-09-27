// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

// Package netlib exposes the Homebrew reference BLAS as an optional benchmark oracle.
package netlib

/*
#cgo CFLAGS: -I/opt/homebrew/opt/lapack/include
#cgo LDFLAGS: -L/opt/homebrew/opt/lapack/lib -Wl,-rpath,/opt/homebrew/opt/lapack/lib -lblas
#include <complex.h>
#include <stdint.h>

typedef int32_t fint;
typedef struct { double sum, a, b, c, s, p[5]; fint idx; } real_result;
typedef struct { double sumr, sumi, real; fint idx; } complex_result;

#define DOT(NAME,T,R) \
extern R NAME##_(fint*,T*,fint*,T*,fint*); \
static real_result run_##NAME(fint q,fint n,T*x,fint ix,T*y,fint iy) { \
	real_result z={0}; for(fint i=0;i<q;i++) z.sum+=NAME##_(&n,x,&ix,y,&iy); return z; }
#define ONE(NAME,T,R) \
extern R NAME##_(fint*,T*,fint*); \
static real_result run_##NAME(fint q,fint n,T*x,fint ix) { \
	real_result z={0}; for(fint i=0;i<q;i++) z.sum+=NAME##_(&n,x,&ix); return z; }
#define IAMAX(NAME,T) \
extern fint NAME##_(fint*,T*,fint*); \
static real_result run_##NAME(fint q,fint n,T*x,fint ix) { \
	real_result z={0}; for(fint i=0;i<q;i++) z.idx+=NAME##_(&n,x,&ix); return z; }
#define MUT2(NAME,T) \
extern void NAME##_(fint*,T*,fint*,T*,fint*); \
static void run_##NAME(fint q,fint n,T*x,fint ix,T*y,fint iy) { \
	for(fint i=0;i<q;i++) NAME##_(&n,x,&ix,y,&iy); }
#define AXPY(NAME,T) \
extern void NAME##_(fint*,T*,T*,fint*,T*,fint*); \
static void run_##NAME(fint q,fint n,T a,T*x,fint ix,T*y,fint iy) { \
	for(fint i=0;i<q;i++) { T v=(i&1)?-a:a; NAME##_(&n,&v,x,&ix,y,&iy); } }
#define SCAL(NAME,T,A) \
extern void NAME##_(fint*,A*,T*,fint*); \
static void run_##NAME(fint q,fint n,A a,T*x,fint ix) { \
	for(fint i=0;i<q;i++) NAME##_(&n,&a,x,&ix); }
#define ROT(NAME,T) \
extern void NAME##_(fint*,T*,fint*,T*,fint*,T*,T*); \
static void run_##NAME(fint q,fint n,T*x,fint ix,T*y,fint iy,T c,T s) { \
	for(fint i=0;i<q;i++) NAME##_(&n,x,&ix,y,&iy,&c,&s); }

DOT(sdot,float,float) DOT(dsdot,float,double) DOT(ddot,double,double)
ONE(snrm2,float,float) ONE(sasum,float,float) ONE(dnrm2,double,double) ONE(dasum,double,double)
IAMAX(isamax,float) IAMAX(idamax,double)
MUT2(sswap,float) MUT2(scopy,float) MUT2(dswap,double) MUT2(dcopy,double)
AXPY(saxpy,float) AXPY(daxpy,double)
SCAL(sscal,float,float) SCAL(dscal,double,double)
ROT(srot,float) ROT(drot,double)

extern float sdsdot_(fint*,float*,float*,fint*,float*,fint*);
static real_result run_sdsdot(fint q,fint n,float a,float*x,fint ix,float*y,fint iy){real_result z={0};for(fint i=0;i<q;i++)z.sum+=sdsdot_(&n,&a,x,&ix,y,&iy);return z;}
extern void srotg_(float*,float*,float*,float*); extern void drotg_(double*,double*,double*,double*);
static real_result run_srotg(fint q,float a,float b){real_result z={0};for(fint i=0;i<q;i++){float x=a,y=b,c,s;srotg_(&x,&y,&c,&s);z.a=x;z.b=y;z.c=c;z.s=s;}return z;}
static real_result run_drotg(fint q,double a,double b){real_result z={0};for(fint i=0;i<q;i++){z.a=a;z.b=b;drotg_(&z.a,&z.b,&z.c,&z.s);}return z;}
extern void srotmg_(float*,float*,float*,float*,float*); extern void drotmg_(double*,double*,double*,double*,double*);
static void run_srotmg(fint q,float*d1,float*d2,float*x1,float y1,float*p){float i1=*d1,i2=*d2,ix=*x1;for(fint i=0;i<q;i++){float a=i1,b=i2,c=ix;srotmg_(&a,&b,&c,&y1,p);*d1=a;*d2=b;*x1=c;}}
static void run_drotmg(fint q,double*d1,double*d2,double*x1,double y1,double*p){double i1=*d1,i2=*d2,ix=*x1;for(fint i=0;i<q;i++){double a=i1,b=i2,c=ix;drotmg_(&a,&b,&c,&y1,p);*d1=a;*d2=b;*x1=c;}}
extern void srotm_(fint*,float*,fint*,float*,fint*,float*); extern void drotm_(fint*,double*,fint*,double*,fint*,double*);
static void run_srotm(fint q,fint n,float*x,fint ix,float*y,fint iy,float*p){for(fint i=0;i<q;i++)srotm_(&n,x,&ix,y,&iy,p);}
static void run_drotm(fint q,fint n,double*x,fint ix,double*y,fint iy,double*p){for(fint i=0;i<q;i++)drotm_(&n,x,&ix,y,&iy,p);}

#define CDOT(NAME,T,CREAL,CIMAG) \
extern T NAME##_(fint*,T*,fint*,T*,fint*); \
static complex_result run_##NAME(fint q,fint n,T*x,fint ix,T*y,fint iy) { \
	complex_result z={0}; for(fint i=0;i<q;i++) { T v=NAME##_(&n,x,&ix,y,&iy); z.sumr+=CREAL(v); z.sumi+=CIMAG(v); } return z; }
#define CONE(NAME,T,R) \
extern R NAME##_(fint*,T*,fint*); \
static complex_result run_##NAME(fint q,fint n,T*x,fint ix) { \
	complex_result z={0}; for(fint i=0;i<q;i++) z.real+=NAME##_(&n,x,&ix); return z; }
#define CIAMAX(NAME,T) \
extern fint NAME##_(fint*,T*,fint*); \
static complex_result run_##NAME(fint q,fint n,T*x,fint ix) { \
	complex_result z={0}; for(fint i=0;i<q;i++) z.idx+=NAME##_(&n,x,&ix); return z; }
#define CMUT2(NAME,T) MUT2(NAME,T)
#define CAXPY(NAME,T) AXPY(NAME,T)
#define CSCAL(NAME,T,A) SCAL(NAME,T,A)
CDOT(cdotu,float _Complex,crealf,cimagf) CDOT(cdotc,float _Complex,crealf,cimagf)
CDOT(zdotu,double _Complex,creal,cimag) CDOT(zdotc,double _Complex,creal,cimag)
CONE(scnrm2,float _Complex,float) CONE(scasum,float _Complex,float)
CONE(dznrm2,double _Complex,double) CONE(dzasum,double _Complex,double)
CIAMAX(icamax,float _Complex) CIAMAX(izamax,double _Complex)
CMUT2(cswap,float _Complex) CMUT2(ccopy,float _Complex) CMUT2(zswap,double _Complex) CMUT2(zcopy,double _Complex)
CAXPY(caxpy,float _Complex) CAXPY(zaxpy,double _Complex)
CSCAL(cscal,float _Complex,float _Complex) CSCAL(csscal,float _Complex,float)
CSCAL(zscal,double _Complex,double _Complex) CSCAL(zdscal,double _Complex,double)
*/
import "C"

import (
	"unsafe"

	"plantcontrol.org/v1/gonum/blas"
)

// Implementation calls the reference BLAS. Repeat greater than one batches
// benchmark work in C; axpy alternates alpha and -alpha and reductions return
// accumulated checksums. Callers must keep Repeat and repeated IAMAX checksum
// values within int32. Repeat less than two has exact one-call BLAS semantics.
type Implementation struct{ Repeat int }

func (i Implementation) repeats() C.fint {
	if i.Repeat > 1 {
		return C.fint(i.Repeat)
	}
	return 1
}
func fp32(x []float32) *C.float {
	if len(x) == 0 {
		return nil
	}
	return (*C.float)(unsafe.Pointer(&x[0]))
}
func fp64(x []float64) *C.double {
	if len(x) == 0 {
		return nil
	}
	return (*C.double)(unsafe.Pointer(&x[0]))
}
func cp64(x []complex64) *C.complexfloat {
	if len(x) == 0 {
		return nil
	}
	return (*C.complexfloat)(unsafe.Pointer(&x[0]))
}
func cp128(x []complex128) *C.complexdouble {
	if len(x) == 0 {
		return nil
	}
	return (*C.complexdouble)(unsafe.Pointer(&x[0]))
}

func (i Implementation) Sdot(n int, x []float32, ix int, y []float32, iy int) float32 {
	return float32(C.run_sdot(i.repeats(), C.fint(n), fp32(x), C.fint(ix), fp32(y), C.fint(iy)).sum)
}
func (i Implementation) Dsdot(n int, x []float32, ix int, y []float32, iy int) float64 {
	return float64(C.run_dsdot(i.repeats(), C.fint(n), fp32(x), C.fint(ix), fp32(y), C.fint(iy)).sum)
}
func (i Implementation) Sdsdot(n int, a float32, x []float32, ix int, y []float32, iy int) float32 {
	return float32(C.run_sdsdot(i.repeats(), C.fint(n), C.float(a), fp32(x), C.fint(ix), fp32(y), C.fint(iy)).sum)
}
func (i Implementation) Snrm2(n int, x []float32, ix int) float32 {
	return float32(C.run_snrm2(i.repeats(), C.fint(n), fp32(x), C.fint(ix)).sum)
}
func (i Implementation) Sasum(n int, x []float32, ix int) float32 {
	return float32(C.run_sasum(i.repeats(), C.fint(n), fp32(x), C.fint(ix)).sum)
}
func (i Implementation) Isamax(n int, x []float32, ix int) int {
	if n <= 0 || ix <= 0 {
		return -1
	}
	return int(C.run_isamax(i.repeats(), C.fint(n), fp32(x), C.fint(ix)).idx) - i.RepeatOrOne()
}
func (i Implementation) Sswap(n int, x []float32, ix int, y []float32, iy int) {
	C.run_sswap(i.repeats(), C.fint(n), fp32(x), C.fint(ix), fp32(y), C.fint(iy))
}
func (i Implementation) Scopy(n int, x []float32, ix int, y []float32, iy int) {
	C.run_scopy(i.repeats(), C.fint(n), fp32(x), C.fint(ix), fp32(y), C.fint(iy))
}
func (i Implementation) Saxpy(n int, a float32, x []float32, ix int, y []float32, iy int) {
	C.run_saxpy(i.repeats(), C.fint(n), C.float(a), fp32(x), C.fint(ix), fp32(y), C.fint(iy))
}
func (i Implementation) Sscal(n int, a float32, x []float32, ix int) {
	C.run_sscal(i.repeats(), C.fint(n), C.float(a), fp32(x), C.fint(ix))
}
func (i Implementation) Srot(n int, x []float32, ix int, y []float32, iy int, c, s float32) {
	C.run_srot(i.repeats(), C.fint(n), fp32(x), C.fint(ix), fp32(y), C.fint(iy), C.float(c), C.float(s))
}
func (i Implementation) Srotg(a, b float32) (c, s, r, z float32) {
	v := C.run_srotg(i.repeats(), C.float(a), C.float(b))
	return float32(v.c), float32(v.s), float32(v.a), float32(v.b)
}
func (i Implementation) Srotmg(d1, d2, x1, y1 float32) (p blas.SrotmParams, rd1, rd2, rx1 float32) {
	q := i.repeats()
	a, b, x := C.float(d1), C.float(d2), C.float(x1)
	var v [5]C.float
	C.run_srotmg(q, &a, &b, &x, C.float(y1), &v[0])
	p.Flag = blas.Flag(v[0])
	for j := range p.H {
		p.H[j] = float32(v[j+1])
	}
	return p, float32(a), float32(b), float32(x)
}
func (i Implementation) Srotm(n int, x []float32, ix int, y []float32, iy int, p blas.SrotmParams) {
	v := [5]C.float{C.float(p.Flag), C.float(p.H[0]), C.float(p.H[1]), C.float(p.H[2]), C.float(p.H[3])}
	C.run_srotm(i.repeats(), C.fint(n), fp32(x), C.fint(ix), fp32(y), C.fint(iy), &v[0])
}

func (i Implementation) Ddot(n int, x []float64, ix int, y []float64, iy int) float64 {
	return float64(C.run_ddot(i.repeats(), C.fint(n), fp64(x), C.fint(ix), fp64(y), C.fint(iy)).sum)
}
func (i Implementation) Dnrm2(n int, x []float64, ix int) float64 {
	return float64(C.run_dnrm2(i.repeats(), C.fint(n), fp64(x), C.fint(ix)).sum)
}
func (i Implementation) Dasum(n int, x []float64, ix int) float64 {
	return float64(C.run_dasum(i.repeats(), C.fint(n), fp64(x), C.fint(ix)).sum)
}
func (i Implementation) Idamax(n int, x []float64, ix int) int {
	if n <= 0 || ix <= 0 {
		return -1
	}
	return int(C.run_idamax(i.repeats(), C.fint(n), fp64(x), C.fint(ix)).idx) - i.RepeatOrOne()
}
func (i Implementation) Dswap(n int, x []float64, ix int, y []float64, iy int) {
	C.run_dswap(i.repeats(), C.fint(n), fp64(x), C.fint(ix), fp64(y), C.fint(iy))
}
func (i Implementation) Dcopy(n int, x []float64, ix int, y []float64, iy int) {
	C.run_dcopy(i.repeats(), C.fint(n), fp64(x), C.fint(ix), fp64(y), C.fint(iy))
}
func (i Implementation) Daxpy(n int, a float64, x []float64, ix int, y []float64, iy int) {
	C.run_daxpy(i.repeats(), C.fint(n), C.double(a), fp64(x), C.fint(ix), fp64(y), C.fint(iy))
}
func (i Implementation) Dscal(n int, a float64, x []float64, ix int) {
	C.run_dscal(i.repeats(), C.fint(n), C.double(a), fp64(x), C.fint(ix))
}
func (i Implementation) Drot(n int, x []float64, ix int, y []float64, iy int, c, s float64) {
	C.run_drot(i.repeats(), C.fint(n), fp64(x), C.fint(ix), fp64(y), C.fint(iy), C.double(c), C.double(s))
}
func (i Implementation) Drotg(a, b float64) (c, s, r, z float64) {
	v := C.run_drotg(i.repeats(), C.double(a), C.double(b))
	return float64(v.c), float64(v.s), float64(v.a), float64(v.b)
}
func (i Implementation) Drotmg(d1, d2, x1, y1 float64) (p blas.DrotmParams, rd1, rd2, rx1 float64) {
	a, b, x := C.double(d1), C.double(d2), C.double(x1)
	var v [5]C.double
	C.run_drotmg(i.repeats(), &a, &b, &x, C.double(y1), &v[0])
	p.Flag = blas.Flag(v[0])
	for j := range p.H {
		p.H[j] = float64(v[j+1])
	}
	return p, float64(a), float64(b), float64(x)
}
func (i Implementation) Drotm(n int, x []float64, ix int, y []float64, iy int, p blas.DrotmParams) {
	v := [5]C.double{C.double(p.Flag), C.double(p.H[0]), C.double(p.H[1]), C.double(p.H[2]), C.double(p.H[3])}
	C.run_drotm(i.repeats(), C.fint(n), fp64(x), C.fint(ix), fp64(y), C.fint(iy), &v[0])
}
func (i Implementation) RepeatOrOne() int {
	if i.Repeat > 1 {
		return i.Repeat
	}
	return 1
}

func (i Implementation) Cdotu(n int, x []complex64, ix int, y []complex64, iy int) complex64 {
	v := C.run_cdotu(i.repeats(), C.fint(n), cp64(x), C.fint(ix), cp64(y), C.fint(iy))
	return complex(float32(v.sumr), float32(v.sumi))
}
func (i Implementation) Cdotc(n int, x []complex64, ix int, y []complex64, iy int) complex64 {
	v := C.run_cdotc(i.repeats(), C.fint(n), cp64(x), C.fint(ix), cp64(y), C.fint(iy))
	return complex(float32(v.sumr), float32(v.sumi))
}
func (i Implementation) Scnrm2(n int, x []complex64, ix int) float32 {
	return float32(C.run_scnrm2(i.repeats(), C.fint(n), cp64(x), C.fint(ix)).real)
}
func (i Implementation) Scasum(n int, x []complex64, ix int) float32 {
	return float32(C.run_scasum(i.repeats(), C.fint(n), cp64(x), C.fint(ix)).real)
}
func (i Implementation) Icamax(n int, x []complex64, ix int) int {
	if n <= 0 || ix <= 0 {
		return -1
	}
	return int(C.run_icamax(i.repeats(), C.fint(n), cp64(x), C.fint(ix)).idx) - i.RepeatOrOne()
}
func (i Implementation) Cswap(n int, x []complex64, ix int, y []complex64, iy int) {
	C.run_cswap(i.repeats(), C.fint(n), cp64(x), C.fint(ix), cp64(y), C.fint(iy))
}
func (i Implementation) Ccopy(n int, x []complex64, ix int, y []complex64, iy int) {
	C.run_ccopy(i.repeats(), C.fint(n), cp64(x), C.fint(ix), cp64(y), C.fint(iy))
}
func (i Implementation) Caxpy(n int, a complex64, x []complex64, ix int, y []complex64, iy int) {
	ca := *(*C.complexfloat)(unsafe.Pointer(&a))
	C.run_caxpy(i.repeats(), C.fint(n), ca, cp64(x), C.fint(ix), cp64(y), C.fint(iy))
}
func (i Implementation) Cscal(n int, a complex64, x []complex64, ix int) {
	ca := *(*C.complexfloat)(unsafe.Pointer(&a))
	C.run_cscal(i.repeats(), C.fint(n), ca, cp64(x), C.fint(ix))
}
func (i Implementation) Csscal(n int, a float32, x []complex64, ix int) {
	C.run_csscal(i.repeats(), C.fint(n), C.float(a), cp64(x), C.fint(ix))
}

func (i Implementation) Zdotu(n int, x []complex128, ix int, y []complex128, iy int) complex128 {
	v := C.run_zdotu(i.repeats(), C.fint(n), cp128(x), C.fint(ix), cp128(y), C.fint(iy))
	return complex(float64(v.sumr), float64(v.sumi))
}
func (i Implementation) Zdotc(n int, x []complex128, ix int, y []complex128, iy int) complex128 {
	v := C.run_zdotc(i.repeats(), C.fint(n), cp128(x), C.fint(ix), cp128(y), C.fint(iy))
	return complex(float64(v.sumr), float64(v.sumi))
}
func (i Implementation) Dznrm2(n int, x []complex128, ix int) float64 {
	return float64(C.run_dznrm2(i.repeats(), C.fint(n), cp128(x), C.fint(ix)).real)
}
func (i Implementation) Dzasum(n int, x []complex128, ix int) float64 {
	return float64(C.run_dzasum(i.repeats(), C.fint(n), cp128(x), C.fint(ix)).real)
}
func (i Implementation) Izamax(n int, x []complex128, ix int) int {
	if n <= 0 || ix <= 0 {
		return -1
	}
	return int(C.run_izamax(i.repeats(), C.fint(n), cp128(x), C.fint(ix)).idx) - i.RepeatOrOne()
}
func (i Implementation) Zswap(n int, x []complex128, ix int, y []complex128, iy int) {
	C.run_zswap(i.repeats(), C.fint(n), cp128(x), C.fint(ix), cp128(y), C.fint(iy))
}
func (i Implementation) Zcopy(n int, x []complex128, ix int, y []complex128, iy int) {
	C.run_zcopy(i.repeats(), C.fint(n), cp128(x), C.fint(ix), cp128(y), C.fint(iy))
}
func (i Implementation) Zaxpy(n int, a complex128, x []complex128, ix int, y []complex128, iy int) {
	ca := *(*C.complexdouble)(unsafe.Pointer(&a))
	C.run_zaxpy(i.repeats(), C.fint(n), ca, cp128(x), C.fint(ix), cp128(y), C.fint(iy))
}
func (i Implementation) Zscal(n int, a complex128, x []complex128, ix int) {
	ca := *(*C.complexdouble)(unsafe.Pointer(&a))
	C.run_zscal(i.repeats(), C.fint(n), ca, cp128(x), C.fint(ix))
}
func (i Implementation) Zdscal(n int, a float64, x []complex128, ix int) {
	C.run_zdscal(i.repeats(), C.fint(n), C.double(a), cp128(x), C.fint(ix))
}

var _ blas.Float32Level1 = Implementation{}
var _ blas.Float64Level1 = Implementation{}
var _ blas.Complex64Level1 = Implementation{}
var _ blas.Complex128Level1 = Implementation{}
