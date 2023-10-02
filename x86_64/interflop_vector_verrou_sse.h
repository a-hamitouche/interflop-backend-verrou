
/*--------------------------------------------------------------------*/
/*--- Verrou: a FPU instrumentation tool.                          ---*/
/*--- Interface for floating-point operations overloading.         ---*/
/*---                                           interflop_verrou.h ---*/
/*--------------------------------------------------------------------*/

/*
   This file is part of Verrou, a FPU instrumentation tool.

   Copyright (C) 2014-2021 EDF
     F. Févotte     <francois.fevotte@edf.fr>
     B. Lathuilière <bruno.lathuiliere@edf.fr>

   This program is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
   02111-1307, USA.

   The GNU Lesser General Public License is contained in the file COPYING.
*/

#ifndef __INTERFLOP_VECTOR_VERROU_SSE_H
#define __INTERFLOP_VECTOR_VERROU_SSE_H


#ifdef __cplusplus
extern "C" {
#endif

#define INTERFLOP_VECTOR_VERROU_API(FCT) interflop_vector_verrou_##FCT##_sse

void INTERFLOP_VECTOR_VERROU_API(add_float_1)(float *a, float *b, float *c,
                                          void *context);
void INTERFLOP_VECTOR_VERROU_API(add_float_4)(float *a, float *b, float *c,
                                          void *context);
void INTERFLOP_VECTOR_VERROU_API(add_float_8)(float *a, float *b, float *c,
                                          void *context);
void INTERFLOP_VECTOR_VERROU_API(add_float_16)(float *a, float *b, float *c,
                                          void *context);
void INTERFLOP_VECTOR_VERROU_API(sub_float_1)(float *a, float *b, float *c,
                                          void *context);
void INTERFLOP_VECTOR_VERROU_API(sub_float_4)(float *a, float *b, float *c,
                                          void *context);
void INTERFLOP_VECTOR_VERROU_API(sub_float_8)(float *a, float *b, float *c,
                                          void *context);
void INTERFLOP_VECTOR_VERROU_API(sub_float_16)(float *a, float *b, float *c,
                                          void *context);
void INTERFLOP_VECTOR_VERROU_API(mul_float_1)(float *a, float *b, float *c,
                                          void *context);
void INTERFLOP_VECTOR_VERROU_API(mul_float_4)(float *a, float *b, float *c,
                                          void *context);
void INTERFLOP_VECTOR_VERROU_API(mul_float_8)(float *a, float *b, float *c,
                                          void *context);
void INTERFLOP_VECTOR_VERROU_API(mul_float_16)(float *a, float *b, float *c,
                                          void *context);
void INTERFLOP_VECTOR_VERROU_API(div_float_1)(float *a, float *b, float *c,
                                          void *context);
void INTERFLOP_VECTOR_VERROU_API(div_float_4)(float *a, float *b, float *c,
                                          void *context);
void INTERFLOP_VECTOR_VERROU_API(div_float_8)(float *a, float *b, float *c,
                                          void *context);
void INTERFLOP_VECTOR_VERROU_API(div_float_16)(float *a, float *b, float *c,
                                          void *context);
struct interflop_vector_type_t INTERFLOP_VECTOR_VERROU_API(init)(void *context);

#ifdef __cplusplus
}
#endif

#endif