
/*--------------------------------------------------------------------*/
/*--- Verrou: a FPU instrumentation tool.                          ---*/
/*--- Utilities for easier manipulation of floating-point values.  ---*/
/*---                                                vr_isNan.hxx ---*/
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

#pragma once

#include <cstdio>
#include <cfloat>
#include <stdint.h>
#include <immintrin.h>

#include "interflop/interflop_stdlib.h"
#include "interflop_verrou.h"

template <class REALTYPE> inline __m128i areNan(const REALTYPE &x) {
  interflop_panic("isNan called on an unknown type");
  return _mm_set1_epi32( 0);
}

/*
template <> inline __m128i isNan<double>(const double &x) {

  static const uint64_t maskSpecial = 0x7ff0000000000000;
  static const uint64_t maskInf = 0x000fffffffffffff;
  const uint64_t *X = reinterpret_cast<const uint64_t *>(&x);
  if ((*X & maskSpecial) == maskSpecial) {
    if ((*X & maskInf) != 0) {
      return true;
    }
  }
  return false;
}


template <> inline __m128i areNan<__m128>(const __m128 &x) {

  static const __m128i maskSpecial = _mm_set1_epi32(0x7f800000);
  static const __m128i maskInf = _mm_set1_epi32(0x007fffff);
  const __m128i X = _mm_castps_si128 (x);
  if ((*X & maskSpecial) == maskSpecial) {
    if ((*X & maskInf) != 0) {
      return true;
    }
  }
  return false;
}
*/
template <class REALTYPE> inline __m128i hasNanInf(const REALTYPE &x) {
  interflop_panic("isNanInf called on an unknown type");
  return _mm_set1_epi32( 0);
}


#if defined(__SSE4_2__)
template <> inline __m128i hasNanInf<__m128d>(const __m128d &x) {
  static const __m128i mask = _mm_set1_epi64x (0x7ff0000000000000);
  const __m128i X = _mm_castpd_si128(x);
  return _mm_cmpeq_epi64 (_mm_and_si128 (X, mask), mask);
}

template <> inline __m128i hasNanInf<__m128>(const __m128 &x) {

  static const __m128i v_mask = _mm_set1_epi32 (0x7f800000);
  const __m128i  v_X = _mm_castps_si128 (x);
  return _mm_cmpeq_epi32 (_mm_and_si128 ( v_X, v_mask), v_mask);
}
#endif

#if defined(__AVX2__)
template <> inline __m128i hasNanInf<__m256>(const __m256 &x) {
  static const __m256i v_mask = _mm256_set1_epi32 (0x7f800000);
  const __m256i  v_X = _mm256_castps_si256 (x);
  __m256i eqNan = _mm256_cmpeq_epi32 (_mm256_and_si256 ( v_X, v_mask), v_mask);

  uint32_t acc_gather[8];
  uint16_t acc_reduce[8]; 
  _mm256_storeu_si256 ( (__m256i_u *)acc_gather, eqNan);
  for (int i = 0; i < 8; i++) acc_reduce[i] = acc_gather[i];

  return _mm_loadu_si128((__m128i_u *)acc_reduce);
}
#endif
