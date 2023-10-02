
/*--------------------------------------------------------------------*/
/*--- Verrou: a FPU instrumentation tool.                          ---*/
/*--- Utilities for easier manipulation of floating-point values.  ---*/
/*---                                                vr_nextUlp.hxx ---*/
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
#include <immintrin.h>

#include "../vr_nextUlp.hxx"

#if defined(__SSE2__)
template<> inline __m128 nextAwayFromZero<__m128>(__m128 a) {
  static const __m128i c_one128 = _mm_set1_epi32 (1);
  __m128 x = a;
  __m128i u = _mm_castps_si128 (x);
  u = _mm_add_epi32 (u, c_one128);
  x = _mm_castsi128_ps (u);
  return x;
}
#endif

#if defined(__AVX2__)
template<> inline __m256 nextAwayFromZero<__m256>(__m256 a) {
  static const __m256i c_one256 = _mm256_set1_epi32 (1);
  __m256 x = a;
  __m256i u = _mm256_castps_si256 (x);
  u = _mm256_add_epi32 (u, c_one256);
  x = _mm256_castsi256_ps (u);
  return x;
}
#endif


#if defined(__SSE2__)
template<> inline __m128 nextTowardZero<__m128>(__m128 a) {
  static const __m128i c_one128 = _mm_set1_epi32 (1);
  __m128 x = a;
  __m128i u = _mm_castps_si128 (x);
  u = _mm_sub_epi32 (u, c_one128);
  x = _mm_castsi128_ps (u);
  return x;
}
#endif

#if defined(__AVX2__)
template<> inline __m256 nextTowardZero<__m256>(__m256 a) {
  static const __m256i c_one256 = _mm256_set1_epi32 (1);
  __m256 x = a;
  __m256i u = _mm256_castps_si256 (x);
  u = _mm256_sub_epi32 (u, c_one256);
  x = _mm256_castsi256_ps (u);
  return x;
}
#endif

#if defined(__SSE2__)
template <> inline __m128 nextAfter<__m128>(__m128 a) {
  __m128 ge_zero = _mm_cmpeq_ps (a, _mm_setzero_ps());
  __m128 ret = _mm_blendv_ps (nextTowardZero(a), nextAwayFromZero(a), ge_zero);
  return ret;
}
#endif

#if defined(__AVX2__)
template <> inline __m256 nextAfter<__m256>(__m256 a) {
  __m256 ge_zero = _mm256_cmp_ps (a, _mm256_setzero_ps(), _CMP_EQ_OQ);
  __m256 ret = _mm256_blendv_ps (nextTowardZero(a), nextAwayFromZero(a), ge_zero);
  return ret;
}
#endif

#if defined(__SSE2__)
template<> inline __m128 nextPrev (__m128 a) {
  __m128 eq_zero = _mm_cmpeq_ps (a, _mm_setzero_ps());
  __m128 gt_zero = _mm_cmpgt_ps (a, _mm_setzero_ps());
  __m128 lt_zero = _mm_cmplt_ps (a, _mm_setzero_ps());
  __m128 res     = _mm_blendv_ps (a, _mm_set1_ps(-std::numeric_limits<float>::denorm_min()), eq_zero);
  res            = _mm_blendv_ps (res, nextTowardZero(a), gt_zero);
  res            = _mm_blendv_ps (res, nextAwayFromZero(a), lt_zero);
  return res;
}
#endif

#if defined(__AVX2__)
template<> inline __m256 nextPrev (__m256 a) {
  __m256 eq_zero = _mm256_cmp_ps (a, _mm256_setzero_ps(), _CMP_EQ_OQ);
  __m256 gt_zero = _mm256_cmp_ps (a, _mm256_setzero_ps(), _CMP_GT_OQ);
  __m256 lt_zero = _mm256_cmp_ps (a, _mm256_setzero_ps(), _CMP_LT_OQ);
  __m256 res     = _mm256_blendv_ps (a, _mm256_set1_ps(-std::numeric_limits<float>::denorm_min()), eq_zero);
  res            = _mm256_blendv_ps (res, nextTowardZero(a), gt_zero);
  res            = _mm256_blendv_ps (res, nextAwayFromZero(a), lt_zero);
  return res;
}
#endif