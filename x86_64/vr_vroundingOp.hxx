/*--------------------------------------------------------------------*/
/*--- Verrou: a FPU instrumentation tool.                          ---*/
/*--- Implementation of the software implementation of rounding    ---*/
/*--- mode switching.                                              ---*/
/*---                                            vr_roundingOp.hxx ---*/
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
#include "../vr_roundingOp.hxx"
#include "vr_areNan.hxx"
#include "vr_nextUlps.hxx"

#include "vr_vop.hxx"
#include "../vr_rand_implem.h"

#if defined(__SSE4_2__)
template<template<class REAL> class OP>
class RoundingNearest<OP<__m128>>
{
public:
  typedef __m128 RealType;
  typedef typename OP<__m128>::PackArgs PackArgs;

  static inline RealType apply(const PackArgs &p) {
    const RealType res = OP<__m128>::nearestOp(p);
    OP<__m128>::check(p, res);
    return res;    
  };
};
#endif

#if defined(__AVX2__)
template<template<class REAL> class OP>
class RoundingNearest<OP<__m256>>
{
public:
  typedef __m256 RealType;
  typedef typename OP<__m256>::PackArgs PackArgs;

  static inline RealType apply(const PackArgs &p) {
    const RealType res = OP<__m256>::nearestOp(p);
    OP<__m256>::check(p, res);
    return res;    
  };
};
#endif

#if defined(__SSE4_2__)
template<template<class REAL> class OP, class RAND>
class RoundingUpward<OP<__m128>, RAND>
{
public:
  typedef __m128 RealType;
  typedef typename OP<__m128>::PackArgs PackArgs;
  static inline RealType apply(const PackArgs &p) {
    /*const*/ RealType res = OP<__m128>::nearestOp(p);
    OP<__m128>::check(p, res);
    INC_OP; // Sould exist for simd ops OR called 4 times
#ifndef VERROU_IGNORE_NANINF_CHECK
//  Check Nans and Infs
    __m128 simd_is_isNanInf = _mm_castsi128_ps( hasNanInf<RealType>(res));    
    __m128 simd_is_eq_neg_inf = _mm_cmpneq_ps (res, _mm_set1_ps(-std::numeric_limits<float>::infinity()));
    __m128 simd_is_areInfNotSpecificToNearest = _mm_castsi128_ps (OP<__m128>::areInfNotSpecificToNearest(p));

    __m128 simd_tmp1 = _mm_or_ps (_mm_or_ps (simd_is_eq_neg_inf, simd_is_areInfNotSpecificToNearest), simd_is_isNanInf);
    __m128 simd_tmp2 = _mm_or_ps ( _mm_andnot_ps(simd_tmp1, simd_tmp1), simd_is_isNanInf);

    res = _mm_blendv_ps (res, _mm_set1_ps(-std::numeric_limits<float>::max()), simd_tmp2);
#endif
    __m128 v_signError = OP<__m128>::sameSignOfError(p, res);
    __m128 v_fzero = _mm_setzero_ps ();
    __m128 v_fdenorm_min = _mm_set1_ps (std::numeric_limits<float>::denorm_min());
#ifdef PROFILING_EXACT
    __m128 simd_is_signError_eq_fzero = _mm_cmpeq_ps (v_signError, v_fzero);
#endif
    __m128 simd_is_signError_gt_fzero    = _mm_cmpgt_ps (v_signError, v_fzero);

    if (_mm_movemask_ps (simd_is_signError_gt_fzero) == 0) return res; // Check if at least one has error > 0.

    __m128 res_nextAfter  = nextAfter<RealType> (res);

    __m128 simd_is_res_eq_fzero          = _mm_cmpeq_ps (res, v_fzero);
    __m128 simd_is_res_eq_neg_denorm_min = _mm_cmpeq_ps (res, _mm_set1_ps(-std::numeric_limits<float>::denorm_min()));

    if (_mm_movemask_ps (_mm_or_ps (simd_is_res_eq_fzero, simd_is_res_eq_neg_denorm_min)))
    {
      res_nextAfter         = _mm_blendv_ps (res_nextAfter , v_fdenorm_min, simd_is_res_eq_fzero);
      res_nextAfter         = _mm_blendv_ps (res_nextAfter , v_fzero, simd_is_res_eq_neg_denorm_min);
    }

    const __m128 v_res = _mm_blendv_ps(res, res_nextAfter, simd_is_signError_gt_fzero);

    return v_res;    
  }
};
#endif

#if defined(__AVX2__)
template<template<class REAL> class OP, class RAND>
class RoundingUpward<OP<__m256>, RAND>
{
public:
  typedef __m256 RealType;
  typedef typename OP<__m256>::PackArgs PackArgs;
  static inline RealType apply(const PackArgs &p) {
    const RealType res = OP<__m256>::nearestOp(p);
    OP<__m256>::check(p, res);
    INC_OP; // Sould exist for simd ops OR called 4 times
#ifndef VERROU_IGNORE_NANINF_CHECK

#endif
    __m256 v_signError = OP<__m256>::sameSignOfError(p, res);
    __m256 v_fzero = _mm256_setzero_ps ();
    __m256 v_fdenorm_min = _mm256_set1_ps (std::numeric_limits<float>::denorm_min());
#ifdef PROFILING_EXACT
    __m256 simd_is_signError_eq_fzero = _mm256_cmp_ps (v_signError, v_fzero);
#endif
    __m256 simd_is_signError_gt_fzero    = _mm256_cmp_ps (v_signError, v_fzero, _CMP_GT_OQ);

    if (_mm256_movemask_ps (simd_is_signError_gt_fzero) == 0) return res; // Check if at least one has error > 0.

    __m256 res_nextAfter  = nextAfter<RealType> (res);

    __m256 simd_is_res_eq_fzero          = _mm256_cmp_ps (res, v_fzero, _CMP_EQ_OQ);
    __m256 simd_is_res_eq_neg_denorm_min = _mm256_cmp_ps (res, _mm256_set1_ps(-std::numeric_limits<float>::denorm_min()), _CMP_EQ_OQ);

    if (_mm256_movemask_ps (_mm256_or_ps (simd_is_res_eq_fzero, simd_is_res_eq_neg_denorm_min)))
    {
      res_nextAfter         = _mm256_blendv_ps (res_nextAfter , v_fdenorm_min, simd_is_res_eq_fzero);
      res_nextAfter         = _mm256_blendv_ps (res_nextAfter , v_fzero, simd_is_res_eq_neg_denorm_min);
    }

    const __m256 v_res = _mm256_blendv_ps(res, res_nextAfter, simd_is_signError_gt_fzero);

    return v_res;    
  }
};
#endif

#if defined(__SSE4_2__)
template<template<class REAL> class OP, class RAND>
class RoundingDownward<OP<__m128>, RAND>
{
public:
  typedef __m128 RealType;
  typedef typename OP<__m128>::PackArgs PackArgs;

  static inline RealType apply(const PackArgs &p) {
    /*const*/ RealType res = OP<__m128>::nearestOp(p);
    OP<__m128>::check(p, res);
    INC_OP; // Sould exist for simd ops OR called 4 times
#ifndef VERROU_IGNORE_NANINF_CHECK
//  Check Nans and Infs
    __m128 simd_is_isNanInf = _mm_castsi128_ps( hasNanInf<RealType>(res));    
    __m128 simd_is_eq_neg_inf = _mm_cmpneq_ps (res, _mm_set1_ps(std::numeric_limits<float>::infinity()));
    __m128 simd_is_areInfNotSpecificToNearest = _mm_castsi128_ps (OP<__m128>::areInfNotSpecificToNearest(p));

    __m128 simd_tmp1 = _mm_or_ps (_mm_or_ps (simd_is_eq_neg_inf, simd_is_areInfNotSpecificToNearest), simd_is_isNanInf);
    __m128 simd_tmp2 = _mm_or_ps ( _mm_andnot_ps(simd_tmp1, simd_tmp1), simd_is_isNanInf);

    res = _mm_blendv_ps (res, _mm_set1_ps(std::numeric_limits<float>::max()), simd_tmp2);
  
#endif
    __m128 v_signError = OP<__m128>::sameSignOfError(p, res);
    __m128 v_fzero = _mm_setzero_ps ();
    __m128 v_fdenorm_min = _mm_set1_ps (-std::numeric_limits<float>::denorm_min());
#ifdef PROFILING_EXACT
    __m128 simd_is_signError_eq_fzero = _mm_cmpeq_ps (v_signError, v_fzero);
#endif
    __m128 simd_is_signError_lt_fzero    = _mm_cmplt_ps (v_signError, v_fzero);

    if (_mm_movemask_ps (simd_is_signError_lt_fzero) == 0) return res; // Check if at least one has error > 0.

    __m128 res_nextPrev  = nextPrev<RealType> (res);

    __m128 simd_is_res_eq_fzero          = _mm_cmpeq_ps (res, v_fzero);
    __m128 simd_is_res_eq_neg_denorm_min = _mm_cmpeq_ps (res, _mm_set1_ps(std::numeric_limits<float>::denorm_min()));

    if (_mm_movemask_ps (_mm_or_ps (simd_is_res_eq_fzero, simd_is_res_eq_neg_denorm_min)))
    {
      res_nextPrev         = _mm_blendv_ps (res_nextPrev , v_fdenorm_min, simd_is_res_eq_fzero);
      res_nextPrev         = _mm_blendv_ps (res_nextPrev , v_fzero, simd_is_res_eq_neg_denorm_min);
    }

    const __m128 v_res = _mm_blendv_ps(res, res_nextPrev, simd_is_signError_lt_fzero);
    return v_res;
  };
};
#endif

#if defined(__AVX2__)
template<template<class REAL> class OP, class RAND>
class RoundingDownward<OP<__m256>, RAND>
{
public:
  typedef __m256 RealType;
  typedef typename OP<__m256>::PackArgs PackArgs;

  static inline RealType apply(const PackArgs &p) {
    const RealType res = OP<__m256>::nearestOp(p);
    OP<__m256>::check(p, res);
    INC_OP; // Sould exist for simd ops OR called 4 times
#ifndef VERROU_IGNORE_NANINF_CHECK
 
#endif
    __m256 v_signError = OP<__m256>::sameSignOfError(p, res);
    __m256 v_fzero = _mm256_setzero_ps ();
    __m256 v_fdenorm_min = _mm256_set1_ps (-std::numeric_limits<float>::denorm_min());
#ifdef PROFILING_EXACT
    __m256 simd_is_signError_eq_fzero = _mm256_cmp_ps (v_signError, v_fzero);
#endif
    __m256 simd_is_signError_lt_fzero    = _mm256_cmp_ps (v_signError, v_fzero, _CMP_LT_OQ);

    if (_mm256_movemask_ps (simd_is_signError_lt_fzero) == 0) return res; // Check if at least one has error > 0.

    __m256 res_nextPrev  = nextPrev<RealType> (res);

    __m256 simd_is_res_eq_fzero          = _mm256_cmp_ps (res, v_fzero, _CMP_EQ_OQ);
    __m256 simd_is_res_eq_neg_denorm_min = _mm256_cmp_ps (res, _mm256_set1_ps(std::numeric_limits<float>::denorm_min()), _CMP_EQ_OQ);

    if (_mm256_movemask_ps (_mm256_or_ps (simd_is_res_eq_fzero, simd_is_res_eq_neg_denorm_min)))
    {
      res_nextPrev         = _mm256_blendv_ps (res_nextPrev , v_fdenorm_min, simd_is_res_eq_fzero);
      res_nextPrev         = _mm256_blendv_ps (res_nextPrev , v_fzero, simd_is_res_eq_neg_denorm_min);
    }

    const __m256 v_res = _mm256_blendv_ps(res, res_nextPrev, simd_is_signError_lt_fzero);
    return v_res;
  };
};
#endif

#include "vr_vop.hxx"

template<class REALTYPE>
static inline REALTYPE ret_zero() {
  return 0;
};

#if defined(__SSE4_2__)
template <>
inline __m128 ret_zero<__m128>() {
  return _mm_setzero_ps();
};
#endif

#if defined(__AVX2__)
template <>
inline __m256 ret_zero<__m256>() {
  return _mm256_setzero_ps();
};
#endif

template <class OP> class VOpWithSelectedRoundingMode {
public:
  typedef typename OP::RealType RealType;
  typedef typename OP::PackArgs PackArgs;

  static inline void apply(const PackArgs &p, RealType *res, void *context) {
    *res = applySeq(p, context);
#ifdef DEBUG_PRINT_OP
    print_debug(p, res);
#endif
#ifndef VERROU_IGNORE_NANINF_CHECK
/*    if (isNanInf(*res)) {
      if (isNan(*res)) {
        interflop_nanHandler();
      }
      if (isinf(*res)) {
        interflop_infHandler();
      }
    }*/
#endif
  }

#ifdef DEBUG_PRINT_OP
  static inline void print_debug(const PackArgs &p, const RealType *res) {
    static const int nbParam = OP::PackArgs::nb;

    double args[nbParam];
    const double resDouble(*res);
    p.serialyzeDouble(args);
    if (vr_debug_print_op == NULL)
      return;
    vr_debug_print_op(nbParam, OP::OpName(), args, &resDouble);
  }
#endif

  static inline RealType applySeq(const PackArgs &p, void *context) {
//    return RoundingNearest<OP>::apply(p);

    verrou_context_t *ctx = (verrou_context_t *)context;
    switch (ctx->rounding_mode) {
    case VR_NEAREST:
      return RoundingNearest<OP>::apply(p);

    case VR_UPWARD:
      return RoundingUpward<OP>::apply(p);

    case VR_DOWNWARD:
      return RoundingDownward<OP>::apply(p);
   default:
     interflop_panic("Rounding mode not implemented !");
    }
    return ret_zero<RealType>();
  }
};

//#endif
