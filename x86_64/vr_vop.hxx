
/*--------------------------------------------------------------------*/
/*--- Verrou: a FPU instrumentation tool.                          ---*/
/*--- Implementation of error estimation for all FP operations     ---*/
/*---                                                    vr_op.hxx ---*/
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

#include "../vr_op.hxx"
#include "vr_areNan.hxx"


// Vector specialization
// vr_packArg
#if defined(__SSE4_2__)
template<>
inline __m128i vr_packArg<__m128, 1>::hasOneArgNanInf() const {
  return hasNanInf(this->arg1);
}

template<>
inline __m128i vr_packArg<__m128, 2>::hasOneArgNanInf() const {
  return _mm_or_si128( hasNanInf(this->arg1), hasNanInf(this->arg2));
}

template<>
inline __m128i vr_packArg<__m128, 3>::hasOneArgNanInf() const {
  return _mm_or_si128( _mm_or_si128( hasNanInf(this->arg1), hasNanInf(this->arg2)), hasNanInf(this->arg3));
}
#endif

#if defined(__AVX2__)
template<>
inline __m128i vr_packArg<__m256, 1>::hasOneArgNanInf() const {
  return hasNanInf(this->arg1);
}

template<>
inline __m128i vr_packArg<__m256, 2>::hasOneArgNanInf() const {
  return _mm_or_si128( hasNanInf(this->arg1), hasNanInf(this->arg2));
}

template<>
inline __m128i vr_packArg<__m256, 3>::hasOneArgNanInf() const {
  return _mm_or_si128( _mm_or_si128( hasNanInf(this->arg1), hasNanInf(this->arg2)), hasNanInf(this->arg3));
}
#endif
// AddOp
#if defined(__SSE4_2__)
template<>
inline __m128 AddOp<__m128>::nearestOp(const PackArgs &p) {
  const RealType &a(p.arg1);
  const RealType &b(p.arg2);
  return _mm_add_ps (a, b);
}

template<>
inline __m128 AddOp<__m128>::error (const PackArgs& p, const RealType& x) {
  const RealType & a(p.arg1);
  const RealType & b(p.arg2);
  const RealType z = _mm_sub_ps (x, a);
  return _mm_add_ps (
            _mm_sub_ps (a, _mm_sub_ps(x, z)),
            _mm_sub_ps (b, z)
          );
}
#endif

#if defined(__AVX2__)
template<>
inline __m256 AddOp<__m256>::nearestOp(const PackArgs &p) {
  const RealType &a(p.arg1);
  const RealType &b(p.arg2);
  return _mm256_add_ps (a, b);
}

template<>
inline __m256 AddOp<__m256>::error (const PackArgs& p, const RealType& x) {
  const RealType & a(p.arg1);
  const RealType & b(p.arg2);
  const RealType z = _mm256_sub_ps (x, a);
  return _mm256_add_ps (
            _mm256_sub_ps (a, _mm256_sub_ps(x, z)),
            _mm256_sub_ps (b, z)
          );
}
#endif

// SubOp
#if defined(__SSE4_2__)
template<>
inline __m128 SubOp<__m128>::nearestOp(const PackArgs &p) {
  const RealType &a(p.arg1);
  const RealType &b(p.arg2);
  return _mm_sub_ps (a, b);
}

template<>
inline __m128 SubOp<__m128>::error (const PackArgs& p, const RealType& x) {
  const RealType & a(p.arg1);
  const RealType & b(_mm_sub_ps (_mm_setzero_ps(), p.arg2));
  const RealType z = _mm_sub_ps (x, a);
  return _mm_add_ps (
            _mm_sub_ps (a, _mm_sub_ps(x, z)),
            _mm_sub_ps (b, z)
          );
}
#endif

#if defined(__AVX2__)
template<>
inline __m256 SubOp<__m256>::nearestOp(const PackArgs &p) {
  const RealType &a(p.arg1);
  const RealType &b(p.arg2);
  return _mm256_sub_ps (a, b);
}

template<>
inline __m256 SubOp<__m256>::error (const PackArgs& p, const RealType& x) {
  const RealType & a(p.arg1);
  const RealType & b(_mm256_sub_ps( _mm256_setzero_ps(), p.arg2));
  const RealType z = _mm256_sub_ps (x, a);
  return _mm256_add_ps (
            _mm256_sub_ps (a, _mm256_sub_ps(x, z)),
            _mm256_sub_ps (b, z)
          );
}
#endif

// MulOp
#if defined(__SSE4_2__)
template <> inline __m128 splitFactor<__m128>() {
  return _mm_set1_ps((float)4097); //((2^12)+1); / 24/2 en float/
}
#endif

#if defined(__AVX2__)
template <> inline __m256 splitFactor<__m256>() {
  return _mm256_set1_ps((float)4097); //((2^12)+1); / 24/2 en float/
}
#endif

#if defined(__SSE4_2__)
template <> class MulOp<__m128> {
public:
  typedef __m128 RealType;
  typedef vr_packArg<RealType, 2> PackArgs;

  static const char *OpName() { return "mul"; }
  static inline uint64_t getHash() {
    return opHash::mulHash * typeHash::nbTypeHash + getTypeHash<RealType>();
  }

  static inline RealType nearestOp(const PackArgs &p) {
    const RealType &a(p.arg1);
    const RealType &b(p.arg2);
    return _mm_mul_ps (a, b);
  };

  static inline RealType error(const PackArgs &p, const RealType &x) {
    /*Provient de "Accurate Sum and dot product" OGITA RUMP OISHI */
    const RealType a(p.arg1);
    const RealType b(p.arg2);
    RealType a1,a2;
    RealType b1,b2;
    MulOp<RealType>::split(a,a1,a2);
    MulOp<RealType>::split(b,b1,b2);

    return _mm_add_ps (_mm_add_ps (_mm_sub_ps (_mm_mul_ps (a1, b1), x),
                                   _mm_add_ps ( _mm_mul_ps (a1, b2), _mm_mul_ps (a2, b1))),
                        _mm_mul_ps (a2, b2));
  };

  static inline void split(RealType a, RealType &x, RealType &y) {
    //    const RealType factor=134217729; //((2^27)+1); /*27 en double*/
    const RealType factor(splitFactor<RealType>());
    const RealType c = _mm_mul_ps (factor , a);
    x = _mm_sub_ps (c, _mm_sub_ps(c, a));
    y = _mm_sub_ps (a, x);
  }

  static inline RealType sameSignOfError(const PackArgs &p, const RealType &c) {
    __m128 res = MulOp<__m128>::error(p, c);// Call for double later

    __m128 res_lt_0 = _mm_cmplt_ps (res, _mm_setzero_ps());
    __m128 res_gt_0 = _mm_cmpgt_ps (res, _mm_setzero_ps());

    __m128 ret = _mm_setzero_ps();
    ret = _mm_blendv_ps (ret, _mm_set1_ps(-1), res_lt_0);
    ret = _mm_blendv_ps (ret, _mm_set1_ps( 1), res_gt_0);
    return ret;
  };

  static inline const PackArgs comdetPack(const PackArgs &p) {
    return PackArgs(_mm_min_ps (p.arg1, p.arg2), _mm_max_ps(p.arg1, p.arg2));
  }
  static inline uint64_t getComdetHash() { return getHash(); };

  static inline bool isInfNotSpecificToNearest(const PackArgs &p) {
    return p.isOneArgNanInf();
  }

  static inline __m128i areInfNotSpecificToNearest(const PackArgs &p) {
    return p.hasOneArgNanInf();
  }

  static inline void check([[maybe_unused]] const PackArgs &p,
                           [[maybe_unused]] const RealType &c){};

  static inline void twoProd(const RealType &a, const RealType &b, RealType &x,
                             RealType &y) {
    const PackArgs p(a, b);
    x = MulOp<__m128>::nearestOp(p);
    y = MulOp<__m128>::error(p, x);
  }
};
#endif

#if defined(__AVX2__)
template <> class MulOp<__m256> {
public:
  typedef __m256 RealType;
  typedef vr_packArg<RealType, 2> PackArgs;

  static const char *OpName() { return "mul"; }
  static inline uint64_t getHash() {
    return opHash::mulHash * typeHash::nbTypeHash + getTypeHash<RealType>();
  }

  static inline RealType nearestOp(const PackArgs &p) {
    const RealType &a(p.arg1);
    const RealType &b(p.arg2);
    return _mm256_mul_ps (a, b);
  };

  static inline RealType error(const PackArgs &p, const RealType &x) {
    /*Provient de "Accurate Sum and dot product" OGITA RUMP OISHI */
    const RealType a(p.arg1);
    const RealType b(p.arg2);
    return _mm256_fmadd_ps (a, b, _mm256_sub_ps(_mm256_setzero_ps(), x));

    RealType a1,a2;
    RealType b1,b2;
    MulOp<RealType>::split(a,a1,a2);
    MulOp<RealType>::split(b,b1,b2);
    return _mm256_add_ps (_mm256_add_ps ( _mm256_sub_ps (_mm256_mul_ps (a1, b1), x), 
                                                         _mm256_add_ps (_mm256_mul_ps (a1, b2),
                                                                        _mm256_mul_ps (a2, b1))),
                          _mm256_mul_ps (a2, b2));
  };

  static inline void split(RealType a, RealType &x, RealType &y) {
    //    const RealType factor=134217729; //((2^27)+1); /*27 en double*/
    const RealType factor(splitFactor<RealType>());
    const RealType c = _mm256_mul_ps (factor , a);
    x = _mm256_sub_ps (c, _mm256_sub_ps(c, a));
    y = _mm256_sub_ps (a, x);
  }

  static inline RealType sameSignOfError(const PackArgs &p, const RealType &c) {
    __m256 res = MulOp<__m256>::error(p, c);// Call for double later

    __m256 res_lt_0 = _mm256_cmp_ps (res, _mm256_setzero_ps(), _CMP_LT_OQ);
    __m256 res_gt_0 = _mm256_cmp_ps (res, _mm256_setzero_ps(), _CMP_GT_OQ);

    __m256 ret = _mm256_setzero_ps();
    ret = _mm256_blendv_ps (ret, _mm256_set1_ps(-1), res_lt_0);
    ret = _mm256_blendv_ps (ret, _mm256_set1_ps( 1), res_gt_0);
    return ret;
  };

  static inline const PackArgs comdetPack(const PackArgs &p) {
    return PackArgs(_mm256_min_ps (p.arg1, p.arg2), _mm256_max_ps(p.arg1, p.arg2));
  }
  static inline uint64_t getComdetHash() { return getHash(); };

  static inline bool isInfNotSpecificToNearest(const PackArgs &p) {
    return p.isOneArgNanInf();
  }

  static inline __m128i areInfNotSpecificToNearest(const PackArgs &p) {
    return p.hasOneArgNanInf();
  }

  static inline void check([[maybe_unused]] const PackArgs &p,
                           [[maybe_unused]] const RealType &c){};

  static inline void twoProd(const RealType &a, const RealType &b, RealType &x,
                             RealType &y) {
    const PackArgs p(a, b);
    x = MulOp<__m256>::nearestOp(p);
    y = MulOp<__m256>::error(p, x);
  }
};
#endif