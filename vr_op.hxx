
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

#include <algorithm>
#include <cstring>
#include <type_traits>
#include <immintrin.h>

#include "interflop/fma/interflop_fma.h"
#include "vr_isNan.hxx"

enum opHash : uint32_t {
  addHash = 0,
  subHash = 1,
  mulHash = 2,
  divHash = 3,
  maddHash = 4,
  castHash = 5,
  nbOpHash = 6
};

enum typeHash : uint32_t {
  floatHash = 0,
  doubleHash = 1,
  otherHash = 2,
  nbTypeHash = 3
};

template <class> inline uint64_t getTypeHash() { return typeHash::otherHash; }
template <> inline uint64_t getTypeHash<float>() { return typeHash::floatHash; }
template <> inline uint64_t getTypeHash<double>() {
  return typeHash::doubleHash;
}

template <class REALTYPE, int NB> struct vr_packArg;

template <typename REAL>
REAL __verrou_internal_fma(const REAL &a, const REAL &b, const REAL &c);

template <>
inline float __verrou_internal_fma(const float &a, const float &b, const float &c) {
  return interflop_fma_binary32(a, b, c);
}

template <>
inline double __verrou_internal_fma(const double &a, const double &b,
                             const double &c) {
  return interflop_fma_binary64(a, b, c);
}
template <>
inline __float128 __verrou_internal_fma(const __float128 &a, const __float128 &b,
                                 const __float128 &c) {
  return interflop_fma_binary128(a, b, c);
}

/*
 * takes a real number and returns a uint64_t by reinterpreting its bits, NOT
 * casting it used by the getHash function in the vr_packArg classes
 */
template <class REALTYPE>
inline uint64_t realToUint64_reinterpret_cast(const REALTYPE x) {
  double x_double = static_cast<double>(x);
  uint64_t u;
  std::memcpy(&u, &x_double, sizeof(x_double));
  return u;
}

inline uint32_t realToUint32_reinterpret_cast(const float x) {
  // insures we have a 32 bits representation
  float x_float = x;
  uint32_t u;
  std::memcpy(&u, &x_float, sizeof(float));
  return u;
}

template <class REALTYPE> struct vr_packArg<REALTYPE, 1> {
  static const int nb = 1;
  typedef REALTYPE RealType;

  inline vr_packArg(const RealType &v1) : arg1(v1){};

  inline void serialyzeDouble(double *res) const { res[0] = (double)arg1; }

  inline bool isOneArgNanInf() const { return isNanInf<RealType>(arg1); }
  
  inline __m128i hasOneArgNanInf() const {
    interflop_panic ("Not implemented");
    return _mm_set1_epi8 ( (char) 1);
  }
  
  const RealType &arg1;
};

template <class REALTYPE> struct vr_packArg<REALTYPE, 2> {
  static const int nb = 2;
  typedef REALTYPE RealType;

  vr_packArg(const RealType &v1, const RealType &v2) : arg1(v1), arg2(v2){};

  inline void serialyzeDouble(double *res) const {
    res[0] = (double)arg1;
    res[1] = (double)arg2;
  }

  inline bool isOneArgNanInf() const {
    return (isNanInf<RealType>(arg1) || isNanInf<RealType>(arg2));
  }

  inline __m128i hasOneArgNanInf() const {
    interflop_panic ("Not implemented");
    return _mm_set1_epi8 ( (char) 1);
  }

  const RealType &arg1;
  const RealType &arg2;
};

template <class REALTYPE> struct vr_packArg<REALTYPE, 3> {
  static const int nb = 3;
  typedef REALTYPE RealType;

  vr_packArg(const RealType &v1, const RealType &v2, const RealType &v3)
      : arg1(v1), arg2(v2), arg3(v3){};

  inline void serialyzeDouble(double *res) const {
    res[0] = (double)arg1;
    res[1] = (double)arg2;
    res[2] = (double)arg3;
  }

  inline bool isOneArgNanInf() const {
    return (isNanInf<RealType>(arg1) || isNanInf<RealType>(arg2) ||
            isNanInf<RealType>(arg3));
  }

  inline __m128i hasOneArgNanInf() const {
    interflop_panic ("Not implemented");
    return _mm_set1_epi8 ( (char) 1);
  }

  const RealType &arg1;
  const RealType &arg2;
  const RealType &arg3;
};

template <class REALTYPE, int NB> class vr_roundFloat;

template <class REALTYPE> struct vr_roundFloat<REALTYPE, 1> {
  vr_roundFloat(const vr_packArg<REALTYPE, 1> &p)
      : arg1(REALTYPE(float(p.arg1))) {}
  vr_packArg<REALTYPE, 1> getPack() const {
    return vr_packArg<REALTYPE, 1>(arg1);
  }

  const REALTYPE arg1;
};

template <class REALTYPE> struct vr_roundFloat<REALTYPE, 2> {
  vr_roundFloat(const vr_packArg<REALTYPE, 2> &p)
      : arg1(REALTYPE(float(p.arg1))), arg2(REALTYPE(float(p.arg2))) {}
  vr_packArg<REALTYPE, 2> getPack() const {
    return vr_packArg<REALTYPE, 2>(arg1, arg2);
  }
  const REALTYPE arg1;
  const REALTYPE arg2;
};

template <class REALTYPE> struct vr_roundFloat<REALTYPE, 3> {
  vr_roundFloat(const vr_packArg<REALTYPE, 3> &p)
      : arg1(REALTYPE(float(p.arg1))), arg2(REALTYPE(float(p.arg2))),
        arg3(REALTYPE(float(p.arg3))) {}
  vr_packArg<REALTYPE, 3> getPack() const {
    return vr_packArg<REALTYPE, 3>(arg1, arg2, arg3);
  }
  const REALTYPE arg1;
  const REALTYPE arg2;
  const REALTYPE arg3;
};

template <typename REAL> class AddOp {
public:
  typedef REAL RealType;
  typedef vr_packArg<RealType, 2> PackArgs;

  static const char *OpName() { return "add"; }
  static inline uint64_t getHash() {
    return opHash::addHash * typeHash::nbTypeHash + getTypeHash<RealType>();
  }

  static inline RealType nearestOp(const PackArgs &p) {
    const RealType &a(p.arg1);
    const RealType &b(p.arg2);
    return a + b;
  }

  static inline RealType error(const PackArgs &p, const RealType &x) {
    const RealType &a(p.arg1);
    const RealType &b(p.arg2);
    const RealType z = x - a;
    return ((a - (x - z)) + (b - z)); // algo TwoSum
  }

  static inline RealType sameSignOfError(const PackArgs &p, const RealType &c) {
    return AddOp<RealType>::error(p, c);
  }

  static inline const PackArgs comdetPack(const PackArgs &p) {
    return PackArgs(std::min(p.arg1, p.arg2), std::max(p.arg1, p.arg2));
  }
  static inline uint64_t getComdetHash() { return AddOp::getHash(); }

  static inline bool isInfNotSpecificToNearest(const PackArgs &p) {
    return p.isOneArgNanInf();
  }

  static inline __m128i areInfNotSpecificToNearest(const PackArgs &p) {
    return p.hasOneArgNanInf();
  }

  static inline void check([[maybe_unused]] const PackArgs &p,
                           [[maybe_unused]] const RealType &c) {}

  static inline void twoSum(const RealType &a, const RealType &b, RealType &x,
                            RealType &y) {
    const PackArgs p(a, b);
    x = AddOp<REAL>::nearestOp(p);
    y = AddOp<REAL>::error(p, x);
  }
};

template <typename REAL> class SubOp {
public:
  typedef REAL RealType;
  typedef vr_packArg<RealType, 2> PackArgs;

  static const char *OpName() { return "sub"; }
  static inline uint64_t getHash() {
    return opHash::subHash * typeHash::nbTypeHash + getTypeHash<RealType>();
  }

  static inline RealType nearestOp(const PackArgs &p) {
    const RealType &a(p.arg1);
    const RealType &b(p.arg2);
    return a - b;
  }

  static inline RealType error(const PackArgs &p, const RealType &x) {
    const RealType &a(p.arg1);
    const RealType &b(-p.arg2);
    const RealType z = x - a;
    return ((a - (x - z)) + (b - z)); // algo TwoSum
  }

  static inline bool isInfNotSpecificToNearest(const PackArgs &p) {
    return p.isOneArgNanInf();
  }

  static inline __m128i areInfNotSpecificToNearest(const PackArgs &p) {
    return p.hasOneArgNanInf();
  }

  static inline RealType sameSignOfError(const PackArgs &p, const RealType &c) {
    return SubOp<RealType>::error(p, c);
  }

  static inline const PackArgs comdetPack(const PackArgs &p) {
    return PackArgs(std::min(p.arg1, -p.arg2), std::max(p.arg1, -p.arg2));
  }
  static inline uint64_t getComdetHash() {
    return opHash::addHash * typeHash::nbTypeHash + getTypeHash<RealType>();
  }

  static inline void check([[maybe_unused]] const PackArgs &p,
                           [[maybe_unused]] const RealType &c) {}
};

// splitFactor used by MulOp
template <class REALTYPE> inline REALTYPE splitFactor() {
  return 0. / 0.; // nan to make sure not used
}

template <> inline double splitFactor<double>() {
  return 134217729; //((2^27)+1); /27 en double  sup(53/2) /
}

template <> inline float splitFactor<float>() {
  return 4097; //((2^12)+1); / 24/2 en float/
}

template <typename REAL> class MulOp {
public:
  typedef REAL RealType;
  typedef vr_packArg<RealType, 2> PackArgs;

  static const char *OpName() { return "mul"; }
  static inline uint64_t getHash() {
    return opHash::mulHash * typeHash::nbTypeHash + getTypeHash<RealType>();
  }

  static inline RealType nearestOp(const PackArgs &p) {
    const RealType &a(p.arg1);
    const RealType &b(p.arg2);
    return a * b;
  };

  static inline RealType error(const PackArgs &p, const RealType &x) {
    /* From "Accurate Sum and dot product" OGITA RUMP OISHI */
    const RealType &a(p.arg1);
    const RealType &b(p.arg2);
    return __verrou_internal_fma(a, b, -x);
  };

  static inline void split(RealType a, RealType &x, RealType &y) {
    //    const RealType factor=134217729; //((2^27)+1); /*27 en double*/
    const RealType factor(splitFactor<RealType>());
    const RealType c = factor * a;
    x = (c - (c - a));
    y = (a - x);
  }

  static inline RealType sameSignOfError(const PackArgs &p, const RealType &c) {
    if (c != 0) {
      return MulOp<RealType>::error(p, c);
    } else {
      if (p.arg1 == 0 || p.arg2 == 0) {
        return 0;
      }
      if (p.arg1 > 0) {
        return p.arg2;
      } else {
        return -p.arg2;
      }
    }
  };
  static inline const PackArgs comdetPack(const PackArgs &p) {
    return PackArgs(std::min(p.arg1, p.arg2), std::max(p.arg1, p.arg2));
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
    x = MulOp<REAL>::nearestOp(p);
    y = MulOp<REAL>::error(p, x);
  }
};

template <> class MulOp<float> {
public:
  typedef float RealType;
  typedef vr_packArg<RealType, 2> PackArgs;

  static const char *OpName() { return "mul"; }
  static inline uint64_t getHash() {
    return opHash::mulHash * typeHash::nbTypeHash + getTypeHash<RealType>();
  }

  static inline RealType nearestOp(const PackArgs &p) {
    const RealType &a(p.arg1);
    const RealType &b(p.arg2);
    return a * b;
  };

  static inline RealType error(const PackArgs &p, const RealType &x) {
    /*Provient de "Accurate Sum and dot product" OGITA RUMP OISHI */
    const RealType a(p.arg1);
    const RealType b(p.arg2);
    return __verrou_internal_fma(a, b, -x);
  };

  static inline void split(RealType a, RealType &x, RealType &y) {
    //    const RealType factor=134217729; //((2^27)+1); /*27 en double*/
    const RealType factor(splitFactor<RealType>());
    const RealType c = factor * a;
    x = (c - (c - a));
    y = (a - x);
  }

  static inline RealType sameSignOfError(const PackArgs &p, const RealType &c) {
    double res = MulOp<double>::error(
        vr_packArg<double, 2>((double)p.arg1, (double)p.arg2), (double)c);
    if (res < 0) {
      return -1;
    }
    if (res > 0) {
      return 1;
    }
    return 0.;
  };

  static inline const PackArgs comdetPack(const PackArgs &p) {
    return PackArgs(std::min(p.arg1, p.arg2), std::max(p.arg1, p.arg2));
  }
  static inline uint64_t getComdetHash() { return getHash(); };

  static inline bool isInfNotSpecificToNearest(const PackArgs &p) {
    return p.isOneArgNanInf();
  }

  static inline void check([[maybe_unused]] const PackArgs &p,
                           [[maybe_unused]] const RealType &c){};

  static inline void twoProd(const RealType &a, const RealType &b, RealType &x,
                             RealType &y) {
    const PackArgs p(a, b);
    x = MulOp<float>::nearestOp(p);
    y = MulOp<float>::error(p, x);
  }
};

template <typename REAL> class DivOp {
public:
  typedef REAL RealType;
  typedef vr_packArg<RealType, 2> PackArgs;

  static const char *OpName() { return "div"; }
  static inline uint64_t getHash() {
    return opHash::divHash * typeHash::nbTypeHash + getTypeHash<RealType>();
  }

  static RealType inline nearestOp(const PackArgs &p) {
    const RealType &a(p.arg1);
    const RealType &b(p.arg2);
    return a / b;
  };

  static inline RealType error(const PackArgs &p, const RealType &c) {
    const RealType &x(p.arg1);
    const RealType &y(p.arg2);
    return -__verrou_internal_fma(c, y, -x);
  };

  static inline RealType sameSignOfError(const PackArgs &p, const RealType &c) {
    const RealType &x(p.arg1);
    const RealType &y(p.arg2);
    return -__verrou_internal_fma(c, y, -x);
  };

  static inline const PackArgs comdetPack(const PackArgs &p) { return p; }
  static inline uint64_t getComdetHash() { return getHash(); };

  static inline void check([[maybe_unused]] const PackArgs &p,
                           [[maybe_unused]] const RealType &c){};

  static inline bool isInfNotSpecificToNearest(const PackArgs &p) {
    return (isNanInf<RealType>(p.arg1)) || (p.arg2 == RealType(0.));
  }
};

template <> class DivOp<float> {
public:
  typedef float RealType;
  typedef vr_packArg<RealType, 2> PackArgs;

  static const char *OpName() { return "div"; }
  static inline uint64_t getHash() {
    return opHash::divHash * typeHash::nbTypeHash + getTypeHash<RealType>();
  }

  static RealType inline nearestOp(const PackArgs &p) {
    const RealType &a(p.arg1);
    const RealType &b(p.arg2);
    return a / b;
  };

  static inline RealType error(const PackArgs &p, const RealType &c) {
    const RealType &x(p.arg1);
    const RealType &y(p.arg2);
    return -__verrou_internal_fma(c, y, -x) / y;
  };

  static inline RealType sameSignOfError(const PackArgs &p, const RealType &c) {
    const double x((double)p.arg1);
    const double y((double)p.arg2);
    const double r = -__verrou_internal_fma((double)c, y, -x);
    if (r > 0) {
      return p.arg2;
    } else if (r < 0) {
      return -p.arg2;
    } else {
      return 0.0;
    }
  };

  static inline const PackArgs comdetPack(const PackArgs &p) { return p; }
  static inline uint64_t getComdetHash() { return getHash(); };

  static inline void check([[maybe_unused]] const PackArgs &p,
                           [[maybe_unused]] const RealType &c){};

  static inline bool isInfNotSpecificToNearest(const PackArgs &p) {
    return (isNanInf<RealType>(p.arg1)) || (p.arg2 == RealType(0.));
  }
};

template <typename REAL> class MAddOp {
public:
  typedef REAL RealType;
  typedef vr_packArg<RealType, 3> PackArgs;

  static const char *OpName() { return "madd"; }
  static inline uint64_t getHash() {
    return opHash::maddHash * typeHash::nbTypeHash + getTypeHash<RealType>();
  }

  static RealType inline nearestOp(const PackArgs &p) {
    const RealType &a(p.arg1);
    const RealType &b(p.arg2);
    const RealType &c(p.arg3);
    return __verrou_internal_fma(a, b, c);
  };

  static inline RealType error(const PackArgs &p, const RealType &z) {
    // ErrFmaApp : Exact and Aproximated Error of the FMA By Boldo and Muller
    const RealType &a(p.arg1);
    const RealType &x(p.arg2);
    const RealType &b(p.arg3);

    RealType ph, pl;
    MulOp<RealType>::twoProd(a, x, ph, pl);

    RealType uh, ul;
    AddOp<RealType>::twoSum(b, ph, uh, ul);

    const RealType t(uh - z);
    return (t + (pl + ul));
  };

  static inline RealType sameSignOfError(const PackArgs &p, const RealType &c) {
    return error(p, c);
  };

  static inline const PackArgs comdetPack(const PackArgs &p) {
    return PackArgs(std::min(p.arg1, p.arg2), std::max(p.arg1, p.arg2), p.arg3);
  }
  static inline uint64_t getComdetHash() { return getHash(); };

  static inline void check([[maybe_unused]] const PackArgs &p,
                           [[maybe_unused]] const RealType &d){};

  static inline bool isInfNotSpecificToNearest(const PackArgs &p) {
    return p.isOneArgNanInf();
  }
};

template <typename REALINPUT, typename REALOUTPUT> class CastOp {
public:
  typedef REALINPUT RealTypeIn;
  typedef REALOUTPUT RealTypeOut;
  typedef RealTypeOut RealType;
  typedef vr_packArg<RealTypeIn, 1> PackArgs;

  static const char *OpName() { return "cast"; }
  static inline uint64_t getHash() {
    return opHash::castHash * typeHash::nbTypeHash + getTypeHash<RealType>();
  }

  static inline RealTypeOut nearestOp(const PackArgs &p) {
    const RealTypeIn &in(p.arg1);
    return (RealTypeOut)in;
  };

  static inline RealTypeOut error(const PackArgs &p, const RealTypeOut &z) {
    const RealTypeIn &a(p.arg1);
    const RealTypeIn errorHo = a - (RealTypeIn)z;
    return (RealTypeOut)errorHo;
  };

  static inline RealTypeOut sameSignOfError(const PackArgs &p,
                                            const RealTypeOut &c) {
    return error(p, c);
  };

  static inline const PackArgs comdetPack(const PackArgs &p) { return p; }
  static inline uint64_t getComdetHash() { return getHash(); };

  static inline bool isInfNotSpecificToNearest(const PackArgs &p) {
    return p.isOneArgNanInf();
  }

  static inline void check([[maybe_unused]] const PackArgs &p,
                           [[maybe_unused]] const RealTypeOut &d){};
};
