
#include <argp.h>
#include <stddef.h>


#include <cstring>
#include "../interflop_verrou.h"
// #include "../static_backends.hxx"
#include "vr_nextUlps.hxx"
#include "vr_vop.hxx"
#include "vr_vroundingOp.hxx"

#include "interflop_vinterface.h"

#include <stdio.h>
#include <immintrin.h>

#if defined(VECT512)
#include "interflop_vector_verrou_avx512.h"
#elif defined(VECT256)
#include "interflop_vector_verrou_avx.h"
#elif defined(VECT128)
#include "interflop_vector_verrou_sse.h"
#elif defined (SCALAR)
#include "interflop_vector_verrou_scalar.h"
#else
#error "Mustn't happened"
#endif

void INTERFLOP_VECTOR_VERROU_API(add_float_1)(float *a, float *b, float *res,
                                          void *context) {
  typedef VOpWithSelectedRoundingMode<AddOp<float>> Op;
  Op::apply(Op::PackArgs(*a, *b), res, context);
}

void INTERFLOP_VECTOR_VERROU_API(add_float_4)(float *a, float *b, float *res,
                                          void *context) {
#if defined(__SSE4_1__)
  typedef VOpWithSelectedRoundingMode<AddOp<__m128>> Op;
  __m128 v_a = _mm_loadu_ps (a);
  __m128 v_b = _mm_loadu_ps (b);
  __m128 v_res = _mm_setzero_ps();
  Op::apply(Op::PackArgs(v_a, v_b), &v_res, context);
  _mm_storeu_ps (res, v_res);
#else
  typedef VOpWithSelectedRoundingMode<AddOp<float>> Op;
  for (size_t i = 0; i < 4; i++)
  {
    Op::apply(Op::PackArgs(a[i], b[i]), res+i, context);
  }
  
#endif
}

void INTERFLOP_VECTOR_VERROU_API(add_float_8)(float *a, float *b, float *res,
                                          void *context) {
#if defined(__AVX2__)
  typedef VOpWithSelectedRoundingMode<AddOp<__m256>> Op;
  __m256 v_a = _mm256_loadu_ps (a);
  __m256 v_b = _mm256_loadu_ps (b);
  __m256 v_res = _mm256_setzero_ps();
  Op::apply(Op::PackArgs(v_a, v_b), &v_res, context);
  _mm256_storeu_ps (res, v_res);
#elif defined(__SSE4_2__)
  for (size_t i = 0; i < 2; i++)
  {
    typedef VOpWithSelectedRoundingMode<AddOp<__m128>> Op;
    __m128 v_a = _mm_loadu_ps (a+4*i);
    __m128 v_b = _mm_loadu_ps (b+4*i);
    __m128 v_res = _mm_setzero_ps();
    Op::apply(Op::PackArgs(v_a, v_b), &v_res, context);
    _mm_storeu_ps (res+4*i, v_res);
  }
#else
  typedef VOpWithSelectedRoundingMode<AddOp<float>> Op;
  for (size_t i = 0; i < 8; i++)
  {
    Op::apply(Op::PackArgs(a[i], b[i]), res+i, context);
  }
#endif
}

void INTERFLOP_VECTOR_VERROU_API(add_float_16)(float *a, float *b, float *res,
                                          void *context) {
#if defined(__AVX2__)
  for (size_t i = 0; i < 2; i++)
  {
    typedef VOpWithSelectedRoundingMode<AddOp<__m256>> Op;
    __m256 v_a = _mm256_loadu_ps (a+8*i);
    __m256 v_b = _mm256_loadu_ps (b+8*i);
    __m256 v_res = _mm256_setzero_ps();
    Op::apply(Op::PackArgs(v_a, v_b), &v_res, context);
    _mm256_storeu_ps (res+8*i, v_res);
  }
#elif defined(__SSE4_2__)
  for (size_t i = 0; i < 4; i++)
  {
    typedef VOpWithSelectedRoundingMode<AddOp<__m128>> Op;
    __m128 v_a = _mm_loadu_ps (a+4*i);
    __m128 v_b = _mm_loadu_ps (b+4*i);
    __m128 v_res = _mm_setzero_ps();
    Op::apply(Op::PackArgs(v_a, v_b), &v_res, context);
    _mm_storeu_ps (res+4*i, v_res);
  }
#else
  typedef VOpWithSelectedRoundingMode<AddOp<float>> Op;
  for (size_t i = 0; i < 16; i++)
  {
    Op::apply(Op::PackArgs(a[i], b[i]), res+i, context);
  }
#endif
}

void INTERFLOP_VECTOR_VERROU_API(sub_float_1)(float *a, float *b, float *res,
                                          void *context) {
  typedef VOpWithSelectedRoundingMode<SubOp<float>> Op;
  Op::apply(Op::PackArgs(*a, *b), res, context);
}

void INTERFLOP_VECTOR_VERROU_API(sub_float_4)(float *a, float *b, float *res,
                                          void *context) {
#if defined(__SSE4_1__)
  typedef VOpWithSelectedRoundingMode<SubOp<__m128>> Op;
  __m128 v_a = _mm_loadu_ps (a);
  __m128 v_b = _mm_loadu_ps (b);
  __m128 v_res = _mm_setzero_ps();
  Op::apply(Op::PackArgs(v_a, v_b), &v_res, context);
  _mm_storeu_ps (res, v_res);
#else
  typedef VOpWithSelectedRoundingMode<SubOp<float>> Op;
  for (size_t i = 0; i < 4; i++)
  {
    Op::apply(Op::PackArgs(a[i], b[i]), res+i, context);
  }
  
#endif
}

void INTERFLOP_VECTOR_VERROU_API(sub_float_8)(float *a, float *b, float *res,
                                          void *context) {
#if defined(__AVX2__)
  typedef VOpWithSelectedRoundingMode<SubOp<__m256>> Op;
  __m256 v_a = _mm256_loadu_ps (a);
  __m256 v_b = _mm256_loadu_ps (b);
  __m256 v_res = _mm256_setzero_ps();
  Op::apply(Op::PackArgs(v_a, v_b), &v_res, context);
  _mm256_storeu_ps (res, v_res);
#elif defined(__SSE4_2__)
  for (size_t i = 0; i < 2; i++)
  {
    typedef VOpWithSelectedRoundingMode<SubOp<__m128>> Op;
    __m128 v_a = _mm_loadu_ps (a+4*i);
    __m128 v_b = _mm_loadu_ps (b+4*i);
    __m128 v_res = _mm_setzero_ps();
    Op::apply(Op::PackArgs(v_a, v_b), &v_res, context);
    _mm_storeu_ps (res+4*i, v_res);
  }
#else
  typedef VOpWithSelectedRoundingMode<SubOp<float>> Op;
  for (size_t i = 0; i < 8; i++)
  {
    Op::apply(Op::PackArgs(a[i], b[i]), res+i, context);
  }
#endif
}

void INTERFLOP_VECTOR_VERROU_API(sub_float_16)(float *a, float *b, float *res,
                                          void *context) {
#if defined(__AVX2__)
  for (size_t i = 0; i < 2; i++)
  {
    typedef VOpWithSelectedRoundingMode<SubOp<__m256>> Op;
    __m256 v_a = _mm256_loadu_ps (a+8*i);
    __m256 v_b = _mm256_loadu_ps (b+8*i);
    __m256 v_res = _mm256_setzero_ps();
    Op::apply(Op::PackArgs(v_a, v_b), &v_res, context);
    _mm256_storeu_ps (res+8*i, v_res);
  }
#elif defined(__SSE4_2__)
  for (size_t i = 0; i < 4; i++)
  {
    typedef VOpWithSelectedRoundingMode<SubOp<__m128>> Op;
    __m128 v_a = _mm_loadu_ps (a+4*i);
    __m128 v_b = _mm_loadu_ps (b+4*i);
    __m128 v_res = _mm_setzero_ps();
    Op::apply(Op::PackArgs(v_a, v_b), &v_res, context);
    _mm_storeu_ps (res+4*i, v_res);
  }
#else
  typedef VOpWithSelectedRoundingMode<SubOp<float>> Op;
  for (size_t i = 0; i < 16; i++)
  {
    Op::apply(Op::PackArgs(a[i], b[i]), res+i, context);
  }
#endif
}
void INTERFLOP_VECTOR_VERROU_API(mul_float_1)(float *a, float *b, float *res,
                                          void *context) {
  typedef VOpWithSelectedRoundingMode<MulOp<float>> Op;
  Op::apply(Op::PackArgs(*a, *b), res, context);
}

void INTERFLOP_VECTOR_VERROU_API(mul_float_4)(float *a, float *b, float *res,
                                          void *context) {
#if defined(__SSE4_1__)
  typedef VOpWithSelectedRoundingMode<MulOp<__m128>> Op;
  __m128 v_a = _mm_loadu_ps (a);
  __m128 v_b = _mm_loadu_ps (b);
  __m128 v_res = _mm_setzero_ps();
  Op::apply(Op::PackArgs(v_a, v_b), &v_res, context);
  _mm_storeu_ps (res, v_res);
#else
  typedef VOpWithSelectedRoundingMode<MulOp<float>> Op;
  for (size_t i = 0; i < 4; i++)
  {
    Op::apply(Op::PackArgs(a[i], b[i]), res+i, context);
  }
  
#endif
}

void INTERFLOP_VECTOR_VERROU_API(mul_float_8)(float *a, float *b, float *res,
                                          void *context) {
#if defined(__AVX2__)
  typedef VOpWithSelectedRoundingMode<MulOp<__m256>> Op;
  __m256 v_a = _mm256_loadu_ps (a);
  __m256 v_b = _mm256_loadu_ps (b);
  __m256 v_res = _mm256_setzero_ps();
  Op::apply(Op::PackArgs(v_a, v_b), &v_res, context);
  _mm256_storeu_ps (res, v_res);
#elif defined(__SSE4_2__)
  for (size_t i = 0; i < 2; i++)
  {
    typedef VOpWithSelectedRoundingMode<MulOp<__m128>> Op;
    __m128 v_a = _mm_loadu_ps (a+4*i);
    __m128 v_b = _mm_loadu_ps (b+4*i);
    __m128 v_res = _mm_setzero_ps();
    Op::apply(Op::PackArgs(v_a, v_b), &v_res, context);
    _mm_storeu_ps (res+4*i, v_res);
  }
#else
  typedef VOpWithSelectedRoundingMode<MulOp<float>> Op;
  for (size_t i = 0; i < 8; i++)
  {
    Op::apply(Op::PackArgs(a[i], b[i]), res+i, context);
  }
#endif
}

void INTERFLOP_VECTOR_VERROU_API(mul_float_16)(float *a, float *b, float *res,
                                          void *context) {
#if defined(__AVX2__)
  for (size_t i = 0; i < 2; i++)
  {
    typedef VOpWithSelectedRoundingMode<MulOp<__m256>> Op;
    __m256 v_a = _mm256_loadu_ps (a+8*i);
    __m256 v_b = _mm256_loadu_ps (b+8*i);
    __m256 v_res = _mm256_setzero_ps();
    Op::apply(Op::PackArgs(v_a, v_b), &v_res, context);
    _mm256_storeu_ps (res+8*i, v_res);
  }
#elif defined(__SSE4_2__)
  for (size_t i = 0; i < 4; i++)
  {
    typedef VOpWithSelectedRoundingMode<MulOp<__m128>> Op;
    __m128 v_a = _mm_loadu_ps (a+4*i);
    __m128 v_b = _mm_loadu_ps (b+4*i);
    __m128 v_res = _mm_setzero_ps();
    Op::apply(Op::PackArgs(v_a, v_b), &v_res, context);
    _mm_storeu_ps (res+4*i, v_res);
  }
#else
  typedef VOpWithSelectedRoundingMode<MulOp<float>> Op;
  for (size_t i = 0; i < 16; i++)
  {
    Op::apply(Op::PackArgs(a[i], b[i]), res+i, context);
  }
#endif
}

void INTERFLOP_VECTOR_VERROU_API(div_float_1)(float *a, float *b, float *res,
                                          void *context) {
  typedef VOpWithSelectedRoundingMode<DivOp<float>> Op;
  Op::apply(Op::PackArgs(*a, *b), res, context);
}

void INTERFLOP_VECTOR_VERROU_API(div_float_4)(float *a, float *b, float *res,
                                          void *context) {
  typedef VOpWithSelectedRoundingMode<DivOp<float>> Op;
  for (size_t i = 0; i < 4; i++)
  {
    Op::apply(Op::PackArgs(a[i], b[i]), res+i, context);
  }
}

void INTERFLOP_VECTOR_VERROU_API(div_float_8)(float *a, float *b, float *res,
                                          void *context) {
  typedef VOpWithSelectedRoundingMode<DivOp<float>> Op;
  for (size_t i = 0; i < 8; i++)
  {
    Op::apply(Op::PackArgs(a[i], b[i]), res+i, context);
  }

}

void INTERFLOP_VECTOR_VERROU_API(div_float_16)(float *a, float *b, float *res,
                                          void *context) {
  typedef VOpWithSelectedRoundingMode<DivOp<float>> Op;
  for (size_t i = 0; i < 16; i++)
  {
    Op::apply(Op::PackArgs(a[i], b[i]), res+i, context);
  }
}

struct interflop_vector_type_t INTERFLOP_VECTOR_VERROU_API(init)(void *context)
{
  struct interflop_vector_type_t vbackend = {
    add : {
      op_vector_float_1 : INTERFLOP_VECTOR_VERROU_API(add_float_1),
      op_vector_float_4 : INTERFLOP_VECTOR_VERROU_API(add_float_4),
      op_vector_float_8 : INTERFLOP_VECTOR_VERROU_API(add_float_8),
      op_vector_float_16 : INTERFLOP_VECTOR_VERROU_API(add_float_16)
    },
    sub : {
      op_vector_float_1 : INTERFLOP_VECTOR_VERROU_API(sub_float_1),
      op_vector_float_4 : INTERFLOP_VECTOR_VERROU_API(sub_float_4),
      op_vector_float_8 : INTERFLOP_VECTOR_VERROU_API(sub_float_8),
      op_vector_float_16 : INTERFLOP_VECTOR_VERROU_API(sub_float_16)
    },
    mul : {
      op_vector_float_1 : INTERFLOP_VECTOR_VERROU_API(mul_float_1),
      op_vector_float_4 : INTERFLOP_VECTOR_VERROU_API(mul_float_4),
      op_vector_float_8 : INTERFLOP_VECTOR_VERROU_API(mul_float_8),
      op_vector_float_16 : INTERFLOP_VECTOR_VERROU_API(mul_float_16)
    },
    div : {
      op_vector_float_1 : INTERFLOP_VECTOR_VERROU_API(div_float_1),
      op_vector_float_4 : INTERFLOP_VECTOR_VERROU_API(div_float_4),
      op_vector_float_8 : INTERFLOP_VECTOR_VERROU_API(div_float_8),
      op_vector_float_16 : INTERFLOP_VECTOR_VERROU_API(div_float_16)
    }
  };
  return vbackend;
}