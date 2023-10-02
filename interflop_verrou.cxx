
/*--------------------------------------------------------------------*/
/*--- Verrou: a FPU instrumentation tool.                          ---*/
/*--- Interface for floating-point operations overloading.         ---*/
/*---                                         interflop_verrou.cxx ---*/
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

#include <argp.h>
#include <stddef.h>

#include "interflop/prng/vr_rand.h"

#include "interflop/fma/interflop_fma.h"
#include "interflop/interflop_stdlib.h"
#include "interflop/iostream/logger.h"
#include "interflop_verrou.h"
#include "static_backends.hxx"
#include "vr_nextUlp.hxx"
#include "vr_op.hxx"
#include "vr_rand_implem.h"
#include "vr_roundingOp.hxx"

#if defined(VECT512)
#include "x86_64/interflop_vector_verrou_avx512.h"
#endif

#if defined(VECT256)
#include "x86_64/interflop_vector_verrou_avx.h"
#endif

#if defined(VECT128)
#include "x86_64/interflop_vector_verrou_sse.h"
#endif

#if defined(SCALAR)
#include "x86_64/interflop_vector_verrou_scalar.h"
#endif
// * Global variables & parameters

static const char backend_name[] = "interflop-verrou";
static const char backend_version[] = "1.x-dev";

typedef enum { KEY_ROUNDING_MODE, KEY_SEED, KEY_STATIC_BACKEND } key_args;

static const char key_rounding_mode_str[] = "rounding-mode";
static const char key_seed_str[] = "seed";
static const char key_static_backend_str[] = "static-backend";

int CHECK_C = 0;
vr_RoundingMode DEFAULTROUNDINGMODE;
vr_RoundingMode ROUNDINGMODE;
unsigned int vr_seed;
TLS Vr_Rand vr_rand;
static File *stderr_stream;

#if defined(__cplusplus)
extern "C" {
#endif

#ifdef PROFILING_EXACT
unsigned int vr_NumOp;
unsigned int vr_NumExactOp;
#endif

/* public functions */

const char *verrou_rounding_mode_name(enum vr_RoundingMode mode) {
  switch (mode) {
  case VR_NEAREST:
    return "NEAREST";
  case VR_UPWARD:
    return "UPWARD";
  case VR_DOWNWARD:
    return "DOWNWARD";
  case VR_ZERO:
    return "TOWARD_ZERO";
  case VR_RANDOM:
    return "RANDOM";
  case VR_RANDOM_DET:
    return "RANDOM_DET";
  case VR_RANDOM_COMDET:
    return "RANDOM_COMDET";
  case VR_AVERAGE:
    return "AVERAGE";
  case VR_AVERAGE_DET:
    return "AVERAGE_DET";
  case VR_AVERAGE_COMDET:
    return "AVERAGE_COMDET";
  case VR_PRANDOM:
    return "PRANDOM";
  case VR_PRANDOM_DET:
    return "PRANDOM_DET";
  case VR_PRANDOM_COMDET:
    return "PRANDOM_COMDET";
  case VR_FARTHEST:
    return "FARTHEST";
  case VR_FLOAT:
    return "FLOAT";
  case VR_NATIVE:
    return "NATIVE";
  case VR_FTZ:
    return "FTZ";
  }

  return "undefined";
}

void verrou_begin_instr(void *context) {
  verrou_context_t *ctx = (verrou_context_t *)context;
  ctx->rounding_mode = ctx->default_rounding_mode;
}

void verrou_end_instr(void *context) {
  verrou_context_t *ctx = (verrou_context_t *)context;
  ctx->rounding_mode = VR_NEAREST;
}

void verrou_init_profiling_exact(void) {
#ifdef PROFILING_EXACT
  vr_NumOp = 0;
  vr_NumExactOp = 0;
#endif
}

void verrou_get_profiling_exact([[maybe_unused]] unsigned int *num,
                                [[maybe_unused]] unsigned int *numExact) {
#ifdef PROFILING_EXACT
  *num = vr_NumOp;
  *numExact = vr_NumExactOp;
#endif
}

void verrou_set_seed(unsigned int seed) {
  vr_seed = vr_rand_next(&vr_rand);
  vr_rand_setSeed(&vr_rand, seed);
}

void verrou_set_random_seed() { vr_rand_setSeed(&vr_rand, vr_seed); }

double verrou_prandom_pvalue(void) { return vr_rand.p; }

void verrou_updatep_prandom(void) {
  const double p = tinymt64_generate_double(&(vr_rand.gen_));
  vr_rand.p = p;
}

void verrou_updatep_prandom_double(double p) { vr_rand.p = p; }

#define IFV_INLINE inline

IFV_INLINE void INTERFLOP_VERROU_API(add_double)(double a, double b,
                                                 double *res, void *context) {
  typedef OpWithSelectedRoundingMode<AddOp<double>> Op;
  Op::apply(Op::PackArgs(a, b), res, context);
}

IFV_INLINE void INTERFLOP_VERROU_API(add_float)(float a, float b, float *res,
                                                void *context) {
  typedef OpWithSelectedRoundingMode<AddOp<float>> Op;
  Op::apply(Op::PackArgs(a, b), res, context);
}

IFV_INLINE void INTERFLOP_VERROU_API(sub_double)(double a, double b,
                                                 double *res, void *context) {
  typedef OpWithSelectedRoundingMode<SubOp<double>> Op;
  Op::apply(Op::PackArgs(a, b), res, context);
}

IFV_INLINE void INTERFLOP_VERROU_API(sub_float)(float a, float b, float *res,
                                                void *context) {
  typedef OpWithSelectedRoundingMode<SubOp<float>> Op;
  Op::apply(Op::PackArgs(a, b), res, context);
}

IFV_INLINE void INTERFLOP_VERROU_API(mul_double)(double a, double b,
                                                 double *res, void *context) {
  typedef OpWithSelectedRoundingMode<MulOp<double>> Op;
  Op::apply(Op::PackArgs(a, b), res, context);
}

IFV_INLINE void INTERFLOP_VERROU_API(mul_float)(float a, float b, float *res,
                                                void *context) {
  typedef OpWithSelectedRoundingMode<MulOp<float>> Op;
  Op::apply(Op::PackArgs(a, b), res, context);
}

IFV_INLINE void INTERFLOP_VERROU_API(div_double)(double a, double b,
                                                 double *res, void *context) {
  typedef OpWithSelectedRoundingMode<DivOp<double>> Op;
  Op::apply(Op::PackArgs(a, b), res, context);
}

IFV_INLINE void INTERFLOP_VERROU_API(div_float)(float a, float b, float *res,
                                                void *context) {
  typedef OpWithSelectedRoundingMode<DivOp<float>> Op;
  Op::apply(Op::PackArgs(a, b), res, context);
}

IFV_INLINE void INTERFLOP_VERROU_API(cast_double_to_float)(double a, float *res,
                                                           void *context) {
  typedef OpWithSelectedRoundingMode<CastOp<double, float>> Op;
  Op::apply(Op::PackArgs(a), res, context);
}

IFV_INLINE void INTERFLOP_VERROU_API(fma_double)(double a, double b, double c,
                                                 double *res, void *context) {
  typedef OpWithSelectedRoundingMode<MAddOp<double>> Op;
  Op::apply(Op::PackArgs(a, b, c), res, context);
}

IFV_INLINE void INTERFLOP_VERROU_API(fma_float)(float a, float b, float c,
                                                float *res, void *context) {
  typedef OpWithSelectedRoundingMode<MAddOp<float>> Op;
  Op::apply(Op::PackArgs(a, b, c), res, context);
}

static void _interflop_usercall_inexact([[maybe_unused]] void *context,
                                        va_list ap) {
  typedef std::underlying_type<enum FTYPES>::type ftypes_t;
  float xf = 0;
  double xd = 0;
  ftypes_t ftype;
  void *value = NULL;
  ftype = va_arg(ap, ftypes_t);
  value = va_arg(ap, void *);
  switch (ftype) {
  case FFLOAT:
    xf = *((float *)value);
    xf = vr_rand_bool(&vr_rand) ? nextAfter<float>(xf) : nextPrev<float>(xf);
    *((float *)value) = xf;
    break;
  case FDOUBLE:
    xd = *((double *)value);
    xd = vr_rand_bool(&vr_rand) ? nextAfter<double>(xd) : nextPrev<double>(xd);
    *((double *)value) = xd;
    break;
  default:
    interflop_fprintf(
        stderr_stream,
        "Uknown type passed to _interflop_usercall_inexact function");
    break;
  }
}

void INTERFLOP_VERROU_API(user_call)(void *context, interflop_call_id id,
                                     va_list ap) {
  switch (id) {
  case INTERFLOP_INEXACT_ID:
    _interflop_usercall_inexact(context, ap);
    break;
  default:
    interflop_fprintf(stderr_stream, "Unknown interflop_call id (=%d)", id);
    break;
  }
}

void INTERFLOP_VERROU_API(finalize)([[maybe_unused]] void *context) {}

const char *INTERFLOP_VERROU_API(get_backend_name)() { return backend_name; }

const char *INTERFLOP_VERROU_API(get_backend_version)() {
  return backend_version;
}

void _verrou_check_stdlib(void) {
  INTERFLOP_CHECK_IMPL(exit);
  INTERFLOP_CHECK_IMPL(fprintf);
  INTERFLOP_CHECK_IMPL(gettid);
  INTERFLOP_CHECK_IMPL(gettimeofday);
  INTERFLOP_CHECK_IMPL(infHandler);
  INTERFLOP_CHECK_IMPL(malloc);
  INTERFLOP_CHECK_IMPL(nanHandler);
  INTERFLOP_CHECK_IMPL(strcasecmp);
  INTERFLOP_CHECK_IMPL(strtol);
}

void _verrou_alloc_context(void **context) {
  *context = (verrou_context_t *)interflop_malloc(sizeof(verrou_context_t));
}

void _verrou_init_context(verrou_context_t *ctx) {
  ctx->default_rounding_mode = VERROU_ROUDING_MODE_DEFAULT;
  ctx->rounding_mode = VERROU_ROUDING_MODE_DEFAULT; // default value
  ctx->static_backend = VERROU_STATIC_BACKEND_DEFAULT;
  ctx->seed = VERROU_SEED_DEFAULT;
  ctx->choose_seed = false;
}

void INTERFLOP_VERROU_API(pre_init)(interflop_panic_t panic, File *stream,
                                    void **context) {
  stderr_stream = stream;
  interflop_set_handler("panic", (void *)panic);
  _verrou_check_stdlib();
  /* Initialize the logger */
  logger_init(panic, stream, backend_name);
  _verrou_alloc_context(context);
  _verrou_init_context((verrou_context_t *)*context);
}

static struct argp_option end_option = {0, 0, 0, 0, 0, 0};

static struct argp_option options[] = {
    {key_rounding_mode_str, KEY_ROUNDING_MODE, "ROUNDING_MODE", 0,
     "select rounding mode among {nearest, upward, downward, toward_zero, "
     "random, random_det, random_comdet, average, average_det,  "
     "average_comdet, farthest, float, native, ftz}",
     0},
    {key_seed_str, KEY_SEED, "SEED", 0, "fix the random generator seed", 0},
    {key_static_backend_str, KEY_STATIC_BACKEND, 0, 0,
     "load the operators directly instead of switching which makes "
     "computations faster",
     0},
    end_option};

static error_t parse_opt(int key, char *arg, struct argp_state *state) {
  verrou_context_t *ctx = (verrou_context_t *)state->input;
  int error = 0;
  switch (key) {
  case KEY_ROUNDING_MODE:
    if (interflop_strcasecmp("nearest", arg) == 0) {
      ctx->rounding_mode = VR_NEAREST;
    } else if (interflop_strcasecmp("upward", arg) == 0) {
      ctx->rounding_mode = VR_UPWARD;
    } else if (interflop_strcasecmp("downward", arg) == 0) {
      ctx->rounding_mode = VR_DOWNWARD;
    } else if (interflop_strcasecmp("toward_zero", arg) == 0) {
      ctx->rounding_mode = VR_ZERO;
    } else if (interflop_strcasecmp("random", arg) == 0) {
      ctx->rounding_mode = VR_RANDOM;
    } else if (interflop_strcasecmp("random_det", arg) == 0) {
      ctx->rounding_mode = VR_RANDOM_DET;
    } else if (interflop_strcasecmp("random_comdet", arg) == 0) {
      ctx->rounding_mode = VR_RANDOM_COMDET;
    } else if (interflop_strcasecmp("average", arg) == 0) {
      ctx->rounding_mode = VR_AVERAGE;
    } else if (interflop_strcasecmp("average_det", arg) == 0) {
      ctx->rounding_mode = VR_AVERAGE_DET;
    } else if (interflop_strcasecmp("average_comdet", arg) == 0) {
      ctx->rounding_mode = VR_AVERAGE_COMDET;
    } else if (interflop_strcasecmp("prandom", arg) == 0) {
      ctx->rounding_mode = VR_PRANDOM;
    } else if (interflop_strcasecmp("prandom_det", arg) == 0) {
      ctx->rounding_mode = VR_PRANDOM_DET;
    } else if (interflop_strcasecmp("prandom_comdet", arg) == 0) {
      ctx->rounding_mode = VR_PRANDOM_COMDET;
    } else if (interflop_strcasecmp("farthest", arg) == 0) {
      ctx->rounding_mode = VR_FARTHEST;
    } else if (interflop_strcasecmp("float", arg) == 0) {
      ctx->rounding_mode = VR_FLOAT;
    } else if (interflop_strcasecmp("native", arg) == 0) {
      ctx->rounding_mode = VR_NATIVE;
    } else if (interflop_strcasecmp("ftz", arg) == 0) {
      ctx->rounding_mode = VR_FTZ;
    } else {
      interflop_fprintf(stderr_stream,
                        "%s invalid value provided, must be one of: "
                        " nearest, upward, downward, toward_zero, random, "
                        "random_det, random_comdet,average, average_det, "
                        "average_comdet,farthest,float,native,ftz.\n",
                        key_rounding_mode_str);
      interflop_exit(42);
    }
    break;

  case KEY_SEED:
    /* seed */
    error = 0;
    char *endptr;
    ctx->seed = (unsigned long)interflop_strtol(arg, &endptr, &error);
    if (error != 0) {
      interflop_fprintf(stderr_stream,
                        "%s invalid value provided, must be an integer\n",
                        key_seed_str);
      interflop_exit(42);
    }
    ctx->choose_seed = true;
    break;

  case KEY_STATIC_BACKEND:
    /* static backend */
    ctx->static_backend = true;
    break;
  default:
    return ARGP_ERR_UNKNOWN;
  }
  return 0;
}

static struct argp argp = {options, parse_opt, "", "", NULL, NULL, NULL};

void INTERFLOP_VERROU_API(cli)(int argc, char **argv, void *context) {
  verrou_context_t *ctx = (verrou_context_t *)context;
  if (interflop_argp_parse != NULL) {
    interflop_argp_parse(&argp, argc, argv, 0, 0, ctx);
  } else {
    interflop_panic("Interflop backend error: argp_parse not implemented\n"
                    "Provide implementation or use interflop_configure to "
                    "configure the backend\n");
  }

  interflop_fprintf(stderr_stream, "VERROU ROUNDING MODE : %s\n",
                    verrou_rounding_mode_name(ctx->rounding_mode));
}

void INTERFLOP_VERROU_API(configure)(void *configure, void *context) {
  verrou_conf_t *conf = (verrou_conf_t *)configure;
  verrou_context_t *ctx = (verrou_context_t *)context;
  ctx->default_rounding_mode = conf->default_rounding_mode;
  ctx->rounding_mode = conf->rounding_mode;
  ctx->seed = conf->seed;
  ctx->choose_seed = conf->choose_seed;
  ctx->static_backend = conf->static_backend;
}

static void _interflop_set_seed(u_int64_t seed, void *context) {
  verrou_context_t *ctx = (verrou_context_t *)context;
  ctx->seed = seed;

  if (ctx->choose_seed == false) {
    struct timeval t1;
    interflop_gettimeofday(&t1, NULL);
    ctx->seed = t1.tv_sec ^ t1.tv_usec ^ interflop_gettid();
  }

  verrou_set_seed(ctx->seed);
}

static void print_information_header(void *context) {
  /* Environnement variable to disable loading message */
  char *silent_load_env = interflop_getenv("VFC_BACKENDS_SILENT_LOAD");
  bool silent_load = ((silent_load_env == NULL) ||
                      (interflop_strcasecmp(silent_load_env, "True") != 0))
                         ? false
                         : true;

  if (silent_load)
    return;

  verrou_context_t *ctx = (verrou_context_t *)context;
  logger_info("load backend with:\n");
  logger_info("%s = %s\n", key_rounding_mode_str,
              verrou_rounding_mode_name(ctx->rounding_mode));
  logger_info("%s = %llu\n", key_seed_str, ctx->seed);
  logger_info("%s = %s\n", key_static_backend_str,
              ctx->static_backend ? "true" : "false");
}

struct interflop_backend_interface_t _verrou_get_dynamic_backend(void) {
  struct interflop_backend_interface_t interflop_backend_verrou = {
    interflop_add_float : INTERFLOP_VERROU_API(add_float),
    interflop_sub_float : INTERFLOP_VERROU_API(sub_float),
    interflop_mul_float : INTERFLOP_VERROU_API(mul_float),
    interflop_div_float : INTERFLOP_VERROU_API(div_float),
    interflop_cmp_float : NULL,
    interflop_add_double : INTERFLOP_VERROU_API(add_double),
    interflop_sub_double : INTERFLOP_VERROU_API(sub_double),
    interflop_mul_double : INTERFLOP_VERROU_API(mul_double),
    interflop_div_double : INTERFLOP_VERROU_API(div_double),
    interflop_cmp_double : NULL,
    interflop_cast_double_to_float : INTERFLOP_VERROU_API(cast_double_to_float),
    interflop_fma_float : INTERFLOP_VERROU_API(fma_float),
    interflop_fma_double : INTERFLOP_VERROU_API(fma_double),
    interflop_enter_function : NULL,
    interflop_exit_function : NULL,
    interflop_user_call : INTERFLOP_VERROU_API(user_call),
    interflop_finalize : INTERFLOP_VERROU_API(finalize),
    vbackend : {
      scalar : interflop_vector_verrou_init_scalar (nullptr),
      vector128 : interflop_vector_verrou_init_sse (nullptr),
      vector256 : interflop_vector_verrou_init_avx (nullptr),
      vector512 : interflop_vector_verrou_init_avx512 (nullptr)
    }
  };
  return interflop_backend_verrou;
}

struct interflop_backend_interface_t INTERFLOP_VERROU_API(init)(void *context) {
  verrou_context_t *ctx = (verrou_context_t *)context;
  _interflop_set_seed(ctx->seed, context);

  print_information_header(ctx);

  struct interflop_backend_interface_t interflop_verrou_backend =
      (ctx->static_backend) ? get_static_backend(ctx)
                            : _verrou_get_dynamic_backend();

  return interflop_verrou_backend;
}

struct interflop_backend_interface_t interflop_init(void *context)
    __attribute__((weak, alias("interflop_verrou_init")));

void interflop_pre_init(interflop_panic_t panic, File *stream, void **context)
    __attribute__((weak, alias("interflop_verrou_pre_init")));

void interflop_cli(int argc, char **argv, void *context)
    __attribute__((weak, alias("interflop_verrou_cli")));

#if defined(__cplusplus)
}
#endif