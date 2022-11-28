ACLOCAL_AMFLAGS=-I m4
lib_LTLIBRARIES = libinterflop_verrou.la libinterflop_verrou_no-tls.la

if ENABLE_LTO
LTO_FLAGS = -flto
else
LTO_FLAGS =
endif

if ENABLE_WARNINGS
WARNING_FLAGS = -Wall -Wextra -Wno-varargs
else
WARNING_FLAGS =
endif

# Backend version with TLS enabled
libinterflop_verrou_la_SOURCES = interflop_verrou.cxx

libinterflop_verrou_la_CFLAGS = \
    -I@INTERFLOP_INCLUDEDIR@/ \
    -DVERROU_DET_HASH=vr_@vg_cv_verrou_det_hash@_hash \
    -DVERROU_NUM_AVG=@VERROU_NUM_AVG@ \
    -DRNG_THREAD_SAFE \
    -fno-stack-protector $(LTO_FLAGS) -O2 \
    $(WARNING_FLAGS)

libinterflop_verrou_la_CXXFLAGS = \
    -I@INTERFLOP_INCLUDEDIR@/ \
    -DVERROU_DET_HASH=vr_@vg_cv_verrou_det_hash@_hash \
    -DVERROU_NUM_AVG=@VERROU_NUM_AVG@ \
    -DRNG_THREAD_SAFE \
    -fno-stack-protector $(LTO_FLAGS) -O2 \
    $(WARNING_FLAGS)

libinterflop_verrou_la_LDFLAGS = $(LTO_FLAGS) -O2

libinterflop_verrou_la_LIBADD = \
    @INTERFLOP_LIBDIR@/libinterflop_prng.la \
    @INTERFLOP_LIBDIR@/libinterflop_fma.la \
    @INTERFLOP_LIBDIR@/libinterflop_logger.la \
    @INTERFLOP_LIBDIR@/libinterflop_stdlib.la

# Backend version with TLS disabled
libinterflop_verrou_no_tls_la_SOURCES = interflop_verrou.cxx

libinterflop_verrou_no_tls_la_CFLAGS = \
    -I@INTERFLOP_INCLUDEDIR@/ \
    -DVERROU_DET_HASH=vr_@vg_cv_verrou_det_hash@_hash \
    -DVERROU_NUM_AVG=@VERROU_NUM_AVG@ \
    -fno-stack-protector $(LTO_FLAGS) -O2 \
    $(WARNING_FLAGS)

libinterflop_verrou_no_tls_la_CXXFLAGS = \
    -I@INTERFLOP_INCLUDEDIR@/ \
    -DVERROU_DET_HASH=vr_@vg_cv_verrou_det_hash@_hash \
    -DVERROU_NUM_AVG=@VERROU_NUM_AVG@ \
    -fno-stack-protector $(LTO_FLAGS) -O2 \
    $(WARNING_FLAGS)

libinterflop_verrou_no_tls_la_LDFLAGS = $(LTO_FLAGS) -O2

libinterflop_verrou_no_tls_la_LIBADD = \
    @INTERFLOP_LIBDIR@/libinterflop_prng.la \
    @INTERFLOP_LIBDIR@/libinterflop_fma.la \
    @INTERFLOP_LIBDIR@/libinterflop_logger.la \
    @INTERFLOP_LIBDIR@/libinterflop_stdlib.la

includesdir=$(includedir)/interflop
includes_HEADERS= interflop_verrou.h

