/*
 * Argon2 CPU feature detection and dispatch
 *
 * Provides runtime selection of optimal Argon2 implementation based on
 * available CPU features (AVX2, SSSE3, SSE2, or reference fallback).
 *
 * Adapted for Rincoin project
 */

#ifndef ARGON2_DISPATCH_H
#define ARGON2_DISPATCH_H

#include "argon2.h"
#include "core.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * CPU feature flags
 */
typedef enum {
    ARGON2_CPU_FEATURE_NONE   = 0,
    ARGON2_CPU_FEATURE_SSE2   = (1 << 0),
    ARGON2_CPU_FEATURE_SSSE3  = (1 << 1),
    ARGON2_CPU_FEATURE_SSE41  = (1 << 2),
    ARGON2_CPU_FEATURE_AVX    = (1 << 3),
    ARGON2_CPU_FEATURE_AVX2   = (1 << 4),
    ARGON2_CPU_FEATURE_AVX512F = (1 << 5)
} argon2_cpu_features_t;

/*
 * Implementation type enum
 */
typedef enum {
    ARGON2_IMPL_REF = 0,     /* Reference (portable) implementation */
    ARGON2_IMPL_SSE2,        /* SSE2 optimized (128-bit basic SIMD) */
    ARGON2_IMPL_SSSE3,       /* SSSE3 optimized (128-bit with shuffle) */
    ARGON2_IMPL_AVX2,        /* AVX2 optimized (256-bit SIMD) */
    ARGON2_IMPL_AVX512,      /* AVX-512 optimized (512-bit SIMD) */
    ARGON2_IMPL_COUNT
} argon2_impl_type_t;

/*
 * Function pointer type for fill_segment implementations
 */
typedef void (*argon2_fill_segment_fn)(const argon2_instance_t *instance,
                                       argon2_position_t position);

/*
 * Detect available CPU features
 * Returns a bitmask of argon2_cpu_features_t flags
 */
ARGON2_PUBLIC int argon2_detect_cpu_features(void);

/*
 * Get the best available implementation based on CPU features
 * Returns the implementation type that will be used
 */
ARGON2_PUBLIC argon2_impl_type_t argon2_get_best_impl(void);

/*
 * Get human-readable name of current implementation
 */
ARGON2_PUBLIC const char *argon2_get_impl_name(void);

/*
 * Initialize the dispatch system
 * Must be called before first use (called automatically on first use)
 * Thread-safe, can be called multiple times
 */
ARGON2_PUBLIC void argon2_dispatch_init(void);

/*
 * Force a specific implementation (for testing/debugging)
 * Set to ARGON2_IMPL_REF to force reference implementation
 * Set to ARGON2_IMPL_COUNT to auto-detect (default)
 */
ARGON2_PUBLIC void argon2_set_impl(argon2_impl_type_t impl);

/*
 * Dispatched fill_segment function
 * Automatically selects the best available implementation
 */
ARGON2_PUBLIC void fill_segment_dispatch(const argon2_instance_t *instance,
                                         argon2_position_t position);

/*
 * Individual implementation declarations
 */

/* Reference implementation (always available) */
void fill_segment_ref(const argon2_instance_t *instance,
                      argon2_position_t position);

/* SSSE3 implementation (x86/x86-64 only) */
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
void fill_segment_ssse3(const argon2_instance_t *instance,
                        argon2_position_t position);
#endif

/* AVX2 implementation (x86-64 only) */
#if defined(__x86_64__) || defined(_M_X64)
void fill_segment_avx2(const argon2_instance_t *instance,
                       argon2_position_t position);
#endif

/* AVX-512 implementation (x86-64 only) */
#if defined(__x86_64__) || defined(_M_X64)
void fill_segment_avx512(const argon2_instance_t *instance,
                         argon2_position_t position);
#endif

#ifdef __cplusplus
}
#endif

#endif /* ARGON2_DISPATCH_H */
