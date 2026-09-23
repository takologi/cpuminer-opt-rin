/*
 * Argon2 CPU feature detection and runtime dispatch
 *
 * Provides runtime selection of optimal Argon2 implementation based on
 * available CPU features. Selection priority: AVX512 > AVX2 > SSSE3 > reference.
 *
 * PERFORMANCE IMPACT:
 * - AVX2:    ~50% faster than reference on Intel/AMD x86-64
 * - AVX512:  ~10-20% faster than AVX2 on supported CPUs
 * - SSSE3:   ~20-30% faster than reference
 *
 * The dispatch is initialized once per process and cached. The selected
 * fill_segment function pointer is called directly without additional overhead.
 *
 * Adapted for Rincoin RinHash algorithm.
 */

#include <stdint.h>
#include <string.h>
#include <stdio.h>

#include "argon2_dispatch.h"
#include "argon2.h"
#include "core.h"

/*
 * Check which implementations are compiled in
 * These are defined by the build system
 */
#if defined(__x86_64__) || defined(_M_X64)
#define HAVE_AVX512_IMPL 1
#define HAVE_AVX2_IMPL 1
#define HAVE_SSSE3_IMPL 1
#elif defined(__i386__) || defined(_M_IX86)
#define HAVE_AVX512_IMPL 0
#define HAVE_AVX2_IMPL 0
#define HAVE_SSSE3_IMPL 1
#else
#define HAVE_AVX512_IMPL 0
#define HAVE_AVX2_IMPL 0
#define HAVE_SSSE3_IMPL 0
#endif

/*
 * CPU feature detection for x86/x86-64
 */
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif

/* CPUID helpers */
static void get_cpuid(uint32_t leaf, uint32_t subleaf, uint32_t *eax,
                      uint32_t *ebx, uint32_t *ecx, uint32_t *edx) {
#ifdef _MSC_VER
    int regs[4];
    __cpuidex(regs, (int)leaf, (int)subleaf);
    *eax = (uint32_t)regs[0];
    *ebx = (uint32_t)regs[1];
    *ecx = (uint32_t)regs[2];
    *edx = (uint32_t)regs[3];
#else
    __cpuid_count(leaf, subleaf, *eax, *ebx, *ecx, *edx);
#endif
}

/* Read XCR0 (extended control register) for AVX state check */
static uint64_t get_xcr0(void) {
#ifdef _MSC_VER
    return _xgetbv(0);
#else
    uint32_t eax, edx;
    __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return ((uint64_t)edx << 32) | eax;
#endif
}

int argon2_detect_cpu_features(void) {
    int features = ARGON2_CPU_FEATURE_NONE;
    uint32_t eax, ebx, ecx, edx;
    uint32_t max_leaf;

    /* Get maximum supported CPUID leaf */
    get_cpuid(0, 0, &max_leaf, &ebx, &ecx, &edx);
    if (max_leaf < 1) {
        return features;
    }

    /* CPUID leaf 1: Basic feature flags */
    get_cpuid(1, 0, &eax, &ebx, &ecx, &edx);

    /* SSE2: EDX bit 26 */
    if (edx & (1U << 26)) {
        features |= ARGON2_CPU_FEATURE_SSE2;
    }

    /* SSSE3: ECX bit 9 */
    if (ecx & (1U << 9)) {
        features |= ARGON2_CPU_FEATURE_SSSE3;
    }

    /* SSE4.1: ECX bit 19 */
    if (ecx & (1U << 19)) {
        features |= ARGON2_CPU_FEATURE_SSE41;
    }

    /* Check for AVX support (requires OS support via XSAVE) */
    /* OSXSAVE: ECX bit 27, AVX: ECX bit 28 */
    if ((ecx & (1U << 27)) && (ecx & (1U << 28))) {
        /* Verify OS has enabled AVX state saving */
        uint64_t xcr0 = get_xcr0();
        if ((xcr0 & 0x6) == 0x6) {  /* XMM and YMM state enabled */
            features |= ARGON2_CPU_FEATURE_AVX;

            /* Check for AVX2 (CPUID leaf 7) */
            if (max_leaf >= 7) {
                get_cpuid(7, 0, &eax, &ebx, &ecx, &edx);

                /* AVX2: EBX bit 5 */
                if (ebx & (1U << 5)) {
                    features |= ARGON2_CPU_FEATURE_AVX2;
                }

                /* AVX-512F: EBX bit 16 (for future use) */
                if ((xcr0 & 0xE0) == 0xE0) {  /* ZMM state enabled */
                    if (ebx & (1U << 16)) {
                        features |= ARGON2_CPU_FEATURE_AVX512F;
                    }
                }
            }
        }
    }

    return features;
}

#elif defined(__aarch64__) || defined(_M_ARM64)
/* ARM64: No SIMD optimization currently, return no features */
int argon2_detect_cpu_features(void) {
    return ARGON2_CPU_FEATURE_NONE;
}

#elif defined(__arm__) || defined(_M_ARM)
/* ARM32: No SIMD optimization currently, return no features */
int argon2_detect_cpu_features(void) {
    return ARGON2_CPU_FEATURE_NONE;
}

#else
/* Unknown architecture: No SIMD optimization */
int argon2_detect_cpu_features(void) {
    return ARGON2_CPU_FEATURE_NONE;
}

#endif /* Architecture detection */

/*
 * Global dispatch state
 */
static volatile int g_dispatch_initialized = 0;
static argon2_impl_type_t g_current_impl = ARGON2_IMPL_REF;
static argon2_fill_segment_fn g_fill_segment_fn = NULL;
static int g_cpu_features = 0;

/* Forward declaration of reference implementation */
extern void fill_segment(const argon2_instance_t *instance,
                         argon2_position_t position);

/*
 * Implementation name strings
 */
static const char *impl_names[] = {
    "reference",
    "SSE2",
    "SSSE3",
    "AVX2",
    "AVX-512"
};

const char *argon2_get_impl_name(void) {
    if (!g_dispatch_initialized) {
        argon2_dispatch_init();
    }
    return impl_names[g_current_impl];
}

argon2_impl_type_t argon2_get_best_impl(void) {
    if (!g_dispatch_initialized) {
        argon2_dispatch_init();
    }
    return g_current_impl;
}

/*
 * Select best implementation based on detected CPU features
 * and compiled-in implementations
 */
static argon2_fill_segment_fn select_impl(int features, argon2_impl_type_t *impl_type) {
    *impl_type = ARGON2_IMPL_REF;

#if HAVE_AVX512_IMPL
    /* x86-64: Check for AVX-512 first (best) */
    if (features & ARGON2_CPU_FEATURE_AVX512F) {
        *impl_type = ARGON2_IMPL_AVX512;
        return fill_segment_avx512;
    }
#endif

#if HAVE_AVX2_IMPL
    /* x86-64: Check for AVX2 */
    if (features & ARGON2_CPU_FEATURE_AVX2) {
        *impl_type = ARGON2_IMPL_AVX2;
        return fill_segment_avx2;
    }
#endif

#if HAVE_SSSE3_IMPL
    /* x86/x86-64: Check for SSSE3 (better shuffles) or SSE2 */
    if (features & ARGON2_CPU_FEATURE_SSSE3) {
        *impl_type = ARGON2_IMPL_SSSE3;
        return fill_segment_ssse3;
    }
    /* SSE2-only CPUs use the reference path: argon2_ssse3.c is built with
     * -mssse3 and would fault on pshufb */
#endif

    /* Fallback to reference implementation */
    *impl_type = ARGON2_IMPL_REF;
    return fill_segment;  /* Original reference fill_segment from ref.c */
}

/*
 * Initialize dispatch system
 */
void argon2_dispatch_init(void) {
    /* Simple double-check locking pattern */
    if (g_dispatch_initialized) {
        return;
    }

    /* Detect CPU features */
    g_cpu_features = argon2_detect_cpu_features();

    /* Select best implementation */
    g_fill_segment_fn = select_impl(g_cpu_features, &g_current_impl);

    /* Memory barrier to ensure all writes are visible */
#ifdef __GNUC__
    __sync_synchronize();
#elif defined(_MSC_VER)
    _ReadWriteBarrier();
#endif

    g_dispatch_initialized = 1;
}

/*
 * Force a specific implementation
 */
void argon2_set_impl(argon2_impl_type_t impl) {
    if (!g_dispatch_initialized) {
        argon2_dispatch_init();
    }

    switch (impl) {
    case ARGON2_IMPL_REF:
        g_current_impl = ARGON2_IMPL_REF;
        g_fill_segment_fn = fill_segment;
        break;

#if HAVE_SSSE3_IMPL
    case ARGON2_IMPL_SSE2:
    case ARGON2_IMPL_SSSE3:
        if (g_cpu_features & ARGON2_CPU_FEATURE_SSSE3) {
            g_current_impl = ARGON2_IMPL_SSSE3;
            g_fill_segment_fn = fill_segment_ssse3;
        }
        break;
#endif

#if HAVE_AVX2_IMPL
    case ARGON2_IMPL_AVX2:
        if (g_cpu_features & ARGON2_CPU_FEATURE_AVX2) {
            g_current_impl = ARGON2_IMPL_AVX2;
            g_fill_segment_fn = fill_segment_avx2;
        }
        break;
#endif

#if HAVE_AVX512_IMPL
    case ARGON2_IMPL_AVX512:
        if (g_cpu_features & ARGON2_CPU_FEATURE_AVX512F) {
            g_current_impl = ARGON2_IMPL_AVX512;
            g_fill_segment_fn = fill_segment_avx512;
        }
        break;
#endif

    default:
        /* Auto-detect: re-run selection */
        g_fill_segment_fn = select_impl(g_cpu_features, &g_current_impl);
        break;
    }
}

/*
 * Dispatched fill_segment function
 * This is the main entry point called by core.c
 */
void fill_segment_dispatch(const argon2_instance_t *instance,
                           argon2_position_t position) {
    if (__builtin_expect(!g_dispatch_initialized, 0)) {
        argon2_dispatch_init();
    }

    /* Call selected implementation */
    g_fill_segment_fn(instance, position);
}

/*
 * Wrapper for reference implementation to match expected signature
 * The original fill_segment from ref.c is used directly
 */
void fill_segment_ref(const argon2_instance_t *instance,
                      argon2_position_t position) {
    fill_segment(instance, position);
}
