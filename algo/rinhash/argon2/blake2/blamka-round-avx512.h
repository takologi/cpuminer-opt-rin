/*
 * Argon2 AVX-512 optimized BlaMka round functions
 *
 * Based on Argon2 reference source code package
 * Copyright 2015 Daniel Dinu, Dmitry Khovratovich, Jean-Philippe Aumasson, Samuel Neves
 *
 * Optimized for AVX-512 (512-bit SIMD)
 * Adapted for Rincoin project
 *
 * Fixed: Uses proper 8-argument macros with SWAP_HALVES/SWAP_QUARTERS
 * for correct inter-register data movement (same as PHC reference)
 */

#ifndef BLAMKA_ROUND_AVX512_H
#define BLAMKA_ROUND_AVX512_H

#include "blake2-impl.h"

#if defined(__AVX512F__)

#include <immintrin.h>

/* Rotation macro for AVX-512 */
#define ror64_avx512(x, n) _mm512_ror_epi64((x), (n))

/* BlaMka multiply-add function: 2*a*b + a + b */
static BLAKE2_INLINE __m512i muladd_avx512(__m512i x, __m512i y) {
    __m512i z = _mm512_mul_epu32(x, y);
    return _mm512_add_epi64(_mm512_add_epi64(x, y), _mm512_add_epi64(z, z));
}

/*
 * 8-argument G functions for AVX-512
 * Processes two groups of 4 registers in parallel
 */
#define G1_AVX512(A0, B0, C0, D0, A1, B1, C1, D1) \
    do { \
        A0 = muladd_avx512(A0, B0); \
        A1 = muladd_avx512(A1, B1); \
        \
        D0 = _mm512_xor_si512(D0, A0); \
        D1 = _mm512_xor_si512(D1, A1); \
        \
        D0 = ror64_avx512(D0, 32); \
        D1 = ror64_avx512(D1, 32); \
        \
        C0 = muladd_avx512(C0, D0); \
        C1 = muladd_avx512(C1, D1); \
        \
        B0 = _mm512_xor_si512(B0, C0); \
        B1 = _mm512_xor_si512(B1, C1); \
        \
        B0 = ror64_avx512(B0, 24); \
        B1 = ror64_avx512(B1, 24); \
    } while ((void)0, 0)

#define G2_AVX512(A0, B0, C0, D0, A1, B1, C1, D1) \
    do { \
        A0 = muladd_avx512(A0, B0); \
        A1 = muladd_avx512(A1, B1); \
        \
        D0 = _mm512_xor_si512(D0, A0); \
        D1 = _mm512_xor_si512(D1, A1); \
        \
        D0 = ror64_avx512(D0, 16); \
        D1 = ror64_avx512(D1, 16); \
        \
        C0 = muladd_avx512(C0, D0); \
        C1 = muladd_avx512(C1, D1); \
        \
        B0 = _mm512_xor_si512(B0, C0); \
        B1 = _mm512_xor_si512(B1, C1); \
        \
        B0 = ror64_avx512(B0, 63); \
        B1 = ror64_avx512(B1, 63); \
    } while ((void)0, 0)

/*
 * 8-argument diagonal shuffles for AVX-512
 * These shuffle within each 512-bit register (within-lane shuffle)
 */
#define DIAGONALIZE_AVX512(A0, B0, C0, D0, A1, B1, C1, D1) \
    do { \
        B0 = _mm512_permutex_epi64(B0, _MM_SHUFFLE(0, 3, 2, 1)); \
        B1 = _mm512_permutex_epi64(B1, _MM_SHUFFLE(0, 3, 2, 1)); \
        \
        C0 = _mm512_permutex_epi64(C0, _MM_SHUFFLE(1, 0, 3, 2)); \
        C1 = _mm512_permutex_epi64(C1, _MM_SHUFFLE(1, 0, 3, 2)); \
        \
        D0 = _mm512_permutex_epi64(D0, _MM_SHUFFLE(2, 1, 0, 3)); \
        D1 = _mm512_permutex_epi64(D1, _MM_SHUFFLE(2, 1, 0, 3)); \
    } while ((void)0, 0)

#define UNDIAGONALIZE_AVX512(A0, B0, C0, D0, A1, B1, C1, D1) \
    do { \
        B0 = _mm512_permutex_epi64(B0, _MM_SHUFFLE(2, 1, 0, 3)); \
        B1 = _mm512_permutex_epi64(B1, _MM_SHUFFLE(2, 1, 0, 3)); \
        \
        C0 = _mm512_permutex_epi64(C0, _MM_SHUFFLE(1, 0, 3, 2)); \
        C1 = _mm512_permutex_epi64(C1, _MM_SHUFFLE(1, 0, 3, 2)); \
        \
        D0 = _mm512_permutex_epi64(D0, _MM_SHUFFLE(0, 3, 2, 1)); \
        D1 = _mm512_permutex_epi64(D1, _MM_SHUFFLE(0, 3, 2, 1)); \
    } while ((void)0, 0)

/*
 * Basic BlaMka round with 8 arguments
 */
#define BLAKE2_ROUND_AVX512(A0, B0, C0, D0, A1, B1, C1, D1) \
    do { \
        G1_AVX512(A0, B0, C0, D0, A1, B1, C1, D1); \
        G2_AVX512(A0, B0, C0, D0, A1, B1, C1, D1); \
        \
        DIAGONALIZE_AVX512(A0, B0, C0, D0, A1, B1, C1, D1); \
        \
        G1_AVX512(A0, B0, C0, D0, A1, B1, C1, D1); \
        G2_AVX512(A0, B0, C0, D0, A1, B1, C1, D1); \
        \
        UNDIAGONALIZE_AVX512(A0, B0, C0, D0, A1, B1, C1, D1); \
    } while ((void)0, 0)

/*
 * Inter-register shuffle macros for row/column processing
 * These move data between the two 512-bit registers of a pair
 */
#define SWAP_HALVES_AVX512(A0, A1) \
    do { \
        __m512i t0, t1; \
        t0 = _mm512_shuffle_i64x2(A0, A1, _MM_SHUFFLE(1, 0, 1, 0)); \
        t1 = _mm512_shuffle_i64x2(A0, A1, _MM_SHUFFLE(3, 2, 3, 2)); \
        A0 = t0; \
        A1 = t1; \
    } while((void)0, 0)

#define SWAP_QUARTERS_AVX512(A0, A1) \
    do { \
        SWAP_HALVES_AVX512(A0, A1); \
        A0 = _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 1, 4, 5, 2, 3, 6, 7), A0); \
        A1 = _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 1, 4, 5, 2, 3, 6, 7), A1); \
    } while((void)0, 0)

#define UNSWAP_QUARTERS_AVX512(A0, A1) \
    do { \
        A0 = _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 1, 4, 5, 2, 3, 6, 7), A0); \
        A1 = _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 1, 4, 5, 2, 3, 6, 7), A1); \
        SWAP_HALVES_AVX512(A0, A1); \
    } while((void)0, 0)

/*
 * BLAKE2_ROUND_1: For processing rows
 * Uses SWAP_HALVES to reorganize data for row processing
 */
#define BLAKE2_ROUND_1_AVX512(A0, C0, B0, D0, A1, C1, B1, D1) \
    do { \
        SWAP_HALVES_AVX512(A0, B0); \
        SWAP_HALVES_AVX512(C0, D0); \
        SWAP_HALVES_AVX512(A1, B1); \
        SWAP_HALVES_AVX512(C1, D1); \
        BLAKE2_ROUND_AVX512(A0, B0, C0, D0, A1, B1, C1, D1); \
        SWAP_HALVES_AVX512(A0, B0); \
        SWAP_HALVES_AVX512(C0, D0); \
        SWAP_HALVES_AVX512(A1, B1); \
        SWAP_HALVES_AVX512(C1, D1); \
    } while ((void)0, 0)

/*
 * BLAKE2_ROUND_2: For processing columns
 * Uses SWAP_QUARTERS for column-wise data reorganization
 */
#define BLAKE2_ROUND_2_AVX512(A0, A1, B0, B1, C0, C1, D0, D1) \
    do { \
        SWAP_QUARTERS_AVX512(A0, A1); \
        SWAP_QUARTERS_AVX512(B0, B1); \
        SWAP_QUARTERS_AVX512(C0, C1); \
        SWAP_QUARTERS_AVX512(D0, D1); \
        BLAKE2_ROUND_AVX512(A0, B0, C0, D0, A1, B1, C1, D1); \
        UNSWAP_QUARTERS_AVX512(A0, A1); \
        UNSWAP_QUARTERS_AVX512(B0, B1); \
        UNSWAP_QUARTERS_AVX512(C0, C1); \
        UNSWAP_QUARTERS_AVX512(D0, D1); \
    } while ((void)0, 0)

/* Number of 512-bit words in an Argon2 block (1024 bytes / 64 bytes = 16) */
#define ARGON2_512BIT_WORDS_IN_BLOCK 16

#endif /* __AVX512F__ */

#endif /* BLAMKA_ROUND_AVX512_H */
