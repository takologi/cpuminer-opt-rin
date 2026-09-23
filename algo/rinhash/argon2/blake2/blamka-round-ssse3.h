/*
 * Argon2 SSSE3 optimized BlaMka round functions
 *
 * Based on Argon2 reference source code package
 * Copyright 2015 Daniel Dinu, Dmitry Khovratovich, Jean-Philippe Aumasson, Samuel Neves
 *
 * Optimized for SSSE3 (128-bit SIMD with shuffle)
 * Adapted for Rincoin project
 */

#ifndef BLAMKA_ROUND_SSSE3_H
#define BLAMKA_ROUND_SSSE3_H

#include "blake2-impl.h"

#if defined(__SSSE3__) || defined(__SSE2__)

#include <emmintrin.h>  /* SSE2 */
#ifdef __SSSE3__
#include <tmmintrin.h>  /* SSSE3 */
#endif

/*
 * Rotation macros for SSSE3/SSE2
 * SSSE3 uses pshufb for efficient byte shuffles
 * SSE2 falls back to shift/or for rotations
 */
#ifdef __SSSE3__

/* SSSE3: Use shuffle for 16 and 24-bit rotations of 64-bit lanes */
#define r16_ssse3 \
    (_mm_setr_epi8(2, 3, 4, 5, 6, 7, 0, 1, 10, 11, 12, 13, 14, 15, 8, 9))
#define r24_ssse3 \
    (_mm_setr_epi8(3, 4, 5, 6, 7, 0, 1, 2, 11, 12, 13, 14, 15, 8, 9, 10))

#define rotr32_ssse3(x) _mm_shuffle_epi32((x), _MM_SHUFFLE(2, 3, 0, 1))
#define rotr24_ssse3(x) _mm_shuffle_epi8((x), r24_ssse3)
#define rotr16_ssse3(x) _mm_shuffle_epi8((x), r16_ssse3)
#define rotr63_ssse3(x) _mm_xor_si128(_mm_srli_epi64((x), 63), _mm_add_epi64((x), (x)))

#else /* SSE2 fallback */

#define rotr32_ssse3(x) _mm_shuffle_epi32((x), _MM_SHUFFLE(2, 3, 0, 1))
#define rotr24_ssse3(x) _mm_xor_si128(_mm_srli_epi64((x), 24), _mm_slli_epi64((x), 40))
#define rotr16_ssse3(x) _mm_xor_si128(_mm_srli_epi64((x), 16), _mm_slli_epi64((x), 48))
#define rotr63_ssse3(x) _mm_xor_si128(_mm_srli_epi64((x), 63), _mm_add_epi64((x), (x)))

#endif /* __SSSE3__ */

/*
 * BlaMka fBlaMka function: 2*a*b + a + b
 * This is the core mixing function used in Argon2
 */
static BLAKE2_INLINE __m128i fBlaMka_ssse3(__m128i x, __m128i y) {
    const __m128i z = _mm_mul_epu32(x, y);
    return _mm_add_epi64(_mm_add_epi64(x, y), _mm_add_epi64(z, z));
}

/*
 * BlaMka G function for SSSE3
 * Processes 2 64-bit values in parallel using 128-bit registers
 */
#define G1_SSSE3(A0, B0, C0, D0, A1, B1, C1, D1) \
    do { \
        A0 = fBlaMka_ssse3(A0, B0); \
        A1 = fBlaMka_ssse3(A1, B1); \
        \
        D0 = _mm_xor_si128(D0, A0); \
        D1 = _mm_xor_si128(D1, A1); \
        \
        D0 = rotr32_ssse3(D0); \
        D1 = rotr32_ssse3(D1); \
        \
        C0 = fBlaMka_ssse3(C0, D0); \
        C1 = fBlaMka_ssse3(C1, D1); \
        \
        B0 = _mm_xor_si128(B0, C0); \
        B1 = _mm_xor_si128(B1, C1); \
        \
        B0 = rotr24_ssse3(B0); \
        B1 = rotr24_ssse3(B1); \
    } while ((void)0, 0)

#define G2_SSSE3(A0, B0, C0, D0, A1, B1, C1, D1) \
    do { \
        A0 = fBlaMka_ssse3(A0, B0); \
        A1 = fBlaMka_ssse3(A1, B1); \
        \
        D0 = _mm_xor_si128(D0, A0); \
        D1 = _mm_xor_si128(D1, A1); \
        \
        D0 = rotr16_ssse3(D0); \
        D1 = rotr16_ssse3(D1); \
        \
        C0 = fBlaMka_ssse3(C0, D0); \
        C1 = fBlaMka_ssse3(C1, D1); \
        \
        B0 = _mm_xor_si128(B0, C0); \
        B1 = _mm_xor_si128(B1, C1); \
        \
        B0 = rotr63_ssse3(B0); \
        B1 = rotr63_ssse3(B1); \
    } while ((void)0, 0)

/*
 * Complete BlaMka round macro for SSSE3
 * Processes 8 128-bit registers (16 64-bit values)
 * 
 * NOTE: The diagonalize/undiagonalize operates ACROSS register pairs (A0/A1, B0/B1, etc.)
 * using alignr to move 64-bit values between registers
 */
#ifdef __SSSE3__
#define DIAGONALIZE_SSSE3(A0, B0, C0, D0, A1, B1, C1, D1)                      \
    do {                                                                       \
        __m128i t0 = _mm_alignr_epi8(B1, B0, 8);                               \
        __m128i t1 = _mm_alignr_epi8(B0, B1, 8);                               \
        B0 = t0;                                                               \
        B1 = t1;                                                               \
                                                                               \
        t0 = C0;                                                               \
        C0 = C1;                                                               \
        C1 = t0;                                                               \
                                                                               \
        t0 = _mm_alignr_epi8(D1, D0, 8);                                       \
        t1 = _mm_alignr_epi8(D0, D1, 8);                                       \
        D0 = t1;                                                               \
        D1 = t0;                                                               \
    } while ((void)0, 0)

#define UNDIAGONALIZE_SSSE3(A0, B0, C0, D0, A1, B1, C1, D1)                    \
    do {                                                                       \
        __m128i t0 = _mm_alignr_epi8(B0, B1, 8);                               \
        __m128i t1 = _mm_alignr_epi8(B1, B0, 8);                               \
        B0 = t0;                                                               \
        B1 = t1;                                                               \
                                                                               \
        t0 = C0;                                                               \
        C0 = C1;                                                               \
        C1 = t0;                                                               \
                                                                               \
        t0 = _mm_alignr_epi8(D0, D1, 8);                                       \
        t1 = _mm_alignr_epi8(D1, D0, 8);                                       \
        D0 = t1;                                                               \
        D1 = t0;                                                               \
    } while ((void)0, 0)
#else /* SSE2 fallback */
#define DIAGONALIZE_SSSE3(A0, B0, C0, D0, A1, B1, C1, D1)                      \
    do {                                                                       \
        __m128i t0 = D0;                                                       \
        __m128i t1 = B0;                                                       \
        D0 = C0;                                                               \
        C0 = C1;                                                               \
        C1 = D0;                                                               \
        D0 = _mm_unpackhi_epi64(D1, _mm_unpacklo_epi64(t0, t0));               \
        D1 = _mm_unpackhi_epi64(t0, _mm_unpacklo_epi64(D1, D1));               \
        B0 = _mm_unpackhi_epi64(B0, _mm_unpacklo_epi64(B1, B1));               \
        B1 = _mm_unpackhi_epi64(B1, _mm_unpacklo_epi64(t1, t1));               \
    } while ((void)0, 0)

#define UNDIAGONALIZE_SSSE3(A0, B0, C0, D0, A1, B1, C1, D1)                    \
    do {                                                                       \
        __m128i t0, t1;                                                        \
        t0 = C0;                                                               \
        C0 = C1;                                                               \
        C1 = t0;                                                               \
        t0 = B0;                                                               \
        t1 = D0;                                                               \
        B0 = _mm_unpackhi_epi64(B1, _mm_unpacklo_epi64(B0, B0));               \
        B1 = _mm_unpackhi_epi64(t0, _mm_unpacklo_epi64(B1, B1));               \
        D0 = _mm_unpackhi_epi64(D0, _mm_unpacklo_epi64(D1, D1));               \
        D1 = _mm_unpackhi_epi64(D1, _mm_unpacklo_epi64(t1, t1));               \
    } while ((void)0, 0)
#endif

#define BLAKE2_ROUND_SSSE3(A0, A1, B0, B1, C0, C1, D0, D1)                     \
    do {                                                                       \
        G1_SSSE3(A0, B0, C0, D0, A1, B1, C1, D1);                              \
        G2_SSSE3(A0, B0, C0, D0, A1, B1, C1, D1);                              \
                                                                               \
        DIAGONALIZE_SSSE3(A0, B0, C0, D0, A1, B1, C1, D1);                     \
                                                                               \
        G1_SSSE3(A0, B0, C0, D0, A1, B1, C1, D1);                              \
        G2_SSSE3(A0, B0, C0, D0, A1, B1, C1, D1);                              \
                                                                               \
        UNDIAGONALIZE_SSSE3(A0, B0, C0, D0, A1, B1, C1, D1);                   \
    } while ((void)0, 0)

/* Number of 128-bit words in an Argon2 block (1024 bytes / 16 bytes = 64) */
#define ARGON2_OWORDS_IN_BLOCK 64

#endif /* __SSSE3__ || __SSE2__ */

#endif /* BLAMKA_ROUND_SSSE3_H */
