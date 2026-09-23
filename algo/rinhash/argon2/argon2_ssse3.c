/*
 * Argon2 SSSE3/SSE2 optimized implementation
 *
 * Based on Argon2 reference source code package
 * Copyright 2015 Daniel Dinu, Dmitry Khovratovich, Jean-Philippe Aumasson, Samuel Neves
 *
 * SSSE3/SSE2 optimized version for Rincoin
 * Uses 128-bit SIMD operations for ~1.5-2x speedup over reference
 * Falls back to SSE2 if SSSE3 not available
 */

#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#include "argon2.h"
#include "core.h"

/* Only compile SSSE3/SSE2 code on x86/x86-64 */
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#if defined(__SSSE3__) || defined(__SSE2__) || defined(__GNUC__) || defined(_MSC_VER)

#ifdef __GNUC__
#include <x86intrin.h>
#else
#include <intrin.h>
#endif

#include <emmintrin.h>  /* SSE2 */
#ifdef __SSSE3__
#include <tmmintrin.h>  /* SSSE3 */
#endif

#include "blake2/blamka-round-ssse3.h"
#include "blake2/blake2-impl.h"
#include "blake2/blake2.h"

/* Number of 128-bit words in an Argon2 block */
#ifndef ARGON2_OWORDS_IN_BLOCK
#define ARGON2_OWORDS_IN_BLOCK 64
#endif

/*
 * Fill a memory block using SSSE3/SSE2 SIMD operations
 *
 * @param state     Current state (128-bit registers)
 * @param ref_block Reference block to mix with
 * @param next_block Output block
 * @param with_xor  Whether to XOR with existing next_block content
 */
#if defined(__SSSE3__) || defined(__SSE2__)
static void fill_block_ssse3(__m128i *state, const block *ref_block,
                             block *next_block, int with_xor) {
    __m128i block_XY[ARGON2_OWORDS_IN_BLOCK];
    unsigned int i;

    if (with_xor) {
        for (i = 0; i < ARGON2_OWORDS_IN_BLOCK; i++) {
            state[i] = _mm_xor_si128(
                state[i], _mm_loadu_si128((const __m128i *)ref_block->v + i));
            block_XY[i] = _mm_xor_si128(
                state[i], _mm_loadu_si128((const __m128i *)next_block->v + i));
        }
    } else {
        for (i = 0; i < ARGON2_OWORDS_IN_BLOCK; i++) {
            block_XY[i] = state[i] = _mm_xor_si128(
                state[i], _mm_loadu_si128((const __m128i *)ref_block->v + i));
        }
    }

    /* Apply BlaMka rounds - 8 groups of 8 128-bit words */
    for (i = 0; i < 8; ++i) {
        BLAKE2_ROUND_SSSE3(
            state[8 * i + 0], state[8 * i + 1],
            state[8 * i + 2], state[8 * i + 3],
            state[8 * i + 4], state[8 * i + 5],
            state[8 * i + 6], state[8 * i + 7]);
    }

    /* Apply BlaMka rounds - interleaved */
    for (i = 0; i < 8; ++i) {
        BLAKE2_ROUND_SSSE3(
            state[8 * 0 + i], state[8 * 1 + i],
            state[8 * 2 + i], state[8 * 3 + i],
            state[8 * 4 + i], state[8 * 5 + i],
            state[8 * 6 + i], state[8 * 7 + i]);
    }

    /* XOR and store result */
    for (i = 0; i < ARGON2_OWORDS_IN_BLOCK; i++) {
        state[i] = _mm_xor_si128(state[i], block_XY[i]);
        _mm_storeu_si128((__m128i *)next_block->v + i, state[i]);
    }
}

/*
 * Generate next address block for data-independent addressing (Argon2i/id)
 */
static void next_addresses_ssse3(block *address_block, block *input_block) {
    __m128i zero_block[ARGON2_OWORDS_IN_BLOCK];
    __m128i zero2_block[ARGON2_OWORDS_IN_BLOCK];

    memset(zero_block, 0, sizeof(zero_block));
    memset(zero2_block, 0, sizeof(zero2_block));

    input_block->v[6]++;

    fill_block_ssse3(zero_block, input_block, address_block, 0);
    fill_block_ssse3(zero2_block, address_block, address_block, 0);
}

/*
 * Main segment filling function for SSSE3/SSE2
 * This is the core of Argon2 - fills one segment of the memory array
 */
void fill_segment_ssse3(const argon2_instance_t *instance,
                        argon2_position_t position) {
    block *ref_block = NULL, *curr_block = NULL;
    block address_block, input_block;
    uint64_t pseudo_rand, ref_index, ref_lane;
    uint32_t prev_offset, curr_offset;
    uint32_t starting_index, i;
    __m128i state[ARGON2_OWORDS_IN_BLOCK];
    int data_independent_addressing;

    if (instance == NULL) {
        return;
    }

    data_independent_addressing =
        (instance->type == Argon2_i) ||
        (instance->type == Argon2_id && (position.pass == 0) &&
         (position.slice < ARGON2_SYNC_POINTS / 2));

    if (data_independent_addressing) {
        init_block_value(&input_block, 0);

        input_block.v[0] = position.pass;
        input_block.v[1] = position.lane;
        input_block.v[2] = position.slice;
        input_block.v[3] = instance->memory_blocks;
        input_block.v[4] = instance->passes;
        input_block.v[5] = instance->type;
    }

    starting_index = 0;

    if ((0 == position.pass) && (0 == position.slice)) {
        starting_index = 2; /* First two blocks already generated */

        if (data_independent_addressing) {
            next_addresses_ssse3(&address_block, &input_block);
        }
    }

    /* Offset of current block */
    curr_offset = position.lane * instance->lane_length +
                  position.slice * instance->segment_length + starting_index;

    if (0 == curr_offset % instance->lane_length) {
        prev_offset = curr_offset + instance->lane_length - 1;
    } else {
        prev_offset = curr_offset - 1;
    }

    /* Load initial state from previous block */
    memcpy(state, ((instance->memory + prev_offset)->v), ARGON2_BLOCK_SIZE);

    for (i = starting_index; i < instance->segment_length;
         ++i, ++curr_offset, ++prev_offset) {

        if (curr_offset % instance->lane_length == 1) {
            prev_offset = curr_offset - 1;
        }

        /* Compute reference block index */
        if (data_independent_addressing) {
            if (i % ARGON2_ADDRESSES_IN_BLOCK == 0) {
                next_addresses_ssse3(&address_block, &input_block);
            }
            pseudo_rand = address_block.v[i % ARGON2_ADDRESSES_IN_BLOCK];
        } else {
            pseudo_rand = instance->memory[prev_offset].v[0];
        }

        /* Determine reference lane */
        ref_lane = ((pseudo_rand >> 32)) % instance->lanes;

        if ((position.pass == 0) && (position.slice == 0)) {
            ref_lane = position.lane;
        }

        /* Compute reference index within lane */
        position.index = i;
        ref_index = index_alpha(instance, &position, pseudo_rand & 0xFFFFFFFF,
                                ref_lane == position.lane);

        /* Get reference and current block pointers */
        ref_block =
            instance->memory + instance->lane_length * ref_lane + ref_index;
        curr_block = instance->memory + curr_offset;

        /* Fill the block */
        if (ARGON2_VERSION_10 == instance->version) {
            fill_block_ssse3(state, ref_block, curr_block, 0);
        } else {
            if (0 == position.pass) {
                fill_block_ssse3(state, ref_block, curr_block, 0);
            } else {
                fill_block_ssse3(state, ref_block, curr_block, 1);
            }
        }
    }
}
#endif /* __SSSE3__ || __SSE2__ */

#endif /* x86/x86-64 with SSSE3/SSE2/GNUC/MSVC */
#endif /* x86/x86-64 */
