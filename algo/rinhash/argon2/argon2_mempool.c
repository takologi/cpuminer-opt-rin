/*
 * Argon2 Memory Pool for RinHash
 * Pre-allocates thread-local memory to avoid repeated malloc/free
 *
 * Copyright (c) 2024-2025 The Rincoin developers
 * Distributed under the MIT software license
 */

#include "argon2_mempool.h"
#include "core.h"  /* for secure_wipe_memory */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
#include <malloc.h>
#define aligned_alloc(alignment, size) _aligned_malloc(size, alignment)
#define aligned_free(ptr) _aligned_free(ptr)
#else
#define aligned_free(ptr) free(ptr)
#endif

/*
 * Thread-local memory pool structure.
 * Each thread gets its own independent pool.
 */
typedef struct argon2_mempool {
    uint8_t *buffer;              /* The actual memory buffer (aligned) */
    size_t capacity;              /* Buffer capacity in bytes */
    uint32_t magic_head;          /* Canary at start - detect underflow */
    uint32_t magic_tail;          /* Canary at end - detect overflow */
    int in_use;                   /* Reentrancy guard */
    int initialized;              /* Has this pool been set up? */
} argon2_mempool_t;

/*
 * Thread-local pool instance.
 * Automatically destroyed when thread exits.
 */
#if defined(_WIN32) && defined(_MSC_VER)
static __declspec(thread) argon2_mempool_t g_pool = {0};
#else
static __thread argon2_mempool_t g_pool = {0};
#endif

/*
 * Internal: Verify canaries are intact
 */
static int verify_canaries(void)
{
    if (g_pool.magic_head != ARGON2_MEMPOOL_MAGIC_HEAD) {
        fprintf(stderr, "FATAL: Argon2 mempool head canary corrupted! "
                        "Expected 0x%08X, got 0x%08X\n",
                ARGON2_MEMPOOL_MAGIC_HEAD, g_pool.magic_head);
        return 0;
    }
    if (g_pool.magic_tail != ARGON2_MEMPOOL_MAGIC_TAIL) {
        fprintf(stderr, "FATAL: Argon2 mempool tail canary corrupted! "
                        "Expected 0x%08X, got 0x%08X\n",
                ARGON2_MEMPOOL_MAGIC_TAIL, g_pool.magic_tail);
        return 0;
    }
    return 1;
}

int argon2_mempool_init(void)
{
    if (g_pool.initialized) {
        return 0;  /* Already initialized */
    }

    /* Allocate aligned buffer for SIMD operations */
    g_pool.buffer = (uint8_t *)aligned_alloc(ARGON2_MEMPOOL_ALIGNMENT,
                                              ARGON2_MEMPOOL_SIZE);
    if (!g_pool.buffer) {
        fprintf(stderr, "ERROR: Failed to allocate Argon2 memory pool (%d bytes)\n",
                ARGON2_MEMPOOL_SIZE);
        return -1;
    }

    /* Initialize pool metadata */
    g_pool.capacity = ARGON2_MEMPOOL_SIZE;
    g_pool.magic_head = ARGON2_MEMPOOL_MAGIC_HEAD;
    g_pool.magic_tail = ARGON2_MEMPOOL_MAGIC_TAIL;
    g_pool.in_use = 0;
    g_pool.initialized = 1;

    /* Zero the buffer initially */
    memset(g_pool.buffer, 0, ARGON2_MEMPOOL_SIZE);

    return 0;
}

void argon2_mempool_cleanup(void)
{
    if (!g_pool.initialized) {
        return;
    }

    if (g_pool.in_use) {
        fprintf(stderr, "WARNING: Cleaning up Argon2 mempool while still in use!\n");
    }

    /* Securely wipe before freeing */
    if (g_pool.buffer) {
        secure_wipe_memory(g_pool.buffer, g_pool.capacity);
        aligned_free(g_pool.buffer);
        g_pool.buffer = NULL;
    }

    g_pool.capacity = 0;
    g_pool.magic_head = 0;
    g_pool.magic_tail = 0;
    g_pool.in_use = 0;
    g_pool.initialized = 0;
}

int argon2_mempool_check(void)
{
    if (!g_pool.initialized) {
        return 0;
    }
    return verify_canaries();
}

int argon2_mempool_allocate(uint8_t **memory, size_t size)
{
    /* Validate output pointer */
    if (!memory) {
        fprintf(stderr, "ERROR: argon2_mempool_allocate called with NULL pointer\n");
        return -1;
    }

    /* Lazy initialization */
    if (!g_pool.initialized) {
        if (argon2_mempool_init() != 0) {
            *memory = NULL;
            return -1;
        }
    }

    /* Check for reentrancy (indicates bug in caller) */
    if (g_pool.in_use) {
        fprintf(stderr, "FATAL: Argon2 mempool reentrancy detected! "
                        "Buffer already in use.\n");
        *memory = NULL;
        return -1;
    }

    /* Verify canaries from previous use */
    if (!verify_canaries()) {
        fprintf(stderr, "FATAL: Memory corruption detected in Argon2 mempool!\n");
        *memory = NULL;
        return -1;
    }

    /* Check size fits in our pool */
    if (size > g_pool.capacity) {
        fprintf(stderr, "ERROR: Argon2 requested %zu bytes, pool only has %zu\n",
                size, g_pool.capacity);
        *memory = NULL;
        return -1;
    }

    /* Mark as in use */
    g_pool.in_use = 1;

    /*
     * Zero the buffer before use.
     * This is REQUIRED for Argon2 correctness - the algorithm assumes
     * memory is zeroed before fill_first_blocks().
     */
    memset(g_pool.buffer, 0, size);

    *memory = g_pool.buffer;
    return 0;
}

void argon2_mempool_free(uint8_t *memory, size_t size)
{
    /* Validate we're freeing our buffer */
    if (!g_pool.initialized) {
        fprintf(stderr, "WARNING: argon2_mempool_free called on uninitialized pool\n");
        return;
    }

    if (memory != g_pool.buffer) {
        fprintf(stderr, "ERROR: argon2_mempool_free called with wrong pointer! "
                        "Expected %p, got %p\n",
                (void *)g_pool.buffer, (void *)memory);
        return;
    }

    if (!g_pool.in_use) {
        fprintf(stderr, "WARNING: argon2_mempool_free called but buffer not in use\n");
        return;
    }

    /* Verify canaries - detect buffer overflow */
    if (!verify_canaries()) {
        fprintf(stderr, "FATAL: Buffer overflow detected during Argon2 mempool free!\n");
        /* Continue anyway to mark not-in-use, but this is a serious bug */
    }

    /*
     * Securely wipe the buffer.
     * This prevents sensitive data from lingering in memory.
     * Performance impact is minimal since we're just overwriting
     * 64KB that's already in cache.
     */
    secure_wipe_memory(memory, size);

    /* Mark as available for next use */
    g_pool.in_use = 0;

    /* Do NOT free the buffer - that's the whole point! */
}
