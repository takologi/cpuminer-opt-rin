/*
 * Argon2 Memory Pool for RinHash
 * Pre-allocates thread-local memory to avoid repeated malloc/free
 *
 * Copyright (c) 2024-2025 The Rincoin developers
 * Distributed under the MIT software license
 */

#ifndef ARGON2_MEMPOOL_H
#define ARGON2_MEMPOOL_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Memory pool size - matches RinHash Argon2 parameters:
 * m_cost = 64 KB, but Argon2 rounds up to segment boundaries.
 * With lanes=1, sync_points=4: memory_blocks = 64 / (1 * 4) * (1 * 4) = 64 blocks
 * Each block = 1024 bytes, so 64 * 1024 = 65536 bytes
 */
#define ARGON2_MEMPOOL_SIZE (64 * 1024)

/* Alignment for AVX-512 (64 bytes) */
#define ARGON2_MEMPOOL_ALIGNMENT 64

/* Magic values for corruption detection */
#define ARGON2_MEMPOOL_MAGIC_HEAD 0xDEADBEEFU
#define ARGON2_MEMPOOL_MAGIC_TAIL 0xCAFEBABEU

/*
 * Custom allocator callback for Argon2.
 * Returns pre-allocated thread-local buffer instead of calling malloc.
 *
 * @param memory    Pointer to store allocated memory address
 * @param size      Requested size in bytes
 * @return          0 on success, non-zero on failure
 */
int argon2_mempool_allocate(uint8_t **memory, size_t size);

/*
 * Custom deallocator callback for Argon2.
 * Securely wipes the buffer but does NOT free it (reused for next hash).
 *
 * @param memory    Memory to "free"
 * @param size      Size of memory region
 */
void argon2_mempool_free(uint8_t *memory, size_t size);

/*
 * Initialize the memory pool for the current thread.
 * Called automatically on first use, but can be called explicitly
 * for predictable initialization timing.
 *
 * @return          0 on success, -1 on failure
 */
int argon2_mempool_init(void);

/*
 * Cleanup the memory pool for the current thread.
 * Optional - memory is automatically freed when thread exits.
 * Useful for clean valgrind reports.
 */
void argon2_mempool_cleanup(void);

/*
 * Check if the memory pool is healthy (canaries intact).
 * Useful for debugging.
 *
 * @return          1 if healthy, 0 if corruption detected
 */
int argon2_mempool_check(void);

#ifdef __cplusplus
}
#endif

#endif /* ARGON2_MEMPOOL_H */
