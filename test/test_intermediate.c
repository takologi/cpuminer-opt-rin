// Test program for cpuminer-opt-rin to show intermediate hash values
#include <stdio.h>
#include <string.h>
#include <stdint.h>

// Forward declarations for libraries 
extern void blake3_hasher_init(void*);
extern void blake3_hasher_update(void*, const void*, size_t);
extern void blake3_hasher_finalize(const void*, void*, size_t);
extern int argon2d_ctx(void*);
extern void SHA3_256(uint8_t*, const uint8_t*, size_t);
extern void argon2_dispatch_init();
extern void argon2_mempool_init();
extern void* argon2_mempool_allocate(uint8_t**, size_t);
extern void argon2_mempool_free(uint8_t*, size_t);

// Argon2 structures (from argon2.h)
typedef enum { Argon2_d = 0, Argon2_i = 1, Argon2_id = 2 } argon2_type;
#define ARGON2_VERSION_13 0x13
#define ARGON2_DEFAULT_FLAGS 0

typedef struct {
    uint8_t *out;
    uint32_t outlen;
    uint8_t *pwd;
    uint32_t pwdlen;
    uint8_t *salt;
    uint32_t saltlen;
    uint8_t *secret;
    uint32_t secretlen;
    uint8_t *ad;
    uint32_t adlen;
    uint32_t t_cost;
    uint32_t m_cost;
    uint32_t lanes;
    uint32_t threads;
    uint32_t version;
    void *(*allocate_cbk)(uint8_t**, size_t);
    void (*free_cbk)(uint8_t*, size_t);
    uint32_t flags;
} argon2_context;

// BLAKE3 hasher structure (64-byte aligned)
typedef struct {
    uint8_t data[1912]; // Actual size may vary, but this should be enough
} blake3_hasher;

void print_hex(const char *label, const uint8_t *data, size_t len) {
    printf("%s: ", label);
    for (size_t i = 0; i < len; i++) {
        printf("%02x", data[i]);
    }
    printf("\n");
}

void hex_to_bin(const char *hex, uint8_t *bin, size_t bin_len) {
    for (size_t i = 0; i < bin_len; i++) {
        sscanf(hex + 2*i, "%2hhx", &bin[i]);
    }
}

int main() {
    printf("=== Intermediate Hash Values Test (cpuminer-opt-rin) ===\n\n");
    
    // Initialize argon2 dispatch and mempool
    argon2_dispatch_init();
    argon2_mempool_init();
    
    // Use the EXACT same header as rinpool-proxy test_hash4 (ENDIAN_SWAP_32_BE style)
    // This is from the "ENDIAN_LITTLE with 4-byte word swapped prevhash" test 
    // which produced a valid result
    const char *header_hex = "0000002043eb75f7e0d88f2c70a4adb9ed32b120388c1ecd67f98b2ec066e1a6010000009c0c4edd7b50d73cc036251a20df20624251e7939e6ced986cbd6362ece4a610cc8e62692f8e031d0019d60d";
    
    uint8_t header[80];
    hex_to_bin(header_hex, header, 80);
    
    printf("Step 0 - Input Header (80 bytes):\n");
    print_hex("Header", header, 80);
    printf("\n");
    
    // Step 1: BLAKE3
    printf("Step 1 - BLAKE3 Hash:\n");
    uint8_t blake3_out[32];
    blake3_hasher hasher;
    blake3_hasher_init(&hasher);
    blake3_hasher_update(&hasher, header, 80);
    blake3_hasher_finalize(&hasher, blake3_out, 32);
    print_hex("BLAKE3 output", blake3_out, 32);
    printf("\n");
    
    // Step 2: Argon2d
    printf("Step 2 - Argon2d Hash:\n");
    const char* salt_str = "RinCoinSalt";
    printf("Salt: \"%s\" (len=%zu)\n", salt_str, strlen(salt_str));
    printf("Parameters: t_cost=2, m_cost=64, lanes=1, threads=1, version=0x13\n");
    
    uint8_t argon2_out[32];
    argon2_context context = {0};
    context.out = argon2_out;
    context.outlen = 32;
    context.pwd = blake3_out;
    context.pwdlen = 32;
    context.salt = (uint8_t*)salt_str;
    context.saltlen = strlen(salt_str);
    context.t_cost = 2;
    context.m_cost = 64;
    context.lanes = 1;
    context.threads = 1;
    context.version = ARGON2_VERSION_13;
    context.allocate_cbk = argon2_mempool_allocate;
    context.free_cbk = argon2_mempool_free;
    context.flags = ARGON2_DEFAULT_FLAGS;
    
    int result = argon2d_ctx(&context);
    if (result != 0) {
        printf("Argon2d failed with error: %d\n", result);
        return 1;
    }
    print_hex("Argon2d output", argon2_out, 32);
    printf("\n");
    
    // Step 3: SHA3-256
    printf("Step 3 - SHA3-256 Hash:\n");
    uint8_t sha3_out[32];
    SHA3_256(sha3_out, argon2_out, 32);
    print_hex("SHA3-256 output (FINAL HASH)", sha3_out, 32);
    printf("\n");
    
    // Expected from test_hash4:
    printf("Expected from rinpool-proxy test_hash4:\n");
    printf("Hash: dfda9d782a5dc37d3de781240ca50c3b69954a3d39d5b389892ff5955d030000\n");
    printf("Difficulty: 1.160632839055e-03\n");
    
    printf("\n=== Done ===\n");
    return 0;
}
