// Test using the OLD argon2 from algo/argon2d/argon2d
#include <stdio.h>
#include <string.h>
#include <stdint.h>

// From algo/argon2d/argon2d/argon2.h - OLD version with explicit version parameter
extern int argon2d_hash_raw(const uint32_t t_cost, const uint32_t m_cost,
                            const uint32_t parallelism, const void *pwd,
                            const size_t pwdlen, const void *salt,
                            const size_t saltlen, void *hash,
                            const size_t hashlen,
                            const uint32_t version);

#define ARGON2_VERSION_13 0x13

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
    printf("=== Testing OLD argon2 (from algo/argon2d/argon2d) ===\n\n");
    
    // Use the exact same BLAKE3 output as both tests
    const char *blake3_hex = "794a9b7232e88331760c3ef1e6515c976483895fa7107d2eee0dcb252a069ea9";
    uint8_t blake3_out[32];
    hex_to_bin(blake3_hex, blake3_out, 32);
    
    const char* salt_str = "RinCoinSalt";
    uint8_t argon2_out[32];
    
    printf("Input (BLAKE3 output): %s\n", blake3_hex);
    printf("Salt: \"%s\" (len=%zu)\n", salt_str, strlen(salt_str));
    printf("Parameters: t_cost=2, m_cost=64, parallelism=1, version=0x13\n\n");
    
    int result = argon2d_hash_raw(2, 64, 1, blake3_out, 32, (uint8_t*)salt_str, strlen(salt_str), argon2_out, 32, ARGON2_VERSION_13);
    if (result != 0) {
        printf("Argon2d failed with error: %d\n", result);
        return 1;
    }
    
    print_hex("Argon2d output", argon2_out, 32);
    printf("\nExpected (from proxy libargon2): 79f935420acef5382dc0f254c798d31ee6ac5d40c0d8e6cd9497016c74a1c64a\n");
    printf("New rincoin argon2 produced:     bc3f804853d926132b9104d386db4a1a01fd36e1f05f391654add1d809a589b4\n");
    
    return 0;
}
