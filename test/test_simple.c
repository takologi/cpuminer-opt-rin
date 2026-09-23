// Simple test using exact header from test_hash4
#include <stdio.h>
#include <string.h>

void rinhash(void* output, const void* input);
void argon2_dispatch_init();
void argon2_mempool_init();

void hex2bin_simple(unsigned char *p, const char *hexstr, size_t len) {
    for (size_t i = 0; i < len; i++) {
        sscanf(hexstr + i * 2, "%2hhx", &p[i]);
    }
}

void print_hex_simple(const char *label, const unsigned char *data, size_t len) {
    printf("%s: ", label);
    for (size_t i = 0; i < len; i++) {
        printf("%02x", data[i]);
    }
    printf("\n");
}

int main() {
    printf("=== Simple RinHash Test with Known Header ===\n\n");
    
    // Initialize argon2
    argon2_dispatch_init();
    argon2_mempool_init();
    
    // From test_hash4.cpp - a known working header
    // Using the ENDIAN_SWAP_32_BE style which should match tnn-miner
    const char *header_hex = "20000000f775eb432c8fd8e0b9ada47020b132edcd1e8c382e8bf967a6e166c0000000014f3a8ce37de3a25aca68dbda91e0f05e48a30be7f2d8c53ee91d4e19c9e59769628ecc1d038e2f0dd61900";
    const char *expected_hash = "44e97dc9c7e4aaa89cb40e485bcf7d91d1919e4d94d2a72eb17e3c58a8adb82f";
    
    unsigned char header[80];
    unsigned char hash[32];
    unsigned char expected[32];
    
    hex2bin_simple(header, header_hex, 80);
    hex2bin_simple(expected, expected_hash, 32);
    
    print_hex_simple("Header", header, 80);
    print_hex_simple("Expected", expected, 32);
    
    printf("\nCalling rinhash...\n");
    rinhash(hash, header);
    
    print_hex_simple("Computed", hash, 32);
    
    if (memcmp(hash, expected, 32) == 0) {
        printf("\n✓ SUCCESS! Miner calculates correct hash!\n");
        return 0;
    } else {
        printf("\n✗ FAILURE! Hash mismatch!\n");
        printf("\nDifferences:\n");
        for (int i = 0; i < 32; i++) {
            if (hash[i] != expected[i]) {
                printf("  Byte %2d: computed=%02x expected=%02x\n", 
                       i, hash[i], expected[i]);
            }
        }
        return 1;
    }
}
