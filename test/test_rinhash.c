/*
 * Test program to verify RinHash calculation
 * Compares miner's hash calculation with stratum-proxy's expected result.
 *
 * PURPOSE: Validates that the cpuminer RinHash implementation produces
 * correct hashes that will be accepted by pools/nodes.
 *
 * USAGE: 
 *   cd test && make test_rinhash && ./test_rinhash
 *
 * Expected output: "Hash matches expected value!" for each test case.
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

// Forward declarations
void rinhash(void* output, const void* input);
void sha256_full(void *hash, const void *data, size_t len);
void argon2_dispatch_init();
void argon2_mempool_init();

// Helper function to convert hex string to bytes
void my_hex2bin(unsigned char *p, const char *hexstr, size_t len)
{
    for (size_t i = 0; i < len; i++) {
        sscanf(hexstr + i * 2, "%2hhx", &p[i]);
    }
}

// Helper function to print bytes as hex
void print_hex(const char *label, const unsigned char *data, size_t len)
{
    printf("%s: ", label);
    for (size_t i = 0; i < len; i++) {
        printf("%02x", data[i]);
    }
    printf("\n");
}

int main()
{
    printf("=== RinHash Test Program ===\n\n");
    
    // Initialize argon2 systems
    argon2_dispatch_init();
    argon2_mempool_init();
    
    // From stratum-proxy log:
    // Job data from mining.notify:
    const char *job_id = "1c1a";
    const char *prevhash_hex = "ab6231663d91dc0a091c296404d51b216bccedd29de39372a16ab3cb00000000";
    const char *coinb1_hex = "01000000010000000000000000000000000000000000000000000000000000000000000000ffffffff4e03974f060477c36a6908fabe6d6d00000000000000000000000000000000000000000000000000000000000000000100000000000000";
    const char *coinb2_hex = "0f706f6f6c2e72706c616e742e78797a00000000020000000000000000266a24aa21a9ede2f61c3f71d1defd3fa999dfa36953755c690689799962b48bebd836974e8cf900f9029500000000160014d38de4263824155ef5bfd0719caa765e768f0ffc00000000";
    const char *version_hex = "20000000";
    const char *nbits_hex = "1d011853";
    const char *ntime_hex = "696ac377";
    
    // From mining.submit:
    const char *extranonce2_hex = "00000000";
    const char *submitted_ntime_hex = "696ac377";
    const char *nonce_hex = "03003b2d";
    
    // Expected hash from stratum-proxy:
    const char *expected_hash_hex = "7c91dee85a8b38909372fdc52013f8c8874505632a7fc279d30d9e5cfa222820";
    
    printf("Test data from stratum-proxy log:\n");
    printf("Job ID: %s\n", job_id);
    printf("Nonce: %s\n", nonce_hex);
    printf("NTime: %s\n", ntime_hex);
    printf("Expected hash: %s\n\n", expected_hash_hex);
    
    // Build the block header (80 bytes)
    unsigned char block_header[80];
    
    // Version (4 bytes, little-endian)
    my_hex2bin(block_header + 0, version_hex, 4);
    
    // Previous block hash (32 bytes, need to reverse for little-endian)
    unsigned char prevhash[32];
    my_hex2bin(prevhash, prevhash_hex, 32);
    // Reverse bytes for little-endian
    for (int i = 0; i < 32; i++) {
        block_header[4 + i] = prevhash[31 - i];
    }
    
    // Merkle root (32 bytes) - need to build from coinbase
    // The coinbase transaction is: coinb1 + extranonce1 + extranonce2 + coinb2
    // Then double SHA256 to get txid, which in this case IS the merkle root (single tx)
    
    printf("Building coinbase transaction...\n");
    
    // Extranonce1 from the stratum connection - we need to find it from the log
    // Looking at the log, the proxy doesn't show extranonce1 in the submit
    // But from the mining.subscribe response, extranonce1 would be there
    // For now, let's try to figure out the correct coinbase
    
    // Actually, looking at the stratum-proxy test_hash4.cpp, it builds the header differently
    // Let me try using the full header from test_hash4 which shows working examples
    
    // From test_hash4.cpp line 88-95, it shows the serialization for block 414711:
    // Build header as: version + prev_hash + merkle_root + ntime + nbits + nonce
    
    // Let's build the coinbase properly:
    // coinb1 + extranonce1 + extranonce2 + coinb2
    const char *extranonce1_hex = "06000000"; // This needs to match what the miner got from subscribe
    
    size_t coinb1_len = strlen(coinb1_hex) / 2;
    size_t coinb2_len = strlen(coinb2_hex) / 2;
    size_t extranonce1_len = strlen(extranonce1_hex) / 2;
    size_t extranonce2_len = strlen(extranonce2_hex) / 2;
    size_t coinbase_len = coinb1_len + extranonce1_len + extranonce2_len + coinb2_len;
    
    unsigned char *coinbase = malloc(coinbase_len);
    my_hex2bin(coinbase, coinb1_hex, coinb1_len);
    my_hex2bin(coinbase + coinb1_len, extranonce1_hex, extranonce1_len);
    my_hex2bin(coinbase + coinb1_len + extranonce1_len, extranonce2_hex, extranonce2_len);
    my_hex2bin(coinbase + coinb1_len + extranonce1_len + extranonce2_len, coinb2_hex, coinb2_len);
    
    printf("Coinbase transaction length: %zu bytes\n", coinbase_len);
    print_hex("Coinbase", coinbase, coinbase_len);
    
    // Calculate merkle root: double SHA256 of coinbase
    unsigned char merkle_root[32];
    unsigned char sha256_temp[32];
    
    // Double SHA256
    sha256_full(sha256_temp, coinbase, coinbase_len);
    sha256_full(merkle_root, sha256_temp, 32);
    
    print_hex("Merkle root", merkle_root, 32);
    
    // Copy merkle root to block header (bytes 36-67)
    memcpy(block_header + 36, merkle_root, 32);
    
    printf("\nBuilding block header...\n");
    printf("Version: ");
    for (int i = 0; i < 4; i++) printf("%02x", block_header[i]);
    printf("\n");
    
    printf("PrevHash (reversed): ");
    for (int i = 4; i < 36; i++) printf("%02x", block_header[i]);
    printf("\n");
    
    printf("MerkleRoot: ");
    for (int i = 36; i < 68; i++) printf("%02x", block_header[i]);
    printf("\n");
    
    // NBits (4 bytes, little-endian)
    my_hex2bin(block_header + 72, nbits_hex, 4);
    
    // NTime (4 bytes, little-endian) 
    my_hex2bin(block_header + 68, ntime_hex, 4);
    
    // Nonce (4 bytes, little-endian)
    my_hex2bin(block_header + 76, nonce_hex, 4);
    
    printf("\nNBits: ");
    for (int i = 72; i < 76; i++) printf("%02x", block_header[i]);
    printf("\n");
    
    printf("NTime: ");
    for (int i = 68; i < 72; i++) printf("%02x", block_header[i]);
    printf("\n");
    
    printf("Nonce: ");
    for (int i = 76; i < 80; i++) printf("%02x", block_header[i]);
    printf("\n");
    
    // Calculate hash using the miner's rinhash function
    print_hex("\nFull block header", block_header, 80);
    printf("\n");
    unsigned char computed_hash[32];
    
    printf("Calling rinhash()...\n");
    rinhash(computed_hash, block_header);
    
    printf("\nResults:\n");
    print_hex("Computed hash ", computed_hash, 32);
    
    unsigned char expected_hash[32];
    my_hex2bin(expected_hash, expected_hash_hex, 32);
    print_hex("Expected hash ", expected_hash, 32);
    
    // Compare
    if (memcmp(computed_hash, expected_hash, 32) == 0) {
        printf("\n✓ SUCCESS: Hashes match!\n");
        return 0;
    } else {
        printf("\n✗ FAILURE: Hashes do not match!\n");
        
        // Show differences
        printf("\nDifferences:\n");
        for (int i = 0; i < 32; i++) {
            if (computed_hash[i] != expected_hash[i]) {
                printf("  Byte %2d: computed=%02x expected=%02x\n", 
                       i, computed_hash[i], expected_hash[i]);
            }
        }
        return 1;
    }
    
    free(coinbase);
}
