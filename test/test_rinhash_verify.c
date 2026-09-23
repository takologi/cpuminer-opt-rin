#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "../algo/rinhash/argon2/include/argon2.h"
#include "../algo/rinhash/argon2/argon2_dispatch.h"
#include "../algo/rinhash/argon2/argon2_mempool.h"

void rinhash(void *state, const void *input);
void rinhash_fast(void *state, const void *input);

static void print_hex(const char *label, const uint8_t *data, size_t len)
{
    printf("%s", label);
    for (size_t i = 0; i < len; ++i) {
        printf("%02x", data[i]);
    }
    printf("\n");
}

static int run_case(const char *label,
                    int use_mempool,
                    argon2_impl_type_t impl,
                    const uint8_t *input,
                    const uint8_t *expected)
{
    uint8_t out[32];
    const char *salt = "RinCoinSalt";
    argon2_context ctx;

    memset(&ctx, 0, sizeof(ctx));
    memset(out, 0, sizeof(out));

    ctx.out = out;
    ctx.outlen = sizeof(out);
    ctx.pwd = (uint8_t *)input;
    ctx.pwdlen = 32;
    ctx.salt = (uint8_t *)salt;
    ctx.saltlen = 11;
    ctx.t_cost = 2;
    ctx.m_cost = 64;
    ctx.lanes = 1;
    ctx.threads = 1;
    ctx.version = ARGON2_VERSION_13;
    ctx.flags = ARGON2_DEFAULT_FLAGS;

    if (use_mempool) {
        ctx.allocate_cbk = argon2_mempool_allocate;
        ctx.free_cbk = argon2_mempool_free;
    }

    argon2_set_impl(impl);
    if (argon2d_ctx(&ctx) != ARGON2_OK) {
        fprintf(stderr, "%s: argon2d_ctx failed\n", label);
        return 1;
    }

    printf("[%s] impl=%s allocator=%s\n",
           label,
           argon2_get_impl_name(),
           use_mempool ? "mempool" : "malloc");
    print_hex("  out      = ", out, sizeof(out));
    print_hex("  expected = ", expected, sizeof(out));

    if (memcmp(out, expected, sizeof(out)) != 0) {
        printf("  RESULT   = MISMATCH\n\n");
        return 1;
    }

    printf("  RESULT   = OK\n\n");
    return 0;
}

static int run_hash_case(const char *label,
                         void (*hash_fn)(void *, const void *),
                         const uint8_t *header,
                         const uint8_t *expected)
{
    uint8_t out[32];
    memset(out, 0, sizeof(out));

    hash_fn(out, header);

    printf("[%s]\n", label);
    print_hex("  out      = ", out, sizeof(out));
    print_hex("  expected = ", expected, sizeof(out));

    if (memcmp(out, expected, sizeof(out)) != 0) {
        printf("  RESULT   = MISMATCH\n\n");
        return 1;
    }

    printf("  RESULT   = OK\n\n");
    return 0;
}

int main(void)
{
    static const uint8_t blake3_out[32] = {
        0x79, 0x4a, 0x9b, 0x72, 0x32, 0xe8, 0x83, 0x31,
        0x76, 0x0c, 0x3e, 0xf1, 0xe6, 0x51, 0x5c, 0x97,
        0x64, 0x83, 0x89, 0x5f, 0xa7, 0x10, 0x7d, 0x2e,
        0xee, 0x0d, 0xcb, 0x25, 0x2a, 0x06, 0x9e, 0xa9
    };

    static const uint8_t expected_argon2[32] = {
        0x79, 0xf9, 0x35, 0x42, 0x0a, 0xce, 0xf5, 0x38,
        0x2d, 0xc0, 0xf2, 0x54, 0xc7, 0x98, 0xd3, 0x1e,
        0xe6, 0xac, 0x5d, 0x40, 0xc0, 0xd8, 0xe6, 0xcd,
        0x94, 0x97, 0x01, 0x6c, 0x74, 0xa1, 0xc6, 0x4a
    };

    static const uint8_t header[80] = {
        0x00, 0x00, 0x00, 0x20, 0x43, 0xeb, 0x75, 0xf7,
        0xe0, 0xd8, 0x8f, 0x2c, 0x70, 0xa4, 0xad, 0xb9,
        0xed, 0x32, 0xb1, 0x20, 0x38, 0x8c, 0x1e, 0xcd,
        0x67, 0xf9, 0x8b, 0x2e, 0xc0, 0x66, 0xe1, 0xa6,
        0x01, 0x00, 0x00, 0x00, 0x9c, 0x0c, 0x4e, 0xdd,
        0x7b, 0x50, 0xd7, 0x3c, 0xc0, 0x36, 0x25, 0x1a,
        0x20, 0xdf, 0x20, 0x62, 0x42, 0x51, 0xe7, 0x93,
        0x9e, 0x6c, 0xed, 0x98, 0x6c, 0xbd, 0x63, 0x62,
        0xec, 0xe4, 0xa6, 0x10, 0xcc, 0x8e, 0x62, 0x69,
        0x2f, 0x8e, 0x03, 0x1d, 0x00, 0x19, 0xd6, 0x0d
    };

    static const uint8_t expected_hash[32] = {
        0xdf, 0xda, 0x9d, 0x78, 0x2a, 0x5d, 0xc3, 0x7d,
        0x3d, 0xe7, 0x81, 0x24, 0x0c, 0xa5, 0x0c, 0x3b,
        0x69, 0x95, 0x4a, 0x3d, 0x39, 0xd5, 0xb3, 0x89,
        0x89, 0x2f, 0xf5, 0x95, 0x5d, 0x03, 0x00, 0x00
    };

    int failed = 0;

    argon2_dispatch_init();
    argon2_mempool_init();

    failed |= run_case("ref+malloc", 0, ARGON2_IMPL_REF, blake3_out, expected_argon2);
    failed |= run_case("ref+mempool", 1, ARGON2_IMPL_REF, blake3_out, expected_argon2);
    failed |= run_case("auto+malloc", 0, ARGON2_IMPL_COUNT, blake3_out, expected_argon2);
    failed |= run_case("auto+mempool", 1, ARGON2_IMPL_COUNT, blake3_out, expected_argon2);

    argon2_set_impl(ARGON2_IMPL_REF);
    failed |= run_hash_case("rinhash", rinhash, header, expected_hash);

    argon2_set_impl(ARGON2_IMPL_COUNT);
    failed |= run_hash_case("rinhash_fast", rinhash_fast, header, expected_hash);

    return failed ? 1 : 0;
}
