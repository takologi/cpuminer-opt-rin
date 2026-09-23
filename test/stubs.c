// Stub functions for test program
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

// Stub implementations
bool work_restart[8];
bool opt_debug = false;

bool fulltest(const void *hash, const void *target) {
    return true;
}

bool submit_solution(struct work *work, const void *hash, void *thr) {
    return true;
}

void std_be_build_stratum_request(char *req, struct work *work, const char *user, const char *job_id, 
                                   const char *extra, const char *ntime_str, const char *nonce_str) {
    // Stub
}

struct work {
    uint8_t data[128];
    uint8_t target[32];
};
