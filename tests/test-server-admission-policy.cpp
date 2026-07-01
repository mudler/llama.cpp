#include "../tools/server/server-admission-policy.h"

#include <cstdio>
#include <cstdlib>

static void require(bool ok, const char * what) {
    if (!ok) {
        std::fprintf(stderr, "require failed: %s\n", what);
        std::exit(1);
    }
}

int main() {
    require(!server_admission_should_defer_decode_for_ttft(false, true, 8), "disabled never defers");
    require(!server_admission_should_defer_decode_for_ttft(true, false, 8), "no prompt backlog never defers");
    require(!server_admission_should_defer_decode_for_ttft(true, true, 0), "first decode token is not deferred");
    require(server_admission_should_defer_decode_for_ttft(true, true, 1), "token 2+ decode is deferred");
    require(server_admission_should_defer_decode_for_ttft(true, true, 64), "later decode is deferred");

    return 0;
}
