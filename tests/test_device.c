#include "h3_device.h"

#include <stdio.h>
#include <string.h>

int main(void) {
    h3_device_info info;
    char error[256] = {0};
    if (!h3_device_probe(&info, error, sizeof(error))) {
        fprintf(stderr, "FAIL device probe: %s\n", error);
        return 1;
    }
    if (!info.name[0] || !info.architecture[0] ||
        !info.recommended_working_set || !info.max_buffer_length) {
        fprintf(stderr, "FAIL device probe: incomplete device information\n");
        return 1;
    }
    printf("ok: %s (%s), %.1f GiB GPU memory, unified=%s\n", info.name,
           info.architecture,
           (double)info.recommended_working_set / (1024.0 * 1024.0 * 1024.0),
           info.unified_memory ? "yes" : "no");
    return 0;
}
