#include "h3_log.h"

#include <stdarg.h>

static h3_log *active_log;

int h3_log_open(h3_log *log, const char *path) {
    if (!log) return 0;
    log->file = NULL;
    if (!path || !*path) return 1;
    log->file = fopen(path, "ab");
    if (!log->file) return 0;
    setvbuf(log->file, NULL, _IOLBF, 0);
    return 1;
}

void h3_log_close(h3_log *log) {
    if (!log) return;
    if (active_log == log) active_log = NULL;
    if (log->file) fclose(log->file);
    log->file = NULL;
}

void h3_log_set_active(h3_log *log) {
    active_log = log;
}

int h3_log_fprintf(FILE *stream, const char *format, ...) {
    va_list arguments;
    va_start(arguments, format);
    int result = vfprintf(stream, format, arguments);
    va_end(arguments);
    if (active_log && active_log->file && stream == stderr) {
        va_start(arguments, format);
        vfprintf(active_log->file, format, arguments);
        va_end(arguments);
        fflush(active_log->file);
    }
    return result;
}

void h3_log_progress(const char *phase, int completed, int total) {
    if (!active_log || !active_log->file) return;
    fprintf(active_log->file,
            "progress phase=%s completed=%d total=%d\n",
            phase, completed, total);
    fflush(active_log->file);
}