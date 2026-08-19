#ifndef H3_LOG_H
#define H3_LOG_H

#include <stdio.h>

typedef struct {
    FILE *file;
} h3_log;

int h3_log_open(h3_log *log, const char *path);
void h3_log_close(h3_log *log);
void h3_log_set_active(h3_log *log);
int h3_log_fprintf(FILE *stream, const char *format, ...)
    __attribute__((format(printf, 2, 3)));
void h3_log_progress(const char *phase, int completed, int total);

#define fprintf(stream, ...) h3_log_fprintf((stream), __VA_ARGS__)

#endif