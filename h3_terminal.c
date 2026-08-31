/* Graphical-terminal output, adapted to H3's RGB frame API from Iris. */
#include "h3_terminal.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <spawn.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

/* Match Iris: macOS graphical terminals otherwise display generated pixels at
 * roughly half their intended logical size on Retina screens. */
static int terminal_zoom = 2;

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

h3_terminal_protocol h3_terminal_detect(void) {
    if (getenv("KITTY_WINDOW_ID") || getenv("GHOSTTY_RESOURCES_DIR"))
        return H3_TERM_KITTY;
    const char *program = getenv("TERM_PROGRAM");
    if ((program && !strcmp(program, "iTerm.app")) ||
        getenv("ITERM_SESSION_ID") || getenv("WEZTERM_PANE") ||
        getenv("KONSOLE_VERSION"))
        return H3_TERM_ITERM2;
    return H3_TERM_NONE;
}

const char *h3_terminal_protocol_name(h3_terminal_protocol protocol) {
    switch (protocol) {
        case H3_TERM_KITTY: return "Kitty graphics";
        case H3_TERM_ITERM2: return "iTerm2 inline images";
        default: return "none";
    }
}

int h3_terminal_set_zoom(int zoom) {
    if (zoom < 1) return 0;
    terminal_zoom = zoom;
    return 1;
}

int h3_terminal_display_dimensions(int width, int height,
                                   int *display_width, int *display_height) {
    if (!display_width || !display_height || width < 1 || height < 1 ||
        width > INT_MAX / terminal_zoom || height > INT_MAX / terminal_zoom)
        return 0;
    *display_width = width * terminal_zoom;
    *display_height = height * terminal_zoom;
    return 1;
}

static const char base64_table[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static char *base64_encode(const uint8_t *data, size_t length,
                           size_t *encoded_length) {
    if (length > (SIZE_MAX - 2) / 3) return NULL;
    size_t output_length = 4 * ((length + 2) / 3);
    char *output = malloc(output_length + 1);
    if (!output) return NULL;
    size_t input = 0, encoded = 0;
    while (input < length) {
        uint32_t a = input < length ? data[input++] : 0;
        uint32_t b = input < length ? data[input++] : 0;
        uint32_t c = input < length ? data[input++] : 0;
        uint32_t triple = (a << 16) | (b << 8) | c;
        output[encoded++] = base64_table[(triple >> 18) & 63u];
        output[encoded++] = base64_table[(triple >> 12) & 63u];
        output[encoded++] = base64_table[(triple >> 6) & 63u];
        output[encoded++] = base64_table[triple & 63u];
    }
    if (length % 3) {
        output[output_length - 1] = '=';
        if (length % 3 == 1) output[output_length - 2] = '=';
    }
    output[output_length] = '\0';
    *encoded_length = output_length;
    return output;
}

static int packed_rgb(const uint8_t *rgb, int width, int height, int stride,
                      uint8_t **copy, const uint8_t **pixels,
                      size_t *size, char *error, size_t error_size) {
    if (!rgb || width < 1 || height < 1 || width > INT_MAX / 3 ||
        stride < width * 3 ||
        (size_t)width > SIZE_MAX / 3 ||
        (size_t)width * 3 > SIZE_MAX / (size_t)height) {
        fail(error, error_size, "invalid terminal RGB frame");
        return 0;
    }
    size_t row = (size_t)width * 3;
    *size = row * (size_t)height;
    *copy = NULL;
    if ((size_t)stride == row) {
        *pixels = rgb;
        return 1;
    }
    *copy = malloc(*size);
    if (!*copy) {
        fail(error, error_size, "out of memory packing terminal RGB frame");
        return 0;
    }
    for (int y = 0; y < height; y++)
        memcpy(*copy + (size_t)y * row, rgb + (size_t)y * (size_t)stride, row);
    *pixels = *copy;
    return 1;
}

static int kitty_display(const uint8_t *pixels, size_t size,
                         int width, int height,
                         char *error, size_t error_size) {
    int display_width = 0, display_height = 0;
    if (!h3_terminal_display_dimensions(width, height,
                                        &display_width, &display_height)) {
        fail(error, error_size, "invalid Kitty display dimensions");
        return 0;
    }
    size_t encoded_length = 0;
    char *encoded = base64_encode(pixels, size, &encoded_length);
    if (!encoded) {
        fail(error, error_size, "out of memory encoding Kitty frame");
        return 0;
    }
    const size_t chunk_size = 4096;
    size_t offset = 0;
    int first = 1;
    while (offset < encoded_length) {
        size_t chunk = encoded_length - offset;
        if (chunk > chunk_size) chunk = chunk_size;
        int more = offset + chunk < encoded_length;
        if (first) {
            fprintf(stdout,
                    "\033_Ga=T,f=24,t=d,s=%d,v=%d,w=%d,h=%d,m=%d;",
                    width, height, display_width, display_height, more);
            first = 0;
        } else {
            fprintf(stdout, "\033_Gm=%d;", more);
        }
        if (fwrite(encoded + offset, 1, chunk, stdout) != chunk) {
            fail(error, error_size, "cannot write Kitty frame");
            free(encoded);
            return 0;
        }
        fputs("\033\\", stdout);
        offset += chunk;
    }
    fputc('\n', stdout);
    fflush(stdout);
    free(encoded);
    return 1;
}

static int write_all(int descriptor, const void *data, size_t length) {
    const uint8_t *cursor = data;
    while (length) {
        ssize_t count = write(descriptor, cursor, length);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return 0;
        cursor += (size_t)count;
        length -= (size_t)count;
    }
    return 1;
}

static int encode_png(const uint8_t *pixels, size_t size,
                      int width, int height, char *path,
                      char *error, size_t error_size) {
    char raw_path[] = "/tmp/h3-terminal-rgb-XXXXXX";
    int raw = mkstemp(raw_path);
    if (raw < 0) {
        fail(error, error_size, "cannot create terminal RGB temporary file: %s",
             strerror(errno));
        return 0;
    }
    int ok = write_all(raw, pixels, size);
    int close_error = close(raw);
    if (!ok || close_error) {
        fail(error, error_size, "cannot write terminal RGB temporary file: %s",
             strerror(errno));
        unlink(raw_path);
        return 0;
    }
    int output = mkstemp(path);
    if (output < 0) {
        fail(error, error_size, "cannot create terminal PNG temporary file: %s",
             strerror(errno));
        unlink(raw_path);
        return 0;
    }
    close(output);
    char dimensions[64];
    snprintf(dimensions, sizeof(dimensions), "%dx%d", width, height);
    char *arguments[] = {
        "ffmpeg", "-v", "error", "-y", "-f", "rawvideo",
        "-pixel_format", "rgb24", "-video_size", dimensions,
        "-i", raw_path, "-frames:v", "1", "-f", "image2", path, NULL
    };
    pid_t child = 0;
    int spawn_error = posix_spawnp(&child, "ffmpeg", NULL, NULL,
                                   arguments, environ);
    int status = 0;
    pid_t waited = -1;
    if (!spawn_error) {
        do {
            waited = waitpid(child, &status, 0);
        } while (waited < 0 && errno == EINTR);
    }
    unlink(raw_path);
    if (spawn_error || waited < 0 || !WIFEXITED(status) || WEXITSTATUS(status)) {
        fail(error, error_size, "FFmpeg cannot encode terminal PNG%s%s",
             spawn_error ? ": " : "", spawn_error ? strerror(spawn_error) : "");
        unlink(path);
        return 0;
    }
    return 1;
}

static uint8_t *read_file(const char *path, size_t *size,
                          char *error, size_t error_size) {
    FILE *file = fopen(path, "rb");
    if (!file || fseek(file, 0, SEEK_END)) goto failure;
    long end = ftell(file);
    if (end < 1 || fseek(file, 0, SEEK_SET)) goto failure;
    uint8_t *data = malloc((size_t)end);
    if (!data) {
        fail(error, error_size, "out of memory reading terminal PNG");
        fclose(file);
        return NULL;
    }
    if (fread(data, 1, (size_t)end, file) != (size_t)end) {
        free(data);
        goto failure;
    }
    fclose(file);
    *size = (size_t)end;
    return data;
failure:
    fail(error, error_size, "cannot read terminal PNG: %s", strerror(errno));
    if (file) fclose(file);
    return NULL;
}

static int iterm2_display(const uint8_t *pixels, size_t size,
                          int width, int height,
                          char *error, size_t error_size) {
    int display_width = 0, display_height = 0;
    if (!h3_terminal_display_dimensions(width, height,
                                        &display_width, &display_height)) {
        fail(error, error_size, "invalid iTerm2 display dimensions");
        return 0;
    }
    char path[] = "/tmp/h3-terminal-XXXXXX";
    if (!encode_png(pixels, size, width, height, path, error, error_size))
        return 0;
    size_t png_size = 0;
    uint8_t *png = read_file(path, &png_size, error, error_size);
    unlink(path);
    if (!png) return 0;
    size_t encoded_length = 0;
    char *encoded = base64_encode(png, png_size, &encoded_length);
    free(png);
    if (!encoded) {
        fail(error, error_size, "out of memory encoding iTerm2 frame");
        return 0;
    }
    fprintf(stdout, "\033]1337;File=inline=1;width=%dpx;height=%dpx:",
            display_width, display_height);
    int ok = fwrite(encoded, 1, encoded_length, stdout) == encoded_length;
    fputs("\a\n", stdout);
    fflush(stdout);
    free(encoded);
    if (!ok) fail(error, error_size, "cannot write iTerm2 frame");
    return ok;
}

int h3_terminal_display_rgb24(h3_terminal_protocol protocol,
                              const uint8_t *rgb, int width, int height,
                              int stride, char *error, size_t error_size) {
    uint8_t *copy = NULL;
    const uint8_t *pixels = NULL;
    size_t size = 0;
    if (!packed_rgb(rgb, width, height, stride, &copy, &pixels, &size,
                    error, error_size)) return 0;
    int ok = protocol == H3_TERM_KITTY ?
        kitty_display(pixels, size, width, height, error, error_size) :
        protocol == H3_TERM_ITERM2 ?
        iterm2_display(pixels, size, width, height, error, error_size) : 0;
    if (protocol == H3_TERM_NONE)
        fail(error, error_size, "no supported graphical terminal detected");
    free(copy);
    return ok;
}
