#include "h3_tokenizer.h"

#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unicode/uchar.h>
#include <unicode/unorm2.h>
#include <unicode/ustring.h>
#include <unicode/utf8.h>

typedef struct {
    char *key;
    uint32_t value;
} h3_map_item;

typedef struct {
    h3_map_item *items;
    size_t capacity;
    size_t count;
} h3_map;

struct h3_tokenizer {
    h3_map vocab;
    h3_map merges;
    h3_map added;
    char **inverse_vocab;
    char **inverse_added;
    size_t inverse_count;
    char *byte_encoder[256];
    int16_t byte_decoder[324];
};

typedef struct {
    const char *cursor;
    const char *end;
    char message[160];
} h3_json;

typedef struct {
    char **values;
    size_t count;
    size_t capacity;
} h3_strings;

typedef struct {
    uint32_t *values;
    size_t count;
    size_t capacity;
} h3_ids;

typedef struct {
    uint32_t value;
    size_t offset;
    size_t length;
} h3_codepoint;

static void h3_error(char *error, size_t size, const char *message) {
    if (error && size) snprintf(error, size, "%s", message ? message : "tokenizer failure");
}

static uint64_t h3_hash(const char *text) {
    uint64_t value = UINT64_C(1469598103934665603);
    for (const unsigned char *p = (const unsigned char *)text; *p; p++) {
        value ^= *p;
        value *= UINT64_C(1099511628211);
    }
    return value;
}

static int h3_map_grow(h3_map *map) {
    size_t capacity = map->capacity ? map->capacity * 2 : 1024;
    h3_map_item *items = calloc(capacity, sizeof(*items));
    if (!items) return 0;
    for (size_t index = 0; index < map->capacity; index++) {
        h3_map_item item = map->items[index];
        if (!item.key) continue;
        size_t slot = (size_t)h3_hash(item.key) & (capacity - 1);
        while (items[slot].key) slot = (slot + 1) & (capacity - 1);
        items[slot] = item;
    }
    free(map->items);
    map->items = items;
    map->capacity = capacity;
    return 1;
}

static int h3_map_put(h3_map *map, char *key, uint32_t value) {
    if (!map->capacity || (map->count + 1) * 10 >= map->capacity * 7)
        if (!h3_map_grow(map)) return 0;
    size_t slot = (size_t)h3_hash(key) & (map->capacity - 1);
    while (map->items[slot].key) {
        if (!strcmp(map->items[slot].key, key)) {
            free(key);
            map->items[slot].value = value;
            return 1;
        }
        slot = (slot + 1) & (map->capacity - 1);
    }
    map->items[slot] = (h3_map_item){key, value};
    map->count++;
    return 1;
}

static int h3_map_get(const h3_map *map, const char *key, uint32_t *value) {
    if (!map->capacity) return 0;
    size_t slot = (size_t)h3_hash(key) & (map->capacity - 1);
    while (map->items[slot].key) {
        if (!strcmp(map->items[slot].key, key)) {
            if (value) *value = map->items[slot].value;
            return 1;
        }
        slot = (slot + 1) & (map->capacity - 1);
    }
    return 0;
}

static void h3_map_free(h3_map *map) {
    for (size_t index = 0; index < map->capacity; index++) free(map->items[index].key);
    free(map->items);
    memset(map, 0, sizeof(*map));
}

static void h3_json_space(h3_json *json) {
    while (json->cursor < json->end && isspace((unsigned char)*json->cursor)) json->cursor++;
}

static int h3_json_fail(h3_json *json, const char *message) {
    if (!json->message[0]) snprintf(json->message, sizeof(json->message), "%s", message);
    return 0;
}

static int h3_json_take(h3_json *json, char wanted) {
    h3_json_space(json);
    if (json->cursor >= json->end || *json->cursor != wanted)
        return h3_json_fail(json, "malformed tokenizer JSON");
    json->cursor++;
    return 1;
}

static int h3_utf8_append(char **buffer, size_t *length, size_t *capacity,
                          uint32_t codepoint) {
    unsigned char bytes[4];
    int count;
    if (codepoint <= 0x7f) { bytes[0] = (unsigned char)codepoint; count = 1; }
    else if (codepoint <= 0x7ff) {
        bytes[0] = (unsigned char)(0xc0 | (codepoint >> 6));
        bytes[1] = (unsigned char)(0x80 | (codepoint & 0x3f)); count = 2;
    } else if (codepoint <= 0xffff) {
        bytes[0] = (unsigned char)(0xe0 | (codepoint >> 12));
        bytes[1] = (unsigned char)(0x80 | ((codepoint >> 6) & 0x3f));
        bytes[2] = (unsigned char)(0x80 | (codepoint & 0x3f)); count = 3;
    } else if (codepoint <= 0x10ffff) {
        bytes[0] = (unsigned char)(0xf0 | (codepoint >> 18));
        bytes[1] = (unsigned char)(0x80 | ((codepoint >> 12) & 0x3f));
        bytes[2] = (unsigned char)(0x80 | ((codepoint >> 6) & 0x3f));
        bytes[3] = (unsigned char)(0x80 | (codepoint & 0x3f)); count = 4;
    } else return 0;
    if (*length + (size_t)count + 1 > *capacity) {
        size_t next = *capacity ? *capacity * 2 : 32;
        while (next < *length + (size_t)count + 1) next *= 2;
        char *grown = realloc(*buffer, next);
        if (!grown) return 0;
        *buffer = grown; *capacity = next;
    }
    memcpy(*buffer + *length, bytes, (size_t)count);
    *length += (size_t)count;
    (*buffer)[*length] = '\0';
    return 1;
}

static int h3_hex(char value) {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    if (value >= 'A' && value <= 'F') return value - 'A' + 10;
    return -1;
}

static int h3_json_u16(h3_json *json, uint32_t *value) {
    if (json->end - json->cursor < 4) return h3_json_fail(json, "truncated JSON escape");
    uint32_t result = 0;
    for (int index = 0; index < 4; index++) {
        int digit = h3_hex(json->cursor[index]);
        if (digit < 0) return h3_json_fail(json, "invalid JSON escape");
        result = result * 16 + (uint32_t)digit;
    }
    json->cursor += 4;
    *value = result;
    return 1;
}

static char *h3_json_string(h3_json *json) {
    if (!h3_json_take(json, '"')) return NULL;
    char *result = NULL;
    size_t length = 0, capacity = 0;
    while (json->cursor < json->end && *json->cursor != '"') {
        unsigned char value = (unsigned char)*json->cursor++;
        uint32_t codepoint = value;
        if (value == '\\') {
            if (json->cursor >= json->end) goto malformed;
            char escape = *json->cursor++;
            if (escape == '"' || escape == '\\' || escape == '/') codepoint = (uint32_t)escape;
            else if (escape == 'b') codepoint = '\b';
            else if (escape == 'f') codepoint = '\f';
            else if (escape == 'n') codepoint = '\n';
            else if (escape == 'r') codepoint = '\r';
            else if (escape == 't') codepoint = '\t';
            else if (escape == 'u') {
                if (!h3_json_u16(json, &codepoint)) goto malformed;
                if (codepoint >= 0xd800 && codepoint <= 0xdbff) {
                    if (json->end - json->cursor < 6 || json->cursor[0] != '\\' || json->cursor[1] != 'u') goto malformed;
                    json->cursor += 2;
                    uint32_t low;
                    if (!h3_json_u16(json, &low) || low < 0xdc00 || low > 0xdfff) goto malformed;
                    codepoint = 0x10000 + ((codepoint - 0xd800) << 10) + (low - 0xdc00);
                }
            } else goto malformed;
            if (!h3_utf8_append(&result, &length, &capacity, codepoint)) goto memory;
        } else {
            if (value < 0x20) goto malformed;
            if (length + 2 > capacity) {
                size_t next = capacity ? capacity * 2 : 32;
                char *grown = realloc(result, next);
                if (!grown) goto memory;
                result = grown; capacity = next;
            }
            result[length++] = (char)value;
            result[length] = '\0';
        }
    }
    if (json->cursor >= json->end) goto malformed;
    json->cursor++;
    if (!result) result = calloc(1, 1);
    return result;
memory:
    h3_json_fail(json, "out of memory parsing tokenizer JSON");
    free(result); return NULL;
malformed:
    h3_json_fail(json, "invalid JSON string");
    free(result); return NULL;
}

static int h3_json_uint(h3_json *json, uint32_t *value) {
    h3_json_space(json);
    errno = 0;
    char *stop = NULL;
    unsigned long parsed = strtoul(json->cursor, &stop, 10);
    if (stop == json->cursor || errno || parsed > UINT32_MAX || stop > json->end)
        return h3_json_fail(json, "invalid tokenizer integer");
    json->cursor = stop;
    *value = (uint32_t)parsed;
    return 1;
}

static int h3_json_literal(h3_json *json, const char *literal) {
    h3_json_space(json);
    size_t length = strlen(literal);
    if ((size_t)(json->end - json->cursor) < length ||
        memcmp(json->cursor, literal, length)) return 0;
    json->cursor += length;
    return 1;
}

static int h3_json_skip(h3_json *json);

static int h3_json_skip_array(h3_json *json) {
    if (!h3_json_take(json, '[')) return 0;
    h3_json_space(json);
    if (json->cursor < json->end && *json->cursor == ']') { json->cursor++; return 1; }
    for (;;) {
        if (!h3_json_skip(json)) return 0;
        h3_json_space(json);
        if (json->cursor < json->end && *json->cursor == ']') { json->cursor++; return 1; }
        if (!h3_json_take(json, ',')) return 0;
    }
}

static int h3_json_skip_object(h3_json *json) {
    if (!h3_json_take(json, '{')) return 0;
    h3_json_space(json);
    if (json->cursor < json->end && *json->cursor == '}') { json->cursor++; return 1; }
    for (;;) {
        char *key = h3_json_string(json);
        if (!key) return 0;
        free(key);
        if (!h3_json_take(json, ':') || !h3_json_skip(json)) return 0;
        h3_json_space(json);
        if (json->cursor < json->end && *json->cursor == '}') { json->cursor++; return 1; }
        if (!h3_json_take(json, ',')) return 0;
    }
}

static int h3_json_skip(h3_json *json) {
    h3_json_space(json);
    if (json->cursor >= json->end) return h3_json_fail(json, "truncated JSON value");
    if (*json->cursor == '"') { char *text = h3_json_string(json); free(text); return text != NULL; }
    if (*json->cursor == '{') return h3_json_skip_object(json);
    if (*json->cursor == '[') return h3_json_skip_array(json);
    if (h3_json_literal(json, "true") || h3_json_literal(json, "false") || h3_json_literal(json, "null")) return 1;
    char *stop = NULL;
    (void)strtod(json->cursor, &stop);
    if (stop == json->cursor) return h3_json_fail(json, "invalid JSON value");
    json->cursor = stop;
    return 1;
}

static char *h3_pair_key(const char *left, const char *right) {
    size_t a = strlen(left), b = strlen(right);
    if (a > SIZE_MAX - b - 2) return NULL;
    char *key = malloc(a + b + 2);
    if (!key) return NULL;
    memcpy(key, left, a); key[a] = '\x1f';
    memcpy(key + a + 1, right, b + 1);
    return key;
}

static int h3_parse_vocab(h3_json *json, h3_tokenizer *tokenizer,
                          uint32_t *maximum_id) {
    if (!h3_json_take(json, '{')) return 0;
    h3_json_space(json);
    if (json->cursor < json->end && *json->cursor == '}') { json->cursor++; return 1; }
    for (;;) {
        char *symbol = h3_json_string(json);
        uint32_t identifier;
        if (!symbol || !h3_json_take(json, ':') || !h3_json_uint(json, &identifier)) {
            free(symbol); return 0;
        }
        if (!h3_map_put(&tokenizer->vocab, symbol, identifier))
            return h3_json_fail(json, "out of memory loading vocabulary");
        if (identifier > *maximum_id) *maximum_id = identifier;
        h3_json_space(json);
        if (json->cursor < json->end && *json->cursor == '}') { json->cursor++; return 1; }
        if (!h3_json_take(json, ',')) return 0;
    }
}

static int h3_parse_merges(h3_json *json, h3_tokenizer *tokenizer) {
    if (!h3_json_take(json, '[')) return 0;
    h3_json_space(json);
    if (json->cursor < json->end && *json->cursor == ']') { json->cursor++; return 1; }
    uint32_t rank = 0;
    for (;;) {
        h3_json_space(json);
        char *left = NULL, *right = NULL;
        if (json->cursor < json->end && *json->cursor == '"') {
            char *entry = h3_json_string(json);
            if (!entry) return 0;
            char *space = strchr(entry, ' ');
            if (!space) { free(entry); return h3_json_fail(json, "invalid tokenizer merge"); }
            *space = '\0';
            left = strdup(entry); right = strdup(space + 1); free(entry);
        } else if (json->cursor < json->end && *json->cursor == '[') {
            if (!h3_json_take(json, '[')) return 0;
            left = h3_json_string(json);
            if (!left || !h3_json_take(json, ',')) { free(left); return 0; }
            right = h3_json_string(json);
            if (!right || !h3_json_take(json, ']')) { free(left); free(right); return 0; }
        } else return h3_json_fail(json, "invalid tokenizer merge");
        char *key = left && right ? h3_pair_key(left, right) : NULL;
        free(left); free(right);
        if (!key || !h3_map_put(&tokenizer->merges, key, rank++)) {
            free(key); return h3_json_fail(json, "out of memory loading merges");
        }
        h3_json_space(json);
        if (json->cursor < json->end && *json->cursor == ']') { json->cursor++; return 1; }
        if (!h3_json_take(json, ',')) return 0;
    }
}

static int h3_parse_model(h3_json *json, h3_tokenizer *tokenizer,
                          uint32_t *maximum_id) {
    int type_ok = 0, vocab_ok = 0, merges_ok = 0, unk_null = 0;
    if (!h3_json_take(json, '{')) return 0;
    h3_json_space(json);
    while (json->cursor < json->end && *json->cursor != '}') {
        char *key = h3_json_string(json);
        if (!key || !h3_json_take(json, ':')) { free(key); return 0; }
        if (!strcmp(key, "type")) {
            char *value = h3_json_string(json);
            type_ok = value && !strcmp(value, "BPE"); free(value);
        } else if (!strcmp(key, "unk_token")) {
            unk_null = h3_json_literal(json, "null");
            if (!unk_null) { free(key); return h3_json_fail(json, "tokenizer unk_token must be null"); }
        } else if (!strcmp(key, "vocab")) {
            vocab_ok = h3_parse_vocab(json, tokenizer, maximum_id);
            if (!vocab_ok) { free(key); return 0; }
        } else if (!strcmp(key, "merges")) {
            merges_ok = h3_parse_merges(json, tokenizer);
            if (!merges_ok) { free(key); return 0; }
        } else if (!h3_json_skip(json)) { free(key); return 0; }
        free(key);
        h3_json_space(json);
        if (*json->cursor == ',') { json->cursor++; h3_json_space(json); }
        else break;
    }
    if (!h3_json_take(json, '}')) return 0;
    if (!type_ok || !vocab_ok || !merges_ok || !unk_null)
        return h3_json_fail(json, "unexpected tokenizer model specification");
    return 1;
}

static int h3_json_bool(h3_json *json, int *value) {
    if (h3_json_literal(json, "true")) { *value = 1; return 1; }
    if (h3_json_literal(json, "false")) { *value = 0; return 1; }
    return h3_json_fail(json, "invalid tokenizer boolean");
}

static int h3_parse_added_item(h3_json *json, h3_tokenizer *tokenizer,
                               uint32_t *maximum_id) {
    char *content = NULL;
    uint32_t identifier = 0;
    int has_id = 0, unsupported = 0;
    if (!h3_json_take(json, '{')) return 0;
    h3_json_space(json);
    while (json->cursor < json->end && *json->cursor != '}') {
        char *key = h3_json_string(json);
        if (!key || !h3_json_take(json, ':')) { free(key); free(content); return 0; }
        if (!strcmp(key, "content")) {
            free(content); content = h3_json_string(json);
            if (!content) { free(key); return 0; }
        } else if (!strcmp(key, "id")) {
            has_id = h3_json_uint(json, &identifier);
            if (!has_id) { free(key); free(content); return 0; }
        } else if (!strcmp(key, "single_word") || !strcmp(key, "lstrip") ||
                   !strcmp(key, "rstrip") || !strcmp(key, "normalized")) {
            int enabled;
            if (!h3_json_bool(json, &enabled)) { free(key); free(content); return 0; }
            unsupported |= enabled;
        } else if (!h3_json_skip(json)) { free(key); free(content); return 0; }
        free(key);
        h3_json_space(json);
        if (*json->cursor == ',') { json->cursor++; h3_json_space(json); }
        else break;
    }
    if (!h3_json_take(json, '}')) { free(content); return 0; }
    if (!content || !has_id || unsupported) {
        free(content); return h3_json_fail(json, "unsupported added-token policy");
    }
    if (!h3_map_put(&tokenizer->added, content, identifier))
        return h3_json_fail(json, "out of memory loading added tokens");
    if (identifier > *maximum_id) *maximum_id = identifier;
    return 1;
}

static int h3_parse_added(h3_json *json, h3_tokenizer *tokenizer,
                          uint32_t *maximum_id) {
    if (!h3_json_take(json, '[')) return 0;
    h3_json_space(json);
    if (*json->cursor == ']') { json->cursor++; return 1; }
    for (;;) {
        if (!h3_parse_added_item(json, tokenizer, maximum_id)) return 0;
        h3_json_space(json);
        if (*json->cursor == ']') { json->cursor++; return 1; }
        if (!h3_json_take(json, ',')) return 0;
    }
}

static int h3_parse_normalizer(h3_json *json) {
    int nfc = 0;
    if (!h3_json_take(json, '{')) return 0;
    h3_json_space(json);
    while (json->cursor < json->end && *json->cursor != '}') {
        char *key = h3_json_string(json);
        if (!key || !h3_json_take(json, ':')) { free(key); return 0; }
        if (!strcmp(key, "type")) {
            char *value = h3_json_string(json);
            nfc = value && !strcmp(value, "NFC"); free(value);
        } else if (!h3_json_skip(json)) { free(key); return 0; }
        free(key);
        h3_json_space(json);
        if (*json->cursor == ',') { json->cursor++; h3_json_space(json); }
        else break;
    }
    if (!h3_json_take(json, '}')) return 0;
    return nfc ? 1 : h3_json_fail(json, "tokenizer normalizer is not NFC");
}

static int h3_parse_root(h3_json *json, h3_tokenizer *tokenizer,
                         uint32_t *maximum_id) {
    int model = 0, normalizer = 0;
    if (!h3_json_take(json, '{')) return 0;
    h3_json_space(json);
    while (json->cursor < json->end && *json->cursor != '}') {
        char *key = h3_json_string(json);
        if (!key || !h3_json_take(json, ':')) { free(key); return 0; }
        if (!strcmp(key, "model")) model = h3_parse_model(json, tokenizer, maximum_id);
        else if (!strcmp(key, "normalizer")) normalizer = h3_parse_normalizer(json);
        else if (!strcmp(key, "added_tokens")) {
            if (!h3_parse_added(json, tokenizer, maximum_id)) { free(key); return 0; }
        } else if (!h3_json_skip(json)) { free(key); return 0; }
        free(key);
        if (json->message[0]) return 0;
        h3_json_space(json);
        if (*json->cursor == ',') { json->cursor++; h3_json_space(json); }
        else break;
    }
    if (!h3_json_take(json, '}')) return 0;
    h3_json_space(json);
    if (json->cursor != json->end) return h3_json_fail(json, "trailing tokenizer JSON data");
    if (!model || !normalizer) return h3_json_fail(json, "incomplete tokenizer specification");
    return 1;
}

static char *h3_read_all(const char *path, size_t *size) {
    FILE *file = fopen(path, "rb");
    if (!file) return NULL;
    if (fseek(file, 0, SEEK_END) || ftell(file) < 0) { fclose(file); return NULL; }
    long length = ftell(file);
    if (fseek(file, 0, SEEK_SET)) { fclose(file); return NULL; }
    char *data = malloc((size_t)length + 1);
    if (!data) { fclose(file); return NULL; }
    size_t got = fread(data, 1, (size_t)length, file);
    fclose(file);
    if (got != (size_t)length) { free(data); return NULL; }
    data[got] = '\0'; *size = got;
    return data;
}

static char *h3_codepoint_string(uint32_t codepoint) {
    char *result = NULL;
    size_t length = 0, capacity = 0;
    if (!h3_utf8_append(&result, &length, &capacity, codepoint)) return NULL;
    return result;
}

h3_tokenizer *h3_tokenizer_load(const char *path, char *error,
                                size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!path) { h3_error(error, error_size, "tokenizer path is required"); return NULL; }
    size_t size = 0;
    char *data = h3_read_all(path, &size);
    if (!data) { h3_error(error, error_size, "cannot read tokenizer JSON"); return NULL; }
    h3_tokenizer *tokenizer = calloc(1, sizeof(*tokenizer));
    if (!tokenizer) { free(data); h3_error(error, error_size, "out of memory"); return NULL; }
    h3_json json = {data, data + size, {0}};
    uint32_t maximum_id = 0;
    if (!h3_parse_root(&json, tokenizer, &maximum_id)) {
        h3_error(error, error_size, json.message); free(data);
        h3_tokenizer_free(tokenizer); return NULL;
    }
    free(data);
    tokenizer->inverse_count = (size_t)maximum_id + 1;
    tokenizer->inverse_vocab = calloc(tokenizer->inverse_count, sizeof(char *));
    tokenizer->inverse_added = calloc(tokenizer->inverse_count, sizeof(char *));
    if (!tokenizer->inverse_vocab || !tokenizer->inverse_added) {
        h3_error(error, error_size, "out of memory indexing vocabulary");
        h3_tokenizer_free(tokenizer); return NULL;
    }
    for (size_t index = 0; index < tokenizer->vocab.capacity; index++) {
        h3_map_item item = tokenizer->vocab.items[index];
        if (item.key && item.value < tokenizer->inverse_count)
            tokenizer->inverse_vocab[item.value] = item.key;
    }
    for (size_t index = 0; index < tokenizer->added.capacity; index++) {
        h3_map_item item = tokenizer->added.items[index];
        if (item.key && item.value < tokenizer->inverse_count)
            tokenizer->inverse_added[item.value] = item.key;
    }
    for (size_t index = 0; index < 324; index++) tokenizer->byte_decoder[index] = -1;
    unsigned extra = 0;
    for (unsigned byte = 0; byte < 256; byte++) {
        int visible = (byte >= '!' && byte <= '~') ||
                      (byte >= 0xa1 && byte <= 0xac) ||
                      (byte >= 0xae && byte <= 0xff);
        uint32_t codepoint = visible ? byte : 256 + extra++;
        tokenizer->byte_encoder[byte] = h3_codepoint_string(codepoint);
        if (!tokenizer->byte_encoder[byte]) {
            h3_error(error, error_size, "out of memory building byte codec");
            h3_tokenizer_free(tokenizer); return NULL;
        }
        tokenizer->byte_decoder[codepoint] = (int16_t)byte;
    }
    return tokenizer;
}

void h3_tokenizer_free(h3_tokenizer *tokenizer) {
    if (!tokenizer) return;
    for (size_t index = 0; index < 256; index++) free(tokenizer->byte_encoder[index]);
    free(tokenizer->inverse_vocab); free(tokenizer->inverse_added);
    h3_map_free(&tokenizer->vocab); h3_map_free(&tokenizer->merges);
    h3_map_free(&tokenizer->added); free(tokenizer);
}

static char *h3_nfc(const char *utf8) {
    UErrorCode status = U_ZERO_ERROR;
    int32_t utf16_length = 0;
    u_strFromUTF8(NULL, 0, &utf16_length, utf8, -1, &status);
    if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status)) return NULL;
    status = U_ZERO_ERROR;
    UChar *utf16 = malloc(((size_t)utf16_length + 1) * sizeof(*utf16));
    if (!utf16) return NULL;
    u_strFromUTF8(utf16, utf16_length + 1, NULL, utf8, -1, &status);
    const UNormalizer2 *nfc = unorm2_getNFCInstance(&status);
    int32_t normalized_length = unorm2_normalize(nfc, utf16, utf16_length,
                                                  NULL, 0, &status);
    if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status)) { free(utf16); return NULL; }
    status = U_ZERO_ERROR;
    UChar *normalized = malloc(((size_t)normalized_length + 1) * sizeof(*normalized));
    if (!normalized) { free(utf16); return NULL; }
    unorm2_normalize(nfc, utf16, utf16_length, normalized,
                     normalized_length + 1, &status);
    free(utf16);
    if (U_FAILURE(status)) { free(normalized); return NULL; }
    int32_t output_length = 0;
    status = U_ZERO_ERROR;
    u_strToUTF8(NULL, 0, &output_length, normalized, normalized_length, &status);
    if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status)) { free(normalized); return NULL; }
    status = U_ZERO_ERROR;
    char *output = malloc((size_t)output_length + 1);
    if (!output) { free(normalized); return NULL; }
    u_strToUTF8(output, output_length + 1, NULL, normalized, normalized_length, &status);
    free(normalized);
    if (U_FAILURE(status)) { free(output); return NULL; }
    return output;
}

static int h3_strings_push(h3_strings *strings, char *value) {
    if (strings->count == strings->capacity) {
        size_t capacity = strings->capacity ? strings->capacity * 2 : 16;
        char **values = realloc(strings->values, capacity * sizeof(*values));
        if (!values) return 0;
        strings->values = values; strings->capacity = capacity;
    }
    strings->values[strings->count++] = value;
    return 1;
}

static void h3_strings_free(h3_strings *strings) {
    for (size_t index = 0; index < strings->count; index++) free(strings->values[index]);
    free(strings->values); memset(strings, 0, sizeof(*strings));
}

static int h3_ids_push(h3_ids *ids, uint32_t value) {
    if (ids->count == ids->capacity) {
        size_t capacity = ids->capacity ? ids->capacity * 2 : 32;
        uint32_t *values = realloc(ids->values, capacity * sizeof(*values));
        if (!values) return 0;
        ids->values = values; ids->capacity = capacity;
    }
    ids->values[ids->count++] = value;
    return 1;
}

static int h3_codepoints(const char *text, h3_codepoint **output,
                         size_t *count) {
    size_t bytes = strlen(text), used = 0;
    h3_codepoint *points = malloc((bytes ? bytes : 1) * sizeof(*points));
    if (!points) return 0;
    int32_t index = 0;
    while ((size_t)index < bytes) {
        int32_t start = index;
        UChar32 value;
        U8_NEXT((const uint8_t *)text, index, (int32_t)bytes, value);
        if (value < 0) { free(points); return 0; }
        points[used++] = (h3_codepoint){(uint32_t)value, (size_t)start,
                                       (size_t)(index - start)};
    }
    *output = points; *count = used; return 1;
}

static int h3_letter(uint32_t value) {
    int8_t category = u_charType((UChar32)value);
    return category == U_UPPERCASE_LETTER || category == U_LOWERCASE_LETTER ||
           category == U_TITLECASE_LETTER || category == U_MODIFIER_LETTER ||
           category == U_OTHER_LETTER;
}

static int h3_number(uint32_t value) {
    int8_t category = u_charType((UChar32)value);
    return category == U_DECIMAL_DIGIT_NUMBER || category == U_LETTER_NUMBER ||
           category == U_OTHER_NUMBER;
}

static int h3_space(uint32_t value) {
    return u_isUWhiteSpace((UChar32)value) || (value >= 0x1c && value <= 0x1f);
}

static char *h3_slice(const char *text, const h3_codepoint *points,
                      size_t start, size_t stop) {
    size_t offset = points[start].offset;
    size_t end = points[stop - 1].offset + points[stop - 1].length;
    char *result = malloc(end - offset + 1);
    if (!result) return NULL;
    memcpy(result, text + offset, end - offset); result[end - offset] = '\0';
    return result;
}

static size_t h3_contraction(const h3_codepoint *points, size_t count,
                             size_t index) {
    static const char *values[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
    if (points[index].value != '\'') return 0;
    for (size_t item = 0; item < sizeof(values) / sizeof(values[0]); item++) {
        size_t length = strlen(values[item]);
        if (index + length > count) continue;
        int matches = 1;
        for (size_t offset = 1; offset < length; offset++) {
            uint32_t got = points[index + offset].value;
            if (got >= 'A' && got <= 'Z') got += 'a' - 'A';
            if (got != (unsigned char)values[item][offset]) matches = 0;
        }
        if (matches) return length;
    }
    return 0;
}

static int h3_pretokenize(const char *input, h3_strings *pieces) {
    char *text = h3_nfc(input);
    if (!text) return 0;
    h3_codepoint *points = NULL;
    size_t count = 0;
    if (!h3_codepoints(text, &points, &count)) { free(text); return 0; }
    size_t index = 0;
    while (index < count) {
        size_t contraction = h3_contraction(points, count, index);
        size_t stop = index;
        if (contraction) stop = index + contraction;
        else {
            uint32_t value = points[index].value;
            ptrdiff_t letter_start = (ptrdiff_t)index;
            if (!h3_letter(value)) {
                if (value != '\r' && value != '\n' && !h3_number(value) &&
                    index + 1 < count && h3_letter(points[index + 1].value))
                    letter_start++;
                else letter_start = -1;
            }
            if (letter_start >= 0) {
                stop = (size_t)letter_start;
                while (stop < count && h3_letter(points[stop].value)) stop++;
            } else if (h3_number(value)) stop = index + 1;
            else {
                size_t punct_start = index +
                    (value == ' ' && index + 1 < count &&
                     !h3_space(points[index + 1].value) &&
                     !h3_letter(points[index + 1].value) &&
                     !h3_number(points[index + 1].value));
                stop = punct_start;
                while (stop < count && !h3_space(points[stop].value) &&
                       !h3_letter(points[stop].value) &&
                       !h3_number(points[stop].value)) stop++;
                if (stop > punct_start) {
                    while (stop < count && (points[stop].value == '\r' ||
                                             points[stop].value == '\n')) stop++;
                } else if (h3_space(value)) {
                    size_t whitespace_end = index + 1;
                    while (whitespace_end < count && h3_space(points[whitespace_end].value)) whitespace_end++;
                    ptrdiff_t newline_end = -1;
                    for (size_t cursor = index; cursor < whitespace_end; cursor++)
                        if (points[cursor].value == '\r' || points[cursor].value == '\n')
                            newline_end = (ptrdiff_t)cursor + 1;
                    if (newline_end >= 0) stop = (size_t)newline_end;
                    else if (whitespace_end == count) stop = whitespace_end;
                    else if (whitespace_end - index > 1) stop = whitespace_end - 1;
                    else stop = index + 1;
                } else { free(points); free(text); return 0; }
            }
        }
        char *piece = h3_slice(text, points, index, stop);
        if (!piece || !h3_strings_push(pieces, piece)) {
            free(piece); free(points); free(text); return 0;
        }
        index = stop;
    }
    free(points); free(text); return 1;
}

static int h3_bpe(const h3_tokenizer *tokenizer, const char *piece,
                  h3_ids *output) {
    h3_strings symbols = {0};
    for (const unsigned char *byte = (const unsigned char *)piece; *byte; byte++) {
        char *symbol = strdup(tokenizer->byte_encoder[*byte]);
        if (!symbol || !h3_strings_push(&symbols, symbol)) {
            free(symbol); h3_strings_free(&symbols); return 0;
        }
    }
    while (symbols.count > 1) {
        uint32_t best_rank = UINT32_MAX;
        size_t best = SIZE_MAX;
        for (size_t index = 0; index + 1 < symbols.count; index++) {
            char *key = h3_pair_key(symbols.values[index], symbols.values[index + 1]);
            uint32_t rank;
            int found = key && h3_map_get(&tokenizer->merges, key, &rank);
            free(key);
            if (found && rank < best_rank) { best_rank = rank; best = index; }
        }
        if (best == SIZE_MAX) break;
        const char *left = symbols.values[best], *right = symbols.values[best + 1];
        h3_strings merged = {0};
        for (size_t index = 0; index < symbols.count;) {
            if (index + 1 < symbols.count && !strcmp(symbols.values[index], left) &&
                !strcmp(symbols.values[index + 1], right)) {
                size_t a = strlen(left), b = strlen(right);
                char *value = malloc(a + b + 1);
                if (value) { memcpy(value, left, a); memcpy(value + a, right, b + 1); }
                if (!value || !h3_strings_push(&merged, value)) {
                    free(value); h3_strings_free(&merged); h3_strings_free(&symbols); return 0;
                }
                index += 2;
            } else {
                char *value = strdup(symbols.values[index++]);
                if (!value || !h3_strings_push(&merged, value)) {
                    free(value); h3_strings_free(&merged); h3_strings_free(&symbols); return 0;
                }
            }
        }
        h3_strings_free(&symbols); symbols = merged;
    }
    for (size_t index = 0; index < symbols.count; index++) {
        uint32_t identifier;
        if (!h3_map_get(&tokenizer->vocab, symbols.values[index], &identifier) ||
            !h3_ids_push(output, identifier)) {
            h3_strings_free(&symbols); return 0;
        }
    }
    h3_strings_free(&symbols); return 1;
}

static int h3_encode_plain(const h3_tokenizer *tokenizer, const char *text,
                           h3_ids *output) {
    h3_strings pieces = {0};
    if (!h3_pretokenize(text, &pieces)) return 0;
    for (size_t index = 0; index < pieces.count; index++)
        if (!h3_bpe(tokenizer, pieces.values[index], output)) {
            h3_strings_free(&pieces); return 0;
        }
    h3_strings_free(&pieces); return 1;
}

static int h3_added_match(const h3_tokenizer *tokenizer, const char *text,
                          size_t start, size_t *offset, size_t *length,
                          uint32_t *identifier) {
    int found = 0;
    for (size_t index = 0; index < tokenizer->added.capacity; index++) {
        h3_map_item item = tokenizer->added.items[index];
        if (!item.key) continue;
        const char *match = strstr(text + start, item.key);
        if (!match) continue;
        size_t at = (size_t)(match - text), size = strlen(item.key);
        if (!found || at < *offset || (at == *offset && size > *length)) {
            found = 1; *offset = at; *length = size; *identifier = item.value;
        }
    }
    return found;
}

int h3_tokenizer_encode(const h3_tokenizer *tokenizer, const char *utf8,
                        int pad_empty, uint32_t **ids, size_t *count,
                        char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!tokenizer || !utf8 || !ids || !count) return 0;
    *ids = NULL; *count = 0;
    h3_codepoint *validation = NULL; size_t validation_count = 0;
    if (!h3_codepoints(utf8, &validation, &validation_count)) {
        h3_error(error, error_size, "prompt is not valid UTF-8"); return 0;
    }
    free(validation);
    h3_ids output = {0};
    size_t start = 0, text_length = strlen(utf8);
    while (start < text_length) {
        size_t offset = 0, length = 0; uint32_t identifier = 0;
        if (!h3_added_match(tokenizer, utf8, start, &offset, &length, &identifier)) break;
        if (offset > start) {
            char *plain = strndup(utf8 + start, offset - start);
            int ok = plain && h3_encode_plain(tokenizer, plain, &output);
            free(plain);
            if (!ok) goto failure;
        }
        if (!h3_ids_push(&output, identifier)) goto failure;
        start = offset + length;
    }
    if (start < text_length && !h3_encode_plain(tokenizer, utf8 + start, &output)) goto failure;
    if (!output.count && pad_empty && !h3_ids_push(&output, H3_PAD_TOKEN_ID)) goto failure;
    *ids = output.values; *count = output.count; return 1;
failure:
    free(output.values); h3_error(error, error_size, "unable to encode prompt"); return 0;
}

void h3_tokenizer_ids_free(uint32_t *ids) { free(ids); }

static int h3_bytes_append(char **output, size_t *length, size_t *capacity,
                           const void *data, size_t bytes) {
    if (*length + bytes + 1 > *capacity) {
        size_t next = *capacity ? *capacity * 2 : 64;
        while (next < *length + bytes + 1) next *= 2;
        char *grown = realloc(*output, next);
        if (!grown) return 0;
        *output = grown; *capacity = next;
    }
    memcpy(*output + *length, data, bytes); *length += bytes;
    (*output)[*length] = '\0'; return 1;
}

char *h3_tokenizer_decode(const h3_tokenizer *tokenizer,
                          const uint32_t *ids, size_t count,
                          char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!tokenizer || (!ids && count)) return NULL;
    char *result = NULL;
    size_t length = 0, capacity = 0;
    for (size_t index = 0; index < count; index++) {
        uint32_t identifier = ids[index];
        if (identifier >= tokenizer->inverse_count) {
            h3_error(error, error_size, "token ID is out of range"); free(result); return NULL;
        }
        const char *added = tokenizer->inverse_added[identifier];
        if (added) {
            if (!h3_bytes_append(&result, &length, &capacity, added, strlen(added))) goto memory;
            continue;
        }
        const char *symbol = tokenizer->inverse_vocab[identifier];
        if (!symbol) {
            h3_error(error, error_size, "unknown token ID"); free(result); return NULL;
        }
        int32_t offset = 0, symbol_length = (int32_t)strlen(symbol);
        while (offset < symbol_length) {
            UChar32 codepoint;
            U8_NEXT((const uint8_t *)symbol, offset, symbol_length, codepoint);
            if (codepoint < 0 || codepoint >= 324 ||
                tokenizer->byte_decoder[codepoint] < 0) {
                h3_error(error, error_size, "invalid byte-level token");
                free(result); return NULL;
            }
            unsigned char byte = (unsigned char)tokenizer->byte_decoder[codepoint];
            if (!h3_bytes_append(&result, &length, &capacity, &byte, 1)) goto memory;
        }
    }
    if (!result) result = calloc(1, 1);
    if (!result) goto memory;
    h3_codepoint *validation = NULL; size_t validation_count = 0;
    if (!h3_codepoints(result, &validation, &validation_count)) {
        free(result); result = strdup("\xef\xbf\xbd");
    }
    free(validation);
    return result;
memory:
    h3_error(error, error_size, "out of memory decoding tokens");
    free(result); return NULL;
}
