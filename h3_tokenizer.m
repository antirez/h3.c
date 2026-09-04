#import <Foundation/Foundation.h>

#include "h3_tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unicode/uchar.h>

typedef struct {
    uint32_t value;
    NSUInteger location;
    NSUInteger length;
} H3Codepoint;

@interface H3Tokenizer : NSObject {
@public
    int16_t byteDecoder[324];
}
@property(nonatomic, strong) NSDictionary<NSString *, NSNumber *> *vocab;
@property(nonatomic, strong) NSArray *inverseVocab;
@property(nonatomic, strong) NSDictionary<NSString *, NSNumber *> *mergeRanks;
@property(nonatomic, strong) NSDictionary<NSString *, NSNumber *> *addedTokens;
@property(nonatomic, strong) NSArray *inverseAddedTokens;
@property(nonatomic, strong) NSArray<NSString *> *addedAlternatives;
@property(nonatomic, strong) NSMutableDictionary<NSString *, NSArray<NSNumber *> *> *bpeCache;
@property(nonatomic, strong) NSArray<NSString *> *byteEncoder;
@end
@implementation H3Tokenizer
@end

/* MiniMax declares seven H3 control tokens in tokenizer_config.json under
 * "additional_special_tokens" only. They are absent from tokenizer.json, so the
 * reference tokenizer appends them after the last declared added token, giving
 * <d> the identifier 151669 for the released vocabulary. Without them the BPE
 * merges swallow the markup: "<d>[English] Hi.</d>" becomes nine ordinary
 * pieces whose first token fuses the tag with the following bracket, so the
 * dialogue, cutoff, lyric, and caption markers described in the official
 * prompting guides never reach the DiT. */
static NSString *const h3_extra_special_tokens[] = {
    @"<d>", @"</d>", @"<|cutoff|>", @"<|lyrics_start|>", @"<|lyrics_end|>",
    @"<|caption_start|>", @"<|caption_end|>",
};

static H3Tokenizer *TOK(const h3_tokenizer *tokenizer) {
    return (__bridge H3Tokenizer *)(void *)tokenizer;
}

static void h3_tok_error(char *error, size_t size, NSString *message) {
    if (error && size) snprintf(error, size, "%s", message.UTF8String);
}

static NSString *h3_codepoint_string(uint32_t value) {
    if (value <= 0xffff) {
        unichar unit = (unichar)value;
        return [NSString stringWithCharacters:&unit length:1];
    }
    value -= 0x10000;
    unichar units[2] = {(unichar)(0xd800 + (value >> 10)),
                        (unichar)(0xdc00 + (value & 0x3ff))};
    return [NSString stringWithCharacters:units length:2];
}

static H3Codepoint *h3_codepoints(NSString *text, size_t *count) {
    NSUInteger length = text.length;
    H3Codepoint *points = calloc(length ? length : 1, sizeof(*points));
    if (!points) return NULL;
    size_t used = 0;
    for (NSUInteger index = 0; index < length;) {
        unichar first = [text characterAtIndex:index];
        uint32_t value = first;
        NSUInteger units = 1;
        if (first >= 0xd800 && first <= 0xdbff && index + 1 < length) {
            unichar second = [text characterAtIndex:index + 1];
            if (second >= 0xdc00 && second <= 0xdfff) {
                value = UINT32_C(0x10000) + (((uint32_t)first - 0xd800) << 10) +
                        ((uint32_t)second - 0xdc00);
                units = 2;
            }
        }
        points[used++] = (H3Codepoint){value, index, units};
        index += units;
    }
    *count = used;
    return points;
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
    return u_isUWhiteSpace((UChar32)value) ||
           (value >= 0x1c && value <= 0x1f);
}

static NSString *h3_slice(NSString *text, const H3Codepoint *points,
                          size_t start, size_t stop) {
    NSUInteger location = points[start].location;
    H3Codepoint last = points[stop - 1];
    NSUInteger end = last.location + last.length;
    return [text substringWithRange:NSMakeRange(location, end - location)];
}

static size_t h3_contraction(const H3Codepoint *points, size_t count,
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

static NSArray<NSString *> *h3_pretokenize(NSString *input) {
    NSString *text = input.precomposedStringWithCanonicalMapping;
    size_t count = 0;
    H3Codepoint *points = h3_codepoints(text, &count);
    if (!points) return nil;
    NSMutableArray<NSString *> *pieces = [NSMutableArray array];
    size_t index = 0;
    while (index < count) {
        size_t contraction = h3_contraction(points, count, index);
        if (contraction) {
            [pieces addObject:h3_slice(text, points, index, index + contraction)];
            index += contraction;
            continue;
        }
        uint32_t value = points[index].value;
        ptrdiff_t letter_start = (ptrdiff_t)index;
        if (h3_letter(value)) {
            /* Already at the first letter. */
        } else if (value != '\r' && value != '\n' && !h3_number(value) &&
                   index + 1 < count && h3_letter(points[index + 1].value)) {
            letter_start++;
        } else {
            letter_start = -1;
        }
        if (letter_start >= 0) {
            size_t stop = (size_t)letter_start;
            while (stop < count && h3_letter(points[stop].value)) stop++;
            [pieces addObject:h3_slice(text, points, index, stop)];
            index = stop;
            continue;
        }
        if (h3_number(value)) {
            [pieces addObject:h3_slice(text, points, index, index + 1)];
            index++;
            continue;
        }
        size_t punct_start = index +
            (value == ' ' && index + 1 < count &&
             !h3_space(points[index + 1].value) &&
             !h3_letter(points[index + 1].value) &&
             !h3_number(points[index + 1].value));
        size_t stop = punct_start;
        while (stop < count && !h3_space(points[stop].value) &&
               !h3_letter(points[stop].value) &&
               !h3_number(points[stop].value)) stop++;
        if (stop > punct_start) {
            while (stop < count &&
                   (points[stop].value == '\r' || points[stop].value == '\n')) stop++;
            [pieces addObject:h3_slice(text, points, index, stop)];
            index = stop;
            continue;
        }
        if (h3_space(value)) {
            size_t whitespace_end = index + 1;
            while (whitespace_end < count &&
                   h3_space(points[whitespace_end].value)) whitespace_end++;
            ptrdiff_t newline_end = -1;
            for (size_t cursor = index; cursor < whitespace_end; cursor++) {
                if (points[cursor].value == '\r' || points[cursor].value == '\n') {
                    newline_end = (ptrdiff_t)cursor + 1;
                }
            }
            size_t piece_end;
            if (newline_end >= 0) piece_end = (size_t)newline_end;
            else if (whitespace_end == count) piece_end = whitespace_end;
            else if (whitespace_end - index > 1) piece_end = whitespace_end - 1;
            else piece_end = index + 1;
            [pieces addObject:h3_slice(text, points, index, piece_end)];
            index = piece_end;
            continue;
        }
        free(points);
        return nil;
    }
    free(points);
    return pieces;
}

static NSString *h3_pair_key(NSString *left, NSString *right) {
    return [NSString stringWithFormat:@"%@\uffff%@", left, right];
}

static NSArray<NSNumber *> *h3_bpe(H3Tokenizer *tokenizer, NSString *piece,
                                   NSString **failure) {
    NSData *utf8 = [piece dataUsingEncoding:NSUTF8StringEncoding];
    const unsigned char *bytes = utf8.bytes;
    NSMutableString *encoded = [NSMutableString string];
    for (NSUInteger index = 0; index < utf8.length; index++) {
        [encoded appendString:tokenizer.byteEncoder[bytes[index]]];
    }
    NSArray<NSNumber *> *cached = tokenizer.bpeCache[encoded];
    if (cached) return cached;
    NSMutableArray<NSString *> *symbols = [NSMutableArray array];
    for (NSUInteger index = 0; index < encoded.length; index++) {
        [symbols addObject:[encoded substringWithRange:NSMakeRange(index, 1)]];
    }
    while (symbols.count > 1) {
        NSNumber *best_rank = nil;
        NSUInteger best = 0;
        for (NSUInteger index = 0; index + 1 < symbols.count; index++) {
            NSNumber *rank = tokenizer.mergeRanks[h3_pair_key(symbols[index],
                                                               symbols[index + 1])];
            if (rank && (!best_rank || rank.unsignedIntegerValue <
                                      best_rank.unsignedIntegerValue)) {
                best_rank = rank;
                best = index;
            }
        }
        if (!best_rank) break;
        NSString *left = symbols[best];
        NSString *right = symbols[best + 1];
        NSMutableArray<NSString *> *merged = [NSMutableArray array];
        for (NSUInteger index = 0; index < symbols.count;) {
            if (index + 1 < symbols.count && [symbols[index] isEqual:left] &&
                [symbols[index + 1] isEqual:right]) {
                [merged addObject:[left stringByAppendingString:right]];
                index += 2;
            } else {
                [merged addObject:symbols[index++]];
            }
        }
        symbols = merged;
    }
    NSMutableArray<NSNumber *> *ids = [NSMutableArray arrayWithCapacity:symbols.count];
    for (NSString *symbol in symbols) {
        NSNumber *token_id = tokenizer.vocab[symbol];
        if (!token_id) {
            if (failure) *failure = [NSString stringWithFormat:
                @"BPE symbol is absent from vocabulary: %@", symbol];
            return nil;
        }
        [ids addObject:token_id];
    }
    NSArray<NSNumber *> *result = [ids copy];
    tokenizer.bpeCache[encoded] = result;
    return result;
}

static int h3_encode_plain(H3Tokenizer *tokenizer, NSString *text,
                           NSMutableArray<NSNumber *> *output,
                           NSString **failure) {
    NSArray<NSString *> *pieces = h3_pretokenize(text);
    if (!pieces) {
        if (failure) *failure = @"unable to pre-tokenize input";
        return 0;
    }
    for (NSString *piece in pieces) {
        NSArray<NSNumber *> *ids = h3_bpe(tokenizer, piece, failure);
        if (!ids) return 0;
        [output addObjectsFromArray:ids];
    }
    return 1;
}

static int h3_added_match(H3Tokenizer *tokenizer, NSString *text,
                          NSRange search, NSRange *match, NSString **token) {
    int found = 0;
    for (NSString *candidate in tokenizer.addedAlternatives) {
        NSRange range = [text rangeOfString:candidate options:0 range:search];
        if (range.location == NSNotFound) continue;
        if (!found || range.location < match->location ||
            (range.location == match->location && range.length > match->length)) {
            *match = range;
            *token = candidate;
            found = 1;
        }
    }
    return found;
}

h3_tokenizer *h3_tokenizer_load(const char *path, char *error,
                                size_t error_size) {
    @autoreleasepool {
        if (error && error_size) error[0] = '\0';
        if (!path) {
            h3_tok_error(error, error_size, @"tokenizer path is required");
            return NULL;
        }
        NSData *data = [NSData dataWithContentsOfFile:
            [NSString stringWithUTF8String:path]];
        if (!data) {
            h3_tok_error(error, error_size, [NSString stringWithFormat:
                         @"cannot read tokenizer: %s", path]);
            return NULL;
        }
        NSError *json_error = nil;
        NSDictionary *config = [NSJSONSerialization JSONObjectWithData:data
                                                                options:0
                                                                  error:&json_error];
        if (![config isKindOfClass:NSDictionary.class]) {
            NSString *message = json_error.localizedDescription;
            h3_tok_error(error, error_size, message ? message :
                                              @"invalid tokenizer JSON");
            return NULL;
        }
        NSDictionary *model = config[@"model"];
        NSDictionary *normalizer = config[@"normalizer"];
        if (![model[@"type"] isEqual:@"BPE"] ||
            model[@"unk_token"] != NSNull.null ||
            ![normalizer[@"type"] isEqual:@"NFC"] ||
            ![model[@"vocab"] isKindOfClass:NSDictionary.class] ||
            ![model[@"merges"] isKindOfClass:NSArray.class]) {
            h3_tok_error(error, error_size, @"unexpected tokenizer specification");
            return NULL;
        }
        H3Tokenizer *tokenizer = [[H3Tokenizer alloc] init];
        tokenizer.vocab = model[@"vocab"];
        NSUInteger maximum_id = 0;
        for (NSNumber *number in tokenizer.vocab.allValues) {
            maximum_id = MAX(maximum_id, number.unsignedIntegerValue);
        }
        NSArray *added = config[@"added_tokens"];
        if (!added) added = @[];
        NSMutableSet<NSString *> *declared = [NSMutableSet set];
        for (NSDictionary *token in added) {
            maximum_id = MAX(maximum_id, [token[@"id"] unsignedIntegerValue]);
            NSString *content = token[@"content"];
            if (content) [declared addObject:content];
        }
        NSMutableArray<NSString *> *extra_tokens = [NSMutableArray array];
        for (size_t index = 0; index < sizeof(h3_extra_special_tokens) /
                                       sizeof(h3_extra_special_tokens[0]); index++) {
            NSString *content = h3_extra_special_tokens[index];
            if (![declared containsObject:content]) [extra_tokens addObject:content];
        }
        NSUInteger first_extra_id = maximum_id + 1;
        maximum_id += extra_tokens.count;
        NSMutableArray *inverse_vocab = [NSMutableArray arrayWithCapacity:maximum_id + 1];
        NSMutableArray *inverse_added = [NSMutableArray arrayWithCapacity:maximum_id + 1];
        for (NSUInteger index = 0; index <= maximum_id; index++) {
            [inverse_vocab addObject:NSNull.null];
            [inverse_added addObject:NSNull.null];
        }
        [tokenizer.vocab enumerateKeysAndObjectsUsingBlock:
            ^(NSString *symbol, NSNumber *number, BOOL *stop) {
                (void)stop;
                inverse_vocab[number.unsignedIntegerValue] = symbol;
            }];
        tokenizer.inverseVocab = inverse_vocab;

        NSMutableDictionary *ranks = [NSMutableDictionary dictionary];
        NSUInteger rank = 0;
        for (id entry in model[@"merges"]) {
            NSArray<NSString *> *pair = nil;
            if ([entry isKindOfClass:NSString.class]) {
                NSRange separator = [entry rangeOfString:@" "];
                if (separator.location != NSNotFound) {
                    pair = @[[entry substringToIndex:separator.location],
                             [entry substringFromIndex:separator.location + 1]];
                }
            } else if ([entry isKindOfClass:NSArray.class] && [entry count] == 2) {
                pair = entry;
            }
            if (!pair) {
                h3_tok_error(error, error_size, @"invalid tokenizer merge");
                return NULL;
            }
            ranks[h3_pair_key(pair[0], pair[1])] = @(rank++);
        }
        tokenizer.mergeRanks = ranks;

        NSMutableDictionary *added_tokens = [NSMutableDictionary dictionary];
        for (NSDictionary *token in added) {
            if ([token[@"single_word"] boolValue] || [token[@"lstrip"] boolValue] ||
                [token[@"rstrip"] boolValue] || [token[@"normalized"] boolValue]) {
                h3_tok_error(error, error_size, @"unsupported added-token policy");
                return NULL;
            }
            NSString *content = token[@"content"];
            NSNumber *identifier = token[@"id"];
            added_tokens[content] = identifier;
            inverse_added[identifier.unsignedIntegerValue] = content;
        }
        for (NSUInteger index = 0; index < extra_tokens.count; index++) {
            NSNumber *identifier = @(first_extra_id + index);
            added_tokens[extra_tokens[index]] = identifier;
            inverse_added[identifier.unsignedIntegerValue] = extra_tokens[index];
        }
        tokenizer.addedTokens = added_tokens;
        tokenizer.inverseAddedTokens = inverse_added;
        tokenizer.addedAlternatives = [added_tokens.allKeys
            sortedArrayUsingComparator:^NSComparisonResult(NSString *left,
                                                            NSString *right) {
                if (left.length > right.length) return NSOrderedAscending;
                if (left.length < right.length) return NSOrderedDescending;
                return [left compare:right];
            }];
        tokenizer.bpeCache = [NSMutableDictionary dictionary];

        NSMutableArray<NSString *> *byte_encoder = [NSMutableArray arrayWithCapacity:256];
        int16_t decoder[324];
        for (size_t index = 0; index < 324; index++) decoder[index] = -1;
        unsigned extra = 0;
        for (unsigned byte = 0; byte < 256; byte++) {
            int visible = (byte >= '!' && byte <= '~') ||
                          (byte >= 0xa1 && byte <= 0xac) ||
                          (byte >= 0xae && byte <= 0xff);
            uint32_t codepoint = visible ? byte : 256 + extra++;
            [byte_encoder addObject:h3_codepoint_string(codepoint)];
            decoder[codepoint] = (int16_t)byte;
        }
        tokenizer.byteEncoder = byte_encoder;
        memcpy(tokenizer->byteDecoder, decoder, sizeof(decoder));
        return (__bridge_retained h3_tokenizer *)tokenizer;
    }
}

void h3_tokenizer_free(h3_tokenizer *tokenizer) {
    if (!tokenizer) return;
    H3Tokenizer *object = CFBridgingRelease(tokenizer);
    object.bpeCache = nil;
}

int h3_tokenizer_encode(const h3_tokenizer *opaque, const char *utf8,
                        int pad_empty, uint32_t **ids, size_t *count,
                        char *error, size_t error_size) {
    @autoreleasepool {
        if (error && error_size) error[0] = '\0';
        if (!opaque || !utf8 || !ids || !count) return 0;
        *ids = NULL;
        *count = 0;
        NSString *text = [NSString stringWithUTF8String:utf8];
        if (!text) {
            h3_tok_error(error, error_size, @"prompt is not valid UTF-8");
            return 0;
        }
        H3Tokenizer *tokenizer = TOK(opaque);
        NSMutableArray<NSNumber *> *output = [NSMutableArray array];
        NSUInteger start = 0;
        NSString *failure = nil;
        while (start < text.length) {
            NSRange search = NSMakeRange(start, text.length - start);
            NSRange match;
            NSString *added = nil;
            if (!h3_added_match(tokenizer, text, search, &match, &added)) break;
            if (match.location > start &&
                !h3_encode_plain(tokenizer,
                    [text substringWithRange:NSMakeRange(start, match.location - start)],
                    output, &failure)) goto fail;
            [output addObject:tokenizer.addedTokens[added]];
            start = NSMaxRange(match);
        }
        if (start < text.length &&
            !h3_encode_plain(tokenizer, [text substringFromIndex:start], output,
                             &failure)) goto fail;
        if (output.count == 0 && pad_empty) [output addObject:@(H3_PAD_TOKEN_ID)];
        if (output.count) {
            uint32_t *values = malloc(output.count * sizeof(*values));
            if (!values) {
                failure = @"out of memory encoding prompt";
                goto fail;
            }
            for (NSUInteger index = 0; index < output.count; index++) {
                values[index] = output[index].unsignedIntValue;
            }
            *ids = values;
        }
        *count = output.count;
        return 1;
fail:
        h3_tok_error(error, error_size, failure ? failure : @"tokenizer failure");
        return 0;
    }
}

void h3_tokenizer_ids_free(uint32_t *ids) {
    free(ids);
}

char *h3_tokenizer_decode(const h3_tokenizer *opaque,
                          const uint32_t *ids, size_t count,
                          char *error, size_t error_size) {
    @autoreleasepool {
        if (error && error_size) error[0] = '\0';
        if (!opaque || (!ids && count)) return NULL;
        H3Tokenizer *tokenizer = TOK(opaque);
        NSMutableString *result = [NSMutableString string];
        NSMutableData *bytes = [NSMutableData data];
        void (^flush)(void) = ^{
            if (!bytes.length) return;
            NSString *part = [[NSString alloc] initWithData:bytes
                                                   encoding:NSUTF8StringEncoding];
            [result appendString:part ? part : @"\ufffd"];
            [bytes setLength:0];
        };
        for (size_t index = 0; index < count; index++) {
            uint32_t identifier = ids[index];
            if (identifier >= tokenizer.inverseVocab.count) {
                h3_tok_error(error, error_size, @"token ID is out of range");
                return NULL;
            }
            id added = tokenizer.inverseAddedTokens[identifier];
            if (added != NSNull.null) {
                flush();
                [result appendString:added];
                continue;
            }
            id symbol = tokenizer.inverseVocab[identifier];
            if (symbol == NSNull.null) {
                h3_tok_error(error, error_size, @"unknown token ID");
                return NULL;
            }
            for (NSUInteger offset = 0; offset < [symbol length]; offset++) {
                unichar codepoint = [symbol characterAtIndex:offset];
                if (codepoint >= 324 || tokenizer->byteDecoder[codepoint] < 0) {
                    h3_tok_error(error, error_size, @"invalid byte-level token");
                    return NULL;
                }
                unsigned char byte = (unsigned char)tokenizer->byteDecoder[codepoint];
                [bytes appendBytes:&byte length:1];
            }
        }
        flush();
        const char *utf8 = result.UTF8String;
        char *copy = strdup(utf8 ? utf8 : "");
        if (!copy) h3_tok_error(error, error_size, @"out of memory decoding tokens");
        return copy;
    }
}
