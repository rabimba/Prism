#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#include "ggml-backend.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

static void prism_silent_log(enum ggml_log_level level, const char * text, void * user_data) {
    (void)level;
    (void)text;
    (void)user_data;
}

void prism_llama_init_backend(void) {
    llama_log_set(prism_silent_log, NULL);
    mtmd_helper_log_set(prism_silent_log, NULL);
    if (ggml_backend_dev_count() == 0) {
        ggml_backend_load_all();
    }
#ifdef PRISM_LLAMA_LIB_DIR
    if (ggml_backend_dev_count() == 0) {
        ggml_backend_load_all_from_path(PRISM_LLAMA_LIB_DIR);
    }
#endif
    llama_backend_init();
}

struct llama_model * prism_llama_load_model(const char * path) {
    struct llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = 999;
    return llama_model_load_from_file(path, params);
}

void prism_llama_free_model(struct llama_model * model) {
    if (model) {
        llama_model_free(model);
    }
}

int32_t prism_llama_model_n_embd(const struct llama_model * model) {
    if (!model) return 0;
    return llama_model_n_embd_out(model);
}

struct mtmd_context * prism_mtmd_init(
    const char * mmproj_path,
    const struct llama_model * model,
    int32_t n_threads
) {
    struct mtmd_context_params params = mtmd_context_params_default();
    params.use_gpu = true;
    params.print_timings = false;
    params.n_threads = n_threads;
    params.warmup = true;
    params.image_min_tokens = 1;
    params.image_max_tokens = 1120;
    return mtmd_init_from_file(mmproj_path, model, params);
}

void prism_mtmd_free(struct mtmd_context * mtmd) {
    if (mtmd) {
        mtmd_free(mtmd);
    }
}

struct llama_context * prism_llama_new_context(
    struct llama_model * model,
    uint32_t max_tokens,
    int32_t n_threads
) {
    struct llama_context_params params = llama_context_default_params();
    params.n_ctx = max_tokens;
    params.n_batch = max_tokens;
    params.n_ubatch = max_tokens;
    params.n_seq_max = 16;
    params.n_threads = n_threads;
    params.n_threads_batch = n_threads;
    params.embeddings = true;
    params.pooling_type = LLAMA_POOLING_TYPE_MEAN;
    params.kv_unified = true;
    return llama_init_from_model(model, params);
}

void prism_llama_free_context(struct llama_context * ctx) {
    if (ctx) {
        llama_free(ctx);
    }
}

static void l2_normalize(const float * src, float * dst, int32_t n_embd) {
    double sum_sq = 0.0;
    for (int32_t i = 0; i < n_embd; ++i) {
        double v = (double)src[i];
        sum_sq += v * v;
    }
    if (sum_sq > 0.0) {
        float inv = (float)(1.0 / sqrt(sum_sq));
        for (int32_t i = 0; i < n_embd; ++i) {
            dst[i] = src[i] * inv;
        }
    } else {
        memset(dst, 0, (size_t)n_embd * sizeof(float));
    }
}

static int32_t place_batch_ext_item(struct llama_batch_ext * batch, int32_t idx, llama_pos position) {
    if (idx < 0) return -1;
    llama_pos pos[4] = { position, 0, 0, 0 };
    if (!llama_batch_ext_set_pos(batch, idx, pos)) return -1;
    if (!llama_batch_ext_set_output_embd(batch, idx, true)) return -1;
    return 0;
}

typedef struct {
    size_t orig_index;
    llama_token * tokens;
    int32_t len;
} tokenized_seq_t;

int32_t prism_llama_embed_texts(
    struct llama_context * ctx,
    const struct llama_model * model,
    const char * const * texts,
    const size_t * text_lens,
    size_t n_texts,
    uint32_t max_tokens,
    float * out_embeddings
) {
    if (!ctx || !model) return -1;
    int32_t n_embd = llama_model_n_embd_out(model);
    if (n_embd <= 0) return -1;
    if (n_texts == 0) return 0;

    memset(out_embeddings, 0, n_texts * (size_t)n_embd * sizeof(float));

    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    llama_token eos = llama_vocab_eos(vocab);

    tokenized_seq_t * seqs = (tokenized_seq_t *)calloc(n_texts, sizeof(tokenized_seq_t));
    if (!seqs) return -1;

    size_t n_valid = 0;
    for (size_t i = 0; i < n_texts; ++i) {
        int32_t cap = (int32_t)text_lens[i] + 8;
        if (cap < (int32_t)max_tokens) cap = (int32_t)max_tokens;
        llama_token * buf = (llama_token *)malloc((size_t)cap * sizeof(llama_token));
        if (!buf) continue;

        int32_t count = llama_tokenize(
            vocab,
            texts[i],
            (int32_t)text_lens[i],
            buf,
            cap,
            true,
            true
        );

        if (count < 0) {
            int32_t needed = -count;
            llama_token * full = (llama_token *)realloc(buf, (size_t)needed * sizeof(llama_token));
            if (!full) {
                free(buf);
                continue;
            }
            buf = full;
            count = llama_tokenize(
                vocab,
                texts[i],
                (int32_t)text_lens[i],
                buf,
                needed,
                true,
                true
            );
        }

        if (count > (int32_t)max_tokens) {
            buf[max_tokens - 1] = eos;
            count = (int32_t)max_tokens;
        }

        if (count > 0) {
            seqs[n_valid].orig_index = i;
            seqs[n_valid].tokens = buf;
            seqs[n_valid].len = count;
            n_valid++;
        } else {
            free(buf);
        }
    }

    struct llama_batch_ext * batch = llama_batch_ext_init(ctx);
    if (!batch) {
        for (size_t i = 0; i < n_valid; ++i) free(seqs[i].tokens);
        free(seqs);
        return -1;
    }

    int32_t status = 0;
    size_t cursor = 0;
    const size_t max_batch_seqs = 16;

    while (cursor < n_valid) {
        size_t run_end = cursor;
        int32_t run_tokens = 0;
        while (run_end < n_valid && (run_end - cursor) < max_batch_seqs) {
            int32_t next_len = seqs[run_end].len;
            if (run_tokens > 0 && (uint32_t)(run_tokens + next_len) > max_tokens) {
                break;
            }
            run_tokens += next_len;
            run_end++;
        }

        llama_batch_ext_clear(batch);
        size_t run_count = run_end - cursor;

        for (size_t s = 0; s < run_count; ++s) {
            tokenized_seq_t * seq = &seqs[cursor + s];
            for (int32_t pos = 0; pos < seq->len; ++pos) {
                int32_t idx = llama_batch_ext_add_token(batch, (llama_seq_id)s, seq->tokens[pos]);
                if (place_batch_ext_item(batch, idx, (llama_pos)pos) != 0) {
                    status = -2;
                    break;
                }
            }
            if (status != 0) break;
        }
        if (status != 0) break;

        int32_t rc = llama_process(ctx, LLAMA_PROCESS_TYPE_ENCODE, batch);
        if (rc != 0) {
            status = -3;
            break;
        }

        for (size_t s = 0; s < run_count; ++s) {
            const float * raw = llama_get_embeddings_seq(ctx, (llama_seq_id)s);
            if (!raw) {
                status = -4;
                break;
            }
            size_t out_idx = seqs[cursor + s].orig_index;
            l2_normalize(raw, out_embeddings + out_idx * (size_t)n_embd, n_embd);
        }
        if (status != 0) break;

        cursor = run_end;
    }

    llama_batch_ext_free(batch);
    for (size_t i = 0; i < n_valid; ++i) {
        free(seqs[i].tokens);
    }
    free(seqs);
    return status;
}

struct mtmd_bitmap * prism_mtmd_bitmap_from_rgb(
    uint32_t nx,
    uint32_t ny,
    const unsigned char * rgb_data
) {
    return mtmd_bitmap_init(nx, ny, rgb_data);
}

struct mtmd_bitmap * prism_mtmd_bitmap_from_audio(
    size_t n_samples,
    const float * pcm_f32
) {
    return mtmd_bitmap_init_from_audio(n_samples, pcm_f32);
}

struct mtmd_bitmap * prism_mtmd_bitmap_from_file(
    struct mtmd_context * mtmd,
    const char * path
) {
    struct mtmd_helper_init_opt opt = mtmd_helper_init_opt_default();
    struct mtmd_helper_bitmap_wrapper w = mtmd_helper_bitmap_init_from_file(mtmd, path, false, opt);
    if (w.video_ctx) {
        if (w.bitmap) mtmd_bitmap_free(w.bitmap);
        mtmd_helper_video_free(w.video_ctx);
        return NULL;
    }
    if (!w.bitmap) return NULL;
    if (mtmd_bitmap_is_audio(w.bitmap)) {
        size_t samples = mtmd_bitmap_get_n_bytes(w.bitmap) / sizeof(float);
        size_t limit = 30 * (size_t)mtmd_get_audio_sample_rate(mtmd);
        if (samples > limit) {
            const float * pcm = (const float *)mtmd_bitmap_get_data(w.bitmap);
            struct mtmd_bitmap * prefix = mtmd_bitmap_init_from_audio(limit, pcm);
            mtmd_bitmap_free(w.bitmap);
            return prefix;
        }
    }
    return w.bitmap;
}

void prism_mtmd_bitmap_free(struct mtmd_bitmap * bmp) {
    if (bmp) {
        mtmd_bitmap_free(bmp);
    }
}

int32_t prism_mtmd_embed_bitmaps(
    struct llama_context * ctx,
    struct mtmd_context * mtmd,
    const struct llama_model * model,
    struct mtmd_bitmap * const * bitmaps,
    size_t n_bitmaps,
    uint32_t max_tokens,
    float * out_embeddings
) {
    (void)max_tokens;
    if (!ctx || !mtmd || !model) return -1;
    int32_t n_embd = llama_model_n_embd_out(model);
    int32_t row_size = llama_model_n_embd_inp(model);
    if (n_embd <= 0 || row_size <= 0) return -1;
    if (n_bitmaps == 0) return 0;

    memset(out_embeddings, 0, n_bitmaps * (size_t)n_embd * sizeof(float));

    struct llama_batch_ext * batch = llama_batch_ext_init(ctx);
    if (!batch) return -1;

    const char * marker = mtmd_default_marker();
    size_t marker_len = strlen(marker);
    int32_t status = 0;

    for (size_t i = 0; i < n_bitmaps; ++i) {
        struct mtmd_input_chunks * chunks = mtmd_input_chunks_init();
        if (!chunks) {
            status = -1;
            break;
        }

        struct mtmd_input_text text;
        memset(&text, 0, sizeof(text));
        text.text = marker;
        text.text_len = marker_len;
        text.add_special = true;
        text.parse_special = true;

        const struct mtmd_bitmap * bmp_ptr = bitmaps[i];
        int32_t rc = mtmd_tokenize(mtmd, chunks, &text, &bmp_ptr, 1);
        if (rc != 0) {
            mtmd_input_chunks_free(chunks);
            status = -2;
            break;
        }

        llama_batch_ext_clear(batch);
        llama_pos position = 0;
        size_t n_chunks = mtmd_input_chunks_size(chunks);

        for (size_t c = 0; c < n_chunks; ++c) {
            const struct mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, c);
            if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                size_t count = 0;
                const llama_token * tokens = mtmd_input_chunk_get_tokens_text(chunk, &count);
                if (!tokens) continue;
                for (size_t t = 0; t < count; ++t) {
                    int32_t idx = llama_batch_ext_add_token(batch, 0, tokens[t]);
                    if (place_batch_ext_item(batch, idx, position) != 0) {
                        status = -3;
                        break;
                    }
                    position++;
                }
            } else {
                if (mtmd_encode_chunk(mtmd, chunk) != 0) {
                    status = -4;
                    break;
                }
                float * rows = mtmd_get_output_embd(mtmd);
                if (!rows) {
                    status = -5;
                    break;
                }
                size_t n_rows = mtmd_input_chunk_get_n_tokens(chunk);
                for (size_t r = 0; r < n_rows; ++r) {
                    struct llama_embd embd;
                    embd.data = rows + r * (size_t)row_size;
                    embd.n_rows = 1;
                    embd.n_embd = (size_t)row_size;
                    int32_t idx = llama_batch_ext_add_embd(batch, 0, embd);
                    if (place_batch_ext_item(batch, idx, position) != 0) {
                        status = -6;
                        break;
                    }
                    position++;
                }
            }
            if (status != 0) break;
        }

        mtmd_input_chunks_free(chunks);
        if (status != 0) break;

        if (llama_process(ctx, LLAMA_PROCESS_TYPE_ENCODE, batch) != 0) {
            status = -7;
            break;
        }

        const float * raw = llama_get_embeddings_seq(ctx, 0);
        if (!raw) {
            status = -8;
            break;
        }
        l2_normalize(raw, out_embeddings + i * (size_t)n_embd, n_embd);
    }

    llama_batch_ext_free(batch);
    return status;
}
