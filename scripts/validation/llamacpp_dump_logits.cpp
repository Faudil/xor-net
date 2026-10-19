// Reference logit dumper for comparing xor-net against llama.cpp.
//
//   dump_logits tokenize <model.gguf> <text.txt> <out_tokens.u32>
//   dump_logits logits   <model.gguf> <tokens.u32> <out_logits.f32> [threads]
//
// `tokenize` writes llama.cpp's tokenization (no BOS/special handling, to
// match xor-net's add_special_tokens=false). `logits` evaluates the given
// token ids in one batch and writes logits for EVERY position, row-major
// [n_tokens, n_vocab] f32.
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

static std::vector<char> slurp(const char *p) {
    std::ifstream f(p, std::ios::binary);
    return std::vector<char>(std::istreambuf_iterator<char>(f), {});
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s tokenize|logits <model> <in> <out> [threads]\n", argv[0]);
        return 1;
    }
    std::string mode = argv[1];
    int threads = argc > 5 ? atoi(argv[5]) : 6;
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model *model = llama_model_load_from_file(argv[2], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab *vocab = llama_model_get_vocab(model);

    if (mode == "tokenize") {
        auto txt = slurp(argv[3]);
        std::vector<llama_token> toks(txt.size() + 16);
        int n = llama_tokenize(vocab, txt.data(), (int)txt.size(), toks.data(), (int)toks.size(),
                               /*add_special*/ false, /*parse_special*/ true);
        if (n < 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
        FILE *o = fopen(argv[4], "wb");
        fwrite(toks.data(), 4, n, o);
        fclose(o);
        fprintf(stderr, "tokens: %d\n", n);
        return 0;
    }

    auto raw = slurp(argv[3]);
    int n = (int)(raw.size() / 4);
    std::vector<llama_token> toks(n);
    memcpy(toks.data(), raw.data(), n * 4);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n + 16;
    cp.n_batch = n;
    cp.n_ubatch = n;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }
    llama_batch b = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; i++) {
        b.token[i] = toks[i];
        b.pos[i] = i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i] = 1;
    }
    b.n_tokens = n;
    if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
    int nv = llama_vocab_n_tokens(vocab);
    FILE *o = fopen(argv[4], "wb");
    for (int i = 0; i < n; i++) fwrite(llama_get_logits_ith(ctx, i), 4, nv, o);
    fclose(o);
    fprintf(stderr, "wrote %d x %d logits\n", n, nv);
    llama_batch_free(b);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
