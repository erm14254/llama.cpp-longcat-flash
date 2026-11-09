#include "models.h"

// LongCat-Flash-Sparse is LongCat-Flash-Ngram with a lightning indexer (LSA) on every other layer;
// the graph is shared, see longcat-flash-ngram.cpp

void llama_model_longcat_flash_sparse::load_arch_hparams(llama_model_loader & ml) {
    llama_model_longcat_flash_ngram::load_arch_hparams(ml);

    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,      hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,      hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,           hparams.indexer_top_k);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_INIT_TOKENS,     hparams.indexer_init_tokens);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_LOCAL_TOKENS,    hparams.indexer_local_tokens);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_K_NORM_EPS,      hparams.indexer_k_norm_eps,      false);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_ROPE_INTERLEAVE, hparams.indexer_rope_interleave, false);

    // layers that do not own an indexer reuse the top-k of the previous owner
    ml.get_key_or_arr(LLM_KV_ATTENTION_INDEXER_TYPES, hparams.is_indexer_full_impl, hparams.n_layer());

    std::string k_norm_type = "rms";
    ml.get_key(LLM_KV_ATTENTION_INDEXER_K_NORM_TYPE, k_norm_type, false);
    if (k_norm_type != "rms") {
        throw std::runtime_error("unsupported indexer key norm type: " + k_norm_type);
    }
    if (hparams.n_lora_q == 0) {
        throw std::runtime_error("the indexer requires the Q LoRA");
    }
    if (hparams.indexer_head_size <= hparams.n_rot()) {
        throw std::runtime_error("the indexer head size must be larger than the RoPE dimension");
    }
    if (!hparams.is_indexer_full(0)) {
        throw std::runtime_error("the first layer must own an indexer");
    }
    if (hparams.indexer_init_tokens + hparams.indexer_local_tokens > hparams.indexer_top_k) {
        throw std::runtime_error("the forced sink and local keys exceed the indexer top-k");
    }
}

void llama_model_longcat_flash_sparse::load_arch_tensors(llama_model_loader & ml) {
    llama_model_longcat_flash_ngram::load_arch_tensors(ml);

    LLAMA_LOAD_LOCALS;

    const int64_t q_lora_rank = hparams.n_lora_q;

    // the NextN/MTP block also owns an indexer, it is not loaded since the MTP head attends densely
    for (int i = 0; i < n_layer_all; ++i) {
        const bool is_nextn = i >= n_layer;

        if (!is_nextn && !hparams.is_indexer_full(i)) {
            continue;
        }

        const int flags = is_nextn ? TENSOR_SKIP : 0;

        auto & layer = layers[i];

        layer.indexer_k_norm   = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,   "weight", i), {hparams.indexer_head_size}, flags);
        layer.indexer_proj     = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ,     "weight", i), {n_embd, hparams.indexer_n_head}, flags);
        layer.indexer_attn_k   = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_K,   "weight", i), {n_embd, hparams.indexer_head_size}, flags);
        layer.indexer_attn_q_b = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", i), {q_lora_rank, hparams.indexer_n_head * hparams.indexer_head_size}, flags);
    }
}

llama_rope_params llama_model_longcat_flash_sparse::get_rope_params_indexer(const llama_cparams & cparams, int il) const {
    auto res = llama_model::get_rope_params_indexer(cparams, il);

    if (hparams.indexer_rope_interleave) {
        res.type = LLAMA_ROPE_TYPE_NORM;
    }

    return res;
}
