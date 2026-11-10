#include "models.h"

#include "../llama-kv-cache-dsa.h"

namespace {

// the router also picks n_expert_zero identity ("zero") experts: their ids become -1 (skipped by ggml_mul_mat_id)
// and their weights are summed into identity_weight, which scales the input instead
struct longcat_moe_route {
    ggml_tensor * ids;             // I32 [n_expert_used, n_tokens], -1 for zero experts
    ggml_tensor * weights;         // F32 [1, n_expert_used, n_tokens], 0 for zero experts
    ggml_tensor * identity_weight; // F32 [1, n_tokens], summed weight of the zero experts
};

longcat_moe_route build_longcat_moe_route(
        ggml_context * ctx,
        ggml_tensor  * logits,
        ggml_tensor  * exp_probs_b,
        int64_t        n_tokens,
        int64_t        n_expert,
        int64_t        n_expert_used,
        float          expert_weights_scale) {
    const int64_t n_expert_total = logits->ne[0];

    ggml_tensor * probs = ggml_soft_max(ctx, logits);

    // the correction bias only affects the selection, the weights come from the unbiased probabilities
    ggml_tensor * selection_probs = exp_probs_b ? ggml_add(ctx, probs, exp_probs_b) : probs;

    ggml_tensor * selected = ggml_argsort_top_k(ctx, selection_probs, n_expert_used);

    ggml_tensor * weights = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, n_expert_total, n_tokens), selected);
    weights = ggml_scale(ctx, weights, expert_weights_scale);

    // is_zero = 1 for ids >= n_expert, 0 otherwise
    ggml_tensor * selected_f = ggml_cast(ctx, selected, GGML_TYPE_F32);
    ggml_tensor * is_zero    = ggml_step(ctx, ggml_scale_bias(ctx, selected_f, 1.0f, 0.5f - (float) n_expert));
    ggml_tensor * is_real    = ggml_scale_bias(ctx, is_zero, -1.0f, 1.0f);

    longcat_moe_route res;

    res.identity_weight = ggml_sum_rows(ctx, ggml_reshape_2d(ctx,
            ggml_mul(ctx, weights, ggml_reshape_3d(ctx, is_zero, 1, n_expert_used, n_tokens)),
            n_expert_used, n_tokens));

    res.weights = ggml_mul(ctx, weights, ggml_reshape_3d(ctx, is_real, 1, n_expert_used, n_tokens));

    // id for the real experts, -1 for the zero experts
    res.ids = ggml_cast(ctx, ggml_sub(ctx, ggml_mul(ctx, selected_f, is_real), is_zero), GGML_TYPE_I32);

    return res;
}

} // namespace

void llama_model_longcat_flash_ngram::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,       hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,      hparams.n_lora_kv);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,    hparams.n_embd_head_k_mla_impl, false);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA,  hparams.n_embd_head_v_mla_impl, false);
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale);
    ml.get_key(LLM_KV_EXPERT_ZERO_COUNT,           hparams.n_expert_zero, false);
    ml.get_key(LLM_KV_NGRAM_NEIGHBOR_NUM,          hparams.ngram_neighbor_num);
    ml.get_key(LLM_KV_NGRAM_SPLIT_NUM,             hparams.ngram_split_num);
    ml.get_key(LLM_KV_NGRAM_VOCAB_SIZE_RATIO,      hparams.ngram_vocab_size_ratio);

    if (hparams.ngram_neighbor_num < 2 || hparams.ngram_split_num == 0) {
        throw std::runtime_error("invalid n-gram embedding parameters");
    }

    if (ml.get_key(LLM_KV_ROPE_SCALING_YARN_LOG_MUL, hparams.rope_yarn_log_mul, false)) {
        // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
        hparams.rope_yarn_log_mul /= 0.1f;
    }

    switch (hparams.n_layer()) {
        case 28: type = LLM_TYPE_68B_A3B; break; // 14 layers, each split into two attention blocks
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_longcat_flash_ngram::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v_mla = hparams.n_embd_head_v_mla();

    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;

    const int64_t q_lora_rank  = hparams.n_lora_q;
    const int64_t kv_lora_rank = hparams.n_lora_kv;
    const int64_t n_ff_exp     = hparams.n_ff_exp();

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (!output) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    // flags of the trunk and of the NextN/MTP block, which is stored after the trunk
    const auto nf = nextn_flags(ml);

    // n-gram embedding j has (ngram_vocab_size_ratio * n_vocab + 2*j + 1) rows
    {
        const uint32_t n_ngram = (hparams.ngram_neighbor_num - 1) * hparams.ngram_split_num;

        const int64_t n_embd_ngram = n_embd / n_ngram;
        const int64_t ngram_m      = (int64_t) hparams.ngram_vocab_size_ratio * n_vocab;

        ngram_embd.resize(n_ngram);
        ngram_proj.resize(n_ngram);

        for (uint32_t j = 0; j < n_ngram; ++j) {
            // token-indexed like tok_embd, so the tables stay with the input instead of on the device of layer j
            ngram_embd[j] = create_tensor_input(tn(LLM_TENSOR_NGRAM_EMBD, "weight", j), {n_embd_ngram, ngram_m + 2*j + 1}, nf.trunk);
            ngram_proj[j] = create_tensor(tn(LLM_TENSOR_NGRAM_PROJ, "weight", j), {n_embd_ngram, n_embd}, nf.trunk);
        }
    }

    for (int i = 0; i < n_layer_all; ++i) {
        auto & layer = layers[i];

        const bool is_nextn = i >= n_layer;
        const int  flags    = is_nextn ? nf.mtp : nf.trunk;

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, flags);

        if (q_lora_rank > 0) {
            layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, flags);
            layer.wq_a          = create_tensor(tn(LLM_TENSOR_ATTN_Q_A,      "weight", i), {n_embd, q_lora_rank}, flags);
            layer.wq_b          = create_tensor(tn(LLM_TENSOR_ATTN_Q_B,      "weight", i), {q_lora_rank, n_head * n_embd_head_k_mla}, flags);
        } else {
            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_head * n_embd_head_k_mla}, flags);
        }

        layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {kv_lora_rank}, flags);
        layer.wkv_a_mqa      = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA,  "weight", i), {n_embd, kv_lora_rank + n_embd_head_qk_rope}, flags);
        layer.wk_b           = create_tensor(tn(LLM_TENSOR_ATTN_K_B,       "weight", i), {n_embd_head_qk_nope, kv_lora_rank, n_head}, flags);
        layer.wv_b           = create_tensor(tn(LLM_TENSOR_ATTN_V_B,       "weight", i), {kv_lora_rank, n_embd_head_v_mla, n_head}, flags);
        layer.wo             = create_tensor(tn(LLM_TENSOR_ATTN_OUT,       "weight", i), {n_head * n_embd_head_v_mla, n_embd}, flags);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, flags);

        // even blocks hold the MoE (output added on the odd block) and a dense MLP stored as the shared expert
        if (!is_nextn && i % 2 == 0) {
            layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert + (int64_t) hparams.n_expert_zero}, flags);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert + (int64_t) hparams.n_expert_zero}, TENSOR_NOT_REQUIRED | flags);

            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff_exp, n_expert}, flags);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd, n_expert}, flags);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd, n_ff_exp, n_expert}, flags);

            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff}, flags);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff, n_embd}, flags);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff}, flags);
        } else {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, flags);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, flags);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, flags);
        }

        if (is_nextn) {
            layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,          "weight", i), {2 * n_embd, n_embd}, flags);
            layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,            "weight", i), {n_embd}, flags);
            layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,            "weight", i), {n_embd}, flags);
            layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", i), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED | flags);
            layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", i), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED | flags);
            layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i), {n_embd}, flags);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_longcat_flash_ngram::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

llama_model_longcat_flash_ngram::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    GGML_ASSERT(hparams.is_mla());

    const int64_t n_embd_head_k       = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope;

    const int64_t kv_lora_rank = hparams.n_lora_kv;

    // LongCat-Flash-Sparse: lightning indexer (LSA) over the MLA cache
    const bool     use_lsa                 = model.arch == LLM_ARCH_LONGCAT_FLASH_SPARSE;
    const int64_t  n_indexer_head          = hparams.indexer_n_head;
    const int64_t  n_embd_indexer_head     = hparams.indexer_head_size;
    const uint32_t n_indexer_top_k         = hparams.indexer_top_k;

    // the MLA LoRA outputs are scaled by sqrt(n_embd / lora_rank)
    const float mla_scale_q  = sqrtf((float) n_embd / (float) hparams.n_lora_q);
    const float mla_scale_kv = sqrtf((float) n_embd / (float) kv_lora_rank);

    // YaRN mscale, as in DeepSeek2
    GGML_ASSERT(ext_factor >= 0.0f);
    const float attn_factor_org = attn_factor * (1.0f + 0.1f * logf(1.0f / freq_scale));
    const float mscale   = attn_factor_org * (1.0f + 0.1f * hparams.rope_yarn_log_mul * logf(1.0f / freq_scale));
    const float kq_scale = 1.0f * mscale * mscale / sqrtf(float(n_embd_head_k));

    ggml_tensor * cur;
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);

    // n-gram embeddings: hashes of the token and the ones before it index extra tables, projected and averaged with tok_embd
    if (ubatch.token) {
        const uint32_t n_ngram    = (hparams.ngram_neighbor_num - 1) * hparams.ngram_split_num;
        const int64_t  vocab_size = model.tok_embd->ne[1];

        auto inp = std::make_unique<llm_graph_input_ngram>(
                (int32_t) hparams.ngram_neighbor_num, (int32_t) hparams.ngram_split_num,
                (int32_t) vocab_size, (int64_t) hparams.ngram_vocab_size_ratio * vocab_size,
                model.vocab.token_eos(), use_lsa, mctx);

        inp->ngram_ids.resize(n_ngram);

        for (uint32_t j = 0; j < n_ngram; ++j) {
            inp->ngram_ids[j] = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
            ggml_set_input(inp->ngram_ids[j]);

            ggml_tensor * emb = ggml_get_rows(ctx0, model.ngram_embd[j], inp->ngram_ids[j]);
            cb(emb, "ngram_emb", j);

            emb = ggml_mul_mat(ctx0, model.ngram_proj[j], emb);
            cb(emb, "ngram_proj", j);

            inpL = ggml_add(ctx0, inpL, emb);
        }

        inpL = ggml_scale(ctx0, inpL, 1.0f / (1.0f + (float) n_ngram));
        cb(inpL, "inp_embd_ngram", -1);

        res->add_input(std::move(inp));
    }

    ggml_tensor * inp_pos = build_inp_pos();

    llm_graph_input_attn_k     * inp_attn_k   = nullptr;
    llm_graph_input_attn_k_dsa * inp_attn_dsa = nullptr;
    if (use_lsa) {
        inp_attn_dsa = build_attn_inp_k_dsa();
    } else {
        inp_attn_k = build_attn_inp_k();
    }

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // while n_kv <= indexer top-k every key is selected, so the indexer scores are skipped
    const bool lsa_sparse = use_lsa && inp_attn_dsa->mctx->get_lid()->get_n_kv() > n_indexer_top_k;

    // MoE output of the even block, added to the output of the following odd block
    ggml_tensor * moe_shortcut = nullptr;

    // top-k of the last indexer owner, reused by the following layers without an indexer
    ggml_tensor * top_k = nullptr;

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention (MLA, absorbed)
        {
            ggml_tensor * q      = nullptr;
            ggml_tensor * q_lora = nullptr;

            if (model.layers[il].wq_a) {
                q = ggml_mul_mat(ctx0, model.layers[il].wq_a, cur);
                cb(q, "q_a", il);

                q = build_norm(q, model.layers[il].attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
                cb(q, "q_a_norm", il);

                q_lora = q;

                q = ggml_mul_mat(ctx0, model.layers[il].wq_b, q);
                q = ggml_scale(ctx0, q, mla_scale_q);
            } else {
                q = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
            }
            cb(q, "q", il);

            if (use_lsa && hparams.is_indexer_full(il)) {
                const auto * mctx_lid = inp_attn_dsa->mctx->get_lid();

                // indexer key: [n_embd_indexer_head, 1, n_tokens], stored in the indexer cache
                ggml_tensor * indexer_k = ggml_mul_mat(ctx0, model.layers[il].indexer_attn_k, cur);
                indexer_k = ggml_rms_norm(ctx0, indexer_k, hparams.indexer_k_norm_eps);
                indexer_k = ggml_mul(ctx0, indexer_k, model.layers[il].indexer_k_norm);
                cb(indexer_k, "indexer_k_norm", il);

                // the head is [rope | nope], the RoPE rotates the first n_rot dims and copies the rest
                indexer_k = ggml_reshape_3d(ctx0, indexer_k, n_embd_indexer_head, 1, n_tokens);
                indexer_k = build_rope_indexer(indexer_k, inp_pos, il);
                cb(indexer_k, "indexer_k", il);

                ggml_build_forward_expand(gf, mctx_lid->cpy_k(ctx0, indexer_k, inp_attn_dsa->get_k_idxs_lid(), il));

                top_k = nullptr;

                if (lsa_sparse) {
                    // indexer query: [n_embd_indexer_head, n_indexer_head, n_tokens]
                    ggml_tensor * indexer_q = ggml_mul_mat(ctx0, model.layers[il].indexer_attn_q_b, q_lora);
                    cb(indexer_q, "indexer_q_b", il);

                    indexer_q = ggml_reshape_3d(ctx0, indexer_q, n_embd_indexer_head, n_indexer_head, n_tokens);
                    indexer_q = build_rope_indexer(indexer_q, inp_pos, il);
                    cb(indexer_q, "indexer_q", il);

                    // per-head weights: [n_indexer_head, n_tokens]
                    ggml_tensor * indexer_weights = ggml_mul_mat(ctx0, model.layers[il].indexer_proj, cur);
                    indexer_weights = ggml_scale(ctx0, indexer_weights, 1.0f / sqrtf(float(n_embd_indexer_head * n_indexer_head)));
                    cb(indexer_weights, "indexer_weights", il);

                    ggml_tensor * indexer_k_cache = mctx_lid->get_k(ctx0, il);

                    const auto n_stream = indexer_k_cache->ne[3];

                    indexer_q = ggml_view_4d(ctx0, indexer_q,
                            indexer_q->ne[0], indexer_q->ne[1], indexer_q->ne[2] / n_stream, n_stream,
                            indexer_q->nb[1], indexer_q->nb[2], indexer_q->nb[3] / n_stream, 0);
                    indexer_weights = ggml_view_4d(ctx0, indexer_weights,
                            indexer_weights->ne[0], indexer_weights->ne[1] / n_stream, indexer_weights->ne[2], n_stream,
                            indexer_weights->nb[1], indexer_weights->nb[2] / n_stream, indexer_weights->nb[3] / n_stream, 0);

                    indexer_q       = ggml_permute(ctx0, indexer_q,       0, 2, 1, 3);
                    indexer_k_cache = ggml_permute(ctx0, indexer_k_cache, 0, 2, 1, 3);

                    // score = sum_h relu(q_h . k) * w_h
                    ggml_tensor * indexer_kq = ggml_mul_mat(ctx0, indexer_k_cache, indexer_q);
                    ggml_mul_mat_set_prec(indexer_kq, GGML_PREC_F32);
                    cb(indexer_kq, "indexer_kq", il);

                    indexer_kq = ggml_cont(ctx0, ggml_permute(ctx0, indexer_kq, 2, 1, 0, 3));

                    ggml_tensor * indexer_score = ggml_relu(ctx0, indexer_kq);
                    indexer_score = ggml_mul(ctx0, indexer_score, indexer_weights);
                    indexer_score = ggml_sum_rows(ctx0, indexer_score);
                    indexer_score = ggml_cont(ctx0, ggml_permute(ctx0, indexer_score, 2, 1, 0, 3));

                    // the LSA mask holds -inf for the invisible keys and +inf for the forced sink and local keys
                    // the mask is F16 when it has no forced keys and the indexer mask is built for the fused path
                    ggml_tensor * lid_mask = inp_attn_dsa->self_attn_mask_lid.raw;
                    if (lid_mask->type != GGML_TYPE_F32) {
                        lid_mask = ggml_cast(ctx0, lid_mask, GGML_TYPE_F32);
                    }
                    indexer_score = ggml_add(ctx0, indexer_score, lid_mask);
                    cb(indexer_score, "indexer_score", il);

                    top_k = ggml_cont(ctx0, ggml_top_k(ctx0, indexer_score, n_indexer_top_k));
                    cb(top_k, "top_k", il);
                }
            }

            ggml_tensor * q_nope = ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens,
                    ggml_row_size(q->type, n_embd_head_k),
                    ggml_row_size(q->type, n_embd_head_k) * n_head, 0);
            cb(q_nope, "q_nope", il);

            ggml_tensor * q_pe = ggml_view_3d(ctx0, q, n_embd_head_qk_rope, n_head, n_tokens,
                    ggml_row_size(q->type, n_embd_head_k),
                    ggml_row_size(q->type, n_embd_head_k) * n_head,
                    ggml_row_size(q->type, n_embd_head_qk_nope));
            cb(q_pe, "q_pe", il);

            ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, model.layers[il].wkv_a_mqa, cur);
            cb(kv_cmpr_pe, "kv_cmpr_pe", il);

            ggml_tensor * kv_cmpr = ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
                    ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);
            cb(kv_cmpr, "kv_cmpr", il);

            ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens,
                    ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                    ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                    ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));
            cb(k_pe, "k_pe", il);

            q_pe = build_rope(q_pe, inp_pos, il);
            cb(q_pe, "q_pe", il);

            k_pe = build_rope(k_pe, inp_pos, il);
            cb(k_pe, "k_pe", il);

            kv_cmpr = build_norm(kv_cmpr, model.layers[il].attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
            kv_cmpr = ggml_scale(ctx0, kv_cmpr, mla_scale_kv);
            cb(kv_cmpr, "kv_cmpr", il);

            // absorb wk_b into q_nope
            q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
            q_nope = ggml_mul_mat(ctx0, model.layers[il].wk_b, q_nope);
            q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
            cb(q_nope, "q_nope_absorbed", il);

            ggml_tensor * Qcur = ggml_concat(ctx0, q_nope, q_pe, 0);
            cb(Qcur, "Qcur", il);

            kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);

            ggml_tensor * Kcur = ggml_concat(ctx0, kv_cmpr, k_pe, 0);
            cb(Kcur, "Kcur", il);

            ggml_tensor * Vcur = kv_cmpr;
            cb(Vcur, "Vcur", il);

            if (use_lsa) {
                cur = build_attn(inp_attn_dsa,
                        model.layers[il].wo, NULL, model.layers[il].wo_s,
                        Qcur, Kcur, Vcur, nullptr, nullptr, model.layers[il].wv_b, top_k, kq_scale, il);
            } else {
                cur = build_attn(inp_attn_k,
                        model.layers[il].wo, NULL, model.layers[il].wo_s,
                        Qcur, Kcur, Vcur, nullptr, nullptr, model.layers[il].wv_b, kq_scale, il);
            }
        }

        if (il == n_layer - 1 && crop_before_nextn(inp_out_ids)) {
            cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            if (moe_shortcut) {
                moe_shortcut = ggml_get_rows(ctx0, moe_shortcut, inp_out_ids);
            }
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if (il % 2 == 0) {
            // MoE with zero experts: computed here, added to the output of the next block
            ggml_tensor * logits = ggml_mul_mat(ctx0, model.layers[il].ffn_gate_inp, cur);
            cb(logits, "ffn_moe_logits", il);

            const longcat_moe_route route = build_longcat_moe_route(ctx0, logits, model.layers[il].ffn_exp_probs_b,
                    n_tokens, n_expert, n_expert_used, hparams.expert_weights_scale);
            cb(route.ids,             "ffn_moe_topk",            il);
            cb(route.weights,         "ffn_moe_weights",         il);
            cb(route.identity_weight, "ffn_moe_identity_weight", il);

            ggml_tensor * cur_moe = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);

            ggml_tensor * up = build_lora_mm_id(model.layers[il].ffn_up_exps, cur_moe, route.ids);
            cb(up, "ffn_moe_up", il);

            ggml_tensor * gate = build_lora_mm_id(model.layers[il].ffn_gate_exps, cur_moe, route.ids);
            cb(gate, "ffn_moe_gate", il);

            ggml_tensor * experts = ggml_swiglu_split(ctx0, gate, up);
            cb(experts, "ffn_moe_swiglu", il);

            experts = build_lora_mm_id(model.layers[il].ffn_down_exps, experts, route.ids);
            cb(experts, "ffn_moe_down", il);

            experts = ggml_mul(ctx0, experts, route.weights);
            cb(experts, "ffn_moe_weighted", il);

            ggml_tensor * moe_out = nullptr;
            for (int64_t i = 0; i < n_expert_used; ++i) {
                ggml_tensor * cur_expert = ggml_view_2d(ctx0, experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1]);
                moe_out = moe_out ? ggml_add(ctx0, moe_out, cur_expert) : cur_expert;
            }
            cb(moe_out, "ffn_moe_out", il);

            moe_shortcut = ggml_add(ctx0, moe_out, ggml_mul(ctx0, cur, route.identity_weight));
            cb(moe_shortcut, "ffn_moe_shortcut", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up_shexp,   NULL, NULL,
                    model.layers[il].ffn_gate_shexp, NULL, NULL,
                    model.layers[il].ffn_down_shexp, NULL, NULL,
                    NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);

            if (moe_shortcut) {
                cur = ggml_add(ctx0, cur, moe_shortcut);
                cb(cur, "ffn_out_moe", il);
                moe_shortcut = nullptr;
            }
        }

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, -1);

    // input of the NextN/MTP head
    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (crop_after_nextn(inp_out_ids)) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;
}

// NextN/MTP draft head: one dense block (MLA attention without the indexer, dense MLP)
llama_model_longcat_flash_ngram::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    GGML_ASSERT(hparams.n_layer_nextn > 0 && "LongCat MTP requires n_layer_nextn > 0");
    GGML_ASSERT(cparams.nextn_layer_offset >= 0 && cparams.nextn_layer_offset < (int) hparams.n_layer_nextn &&
            "nextn_layer_offset out of range [0, n_layer_nextn)");

    const int il = hparams.n_layer() + cparams.nextn_layer_offset;
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && layer.nextn.enorm && layer.nextn.hnorm && "MTP block missing NextN tensors");

    const int64_t n_embd_head_k       = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope;

    const int64_t kv_lora_rank = hparams.n_lora_kv;

    const float mla_scale_q  = hparams.n_lora_q > 0 ? sqrtf((float) n_embd / (float) hparams.n_lora_q) : 1.0f;
    const float mla_scale_kv = sqrtf((float) n_embd / (float) kv_lora_rank);

    GGML_ASSERT(ext_factor >= 0.0f);
    const float attn_factor_org = attn_factor * (1.0f + 0.1f * logf(1.0f / freq_scale));
    const float mscale   = attn_factor_org * (1.0f + 0.1f * hparams.rope_yarn_log_mul * logf(1.0f / freq_scale));
    const float kq_scale = 1.0f * mscale * mscale / sqrtf(float(n_embd_head_k));

    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd_inp(), hparams.n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp->embd);

    ASSERT_EMBD_OR_TOKEN(ubatch);
    ggml_tensor * tok_embd = ubatch.token
        ? ggml_get_rows(ctx0, layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd, inp->tokens)
        : inp->embd;
    cb(tok_embd, "mtp_tok_embd", il);

    inp->h = build_inp_embd_state(inp.get(), hparams.n_embd);

    ggml_tensor * h_embd = inp->h;
    res->add_input(std::move(inp));

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    auto * inp_attn = build_attn_inp_k();

    ggml_tensor * h_norm = build_norm(h_embd, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * cur = build_lora_mm(layer.nextn.eh_proj, ggml_concat(ctx0, e_norm, h_norm, 0), layer.nextn.eh_proj_s);
    cb(cur, "mtp_eh_proj", il);

    ggml_tensor * inpSA = cur;

    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "attn_norm", il);

    ggml_tensor * q = nullptr;
    if (layer.wq_a) {
        q = ggml_mul_mat(ctx0, layer.wq_a, cur);
        q = build_norm(q, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
        q = ggml_mul_mat(ctx0, layer.wq_b, q);
        q = ggml_scale(ctx0, q, mla_scale_q);
    } else {
        q = ggml_mul_mat(ctx0, layer.wq, cur);
    }
    cb(q, "q", il);

    ggml_tensor * q_nope = ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens,
            ggml_row_size(q->type, n_embd_head_k),
            ggml_row_size(q->type, n_embd_head_k) * n_head, 0);

    ggml_tensor * q_pe = ggml_view_3d(ctx0, q, n_embd_head_qk_rope, n_head, n_tokens,
            ggml_row_size(q->type, n_embd_head_k),
            ggml_row_size(q->type, n_embd_head_k) * n_head,
            ggml_row_size(q->type, n_embd_head_qk_nope));

    ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);

    ggml_tensor * kv_cmpr = ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
            ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);

    ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens,
            ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
            ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
            ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));

    q_pe = build_rope(q_pe, inp_pos, il);
    k_pe = build_rope(k_pe, inp_pos, il);

    kv_cmpr = build_norm(kv_cmpr, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
    kv_cmpr = ggml_scale(ctx0, kv_cmpr, mla_scale_kv);

    q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
    q_nope = ggml_mul_mat(ctx0, layer.wk_b, q_nope);
    q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);

    ggml_tensor * Qcur = ggml_concat(ctx0, q_nope, q_pe, 0);

    kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);

    ggml_tensor * Kcur = ggml_concat(ctx0, kv_cmpr, k_pe, 0);
    ggml_tensor * Vcur = kv_cmpr;

    cur = build_attn(inp_attn, layer.wo, nullptr, layer.wo_s,
            Qcur, Kcur, Vcur, nullptr, nullptr, layer.wv_b, kq_scale, il);
    cb(cur, "attn_out", il);

    ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
    cb(ffn_inp, "ffn_inp", il);

    cur = build_norm(ffn_inp, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "ffn_norm", il);

    cur = build_ffn(cur,
            layer.ffn_up,   nullptr, layer.ffn_up_s,
            layer.ffn_gate, nullptr, layer.ffn_gate_s,
            layer.ffn_down, nullptr, layer.ffn_down_s,
            nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(cur, "ffn_out", il);

    cur = ggml_add(ctx0, cur, ffn_inp);

    cur = build_norm(cur, layer.nextn.shared_head_norm ? layer.nextn.shared_head_norm : model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cur = layer.nextn.shared_head_head
        ? build_lora_mm(layer.nextn.shared_head_head, cur, layer.nextn.shared_head_head_s)
        : build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;
}
