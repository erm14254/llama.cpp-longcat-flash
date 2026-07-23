#include "models.h"

#include "../llama-graph.h"
#include "../llama-model.h"

#include <cmath>

void llama_model_longcat_flash_ngram::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,       hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,      hparams.n_lora_kv);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,    hparams.n_embd_head_k_mla_impl, false);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA,  hparams.n_embd_head_v_mla_impl, false);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm, false);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead);
    ml.get_key(LLM_KV_EXPERT_ZERO_COUNT,           hparams.n_expert_zero, false);
    ml.get_key(LLM_KV_NGRAM_NEIGHBOR_NUM,          hparams.ngram_neighbor_num, false);
    ml.get_key(LLM_KV_NGRAM_SPLIT_NUM,             hparams.ngram_split_num, false);
    ml.get_key(LLM_KV_NGRAM_VOCAB_SIZE_RATIO,      hparams.ngram_vocab_size_ratio, false);

    // NextN/MTP: one auxiliary decoder block is appended after the 28 main blocks.
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);
    GGML_ASSERT(hparams.n_layer_nextn <= 1 && "LongCat MTP currently supports one auxiliary block");
    if (hparams.n_layer_nextn > 0) {
        GGML_ASSERT(hparams.n_layer_nextn < hparams.n_layer_all);
    }

    hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX;

    if (ml.get_key(LLM_KV_ROPE_SCALING_YARN_LOG_MUL, hparams.rope_yarn_log_mul, false)) {
        hparams.rope_yarn_log_mul /= 0.1f;
    }

    switch (hparams.n_layer()) {
        case 28: type = LLM_TYPE_65B; break; // 14 logical layers * 2
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_longcat_flash_ngram::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v_mla = hparams.n_embd_head_v_mla();

    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;

    const int64_t q_lora_rank  = hparams.n_lora_q;
    const int64_t kv_lora_rank = hparams.n_lora_kv;
    const int64_t n_ff_exp     = hparams.n_ff_exp;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (!output) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    {
        const uint32_t n_ngram = (hparams.ngram_neighbor_num - 1) * hparams.ngram_split_num;
        GGML_ASSERT(n_ngram > 0 && n_ngram <= (uint32_t) llama_model::NGRAM_MAX);

        const int64_t ngram_emb_dim = n_embd / n_ngram;
        const int64_t ngram_m = (int64_t) hparams.ngram_vocab_size_ratio * n_vocab;

        for (uint32_t j = 0; j < n_ngram; ++j) {
            const int64_t ngram_vocab_j = ngram_m + j * 2 + 1;
            ngram_embd[j] = create_tensor(
                tn(LLM_TENSOR_NGRAM_EMBD, "weight", j),
                {ngram_emb_dim, ngram_vocab_j},
                0);
            ngram_proj[j] = create_tensor(
                tn(LLM_TENSOR_NGRAM_PROJ, "weight", j),
                {ngram_emb_dim, n_embd},
                0);
        }
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        const bool is_moe_layer = (i % 2 == 0);

        layer.attn_norm = create_tensor(
            tn(LLM_TENSOR_ATTN_NORM, "weight", i),
            {n_embd},
            0);

        if (q_lora_rank > 0) {
            layer.attn_q_a_norm = create_tensor(
                tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i),
                {q_lora_rank},
                0);
            layer.wq_a = create_tensor(
                tn(LLM_TENSOR_ATTN_Q_A, "weight", i),
                {n_embd, q_lora_rank},
                0);
            layer.wq_b = create_tensor(
                tn(LLM_TENSOR_ATTN_Q_B, "weight", i),
                {q_lora_rank, n_head * n_embd_head_k_mla},
                0);
        } else {
            layer.wq = create_tensor(
                tn(LLM_TENSOR_ATTN_Q, "weight", i),
                {n_embd, n_head * n_embd_head_k_mla},
                0);
        }

        layer.attn_kv_a_norm = create_tensor(
            tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i),
            {kv_lora_rank},
            0);
        layer.wkv_a_mqa = create_tensor(
            tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", i),
            {n_embd, kv_lora_rank + n_embd_head_qk_rope},
            0);

        layer.wk_b = create_tensor(
            tn(LLM_TENSOR_ATTN_K_B, "weight", i),
            {n_embd_head_qk_nope, kv_lora_rank, n_head},
            0);
        layer.wv_b = create_tensor(
            tn(LLM_TENSOR_ATTN_V_B, "weight", i),
            {kv_lora_rank, n_embd_head_v_mla, n_head},
            0);

        layer.wo = create_tensor(
            tn(LLM_TENSOR_ATTN_OUT, "weight", i),
            {n_head * n_embd_head_v_mla, n_embd},
            0);

        layer.ffn_norm = create_tensor(
            tn(LLM_TENSOR_FFN_NORM, "weight", i),
            {n_embd},
            0);

        if (is_moe_layer) {
            layer.ffn_gate_inp = create_tensor(
                tn(LLM_TENSOR_FFN_GATE_INP, "weight", i),
                {n_embd, n_expert + (int64_t) hparams.n_expert_zero},
                0);
            layer.ffn_exp_probs_b = create_tensor(
                tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i),
                {n_expert + (int64_t) hparams.n_expert_zero},
                TENSOR_NOT_REQUIRED);

            layer.ffn_gate_exps = create_tensor(
                tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i),
                {n_embd, n_ff_exp, n_expert},
                0);
            layer.ffn_down_exps = create_tensor(
                tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i),
                {n_ff_exp, n_embd, n_expert},
                0);
            layer.ffn_up_exps = create_tensor(
                tn(LLM_TENSOR_FFN_UP_EXPS, "weight", i),
                {n_embd, n_ff_exp, n_expert},
                0);

            layer.ffn_gate_shexp = create_tensor(
                tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i),
                {n_embd, n_ff},
                0);
            layer.ffn_down_shexp = create_tensor(
                tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i),
                {n_ff, n_embd},
                0);
            layer.ffn_up_shexp = create_tensor(
                tn(LLM_TENSOR_FFN_UP_SHEXP, "weight", i),
                {n_embd, n_ff},
                0);
        } else {
            layer.ffn_gate = create_tensor(
                tn(LLM_TENSOR_FFN_GATE, "weight", i),
                {n_embd, n_ff},
                0);
            layer.ffn_down = create_tensor(
                tn(LLM_TENSOR_FFN_DOWN, "weight", i),
                {n_ff, n_embd},
                0);
            layer.ffn_up = create_tensor(
                tn(LLM_TENSOR_FFN_UP, "weight", i),
                {n_embd, n_ff},
                0);
        }
    }

    // The MTP decoder block is stored after the effective 28-layer trunk.
    for (int i = n_layer; i < n_layer_all; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(
            tn(LLM_TENSOR_ATTN_NORM, "weight", i),
            {n_embd},
            0);

        if (q_lora_rank > 0) {
            layer.attn_q_a_norm = create_tensor(
                tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i),
                {q_lora_rank},
                0);
            layer.wq_a = create_tensor(
                tn(LLM_TENSOR_ATTN_Q_A, "weight", i),
                {n_embd, q_lora_rank},
                0);
            layer.wq_b = create_tensor(
                tn(LLM_TENSOR_ATTN_Q_B, "weight", i),
                {q_lora_rank, n_head * n_embd_head_k_mla},
                0);
        } else {
            layer.wq = create_tensor(
                tn(LLM_TENSOR_ATTN_Q, "weight", i),
                {n_embd, n_head * n_embd_head_k_mla},
                0);
        }

        layer.attn_kv_a_norm = create_tensor(
            tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i),
            {kv_lora_rank},
            0);
        layer.wkv_a_mqa = create_tensor(
            tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", i),
            {n_embd, kv_lora_rank + n_embd_head_qk_rope},
            0);
        layer.wk_b = create_tensor(
            tn(LLM_TENSOR_ATTN_K_B, "weight", i),
            {n_embd_head_qk_nope, kv_lora_rank, n_head},
            0);
        layer.wv_b = create_tensor(
            tn(LLM_TENSOR_ATTN_V_B, "weight", i),
            {kv_lora_rank, n_embd_head_v_mla, n_head},
            0);
        layer.wo = create_tensor(
            tn(LLM_TENSOR_ATTN_OUT, "weight", i),
            {n_head * n_embd_head_v_mla, n_embd},
            0);

        layer.ffn_norm = create_tensor(
            tn(LLM_TENSOR_FFN_NORM, "weight", i),
            {n_embd},
            0);
        layer.ffn_gate = create_tensor(
            tn(LLM_TENSOR_FFN_GATE, "weight", i),
            {n_embd, n_ff},
            0);
        layer.ffn_down = create_tensor(
            tn(LLM_TENSOR_FFN_DOWN, "weight", i),
            {n_ff, n_embd},
            0);
        layer.ffn_up = create_tensor(
            tn(LLM_TENSOR_FFN_UP, "weight", i),
            {n_embd, n_ff},
            0);

        layer.nextn.eh_proj = create_tensor(
            tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", i),
            {2 * n_embd, n_embd},
            0);
        layer.nextn.enorm = create_tensor(
            tn(LLM_TENSOR_NEXTN_ENORM, "weight", i),
            {n_embd},
            0);
        layer.nextn.hnorm = create_tensor(
            tn(LLM_TENSOR_NEXTN_HNORM, "weight", i),
            {n_embd},
            0);
        layer.nextn.embed_tokens = create_tensor(
            tn(LLM_TENSOR_NEXTN_EMBED_TOKENS, "weight", i),
            {n_embd, n_vocab},
            TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_head = create_tensor(
            tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", i),
            {n_embd, n_vocab},
            TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_norm = create_tensor(
            tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i),
            {n_embd},
            0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_longcat_flash_ngram::build_arch_graph(
        const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

llama_model_longcat_flash_ngram::graph::graph(
        const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {

    const bool is_mla = hparams.is_mla();
    GGML_ASSERT(is_mla && "LongCat-Flash-Ngram requires MLA");

    const int64_t n_embd_head_k = hparams.n_embd_head_k_mla();

    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope;

    const uint32_t kv_lora_rank = hparams.n_lora_kv;

    // MLA LoRA scaling factors (LongCat-Flash-specific, not in DeepSeek2)
    const float mla_scale_q  = sqrtf((float) n_embd / (float) hparams.n_lora_q);
    const float mla_scale_kv = sqrtf((float) n_embd / (float) kv_lora_rank);

    const uint32_t n_expert_real = hparams.n_expert;
    const uint32_t n_expert_zero = hparams.n_expert_zero;
    const uint32_t n_expert_total = n_expert_real + n_expert_zero;

    // YaRN mscale computation (same as DeepSeek2)
    GGML_ASSERT(ext_factor >= 0.0f);
    const float attn_factor_org = attn_factor * (1.0f + 0.1f * logf(1.0f / freq_scale));
    const float mscale = attn_factor_org * (1.0f + 0.1f * hparams.rope_yarn_log_mul * logf(1.0f / freq_scale));
    const float kq_scale = 1.0f * mscale * mscale / sqrtf(float(n_embd_head_k));

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // MoE shortcut: computed on even blocks, added on the following odd block
    ggml_tensor * moe_shortcut = nullptr;

    inpL = build_inp_embd(model.tok_embd);

    // N-gram embedding augmentation
    // Computes polynomial rolling hashes over token ID history, looks up 12 embedding tables,
    // projects each to hidden_size, sums with base embedding, and normalizes by 1/13.
    // Reference: modeling_longcat_ngram.py NgramEmbedding.forward()
    {
        const uint32_t n_neighbor = hparams.ngram_neighbor_num;    // 4
        const uint32_t n_split    = hparams.ngram_split_num;       // 4
        const uint32_t n_ngram    = (n_neighbor - 1) * n_split;    // 12
        const int64_t  vocab_size   = model.tok_embd->ne[1];
        const int64_t  m            = (int64_t)hparams.ngram_vocab_size_ratio * vocab_size;
        const int32_t  eos_token_id = model.vocab.token_eos();
        GGML_ASSERT(eos_token_id == 2);

        // Create n-gram input: 12 I32 tensors of hash IDs, computed on CPU in set_input()
        auto inp = std::make_unique<llm_graph_input_ngram>(
            (int32_t)n_ngram, (int32_t)n_neighbor, (int32_t)n_split,
            (int32_t)vocab_size, m, eos_token_id,
            &res->ngram_token_history);

        for (uint32_t j = 0; j < n_ngram; j++) {
            inp->ngram_ids[j] = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
            ggml_set_input(inp->ngram_ids[j]);
        }

        // For each embedder: lookup embedding table, project to hidden_size, accumulate
        for (uint32_t j = 0; j < n_ngram && j < (uint32_t)llama_model::NGRAM_MAX; j++) {
            // ngram_embd[j] shape: [emb_dim, vocab_j]  (emb_dim = hidden_size / n_ngram = 256)
            // ngram_proj[j] shape: [emb_dim, hidden_size]
            ggml_tensor * emb = ggml_get_rows(ctx0, model.ngram_embd[j], inp->ngram_ids[j]);
            cb(emb, "ngram_emb", j);

            ggml_tensor * proj = ggml_mul_mat(ctx0, model.ngram_proj[j], emb);
            cb(proj, "ngram_proj", j);

            inpL = ggml_add(ctx0, inpL, proj);
        }

        // Normalize: x = (base + sum_of_projections) / (1 + n_ngram)
        inpL = ggml_scale(ctx0, inpL, 1.0f / (1.0f + (float)n_ngram));
        cb(inpL, "inp_embd_ngram", -1);

        res->add_input(std::move(inp));
    }

    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn_k = build_attn_inp_k();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        const bool is_even_block = (il % 2 == 0);

        // norm
        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // MLA self-attention (same as DeepSeek2 with absorption optimization)
        {
            ggml_tensor * q = NULL;

            if (model.layers[il].wq_a) {
                // LoRA Q path
                q = ggml_mul_mat(ctx0, model.layers[il].wq_a, cur);
                cb(q, "q", il);

                q = build_norm(q, model.layers[il].attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
                cb(q, "q", il);

                q = ggml_mul_mat(ctx0, model.layers[il].wq_b, q);
                cb(q, "q", il);

                // MLA LoRA scaling: q *= sqrt(hidden_size / q_lora_rank)
                q = ggml_scale(ctx0, q, mla_scale_q);
                cb(q, "q_scaled", il);
            } else {
                q = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
                cb(q, "q", il);
            }

            // split Q into nope and rope parts
            ggml_tensor * q_nope =
                ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens,
                             ggml_row_size(q->type, n_embd_head_k),
                             ggml_row_size(q->type, n_embd_head_k) * n_head, 0);
            cb(q_nope, "q_nope", il);

            ggml_tensor * q_pe = ggml_view_3d(
                ctx0, q, n_embd_head_qk_rope, n_head, n_tokens,
                ggml_row_size(q->type, n_embd_head_k),
                ggml_row_size(q->type, n_embd_head_k) * n_head,
                ggml_row_size(q->type, n_embd_head_qk_nope));
            cb(q_pe, "q_pe", il);

            // compressed KV + rope
            ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, model.layers[il].wkv_a_mqa, cur);
            cb(kv_cmpr_pe, "kv_cmpr_pe", il);

            ggml_tensor * kv_cmpr =
                ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
                             ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);
            cb(kv_cmpr, "kv_cmpr", il);

            ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens,
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));
            cb(k_pe, "k_pe", il);

            // apply RoPE
            q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                                 freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            cb(q_pe, "q_pe", il);

            k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                                 freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            cb(k_pe, "k_pe", il);

            // normalize compressed KV
            kv_cmpr = build_norm(kv_cmpr, model.layers[il].attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
            cb(kv_cmpr, "kv_cmpr", il);

            // MLA LoRA scaling: kv_cmpr *= sqrt(hidden_size / kv_lora_rank)
            kv_cmpr = ggml_scale(ctx0, kv_cmpr, mla_scale_kv);
            cb(kv_cmpr, "kv_cmpr_scaled", il);

            // MLA absorption optimization: absorb k_b into q_nope
            q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
            cb(q_nope, "q_nope_perm", il);

            ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, model.layers[il].wk_b, q_nope);
            cb(q_nope_absorbed, "q_nope_absorbed", il);

            q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);
            cb(q_nope_absorbed, "q_nope_absorbed_perm", il);

            ggml_tensor * Qcur = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);
            cb(Qcur, "Qcur", il);

            kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
            cb(kv_cmpr, "kv_cmpr_reshape", il);

            ggml_tensor * Kcur = ggml_concat(ctx0, kv_cmpr, k_pe, 0);
            cb(Kcur, "Kcur", il);

            ggml_tensor * Vcur = kv_cmpr;
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn_k,
                    model.layers[il].wo, NULL, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, model.layers[il].wv_b, kq_scale, il);
        }

        if (il == n_layer - 1 && inp_out_ids && cparams.embeddings_nextn_masked) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            // Also filter MoE shortcut from previous even block
            if (moe_shortcut) {
                moe_shortcut = ggml_get_rows(ctx0, moe_shortcut, inp_out_ids);
            }
        }

        // attention residual
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // FFN norm
        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if (is_even_block) {
            // Even block: compute MoE shortcut (saved for next odd block) + dense MLP[0]

            // --- MoE shortcut (computed but NOT added to this block's output) ---
            {
                ggml_tensor * gate_inp = model.layers[il].ffn_gate_inp;

                ggml_tensor * logits = ggml_mul_mat(ctx0, gate_inp, cur); // [n_expert_total, n_tokens]
                cb(logits, "ffn_moe_logits", il);

                auto route = llm_graph_build_longcat_moe_route(
                    ctx0, logits, model.layers[il].ffn_exp_probs_b,
                    n_tokens, n_expert_real, n_expert_total, n_expert_used,
                    hparams.expert_weights_scale);

                cb(route.probs, "ffn_moe_probs", il);
                cb(route.selection_probs, "ffn_moe_probs_biased", il);
                cb(route.selected_experts, "ffn_moe_topk", il);
                cb(route.weights, "ffn_moe_weights_scaled", il);
                cb(route.identity_weight_sum, "identity_weight_sum", il);
                cb(route.weights_real, "ffn_moe_weights_real", il);
                cb(route.selected_real, "ffn_moe_topk_real", il);

                ggml_build_forward_expand(gf, route.weights);

                // Expert FFN dispatch
                ggml_tensor * cur_moe = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);

                ggml_tensor * up = build_lora_mm_id(model.layers[il].ffn_up_exps, cur_moe, route.selected_real);
                cb(up, "ffn_moe_up", il);

                ggml_tensor * gate_proj = build_lora_mm_id(model.layers[il].ffn_gate_exps, cur_moe, route.selected_real);
                cb(gate_proj, "ffn_moe_gate", il);

                ggml_tensor * experts_out = ggml_swiglu_split(ctx0, gate_proj, up);
                cb(experts_out, "ffn_moe_swiglu", il);

                experts_out = build_lora_mm_id(model.layers[il].ffn_down_exps, experts_out, route.selected_real);
                cb(experts_out, "ffn_moe_down", il);

                experts_out = ggml_mul(ctx0, experts_out, route.weights_real);
                cb(experts_out, "ffn_moe_weighted", il);

                // Aggregate expert outputs
                ggml_tensor * cur_experts[LLAMA_MAX_EXPERTS] = { nullptr };
                for (uint32_t i = 0; i < (uint32_t) n_expert_used; ++i) {
                    cur_experts[i] = ggml_view_2d(ctx0, experts_out, n_embd, n_tokens,
                        experts_out->nb[2], i * experts_out->nb[1]);
                    ggml_build_forward_expand(gf, cur_experts[i]);
                }

                ggml_tensor * moe_out = cur_experts[0];
                for (uint32_t i = 1; i < (uint32_t) n_expert_used; ++i) {
                    moe_out = ggml_add(ctx0, moe_out, cur_experts[i]);
                }
                cb(moe_out, "ffn_moe_out", il);

                // Identity residual
                ggml_tensor * identity_residual = ggml_mul(ctx0, cur, route.identity_weight_sum);
                cb(identity_residual, "identity_residual", il);

                moe_shortcut = ggml_add(ctx0, moe_out, identity_residual);
                cb(moe_shortcut, "moe_shortcut", il);
            }

            // --- Dense MLP[0] (stored as "shared expert" in GGUF, actually mlps.0) ---
            cur = build_ffn(cur,
                model.layers[il].ffn_up_shexp, NULL, NULL,
                model.layers[il].ffn_gate_shexp, NULL, NULL,
                model.layers[il].ffn_down_shexp, NULL, NULL,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);

        } else {
            // Odd block: dense MLP[1] + add MoE shortcut from previous even block
            // HF reference: hidden_states = self.mlps[1](hidden_states)
            //               hidden_states = residual + hidden_states + shortcut_mlp_output
            cur = build_ffn(cur,
                model.layers[il].ffn_up, NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);

            // Add MoE shortcut from the paired even block
            if (moe_shortcut) {
                cur = ggml_add(ctx0, cur, moe_shortcut);
                cb(cur, "ffn_out_with_moe", il);
                moe_shortcut = nullptr;
            }
        }

        // residual
        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);

    // The speculative decoder consumes this normalized trunk hidden state.
    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (!cparams.embeddings_nextn_masked && inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

// LLM_GRAPH_TYPE_DECODER_MTP draft head for LongCat-Flash-Ngram.
llama_model_longcat_flash_ngram::graph_mtp::graph_mtp(
        const llama_model & model,
        const llm_graph_params & params) :
    llm_graph_context(params) {
    GGML_ASSERT(hparams.n_layer_nextn > 0 && "LongCat MTP requires nextn_predict_layers > 0");
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "LongCat MTP currently supports one auxiliary block");
    GGML_ASSERT(hparams.is_mla() && "LongCat MTP requires MLA");

    const int il = hparams.n_layer();
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && "LongCat MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm && "LongCat MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm && "LongCat MTP block missing nextn.hnorm");

    const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v_mla = hparams.n_embd_head_v_mla();
    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;
    const uint32_t kv_lora_rank = hparams.n_lora_kv;

    const float mla_scale_q = hparams.n_lora_q > 0
        ? sqrtf((float) n_embd / (float) hparams.n_lora_q)
        : 1.0f;
    const float mla_scale_kv = sqrtf((float) n_embd / (float) kv_lora_rank);

    GGML_ASSERT(ext_factor >= 0.0f);
    const float attn_factor_org =
        attn_factor * (1.0f + 0.1f * logf(1.0f / freq_scale));
    const float mscale =
        attn_factor_org *
        (1.0f + 0.1f * hparams.rope_yarn_log_mul * logf(1.0f / freq_scale));
    const float kq_scale =
        1.0f * mscale * mscale / sqrtf(float(n_embd_head_k_mla));

    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(
        ctx0,
        GGML_TYPE_F32,
        hparams.n_embd_inp(),
        n_tokens);
    ggml_set_input(inp->embd);

    ggml_tensor * tok_embd;
    if (ubatch.token) {
        ggml_tensor * tok_embd_w =
            layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;
        tok_embd = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);
    } else {
        tok_embd = inp->embd;
    }
    cb(tok_embd, "mtp_tok_embd", il);

    inp->h = ggml_new_tensor_2d(
        ctx0,
        GGML_TYPE_F32,
        hparams.n_embd,
        n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    ggml_tensor * h_embd = inp->h;
    res->add_input(std::move(inp));

    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    auto * inp_attn_k = build_attn_inp_k();

    ggml_tensor * h_norm =
        build_norm(h_embd, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm =
        build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, 0);
    cb(concat, "mtp_concat", il);

    ggml_tensor * cur =
        build_lora_mm(layer.nextn.eh_proj, concat, layer.nextn.eh_proj_s);
    cb(cur, "mtp_eh_proj", il);

    ggml_tensor * inpSA = cur;

    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_norm", il);

    ggml_tensor * q = nullptr;
    if (layer.wq_a) {
        q = ggml_mul_mat(ctx0, layer.wq_a, cur);
        q = build_norm(q, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
        q = ggml_mul_mat(ctx0, layer.wq_b, q);
        q = ggml_scale(ctx0, q, mla_scale_q);
    } else {
        q = build_lora_mm(layer.wq, cur, layer.wq_s);
    }
    cb(q, "mtp_q", il);

    ggml_tensor * q_nope = ggml_view_3d(
        ctx0,
        q,
        n_embd_head_qk_nope,
        n_head,
        n_tokens,
        ggml_row_size(q->type, n_embd_head_k_mla),
        ggml_row_size(q->type, n_embd_head_k_mla) * n_head,
        0);

    ggml_tensor * q_pe = ggml_view_3d(
        ctx0,
        q,
        n_embd_head_qk_rope,
        n_head,
        n_tokens,
        ggml_row_size(q->type, n_embd_head_k_mla),
        ggml_row_size(q->type, n_embd_head_k_mla) * n_head,
        ggml_row_size(q->type, n_embd_head_qk_nope));

    ggml_tensor * kv_cmpr_pe =
        ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);

    ggml_tensor * kv_cmpr = ggml_view_2d(
        ctx0,
        kv_cmpr_pe,
        kv_lora_rank,
        n_tokens,
        ggml_row_size(
            kv_cmpr_pe->type,
            kv_lora_rank + n_embd_head_qk_rope),
        0);

    ggml_tensor * k_pe = ggml_view_3d(
        ctx0,
        kv_cmpr_pe,
        n_embd_head_qk_rope,
        1,
        n_tokens,
        ggml_row_size(
            kv_cmpr_pe->type,
            kv_lora_rank + n_embd_head_qk_rope),
        ggml_row_size(
            kv_cmpr_pe->type,
            kv_lora_rank + n_embd_head_qk_rope),
        ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));

    q_pe = ggml_rope_ext(
        ctx0,
        q_pe,
        inp_pos,
        nullptr,
        n_rot,
        rope_type,
        n_ctx_orig,
        freq_base,
        freq_scale,
        ext_factor,
        attn_factor,
        beta_fast,
        beta_slow);

    k_pe = ggml_rope_ext(
        ctx0,
        k_pe,
        inp_pos,
        nullptr,
        n_rot,
        rope_type,
        n_ctx_orig,
        freq_base,
        freq_scale,
        ext_factor,
        attn_factor,
        beta_fast,
        beta_slow);

    kv_cmpr =
        build_norm(kv_cmpr, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
    kv_cmpr = ggml_scale(ctx0, kv_cmpr, mla_scale_kv);

    q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
    q_nope = ggml_mul_mat(ctx0, layer.wk_b, q_nope);
    q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);

    ggml_tensor * Qcur = ggml_concat(ctx0, q_nope, q_pe, 0);
    kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
    ggml_tensor * Kcur = ggml_concat(ctx0, kv_cmpr, k_pe, 0);
    ggml_tensor * Vcur = kv_cmpr;

    cur = build_attn(
        inp_attn_k,
        layer.wo,
        nullptr,
        layer.wo_s,
        Qcur,
        Kcur,
        Vcur,
        nullptr,
        nullptr,
        layer.wv_b,
        kq_scale,
        il);
    cb(cur, "mtp_attn_out", il);

    ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
    cb(ffn_inp, "mtp_ffn_inp", il);

    cur = build_norm(ffn_inp, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_ffn_norm", il);

    cur = build_ffn(
        cur,
        layer.ffn_up,
        nullptr,
        layer.ffn_up_s,
        layer.ffn_gate,
        nullptr,
        layer.ffn_gate_s,
        layer.ffn_down,
        nullptr,
        layer.ffn_down_s,
        nullptr,
        LLM_FFN_SILU,
        LLM_FFN_PAR,
        il);
    cb(cur, "mtp_ffn_out", il);

    cur = ggml_add(ctx0, cur, ffn_inp);
    cb(cur, "mtp_post_ffn", il);

    ggml_tensor * head_norm_w =
        layer.nextn.shared_head_norm
            ? layer.nextn.shared_head_norm
            : model.output_norm;
    GGML_ASSERT(head_norm_w);

    cur = build_norm(cur, head_norm_w, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    ggml_tensor * head_w =
        layer.nextn.shared_head_head
            ? layer.nextn.shared_head_head
            : model.output;
    ggml_tensor * head_s =
        layer.nextn.shared_head_head
            ? layer.nextn.shared_head_head_s
            : model.output_s;
    GGML_ASSERT(head_w);

    cur = build_lora_mm(head_w, cur, head_s);
    cb(cur, "result_output", -1);

    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}
