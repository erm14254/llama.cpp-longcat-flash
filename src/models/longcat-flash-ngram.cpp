#include "models.h"

#include "../llama-graph.h"
#include "../llama-model.h"

#include <cmath>

llm_build_longcat_flash_ngram::llm_build_longcat_flash_ngram(
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
        const int64_t  vocab_size = model.tok_embd->ne[1];
        const int64_t  m          = (int64_t)hparams.ngram_vocab_size_ratio * vocab_size;

        // Create n-gram input: 12 I32 tensors of hash IDs, computed on CPU in set_input()
        auto inp = std::make_unique<llm_graph_input_ngram>(
            (int32_t)n_ngram, (int32_t)n_neighbor, (int32_t)n_split,
            (int32_t)vocab_size, m,
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
                    model.layers[il].wo, NULL,
                    Qcur, Kcur, Vcur, nullptr, nullptr, model.layers[il].wv_b, kq_scale, il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
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

                ggml_tensor * probs = ggml_soft_max(ctx0, logits);
                cb(probs, "ffn_moe_probs", il);

                // Split probs: real [0..256) and identity [256..384)
                ggml_tensor * real_probs = ggml_cont(ctx0,
                    ggml_view_2d(ctx0, probs, n_expert_real, n_tokens, probs->nb[1], 0));
                cb(real_probs, "ffn_moe_real_probs", il);

                ggml_tensor * id_probs = ggml_cont(ctx0,
                    ggml_view_2d(ctx0, probs, n_expert_zero, n_tokens,
                        probs->nb[1], n_expert_real * ggml_element_size(probs)));

                // Sum all identity expert probs per token
                ggml_tensor * identity_weight_sum = ggml_sum_rows(ctx0, id_probs);
                identity_weight_sum = ggml_scale(ctx0, identity_weight_sum, hparams.expert_weights_scale);
                cb(identity_weight_sum, "identity_weight_sum", il);

                // Add bias for selection only
                ggml_tensor * selection_probs = real_probs;
                if (model.layers[il].ffn_exp_probs_b) {
                    ggml_tensor * real_bias = ggml_view_1d(ctx0,
                        model.layers[il].ffn_exp_probs_b, n_expert_real, 0);
                    selection_probs = ggml_add(ctx0, real_probs, real_bias);
                    cb(selection_probs, "ffn_moe_probs_biased", il);
                }

                ggml_tensor * selected_experts = ggml_argsort_top_k(ctx0, selection_probs, n_expert_used);
                cb(selected_experts, "ffn_moe_topk", il);

                // Gather weights from UNBIASED probs
                ggml_tensor * real_probs_3d = ggml_reshape_3d(ctx0, real_probs,
                    1, (int64_t) n_expert_real, n_tokens);
                ggml_tensor * weights = ggml_get_rows(ctx0, real_probs_3d, selected_experts);
                cb(weights, "ffn_moe_weights", il);

                weights = ggml_scale(ctx0, weights, hparams.expert_weights_scale);
                cb(weights, "ffn_moe_weights_scaled", il);

                ggml_build_forward_expand(gf, weights);

                // Expert FFN dispatch
                ggml_tensor * cur_moe = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);

                ggml_tensor * up = build_lora_mm_id(model.layers[il].ffn_up_exps, cur_moe, selected_experts);
                cb(up, "ffn_moe_up", il);

                ggml_tensor * gate_proj = build_lora_mm_id(model.layers[il].ffn_gate_exps, cur_moe, selected_experts);
                cb(gate_proj, "ffn_moe_gate", il);

                ggml_tensor * experts_out = ggml_swiglu_split(ctx0, gate_proj, up);
                cb(experts_out, "ffn_moe_swiglu", il);

                experts_out = build_lora_mm_id(model.layers[il].ffn_down_exps, experts_out, selected_experts);
                cb(experts_out, "ffn_moe_down", il);

                experts_out = ggml_mul(ctx0, experts_out, weights);
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
                ggml_tensor * identity_residual = ggml_mul(ctx0, cur, identity_weight_sum);
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
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
