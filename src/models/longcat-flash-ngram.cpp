#include "models.h"

#include "../llama-graph.h"
#include "../llama-impl.h"
#include "../llama-kv-cache-dsa.h"
#include "../llama-model.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <vector>

// LONGCAT_ROPE_ORACLE (Experiment R1): diagnostic graph input carrying the
// captured HF BF16 rope cos/sin oracles for physical block 0.
//
// Active only when LONGCAT_ROPE_ORACLE_DIR is set. The files are the
// authoritative Blackwell captures rope_cos.bin / rope_sin.bin
// ([512, 64] token-major F32 on the BF16 lattice; the first 32 columns hold
// the per-pair angles, modeling_longcat_flash.py:322-324). SHA256 of the
// files is verified by the run-harness preflight; here the exact byte size
// and the diagnostic preconditions are enforced: 512-token single ubatch
// with pos[i] == i, exactly the frozen capture invocation.
namespace {

class llm_graph_input_longcat_rope_oracle : public llm_graph_input_i {
public:
    explicit llm_graph_input_longcat_rope_oracle(std::string dir_) : dir(std::move(dir_)) {}

    void set_input(const llama_ubatch * ubatch) override {
        const int64_t n_tokens = ubatch->n_tokens;

        GGML_ASSERT(n_tokens == 512 &&
                    "LONGCAT_ROPE_ORACLE: diagnostic is 512-token single-ubatch only");
        GGML_ASSERT(ubatch->pos != nullptr);

        for (int64_t i = 0; i < n_tokens; ++i) {
            GGML_ASSERT(ubatch->pos[i] == i &&
                        "LONGCAT_ROPE_ORACLE: row index must equal position");
        }

        struct { const char * file; ggml_tensor * dst; } feeds[2] = {
            { "rope_cos.bin", cos_half },
            { "rope_sin.bin", sin_half },
        };

        for (const auto & feed : feeds) {
            const std::string path = dir + "/" + feed.file;
            std::ifstream f(path, std::ios::binary);
            if (!f) {
                GGML_ABORT("LONGCAT_ROPE_ORACLE: cannot open %s", path.c_str());
            }
            std::vector<float> full((size_t) n_tokens * 64);
            f.read((char *) full.data(), full.size() * sizeof(float));
            if ((size_t) f.gcount() != full.size() * sizeof(float)) {
                GGML_ABORT("LONGCAT_ROPE_ORACLE: %s short read", path.c_str());
            }
            // first-half columns, laid out [32, 1, n_tokens] (j fastest)
            std::vector<float> staged((size_t) n_tokens * 32);
            for (int64_t t = 0; t < n_tokens; ++t) {
                for (int64_t j = 0; j < 32; ++j) {
                    staged[(size_t) t * 32 + j] = full[(size_t) t * 64 + j];
                }
            }
            ggml_backend_tensor_set(feed.dst, staged.data(), 0,
                                    staged.size() * sizeof(float));
        }
    }

    ggml_tensor * cos_half = nullptr; // [32, 1, n_tokens] F32, BF16-lattice values
    ggml_tensor * sin_half = nullptr;

private:
    std::string dir;
};

} // namespace

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

void llama_model_longcat_flash_sparse::load_arch_hparams(llama_model_loader & ml) {
    // Load the existing LongCat MLA/MoE/N-gram contract first. This also reads
    // nextn_predict_layers into n_layer_nextn: the physical appended MTP block
    // count. It must stay 1; it is not the conceptual MTP step count.
    llama_model_longcat_flash_ngram::load_arch_hparams(ml);

    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,      hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,      hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,           hparams.indexer_top_k);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_INIT_TOKENS,     hparams.indexer_init_tokens);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_LOCAL_TOKENS,    hparams.indexer_local_tokens);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_K_NORM_EPS,      hparams.indexer_k_norm_eps);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_ROPE_INTERLEAVE, hparams.indexer_rope_interleave);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_CLI_FACTOR,      hparams.indexer_cli_factor);

    std::string indexer_k_norm_type;
    ml.get_key(LLM_KV_ATTENTION_INDEXER_K_NORM_TYPE, indexer_k_norm_type);

    ml.get_key(LLM_KV_MTP_NUM_LAYERS,        hparams.mtp_num_layers);
    ml.get_key(LLM_KV_MTP_REPLICATE_MODULES, hparams.mtp_replicate_modules);
    ml.get_key(LLM_KV_MTP_DSA_CLI,           hparams.mtp_dsa_cli);

    // This array covers only the 28 physical trunk blocks. The appended
    // physical MTP block owns its own parameterized indexer separately.
    ml.get_key_or_arr(
        LLM_KV_ATTENTION_INDEXER_TYPES,
        hparams.is_indexer_full_impl,
        hparams.n_layer());

    // Gate 2 supports the exact published LongCat Flash Lite Sparse contract.
    // Fail closed rather than silently interpreting a different variant.
    if (hparams.n_layer() != 28 || hparams.n_layer_nextn != 1) {
        throw std::runtime_error(
            "LongCat Flash Lite Sparse requires 28 trunk blocks and one physical MTP block");
    }
    if (hparams.indexer_n_head != 16 ||
        hparams.indexer_head_size != 128 ||
        hparams.indexer_top_k != 2048 ||
        hparams.indexer_init_tokens != 16 ||
        hparams.indexer_local_tokens != 1024 ||
        hparams.indexer_cli_factor != 2) {
        throw std::runtime_error("unsupported LongCat Flash Lite Sparse LSA metadata");
    }
    if (indexer_k_norm_type != "rms") {
        throw std::runtime_error(
            "LongCat Flash Lite Sparse requires RMS indexer K normalization");
    }
    if (std::fabs(hparams.indexer_k_norm_eps - 1.0e-6f) > 1.0e-12f) {
        throw std::runtime_error(
            "LongCat Flash Lite Sparse requires indexer K RMSNorm epsilon 1e-6");
    }
    if (!hparams.indexer_rope_interleave) {
        throw std::runtime_error(
            "LongCat Flash Lite Sparse requires interleaved indexer RoPE");
    }
    if (hparams.mtp_num_layers != 3 ||
        !hparams.mtp_replicate_modules ||
        !hparams.mtp_dsa_cli) {
        throw std::runtime_error(
            "LongCat Flash Lite Sparse requires one replicated physical MTP "
            "module for three conceptual steps with DSA MTP CLI");
    }

    for (uint32_t i = 0; i < hparams.n_layer(); ++i) {
        const bool expected_owner = (i % 2) == 0;
        if (hparams.is_indexer_full(i) != expected_owner) {
            throw std::runtime_error(
                "LongCat Flash Lite Sparse indexer.types must alternate "
                "owner/reuse as [true,false] across 28 trunk blocks");
        }
    }
}

void llama_model_longcat_flash_sparse::load_arch_tensors(llama_model_loader & ml) {
    // Existing LongCat loads MLA, MoE, N-gram, and the physical MTP block.
    llama_model_longcat_flash_ngram::load_arch_tensors(ml);

    LLAMA_LOAD_LOCALS;

    const int64_t q_lora_rank = hparams.n_lora_q;
    const int64_t indexer_width =
        (int64_t) hparams.indexer_n_head * hparams.indexer_head_size;

    // Trunk owners: blk.0,2,...,26. Reuse blocks have no independent trained
    // indexer parameters. blk.28 is the one physical MTP indexer owner.
    for (int i = 0; i < n_layer_all; ++i) {
        const bool is_trunk_owner =
            i < n_layer && hparams.is_indexer_full((uint32_t) i);
        const bool is_physical_mtp_owner =
            i == n_layer && hparams.n_layer_nextn == 1;

        if (!is_trunk_owner && !is_physical_mtp_owner) {
            continue;
        }

        auto & layer = layers[i];

        layer.indexer_k_norm = create_tensor(
            tn(LLM_TENSOR_INDEXER_K_NORM, "weight", i),
            {hparams.indexer_head_size},
            0);
        layer.indexer_proj = create_tensor(
            tn(LLM_TENSOR_INDEXER_PROJ, "weight", i),
            {n_embd, hparams.indexer_n_head},
            0);
        layer.indexer_attn_k = create_tensor(
            tn(LLM_TENSOR_INDEXER_ATTN_K, "weight", i),
            {n_embd, hparams.indexer_head_size},
            0);
        layer.indexer_attn_q_b = create_tensor(
            tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", i),
            {q_lora_rank, indexer_width},
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

    const bool longcat_lsa = arch == LLM_ARCH_LONGCAT_FLASH_SPARSE;
    const int64_t n_indexer_head = hparams.indexer_n_head;
    const int64_t n_embd_indexer_head = hparams.indexer_head_size;
    const int64_t n_embd_indexer_head_rope = n_embd_head_qk_rope;
    const int64_t n_embd_indexer_head_nope =
        n_embd_indexer_head - n_embd_indexer_head_rope;
    const uint32_t n_indexer_top_k = hparams.indexer_top_k;

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

        // the tokens before the ubatch come from the KV cache cells (the MLA cache for the sparse model)
        inp->mctx = longcat_lsa
            ? static_cast<const llama_kv_cache_dsa_context *>(mctx)->get_mla()
            : static_cast<const llama_kv_cache_context *>(mctx);

        for (uint32_t j = 0; j < n_ngram; j++) {
            inp->ngram_ids[j] = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
            ggml_set_input(inp->ngram_ids[j]);
        }

        // LONGCAT_NGRAM_BF16_PARITY_DIAGNOSTIC:
        // HF keeps the running N-gram embedding state on the BF16 lattice.
        // Round the base embedding before beginning the sequential accumulation.
        inpL = ggml_cast(ctx0, inpL, GGML_TYPE_BF16);

        // For each embedder: lookup embedding table, project to hidden_size, accumulate
        for (uint32_t j = 0; j < n_ngram && j < (uint32_t)llama_model::NGRAM_MAX; j++) {
            // ngram_embd[j] shape: [emb_dim, vocab_j]  (emb_dim = hidden_size / n_ngram = 256)
            // ngram_proj[j] shape: [emb_dim, hidden_size]
            ggml_tensor * emb = ggml_get_rows(ctx0, model.ngram_embd[j], inp->ngram_ids[j]);
            cb(emb, "ngram_emb", j);

            ggml_tensor * proj = ggml_mul_mat(ctx0, model.ngram_proj[j], emb);
            cb(proj, "ngram_proj", j);

            // ggml_mul_mat returns F32. HF parity only requires that each
            // projection be rounded to BF16 before the sequential BF16 add.
            proj = ggml_cast(ctx0, proj, GGML_TYPE_BF16);

            // Perform the arithmetic in F32 using already-BF16-rounded operands,
            // then round the running accumulator back to BF16 after every add.
            // This avoids depending on native BF16 ADD kernel semantics while
            // reproducing the validated standalone HF arithmetic.
            ggml_tensor * inpL_f32 = ggml_cast(ctx0, inpL, GGML_TYPE_F32);
            ggml_tensor * proj_f32 = ggml_cast(ctx0, proj, GGML_TYPE_F32);

            inpL = ggml_add(ctx0, inpL_f32, proj_f32);
            inpL = ggml_cast(ctx0, inpL, GGML_TYPE_BF16);
        }

        // Normalize: x = (base + sum_of_projections) / (1 + n_ngram)
        // Compute from the BF16-rounded accumulator and round the result back
        // to BF16, matching frozen HF NgramEmbedding.forward().
        inpL = ggml_cast(ctx0, inpL, GGML_TYPE_F32);
        inpL = ggml_scale(ctx0, inpL, 1.0f / (1.0f + (float)n_ngram));
        inpL = ggml_cast(ctx0, inpL, GGML_TYPE_BF16);
        cb(inpL, "inp_embd_ngram", -1);

        // LONGCAT_NGRAM_BF16_RESTORE_F32_DIAGNOSTIC:
        // BF16 values are exactly representable in F32. Restore the graph's
        // expected activation type without changing the HF-exact values.
        inpL = ggml_cast(ctx0, inpL, GGML_TYPE_F32);

        res->add_input(std::move(inp));
    }

    ggml_tensor * inp_pos = build_inp_pos();

    llm_graph_input_attn_k * inp_attn_k = nullptr;
    llm_graph_input_attn_k_dsa * inp_attn_dsa = nullptr;
    if (longcat_lsa) {
        inp_attn_dsa = build_attn_inp_k_dsa();
    } else {
        inp_attn_k = build_attn_inp_k();
    }

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * prev_top_k = nullptr;

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        const bool is_even_block = (il % 2 == 0);

        // norm
        if (il == 0) {
            // LONGCAT_ATTN0_HF_RMSNORM_DIAGNOSTIC:
            //
            // Transformers LongCat RMSNorm semantics for BF16 activations:
            //
            //   1. RMS normalization in F32
            //   2. round normalized activation to BF16
            //   3. multiply by the BF16 norm weight
            //   4. round output to BF16
            //
            // The GGUF norm weight is F32 but was proven to be the exact
            // numerical expansion of the HF BF16 weight, so multiplying the
            // BF16-rounded activation by that F32 value is numerically the
            // same pre-output-rounding product.
            //
            // Restore F32 afterward because the existing llama.cpp LongCat
            // trunk expects F32 activations.
            cur = ggml_rms_norm(ctx0, inpL, hparams.f_norm_rms_eps);

            cur = ggml_cast(ctx0, cur, GGML_TYPE_BF16);
            cur = ggml_cast(ctx0, cur, GGML_TYPE_F32);

            cur = ggml_mul(ctx0, cur, model.layers[il].attn_norm);

            cur = ggml_cast(ctx0, cur, GGML_TYPE_BF16);
            cur = ggml_cast(ctx0, cur, GGML_TYPE_F32);
        } else {
            cur = build_norm(
                inpL,
                model.layers[il].attn_norm,
                NULL,
                LLM_NORM_RMS,
                il);
        }

        cb(cur, "attn_norm", il);

        // MLA self-attention (same as DeepSeek2 with absorption optimization)
        {
            ggml_tensor * q = NULL;
            ggml_tensor * q_lora = nullptr;

            if (model.layers[il].wq_a) {
                // LoRA Q path
                if (il == 0) {
                    // LONGCAT_ATTN0_Q_BF16_SEMANTICS_DIAGNOSTIC:
                    //
                    // HF block-0 Q path is BF16 at the Linear/RMSNorm
                    // boundaries. Widen only where ggml RMSNorm / downstream
                    // graph operations require F32.
                    ggml_tensor * q_in_bf16 =
                        ggml_cast(ctx0, cur, GGML_TYPE_BF16);

                    q = ggml_mul_mat(
                        ctx0,
                        model.layers[il].wq_a,
                        q_in_bf16);

                    // q_a_proj output is BF16 in the HF BF16 model.
                    q = ggml_cast(ctx0, q, GGML_TYPE_BF16);
                    // LONGCAT_MLA_STAGE_SURFACE: distinct name so the parity
                    // dump can tell the three block-0 Q boundaries apart.
                    cb(q, "q_a_proj", il);

                    // LONGCAT_ATTN0_QA_NORM_EPS_DIAGNOSTIC:
                    // HF q_a_layernorm uses eps=1e-6 and computes the RMS
                    // reduction in F32.
                    q = ggml_cast(ctx0, q, GGML_TYPE_F32);
                    q = ggml_rms_norm(ctx0, q, 1.0e-6f);

                    // LongcatFlashRMSNorm casts the normalized activation
                    // back to BF16 before multiplying by its BF16 weight.
                    q = ggml_cast(ctx0, q, GGML_TYPE_BF16);
                    q = ggml_cast(ctx0, q, GGML_TYPE_F32);

                    q = ggml_mul(
                        ctx0,
                        q,
                        model.layers[il].attn_q_a_norm);

                    // RMSNorm output is BF16.
                    q = ggml_cast(ctx0, q, GGML_TYPE_BF16);
                    // LONGCAT_MLA_STAGE_SURFACE: HF q_a_layernorm boundary.
                    cb(q, "q_a_norm", il);

                    // LONGCAT_LSA_TAP: the indexer Q path consumes the
                    // standing post-q_a_layernorm tensor. Alias only - the
                    // hardened chain must stay byte-untouched.
                    q_lora = q;

                    // q_b_proj consumes BF16 and produces BF16 in HF.
                    q = ggml_mul_mat(
                        ctx0,
                        model.layers[il].wq_b,
                        q);

                    q = ggml_cast(ctx0, q, GGML_TYPE_BF16);
                    // LONGCAT_MLA_STAGE_SURFACE: HF q_b_proj boundary, before
                    // the MLA LoRA q scaling below.
                    cb(q, "q_b_proj", il);

                    // HF scales the BF16 q_pass/q_rot tensors and therefore
                    // returns to the BF16 lattice here as well. Widen the
                    // rounded result for the existing llama.cpp RoPE path.
                    q = ggml_cast(ctx0, q, GGML_TYPE_F32);
                    q = ggml_scale(ctx0, q, mla_scale_q);
                    q = ggml_cast(ctx0, q, GGML_TYPE_BF16);
                    q = ggml_cast(ctx0, q, GGML_TYPE_F32);
                    cb(q, "q_scaled", il);
                } else {
                    q = ggml_mul_mat(
                        ctx0,
                        model.layers[il].wq_a,
                        cur);

                    // LONGCAT_MLA_BF16_OUTPUT_BOUNDARY (production, il >= 1, A1):
                    // HF q_a_proj is a BF16 nn.Linear. The block-2 MLA walk
                    // proved this F32 GEMM output HF-equivalent at the BF16
                    // output boundary from all-exact inputs (786432/786432
                    // after RNE rounding). Round, then widen so view strides
                    // and the downstream F32 graph are unchanged
                    // (AUDIT_MLA_PRODSCOPE_2026-08-18.md).
                    q = ggml_cast(ctx0, q, GGML_TYPE_BF16);
                    q = ggml_cast(ctx0, q, GGML_TYPE_F32);

                    // LONGCAT_BLOCK2_MLA_STAGE_SURFACE: distinct names at
                    // il == 2 only, so the block-2 MLA walk can dump the
                    // three Q boundaries separately (name plumbing only; the
                    // generic "q" label is reused three times otherwise).
                    cb(q, il == 2 ? "q_a_proj" : "q", il);

                    q = build_norm(
                        q,
                        model.layers[il].attn_q_a_norm,
                        nullptr,
                        LLM_NORM_RMS,
                        il);
                    cb(q, il == 2 ? "q_a_norm" : "q", il);

                    // LONGCAT_LSA_TAP: alias only (see the il == 0 branch).
                    q_lora = q;

                    q = ggml_mul_mat(
                        ctx0,
                        model.layers[il].wq_b,
                        q);

                    // LONGCAT_MLA_BF16_OUTPUT_BOUNDARY (production, il >= 1, A2):
                    // HF q_b_proj output is BF16; the hex reset proved this
                    // GEMM HF-equivalent at the BF16 output boundary from
                    // all-exact inputs (3145728/3145728 after RNE rounding).
                    q = ggml_cast(ctx0, q, GGML_TYPE_BF16);
                    q = ggml_cast(ctx0, q, GGML_TYPE_F32);
                    cb(q, il == 2 ? "q_b_proj" : "q", il);

                    // MLA LoRA scaling: q *= sqrt(hidden_size / q_lora_rank)
                    q = ggml_scale(ctx0, q, mla_scale_q);

                    // LONGCAT_MLA_BF16_OUTPUT_BOUNDARY (production, il >= 1, A3):
                    // HF scales the BF16 q_pass/q_rot tensors after the split
                    // (modeling_longcat_flash.py:424-425), so the scaled q is
                    // stored on the BF16 lattice. Source-audit-derived and
                    // block-0-known-answer-supported (the accepted il == 0
                    // post-scale round above; R1/t=0 rope-entry identity) --
                    // NOT independently causal-frontier-measured at block 2;
                    // gated by the offline T4 target in the stage-A run.
                    q = ggml_cast(ctx0, q, GGML_TYPE_BF16);
                    q = ggml_cast(ctx0, q, GGML_TYPE_F32);
                    cb(q, "q_scaled", il);
                }
            } else {
                q = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
                cb(q, "q", il);
            }

            ggml_tensor * top_k = nullptr;

            if (longcat_lsa) {
                GGML_ASSERT(inp_attn_dsa != nullptr);
                GGML_ASSERT(q_lora != nullptr);
                GGML_ASSERT(hparams.indexer_cli_factor == 2);

                const bool indexer_owner = hparams.is_indexer_full((uint32_t) il);
                const auto * mctx_lid = inp_attn_dsa->mctx->get_lid();
                const uint32_t n_kv_lid = mctx_lid->get_n_kv();
                const bool sparse_active = n_kv_lid > n_indexer_top_k;

                if (indexer_owner) {
                    GGML_ASSERT(model.layers[il].indexer_k_norm);
                    GGML_ASSERT(model.layers[il].indexer_proj);
                    GGML_ASSERT(model.layers[il].indexer_attn_k);
                    GGML_ASSERT(model.layers[il].indexer_attn_q_b);

                    ggml_tensor * indexer_hidden =
                        ggml_cast(ctx0, cur, GGML_TYPE_BF16);
                    ggml_tensor * indexer_k =
                        ggml_mul_mat(ctx0, model.layers[il].indexer_attn_k, indexer_hidden);
                    cb(indexer_k, "lsa_indexer_k_proj", il);

                    indexer_k = ggml_rms_norm(
                        ctx0, indexer_k, hparams.indexer_k_norm_eps);
                    indexer_k = ggml_mul(
                        ctx0, indexer_k, model.layers[il].indexer_k_norm);
                    // Match HF's BF16 projection/norm rounding point, but
                    // CPU ggml_rope supports only F16/F32. Widen the already-
                    // rounded BF16 values solely for RoPE, then round back.
                    indexer_k = ggml_cast(ctx0, indexer_k, GGML_TYPE_BF16);
                    cb(indexer_k, "lsa_indexer_k_norm", il);
                    indexer_k = ggml_cast(ctx0, indexer_k, GGML_TYPE_F32);

                    ggml_tensor * indexer_k_pe =
                        ggml_view_3d(ctx0, indexer_k,
                            n_embd_indexer_head_rope, 1, n_tokens,
                            ggml_row_size(indexer_k->type, n_embd_indexer_head),
                            ggml_row_size(indexer_k->type, n_embd_indexer_head), 0);

                    ggml_tensor * indexer_k_nope =
                        ggml_view_3d(ctx0, indexer_k,
                            n_embd_indexer_head_nope, 1, n_tokens,
                            ggml_row_size(indexer_k->type, n_embd_indexer_head),
                            ggml_row_size(indexer_k->type, n_embd_indexer_head),
                            ggml_row_size(indexer_k->type, n_embd_indexer_head_rope));

                    indexer_k_pe = ggml_rope_ext(
                        ctx0, indexer_k_pe, inp_pos, nullptr,
                        n_embd_indexer_head_rope, LLAMA_ROPE_TYPE_NORM,
                        n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);

                    indexer_k = ggml_concat(
                        ctx0, indexer_k_pe, indexer_k_nope, 0);
                    indexer_k = ggml_cast(ctx0, indexer_k, GGML_TYPE_BF16);
                    cb(indexer_k, "lsa_indexer_k", il);

                    // LONGCAT_LSA_DUMP_SURFACE (measurement round, dump-only):
                    // the canonical lsa_indexer_k is [128, 1, n_tokens] and
                    // the dump helper rejects ne[2] != 1, so a first-owner 2D
                    // contiguous copy is made purely for the dump. It feeds
                    // nothing downstream; ggml_build_forward_expand forces its
                    // evaluation. Values byte-identical; zero arithmetic
                    // change.
                    if (il == 0) {
                        ggml_tensor * indexer_k_dump = ggml_cont_2d(
                            ctx0, indexer_k,
                            n_embd_indexer_head, n_tokens);
                        cb(indexer_k_dump, "lsa_indexer_k_2d", il);
                        ggml_build_forward_expand(gf, indexer_k_dump);
                    }

                    // ggml_set_rows() currently accepts F32/F16 source rows,
                    // not BF16. Preserve the BF16-projected/RoPE'd values above,
                    // then widen exactly for the cache write. The destination
                    // indexer cache remains BF16.
                    ggml_tensor * indexer_k_store =
                        ggml_cast(ctx0, indexer_k, GGML_TYPE_F32);

                    const auto & k_idxs_lid = inp_attn_dsa->get_k_idxs_lid();
                    ggml_build_forward_expand(
                        gf, mctx_lid->cpy_k(ctx0, indexer_k_store, k_idxs_lid, il));

                    if (sparse_active) {
                        ggml_tensor * indexer_q_in =
                            ggml_cast(ctx0, q_lora, GGML_TYPE_BF16);
                        ggml_tensor * indexer_q =
                            ggml_mul_mat(ctx0, model.layers[il].indexer_attn_q_b, indexer_q_in);
                        // Preserve BF16 Q projection rounding, then widen only
                        // for the CPU RoPE kernel exactly as for indexer K.
                        indexer_q = ggml_cast(ctx0, indexer_q, GGML_TYPE_BF16);
                        // LONGCAT_LSA_DUMP_SURFACE (measurement round,
                        // name-only): expose the first-owner pre-RoPE BF16
                        // Q-projection boundary for the 2050 capture. The
                        // cast node is otherwise unnamed; owners 2..26 are
                        // deliberately not named. Zero graph nodes; zero
                        // arithmetic change.
                        if (il == 0) {
                            cb(indexer_q, "lsa_indexer_q_proj", il);
                        }
                        indexer_q = ggml_cast(ctx0, indexer_q, GGML_TYPE_F32);

                        ggml_tensor * indexer_q_pe =
                            ggml_view_3d(ctx0, indexer_q,
                                n_embd_indexer_head_rope, n_indexer_head, n_tokens,
                                ggml_row_size(indexer_q->type, n_embd_indexer_head),
                                ggml_row_size(indexer_q->type, n_embd_indexer_head) * n_indexer_head, 0);

                        ggml_tensor * indexer_q_nope =
                            ggml_view_3d(ctx0, indexer_q,
                                n_embd_indexer_head_nope, n_indexer_head, n_tokens,
                                ggml_row_size(indexer_q->type, n_embd_indexer_head),
                                ggml_row_size(indexer_q->type, n_embd_indexer_head) * n_indexer_head,
                                ggml_row_size(indexer_q->type, n_embd_indexer_head_rope));

                        indexer_q_pe = ggml_rope_ext(
                            ctx0, indexer_q_pe, inp_pos, nullptr,
                            n_embd_indexer_head_rope, LLAMA_ROPE_TYPE_NORM,
                            n_ctx_orig, freq_base, freq_scale,
                            ext_factor, attn_factor, beta_fast, beta_slow);

                        indexer_q = ggml_concat(
                            ctx0, indexer_q_pe, indexer_q_nope, 0);
                        indexer_q = ggml_cast(ctx0, indexer_q, GGML_TYPE_BF16);
                        cb(indexer_q, "lsa_indexer_q", il);

                        // LONGCAT_LSA_DUMP_SURFACE (measurement round,
                        // dump-only): lsa_indexer_q is [128, 16, n_tokens];
                        // a first-owner 2D contiguous copy (head-major rows,
                        // h*128+d) is made purely for the dump, evaluation
                        // forced, nothing consumed downstream. Values
                        // byte-identical; zero arithmetic change.
                        if (il == 0) {
                            ggml_tensor * indexer_q_dump = ggml_cont_2d(
                                ctx0, indexer_q,
                                n_indexer_head * n_embd_indexer_head,
                                n_tokens);
                            cb(indexer_q_dump, "lsa_indexer_q_2d", il);
                            ggml_build_forward_expand(gf, indexer_q_dump);
                        }

                        ggml_tensor * indexer_weights =
                            ggml_mul_mat(ctx0, model.layers[il].indexer_proj, cur);
                        indexer_weights = ggml_scale(
                            ctx0, indexer_weights,
                            1.0f / sqrtf(float(n_embd_indexer_head * n_indexer_head)));
                        cb(indexer_weights, "lsa_indexer_weights", il);

                        ggml_tensor * indexer_k_cached = mctx_lid->get_k(ctx0, il);

                        const auto n_stream = indexer_k_cached->ne[3];
                        indexer_q = ggml_view_4d(
                            ctx0, indexer_q,
                            indexer_q->ne[0], indexer_q->ne[1],
                            indexer_q->ne[2] / n_stream, n_stream,
                            indexer_q->nb[1], indexer_q->nb[2],
                            indexer_q->nb[3] / n_stream, 0);
                        indexer_weights = ggml_view_4d(
                            ctx0, indexer_weights,
                            indexer_weights->ne[0],
                            indexer_weights->ne[1] / n_stream,
                            indexer_weights->ne[2], n_stream,
                            indexer_weights->nb[1],
                            indexer_weights->nb[2] / n_stream,
                            indexer_weights->nb[3] / n_stream, 0);

                        indexer_q = ggml_permute(ctx0, indexer_q, 0, 2, 1, 3);
                        indexer_k_cached =
                            ggml_permute(ctx0, indexer_k_cached, 0, 2, 1, 3);

                        ggml_tensor * indexer_kq =
                            ggml_mul_mat(ctx0, indexer_k_cached, indexer_q);
                        ggml_mul_mat_set_prec(indexer_kq, GGML_PREC_F32);
                        cb(indexer_kq, "lsa_indexer_kq", il);

                        indexer_kq = ggml_cont(
                            ctx0, ggml_permute(ctx0, indexer_kq, 2, 1, 0, 3));

                        ggml_tensor * indexer_score =
                            ggml_relu(ctx0, indexer_kq);
                        indexer_score =
                            ggml_mul(ctx0, indexer_score, indexer_weights);
                        indexer_score =
                            ggml_sum_rows(ctx0, indexer_score);
                        indexer_score = ggml_cont(
                            ctx0, ggml_permute(ctx0, indexer_score, 2, 1, 0, 3));

                        indexer_score = ggml_add(
                            ctx0, indexer_score, inp_attn_dsa->get_kq_mask_lid());
                        cb(indexer_score, "lsa_indexer_score", il);

                        const uint32_t n_top_k =
                            std::min<uint32_t>(
                                (uint32_t) indexer_score->ne[0],
                                n_indexer_top_k);
                        GGML_ASSERT(n_top_k == n_indexer_top_k);

                        top_k = ggml_cont(
                            ctx0, ggml_top_k(ctx0, indexer_score, n_top_k));
                        GGML_ASSERT(top_k != nullptr);
                        GGML_ASSERT(top_k->ne[0] == n_indexer_top_k);

                        prev_top_k = top_k;
                        LLAMA_LOG_DEBUG(
                            "LONGCAT_LSA_AUDIT owner block=%d n_kv=%u "
                            "top_k=%lld tensor=%p\n",
                            il,
                            n_kv_lid,
                            (long long) top_k->ne[0],
                            (void *) top_k);
                        cb(top_k, "lsa_top_k_owner", il);
                    } else {
                        // LONGCAT_LSA_DEVIATION (instrumentation-only, review
                        // 2026-08-18): the historical marker
                        // cb(indexer_k, "lsa_full_owner", il) is removed - in
                        // the sparse-inactive regime it renamed the canonical
                        // lsa_indexer_k-<il> surface required by the planned
                        // <=512 below-threshold indexer-K dump proof. Values,
                        // owner/reuse semantics, and every graph operation are
                        // unchanged; >2048 structural evidence lives in the
                        // LONGCAT_LSA_AUDIT logs.
                        prev_top_k = nullptr;
                    }
                } else {
                    if (sparse_active) {
                        GGML_ASSERT(prev_top_k != nullptr &&
                            "LongCat CLI reuse block must follow an owner top-K");
                        top_k = prev_top_k;
                        GGML_ASSERT(top_k->ne[0] == n_indexer_top_k);

                        LLAMA_LOG_DEBUG(
                            "LONGCAT_LSA_AUDIT reuse block=%d owner_block=%d "
                            "n_kv=%u top_k=%lld tensor=%p\n",
                            il,
                            il - 1,
                            n_kv_lid,
                            (long long) top_k->ne[0],
                            (void *) top_k);
                        cb(top_k, "lsa_top_k_reuse", il);
                    } else {
                        // LONGCAT_LSA_DEVIATION (instrumentation-only, review
                        // 2026-08-18): the historical marker
                        // cb(cur, "lsa_full_reuse", il) is removed - it renamed
                        // the standing attn_norm-<il> tensor of every odd block
                        // and broke the name-keyed attn_norm-3 full-sequence
                        // dump (the Stage-3 inventory failure). Values and
                        // semantics unchanged; >2048 structural evidence lives
                        // in the LONGCAT_LSA_AUDIT logs.
                        GGML_ASSERT(prev_top_k == nullptr);
                    }
                }
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
            if (il == 0) {
                // LONGCAT_ATTN0_KV_BF16_SEMANTICS_DIAGNOSTIC (Experiment A/D1):
                //
                // HF kv_a_proj_with_mqa is a BF16 nn.Linear: its full 576-wide
                // output is BF16 *before* the [512, 64] split
                // (modeling_longcat_flash.py:419-420). The authoritative
                // Blackwell comparison showed all 294912/294912 elements of
                // this F32 GEMM output equal the HF oracle after one
                // round-to-nearest-even BF16 rounding (STATUS_2026-08-17.md).
                // Round the full tensor here so both split views -- kv_cmpr
                // and the RoPE input k_pe -- read the HF lattice, then widen
                // back to F32 so view strides and the downstream graph are
                // unchanged.
                kv_cmpr_pe = ggml_cast(ctx0, kv_cmpr_pe, GGML_TYPE_BF16);
                kv_cmpr_pe = ggml_cast(ctx0, kv_cmpr_pe, GGML_TYPE_F32);
            } else {
                // LONGCAT_MLA_BF16_OUTPUT_BOUNDARY (production, il >= 1, A4):
                // same full-576 boundary as the il == 0 diagnostic branch
                // above, which is preserved literally per the reviewed plan.
                // The block-2 MLA walk proved this GEMM output HF-equivalent
                // at the BF16 boundary from all-exact inputs (294912/294912
                // after RNE rounding); both split views read the HF lattice.
                kv_cmpr_pe = ggml_cast(ctx0, kv_cmpr_pe, GGML_TYPE_BF16);
                kv_cmpr_pe = ggml_cast(ctx0, kv_cmpr_pe, GGML_TYPE_F32);
            }
            // LONGCAT_MLA_STAGE_SURFACE: at il == 0 this callback observes the
            // post-roundtrip tensor -- the dumped surface must represent the
            // gated semantics, not the pre-cast intermediate.
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
            const char * rope_oracle_dir = std::getenv("LONGCAT_ROPE_ORACLE_DIR");
            if (il == 0 && rope_oracle_dir != nullptr && rope_oracle_dir[0] != '\0') {
                // LONGCAT_ROPE_ORACLE (Experiment R1): real in-graph BF16
                // rotation with the captured HF cos/sin as graph inputs.
                //
                // Source-proven C2 semantics (modeling_longcat_flash.py:57-62
                // analog for rope, :322-331): BF16 inputs, BF16 cos/sin, and a
                // BF16 store-round after EVERY binary op -- the unique cast
                // ordering that reproduces the HF function with 0 mismatches
                // offline (single-final-round misses by 162146, no-final-round
                // by 872330). ggml's own F32 trig cannot be used: its values
                // provably round to different BF16 in 3377/16384 sin elements.
                //
                // ggml_rope_ext, shared RoPE kernels, and production behavior
                // are untouched; il > 0 takes the standard path below.
                auto inp_ro = std::make_unique<llm_graph_input_longcat_rope_oracle>(rope_oracle_dir);

                inp_ro->cos_half = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 32, 1, n_tokens);
                ggml_set_input(inp_ro->cos_half);
                ggml_set_name(inp_ro->cos_half, "longcat_rope_cos_half");

                inp_ro->sin_half = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 32, 1, n_tokens);
                ggml_set_input(inp_ro->sin_half);
                ggml_set_name(inp_ro->sin_half, "longcat_rope_sin_half");

                // lossless: oracle values are on the BF16 lattice
                ggml_tensor * cos_b = ggml_cast(ctx0, inp_ro->cos_half, GGML_TYPE_BF16);
                ggml_tensor * sin_b = ggml_cast(ctx0, inp_ro->sin_half, GGML_TYPE_BF16);

                res->add_input(std::move(inp_ro));

                const size_t es = ggml_type_size(GGML_TYPE_BF16);

                auto build_hf_bf16_rope = [&](ggml_tensor * x) -> ggml_tensor * {
                    const int64_t H = x->ne[1];
                    const int64_t T = x->ne[2];

                    // widen-lossless BF16 input (x is F32 on the BF16 lattice)
                    ggml_tensor * xb = ggml_cast(ctx0, ggml_cont(ctx0, x), GGML_TYPE_BF16);

                    // rows [64] -> [2, 32]: ne0 = 2 fastest, so columns are
                    // the interleaved pairs (x_{2j}, x_{2j+1})
                    xb = ggml_reshape_4d(ctx0, xb, 2, 32, H, T);

                    ggml_tensor * x1 = ggml_cont(ctx0,
                        ggml_view_4d(ctx0, xb, 1, 32, H, T, xb->nb[1], xb->nb[2], xb->nb[3], 0));
                    ggml_tensor * x2 = ggml_cont(ctx0,
                        ggml_view_4d(ctx0, xb, 1, 32, H, T, xb->nb[1], xb->nb[2], xb->nb[3], es));

                    x1 = ggml_reshape_3d(ctx0, x1, 32, H, T);
                    x2 = ggml_reshape_3d(ctx0, x2, 32, H, T);

                    // C2: every binary op stores BF16 (f32 internal per element)
                    ggml_tensor * ev = ggml_sub(ctx0,
                        ggml_mul(ctx0, x1, cos_b), ggml_mul(ctx0, x2, sin_b));
                    ggml_tensor * od = ggml_add(ctx0,
                        ggml_mul(ctx0, x2, cos_b), ggml_mul(ctx0, x1, sin_b));

                    // re-interleave: pair-concat on dim 0 (ne0 fastest)
                    ev = ggml_reshape_4d(ctx0, ev, 1, 32, H, T);
                    od = ggml_reshape_4d(ctx0, od, 1, 32, H, T);

                    ggml_tensor * out = ggml_concat(ctx0, ev, od, 0);
                    out = ggml_reshape_3d(ctx0, out, 64, H, T);

                    // widen for the downstream F32 graph (lossless)
                    return ggml_cast(ctx0, out, GGML_TYPE_F32);
                };

                q_pe = build_hf_bf16_rope(q_pe);
                cb(q_pe, "q_pe", il);

                k_pe = build_hf_bf16_rope(k_pe);
                cb(k_pe, "k_pe", il);
            } else {
                q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                                     freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                cb(q_pe, "q_pe", il);

                k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                                     freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                cb(k_pe, "k_pe", il);
            }

            if (il == 0 || il == 2) {
                // LONGCAT_ATTN_PATH_STAGE_SURFACE (localization, dump-only):
                //
                // Post-RoPE Q/K under distinct names -- the existing "q_pe" /
                // "k_pe" labels collide between the pre- and post-RoPE
                // tensors, and the dump helper rejects ne[2] != 1, so 2D
                // contiguous copies are made purely for the dump. The copies
                // feed nothing downstream; ggml_build_forward_expand forces
                // their evaluation. Values are byte-identical to the graph
                // tensors -- no arithmetic is altered. il == 2 is included
                // for the block-2 MLA walk (same dump-only pattern).
                ggml_tensor * q_pe_dump = ggml_cont_2d(
                    ctx0, q_pe, n_embd_head_qk_rope * n_head, n_tokens);
                cb(q_pe_dump, "q_pe_rope", il);
                ggml_build_forward_expand(gf, q_pe_dump);

                ggml_tensor * k_pe_dump = ggml_cont_2d(
                    ctx0, k_pe, n_embd_head_qk_rope, n_tokens);
                cb(k_pe_dump, "k_pe_rope", il);
                ggml_build_forward_expand(gf, k_pe_dump);
            }

            // normalize compressed KV
            if (il == 0) {
                // LONGCAT_ATTN0_KVA_NORM_EPS_DIAGNOSTIC:
                // HF LongCat kv_a_layernorm uses the RMSNorm default eps=1e-6
                // (LongcatFlashRMSNorm.__init__ default at
                // modeling_longcat_flash.py:49; constructed without an eps
                // override at :367). eps is fixed by source: the offline sweep
                // excludes 1e-5 but cannot distinguish 1e-6 from 1e-8.
                kv_cmpr = ggml_rms_norm(ctx0, kv_cmpr, 1.0e-6f);

                // LONGCAT_ATTN0_KV_BF16_SEMANTICS_DIAGNOSTIC (Experiment A/D2):
                //
                // LongcatFlashRMSNorm casts the normalized activation back to
                // BF16 before multiplying by its BF16 weight, and the product
                // is stored in BF16 (modeling_longcat_flash.py:57-62). Mirror
                // of the proven Q-side pattern above, which reproduces the HF
                // oracle byte-exactly at width 1536.
                kv_cmpr = ggml_cast(ctx0, kv_cmpr, GGML_TYPE_BF16);
                kv_cmpr = ggml_cast(ctx0, kv_cmpr, GGML_TYPE_F32);

                kv_cmpr = ggml_mul(
                    ctx0,
                    kv_cmpr,
                    model.layers[il].attn_kv_a_norm);

                // RMSNorm output is BF16.
                kv_cmpr = ggml_cast(ctx0, kv_cmpr, GGML_TYPE_BF16);
                // LONGCAT_MLA_STAGE_SURFACE: HF kv_a_layernorm boundary, before
                // the MLA LoRA kv scaling below. The callback observes the
                // final rounded tensor; the dump helper widens BF16 to F32.
                cb(kv_cmpr, "kv_a_norm", il);

                // Widen the rounded result for the existing F32 scaling and
                // cache path, exactly as the Q path widens after q_b_proj.
                kv_cmpr = ggml_cast(ctx0, kv_cmpr, GGML_TYPE_F32);
            } else {
                kv_cmpr = build_norm(
                    kv_cmpr,
                    model.layers[il].attn_kv_a_norm,
                    nullptr,
                    LLM_NORM_RMS,
                    il);
                // LONGCAT_MLA_STAGE_SURFACE: HF kv_a_layernorm boundary, before
                // the MLA LoRA kv scaling below. Renamed because the pre-norm
                // view at the top of this block already uses "kv_cmpr".
                cb(kv_cmpr, "kv_a_norm", il);
            }

            // MLA LoRA scaling: kv_cmpr *= sqrt(hidden_size / kv_lora_rank)
            kv_cmpr = ggml_scale(ctx0, kv_cmpr, mla_scale_kv);
            if (il == 0) {
                // LONGCAT_ATTN0_KV_BF16_SEMANTICS_DIAGNOSTIC (Experiment B/D3):
                //
                // HF scales the BF16 k_pass tensor, so the scaled result is
                // stored in BF16 (modeling_longcat_flash.py:426), mirroring
                // the accepted Q-side post-scale rounding above. No capture
                // surface gates this boundary; its effect lands in
                // o_proj/residual and is measured as delta(A->B). Widen for
                // the existing F32 cache path. The kv_cmpr_scaled label lands
                // on the rounded tensor (graph label only, not a dump target).
                kv_cmpr = ggml_cast(ctx0, kv_cmpr, GGML_TYPE_BF16);
                kv_cmpr = ggml_cast(ctx0, kv_cmpr, GGML_TYPE_F32);
            } else {
                // LONGCAT_MLA_BF16_OUTPUT_BOUNDARY (production, il >= 1, A5):
                // same post-scale boundary as the il == 0 Experiment B/D3
                // branch above, preserved literally per the reviewed plan.
                // The hex reset proved the scale HF-equivalent at the BF16
                // output boundary from all-exact inputs (262144/262144 after
                // RNE rounding; scale constant f32-bit-identical 0x401cc471).
                kv_cmpr = ggml_cast(ctx0, kv_cmpr, GGML_TYPE_BF16);
                kv_cmpr = ggml_cast(ctx0, kv_cmpr, GGML_TYPE_F32);
            }
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

            if (longcat_lsa) {
                GGML_ASSERT(inp_attn_dsa != nullptr);
                const uint32_t n_kv_lid =
                    inp_attn_dsa->mctx->get_lid()->get_n_kv();
                const bool sparse_expected =
                    n_kv_lid > n_indexer_top_k;
                GGML_ASSERT(sparse_expected == (top_k != nullptr));

                cur = build_attn(inp_attn_dsa,
                        model.layers[il].wo, NULL, model.layers[il].wo_s,
                        Qcur, Kcur, Vcur, nullptr, nullptr,
                        model.layers[il].wv_b, top_k, kq_scale, il);
            } else {
                cur = build_attn(inp_attn_k,
                        model.layers[il].wo, NULL, model.layers[il].wo_s,
                        Qcur, Kcur, Vcur, nullptr, nullptr, model.layers[il].wv_b, kq_scale, il);
            }

            // LONGCAT_MLA_STAGE_SURFACE: HF o_proj boundary.
            //
            // build_attn emits "kqv_out" BEFORE applying wo, so kqv_out-0 is
            // the pre-projection attention output and is n_head*v_head_dim
            // (4096) wide. HF o_proj output is n_embd (3072) wide. This is the
            // post-wo, pre-residual surface and the only one comparable to it.
            cb(cur, "attn_out", il);
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
        // LONGCAT_TRUNK_FFN_NORM_HF_SEMANTICS (production, ALL il, N2): HF
        // post_attention_layernorm is byte-closed as
        // bf16(bf16(x*rsqrt(var+1e-5))*w) at ffn_norm-2 under an exact
        // predecessor (1572864/1572864; runtime-verified eps 1e-5), and HF
        // construction is role-uniform across the trunk. This site was
        // uncorrected at EVERY layer including il == 0 (the block-0
        // corrective stack never touched the FFN half); the reviewed
        // trunk-norm audit scopes N2 to il = 0..27. eps unchanged
        // (f_norm_rms_eps = 1e-5, matching HF).
        cur = ggml_rms_norm(ctx0, ffn_inp, hparams.f_norm_rms_eps);

        cur = ggml_cast(ctx0, cur, GGML_TYPE_BF16);
        cur = ggml_cast(ctx0, cur, GGML_TYPE_F32);

        cur = ggml_mul(ctx0, cur, model.layers[il].ffn_norm);

        cur = ggml_cast(ctx0, cur, GGML_TYPE_BF16);
        cur = ggml_cast(ctx0, cur, GGML_TYPE_F32);
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
