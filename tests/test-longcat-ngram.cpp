#include "llama-graph.h"
#include "testing.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <array>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

static constexpr int64_t LONGCAT_VOCAB = 131072;
static constexpr int64_t LONGCAT_M = 78 * LONGCAT_VOCAB;
static constexpr int32_t LONGCAT_SPLIT = 4;
static constexpr int32_t LONGCAT_NEIGHBOR = 4;
static constexpr int32_t LONGCAT_EMBEDDERS = 12;
static constexpr int32_t LONGCAT_EOS = 2;

struct ngram_expect {
    std::string name;
    std::vector<int32_t> tokens;
    int pos;
    int32_t ng2;
    int32_t ng3;
    int32_t ng4;
};

static llama_ubatch make_ubatch(
        const std::vector<int32_t> & tokens,
        const std::vector<int32_t> & positions,
        const std::vector<int32_t> & seq_ids) {
    llama_ubatch ubatch = {};
    ubatch.data = std::make_shared<llama_ubatch::data_t>();
    auto & data = *ubatch.data;

    const uint32_t n_tokens = (uint32_t) tokens.size();
    data.token.assign(tokens.begin(), tokens.end());
    data.pos.assign(positions.begin(), positions.end());
    data.n_seq_id.assign(n_tokens, 1);
    data.seq_id_data.assign(seq_ids.begin(), seq_ids.end());
    data.seq_id.resize(n_tokens);
    for (uint32_t i = 0; i < n_tokens; ++i) {
        data.seq_id[i] = &data.seq_id_data[i];
    }

    ubatch.n_tokens = n_tokens;
    ubatch.n_seq_tokens = n_tokens;
    ubatch.n_seqs = 1;
    ubatch.n_seqs_unq = 1;
    ubatch.n_pos = 1;
    ubatch.token = data.token.data();
    ubatch.pos = data.pos.data();
    ubatch.n_seq_id = data.n_seq_id.data();
    ubatch.seq_id = data.seq_id.data();
    return ubatch;
}

class production_ngram_runner {
public:
    production_ngram_runner() {
        struct ggml_init_params params = {
            /* .mem_size   = */ ggml_tensor_overhead() * 32,
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true,
        };
        ctx = ggml_init(params);
        GGML_ASSERT(ctx != nullptr);

        input = std::make_unique<llm_graph_input_ngram>(
            LONGCAT_EMBEDDERS, LONGCAT_NEIGHBOR, LONGCAT_SPLIT,
            LONGCAT_VOCAB, LONGCAT_M, LONGCAT_EOS, &history);
        for (int i = 0; i < LONGCAT_EMBEDDERS; ++i) {
            input->ngram_ids[i] = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 16);
        }
        backend = ggml_backend_cpu_init();
        GGML_ASSERT(backend != nullptr);
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        GGML_ASSERT(buffer != nullptr);
    }

    ~production_ngram_runner() {
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
        ggml_free(ctx);
    }

    std::array<int32_t, 3> append_one(int32_t token, int32_t pos, int32_t seq_id = 0) {
        const auto ubatch = make_ubatch({ token }, { pos }, { seq_id });
        input->set_input(&ubatch);
        return read_row(0);
    }

    std::array<int32_t, 3> append_many(
            const std::vector<int32_t> & tokens,
            const std::vector<int32_t> & positions,
            int row,
            int32_t seq_id = 0) {
        const std::vector<int32_t> seq_ids(tokens.size(), seq_id);
        const auto ubatch = make_ubatch(tokens, positions, seq_ids);
        input->set_input(&ubatch);
        return read_row(row);
    }

    std::array<int32_t, 3> run_prefill(const std::vector<int32_t> & tokens, int pos, int32_t seq_id = 0) {
        std::vector<int32_t> positions(tokens.size());
        std::vector<int32_t> seq_ids(tokens.size(), seq_id);
        for (size_t i = 0; i < tokens.size(); ++i) {
            positions[i] = (int32_t) i;
        }
        const auto ubatch = make_ubatch(tokens, positions, seq_ids);
        input->set_input(&ubatch);
        return read_row(pos);
    }

    void seed_history(int32_t seq_id, const std::vector<std::pair<llama_pos, llama_token>> & entries) {
        auto & hist = history[seq_id];
        hist.clear();
        for (const auto & entry : entries) {
            hist.push_back(entry);
        }
    }

    std::string history_trace(int32_t seq_id = 0) const {
        const auto it = history.find(seq_id);
        if (it == history.end()) {
            return "[]";
        }

        std::ostringstream out;
        out << "[";
        bool first = true;
        for (const auto & entry : it->second) {
            if (!first) {
                out << ",";
            }
            out << entry.first << ":" << entry.second;
            first = false;
        }
        out << "]";
        return out.str();
    }

    size_t history_size(int32_t seq_id = 0) const {
        const auto it = history.find(seq_id);
        return it == history.end() ? 0 : it->second.size();
    }

private:
    std::array<int32_t, 3> read_row(int row) const {
        std::array<int32_t, 3> ids = {};
        ggml_backend_tensor_get(input->ngram_ids[0], &ids[0], row * sizeof(int32_t), sizeof(int32_t));
        ggml_backend_tensor_get(input->ngram_ids[4], &ids[1], row * sizeof(int32_t), sizeof(int32_t));
        ggml_backend_tensor_get(input->ngram_ids[8], &ids[2], row * sizeof(int32_t), sizeof(int32_t));
        return ids;
    }

    ggml_context * ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    llm_ngram_token_history history;
    std::unique_ptr<llm_graph_input_ngram> input;
};

static void test_production_ngram(testing & t) {
    const std::vector<ngram_expect> cases = {
        { "no eos",                 {10, 11, 12, 13, 14},       4, 1703950,  736033, 8151286 },
        { "eos token",              {10, 11,  2, 14, 15, 16},   2, 1441794, 2339134, 2204702 },
        { "eos plus one",           {10, 11,  2, 14, 15, 16},   3,      14,      14,      14 },
        { "eos plus two",           {10, 11,  2, 14, 15, 16},   4, 1835023, 1835023, 1835023 },
        { "eos plus three",         {10, 11,  2, 14, 15, 16},   5, 1966096, 9356547, 9168347 },
        { "multiple eos turns",     { 3,  2,  4,  5,  2, 6, 7, 8}, 7, 917512, 5545366, 5464710 },
        { "eos at zero",            { 2, 10, 11, 12},           3, 1441804, 2339144, 2204712 },
        { "consecutive eos",        {10,  2,  2, 11, 12, 13},   5, 1572877, 6649401, 6501529 },
        { "shorter than shift",     {10, 11},                   1, 1310731, 1310731, 1310731 },
        { "speculative replacement",{10, 11, 99, 100},          3, 2752611, 7829127, 6204250 },
    };

    for (const auto & tc : cases) {
        production_ngram_runner runner;
        const auto ids = runner.run_prefill(tc.tokens, tc.pos);
        t.assert_equal(tc.name + " ng2", tc.ng2, ids[0]);
        t.assert_equal(tc.name + " ng3", tc.ng3, ids[1]);
        t.assert_equal(tc.name + " ng4", tc.ng4, ids[2]);
        t.assert_true(tc.name + " bounded history", runner.history_size() <= (size_t) (LONGCAT_NEIGHBOR - 1 + tc.tokens.size()));
    }

    production_ngram_runner runner;
    runner.append_one(10, 0);
    runner.append_one(11, 1);
    runner.append_one(12, 2);
    const auto replacement = runner.append_one(99, 2);
    t.assert_equal("rollback replacement ng2", 1441891, replacement[0]);
    t.assert_equal("rollback replacement ng3", 2339231, replacement[1]);
    t.assert_equal("rollback replacement ng4", 2204799, replacement[2]);
    t.assert_true("rollback replacement bounded history", runner.history_size() <= (size_t) LONGCAT_NEIGHBOR);

    production_ngram_runner speculative;
    speculative.seed_history(0, { { 97, 97 }, { 98, 98 }, { 99, 99 } });
    t.assert_equal("speculative before first call", std::string("[97:97,98:98,99:99]"), speculative.history_trace());
    speculative.append_many({ 100, 888 }, { 100, 101 }, 1);
    t.assert_equal("speculative after draft ubatch", std::string("[97:97,98:98,99:99,100:100,101:888]"), speculative.history_trace());
    const auto repl = speculative.append_one(101, 101);
    t.assert_equal("speculative after rollback replacement", std::string("[98:98,99:99,100:100,101:101]"), speculative.history_trace());
    t.assert_equal("replacement uses p for ng2", 2883684, repl[0]);
    t.assert_equal("replacement uses p and p-1 for ng3", 7677892, repl[1]);
    t.assert_equal("replacement uses p p-1 p-2 for ng4", 4140793, repl[2]);
    t.assert_true("speculative lifecycle bounded history", speculative.history_size() <= (size_t) (2 * LONGCAT_NEIGHBOR));

    production_ngram_runner multi;
    multi.append_one(10, 0, 0);
    multi.append_one(2, 1, 0);
    const auto seq_a = multi.append_one(20, 2, 0);
    const auto seq_b = multi.append_one(32, 2, 1);
    t.assert_equal("independent sequence a ng2", 20, seq_a[0]);
    t.assert_equal("independent sequence b ng2", 32, seq_b[0]);
}

int main() {
    testing t(std::cout);
    t.test("longcat production ngram", test_production_ngram);
    return t.summary();
}
