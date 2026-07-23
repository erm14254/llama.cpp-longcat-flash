#include "llama-graph.h"
#include "testing.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

struct route_result {
    std::vector<int> ids;
    std::vector<double> weights;
    double real = 0.0;
    double identity = 0.0;
};

static std::vector<double> softmax(const std::vector<double> & logits) {
    const double max_logit = *std::max_element(logits.begin(), logits.end());
    std::vector<double> probs(logits.size());
    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum += probs[i];
    }
    for (double & p : probs) {
        p /= sum;
    }
    return probs;
}

static route_result longcat_route(
        const std::vector<double> & logits,
        const std::vector<double> & bias,
        int n_real,
        int top_k,
        double scale) {
    const auto probs = softmax(logits);
    std::vector<int> order(logits.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return probs[a] + bias[a] > probs[b] + bias[b];
    });

    route_result res;
    for (int i = 0; i < top_k; ++i) {
        const int id = order[i];
        const double weight = probs[id] * scale;
        res.ids.push_back(id);
        res.weights.push_back(weight);
        if (id < n_real) {
            res.real += weight;
        } else {
            res.identity += weight;
        }
    }
    return res;
}

static bool near(double a, double b) {
    return std::fabs(a - b) < 1e-6;
}

static void test_router(testing & t) {
    struct router_case {
        std::string name;
        std::vector<double> logits;
        std::vector<double> bias;
        double real;
        double identity;
    };

    const std::vector<double> no_bias(5, 0.0);
    const std::vector<router_case> cases = {
        { "identity highest", {0, 0, 0, 5, 4}, no_bias, 0.0, 5.912626155935248 },
        { "real highest", {5, 4, 3, 0, 0}, no_bias, 5.411305738577755, 0.0 },
        { "mixed topk", {5, 0, 0, 4, 0}, no_bias, 4.3224760735286125, 1.590150082406636 },
        { "bias changes", {0, 4, 3, 0, 0}, {0, 0, 0, 10, 0}, 4.216958708242208, 0.07723629290886723 },
        { "identity outside topk", {5, 4, 3, 2, 1}, no_bias, 5.223181822869405, 0.0 },
        { "extreme", {1000, -1000, 999, 998, -999}, no_bias, 5.459816560977717, 0.0 },
    };

    for (const auto & tc : cases) {
        const auto res = longcat_route(tc.logits, tc.bias, 3, 2, 6.0);
        t.assert_true(tc.name + " real", near(tc.real, res.real));
        t.assert_true(tc.name + " identity", near(tc.identity, res.identity));
        t.assert_true(tc.name + " total bounded", res.real + res.identity <= 6.0 + 1e-9);
    }

    const auto all_equal = longcat_route({0, 0, 0, 0, 0}, no_bias, 3, 2, 6.0);
    t.assert_true("all equal chooses two experts", all_equal.ids.size() == 2);
    t.assert_true("all equal has scaled selected weight", near(all_equal.weights[0], 1.2));
    t.assert_true("all equal has scaled selected weight 2", near(all_equal.weights[1], 1.2));

    const std::vector<std::vector<double>> batch = {
        {5, 0, 0, 4, 0},
        {0, 0, 0, 5, 4},
    };
    double identity = 0.0;
    for (const auto & logits : batch) {
        identity += longcat_route(logits, no_bias, 3, 2, 6.0).identity;
    }
    t.assert_true("multiple token batch identity contribution", identity > 7.0);
}

static route_result run_production_route(
        const std::vector<float> & logits_data,
        const std::vector<float> & bias_data) {
    struct ggml_init_params params = {
        /* .mem_size   = */ 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);

    ggml_tensor * logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5, 1);
    ggml_tensor * bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5);
    auto route = llm_graph_build_longcat_moe_route(ctx, logits, bias, 1, 3, 5, 2, 6.0f);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(gf, route.selected_experts);
    ggml_build_forward_expand(gf, route.selected_real);
    ggml_build_forward_expand(gf, route.weights_real);
    ggml_build_forward_expand(gf, route.identity_weight_sum);

    memcpy(logits->data, logits_data.data(), logits_data.size() * sizeof(float));
    memcpy(bias->data, bias_data.data(), bias_data.size() * sizeof(float));
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    const int32_t * selected = (const int32_t *) route.selected_experts->data;
    const int32_t * selected_real = (const int32_t *) route.selected_real->data;
    const float * weights_real = (const float *) route.weights_real->data;
    const float * identity_sum = (const float *) route.identity_weight_sum->data;

    route_result res;
    for (int i = 0; i < 2; ++i) {
        res.ids.push_back(selected[i]);
        if (selected[i] < 3) {
            res.real += weights_real[i];
        }
        GGML_ASSERT(selected[i] < 3 || selected_real[i] == 0);
        GGML_ASSERT(selected[i] < 3 || weights_real[i] == 0.0f);
    }
    res.identity = identity_sum[0];

    ggml_free(ctx);
    return res;
}

static void test_production_route_helper(testing & t) {
    const std::vector<float> no_bias = { 0, 0, 0, 0, 0 };

    const auto identity = run_production_route({ 0, 0, 0, 5, 4 }, no_bias);
    t.assert_true("production route selects identity first", identity.ids[0] >= 3);
    t.assert_true("production route selects identity second", identity.ids[1] >= 3);
    t.assert_true("identity selections have zero real contribution", near(0.0, identity.real));
    t.assert_true("identity selected mass scaled once", near(5.912626155935248, identity.identity));

    const auto mixed = run_production_route({ 5, 0, 0, 4, 0 }, no_bias);
    t.assert_true("production route mixed real", mixed.ids[0] < 3 || mixed.ids[1] < 3);
    t.assert_true("production route mixed identity", mixed.ids[0] >= 3 || mixed.ids[1] >= 3);
    t.assert_true("mixed real contribution", near(4.3224760735286125, mixed.real));
    t.assert_true("mixed identity contribution excludes unselected", near(1.590150082406636, mixed.identity));

    const auto biased = run_production_route({ 0, 4, 3, 0, 0 }, { 0, 0, 0, 10, 0 });
    t.assert_true("production route bias selects identity", biased.ids[0] >= 3 || biased.ids[1] >= 3);
    t.assert_true("bias does not alter selected weights", near(0.07723629290886723, biased.identity));
}

int main() {
    testing t(std::cout);
    t.test("longcat router", test_router);
    t.test("longcat production route helper", test_production_route_helper);
    return t.summary();
}
