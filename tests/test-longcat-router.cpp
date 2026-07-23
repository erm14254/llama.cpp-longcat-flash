#include "testing.h"

#include <algorithm>
#include <cmath>
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

int main() {
    testing t(std::cout);
    t.test("longcat router", test_router);
    return t.summary();
}
