// tests/models/qwen4exp/batch_rows_value_test.cpp - split_rows_by_value: a batch window over its row budget.
//
//   1. every slot keeps its own row, no slot gets more than it wants, and the budget is used up to what is wanted;
//   2. the split's expected accepted drafts equal the best over every split of the budget (exhaustive, small cases);
//   3. rows follow the chances: confident drafts take rows from doubtful ones;
//   4. a slot's scale (how its drafts run against the table) moves rows toward it;
//   5. ties go to the lower slot, so the split is the same every time.
#include "models/qwen4exp/serve/batch_row_cost.hpp"

#include <cmath>
#include <cstdio>
#include <functional>
#include <random>

using basalt::models::qwen4exp::serve::split_rows_by_value;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-66s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

constexpr int kS = 4, kJ = 8;
struct Case {
    int want[kS];
    float scale[kS];
    float p[kS][kJ];   // acc of slot s's draft j
};

double expected(const Case& c, const int* give) {
    double e = 0.0;
    for (int s = 0; s < kS; ++s) {
        double chain = c.scale[s];
        for (int j = 0; j + 1 < give[s]; ++j) {
            chain *= c.p[s][j];
            e += chain;
        }
    }
    return e;
}

double best_split(const Case& c, int budget) {
    double best = -1.0;
    int g[kS];
    std::function<void(int, int)> rec = [&](int s, int left) {
        if (s == kS) {
            best = std::fmax(best, expected(c, g));
            return;
        }
        for (int k = 1; k <= c.want[s] && k <= left - (kS - 1 - s); ++k) {
            g[s] = k;
            rec(s + 1, left - k);
        }
    };
    rec(0, budget);
    return best;
}

void split(const Case& c, int budget, int* give) {
    split_rows_by_value(c.want, c.scale, kS, budget, [&](int s, int j) { return c.p[s][j]; }, give);
}
}  // namespace

int main() {
    std::printf("split_rows_by_value\n");
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(0.05f, 0.99f), sc(0.5f, 1.5f);
    std::uniform_int_distribution<int> w(1, kJ), b(kS, 16);
    bool bounds = true, optimal = true;
    for (int t = 0; t < 20000; ++t) {
        Case c{};
        int sum = 0;
        for (int s = 0; s < kS; ++s) {
            c.want[s] = w(rng);
            sum += c.want[s];
            c.scale[s] = t % 2 ? sc(rng) : 1.0f;
            for (int j = 0; j < kJ; ++j) c.p[s][j] = u(rng);
        }
        const int budget = b(rng);
        int give[kS];
        split(c, budget, give);
        int got = 0;
        for (int s = 0; s < kS; ++s) {
            got += give[s];
            bounds = bounds && give[s] >= 1 && give[s] <= c.want[s];
        }
        bounds = bounds && got == std::min(budget, sum);
        const double e = expected(c, give), best = best_split(c, std::min(budget, sum));
        optimal = optimal && e >= best - 1e-5 * (1.0 + best);
    }
    check(bounds, "1. one row each, at most what is wanted, the budget used");
    check(optimal, "2. the split's expected drafts equal the exhaustive best");

    {   // a confident slot against three doubtful ones, 8 rows
        Case c{};
        for (int s = 0; s < kS; ++s) {
            c.want[s] = 5;
            c.scale[s] = 1.0f;
            for (int j = 0; j < kJ; ++j) c.p[s][j] = s == 2 ? 0.96f : 0.26f;
        }
        int give[kS];
        split(c, 8, give);
        check(give[2] == 5 && give[0] == 1 && give[1] == 1 && give[3] == 1, "3. a confident slot takes the spare rows");
    }
    {   // equal chances, a slot whose drafts run better than the table
        Case c{};
        for (int s = 0; s < kS; ++s) {
            c.want[s] = 4;
            c.scale[s] = s == 3 ? 1.5f : 0.8f;
            for (int j = 0; j < kJ; ++j) c.p[s][j] = 0.66f;
        }
        int give[kS];
        split(c, 8, give);
        check(give[3] >= 3 && give[3] > give[0], "4. a slot running better than the table gets more rows");
    }
    {   // all equal: the lower slots win the ties, every time
        Case c{};
        for (int s = 0; s < kS; ++s) {
            c.want[s] = 4;
            c.scale[s] = 1.0f;
            for (int j = 0; j < kJ; ++j) c.p[s][j] = 0.79f;
        }
        int a[kS], b2[kS];
        split(c, 6, a);
        split(c, 6, b2);
        bool same = true;
        for (int s = 0; s < kS; ++s) same = same && a[s] == b2[s];
        check(same && a[0] == 2 && a[1] == 2 && a[2] == 1 && a[3] == 1, "5. ties go to the lower slot, the same each time");
    }
    std::printf(g_fail == 0 ? "all passed\n" : "%d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
