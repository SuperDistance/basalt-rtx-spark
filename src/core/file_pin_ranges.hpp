#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <map>
#include <utility>
#include <vector>

namespace basalt::core::detail {

// Select page-aligned intervals in rank order. Register each merged interval at the rank of its hottest expert.
inline std::vector<std::pair<uint64_t, uint64_t>> plan_file_pin_ranges(
    const std::vector<std::pair<uint64_t, uint64_t>>& ranked, uint64_t budget) {
    struct RangeEnd { uint64_t end; size_t rank; };
    std::map<uint64_t, RangeEnd> ranges;
    for (size_t rank = 0; rank < ranked.size(); ++rank) {
        auto [begin, end] = ranked[rank];
        if (begin >= end) continue;
        auto first = ranges.lower_bound(begin);
        if (first != ranges.begin() && std::prev(first)->second.end >= begin) --first;
        auto last = first;
        uint64_t covered = 0;
        size_t hottest = rank;
        for (; last != ranges.end() && last->first <= end; ++last) {
            begin = std::min(begin, last->first);
            end = std::max(end, last->second.end);
            covered += last->second.end - last->first;
            hottest = std::min(hottest, last->second.rank);
        }
        const uint64_t added = end - begin - covered;
        if (added > budget) continue;
        budget -= added;
        ranges.erase(first, last);
        ranges.emplace(begin, RangeEnd{end, hottest});
    }

    std::vector<std::pair<size_t, std::pair<uint64_t, uint64_t>>> prioritized;
    prioritized.reserve(ranges.size());
    for (const auto& [begin, range] : ranges)
        prioritized.push_back({range.rank, {begin, range.end}});
    std::sort(prioritized.begin(), prioritized.end());
    std::vector<std::pair<uint64_t, uint64_t>> plan;
    plan.reserve(prioritized.size());
    for (const auto& entry : prioritized) plan.push_back(entry.second);
    return plan;
}

}  // namespace basalt::core::detail
