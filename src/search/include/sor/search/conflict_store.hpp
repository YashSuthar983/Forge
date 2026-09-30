#pragma once
struct ConflictStoreStats {
    int added = 0;
    int rejected = 0;
    int evicted = 0;
    size_t live = 0;
    size_t bytes = 0;
    int tightenings = 0;
};
