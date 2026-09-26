// MobileLimitsTests.swift - YuE2MemoryManager.mobileLimitsMB sizing and overrides
// Copyright 2026 Vincent Gourbin

import Foundation
import Testing
@testable import YuE2Core

@Suite("MobileLimits", .serialized)
struct MobileLimitsTests {
    /// Without overrides (macOS emulates 6 GB available): cache 1 GB, GC threshold 6 − 1.25 GB.
    @Test func defaultsFollowAvailableMemory() {
        unsetenv("YUE2_CACHE_LIMIT_MB")
        unsetenv("YUE2_MEMORY_LIMIT_MB")
        let (cache, limit) = YuE2MemoryManager.mobileLimitsMB()
        #expect(cache == 1024)
        #expect(limit == 6 * 1024 - 1280)
        #expect(limit >= 3 * 1024)
    }

    @Test func environmentOverridesBothLimits() {
        setenv("YUE2_CACHE_LIMIT_MB", "512", 1)
        setenv("YUE2_MEMORY_LIMIT_MB", "4096", 1)
        defer { unsetenv("YUE2_CACHE_LIMIT_MB"); unsetenv("YUE2_MEMORY_LIMIT_MB") }
        let (cache, limit) = YuE2MemoryManager.mobileLimitsMB()
        #expect(cache == 512)
        #expect(limit == 4096)
    }
}
