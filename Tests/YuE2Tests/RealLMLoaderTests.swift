// RealLMLoaderTests.swift - real-checkpoint LM load, memory and timing budget (tier 2, T-2.3)
// Copyright 2026 Vincent Gourbin

import Foundation
import Testing
@testable import YuE2Core

private func realLMReady() -> Bool {
    guard let dir = modelsDir() else { return false }
    return FileManager.default.fileExists(atPath: dir.appendingPathComponent("YuE2-3B/model.safetensors").path)
}

@Suite("RealLMLoader", .enabled(if: realLMReady()))
struct RealLMLoaderTests {
    /// Loading time isn't asserted strictly (machine-dependent) but is logged for the journal.
    @Test func loadsRealCheckpointWithinMemoryBudget() throws {
        let dir = try #require(modelsDir()).appendingPathComponent("YuE2-3B")
        let config = try YuE2Config.load(from: dir.appendingPathComponent("config.json"))

        let start = Date()
        let model = try LMWeightLoader.load(directory: dir, config: config)
        let seconds = Date().timeIntervalSince(start)
        _ = model

        // `snapshot().activeMB` is binary (1024-based); the fiche's 7.0-7.8 budget is decimal GB
        // (matching model.safetensors' byte count, 7 261 441 640 = 7.26 decimal GB / 6.76 GiB).
        let activeDecimalGB = Double(YuE2MemoryManager.snapshot().activeMB) * 1024 * 1024 / 1_000_000_000
        YuE2Debug.log("LM load: \(seconds) s, \(activeDecimalGB) GB active")
        #expect(activeDecimalGB >= 7.0 && activeDecimalGB <= 7.8, "active \(activeDecimalGB) GB")
        #expect(seconds < 60)
    }
}
