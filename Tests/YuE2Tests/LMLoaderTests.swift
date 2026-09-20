// LMLoaderTests.swift - tier-1 checkpoint coverage for LMWeightLoader (T-2.3)
// Copyright 2026 Vincent Gourbin

import Foundation
import Testing
@testable import YuE2Core

@Suite("LMLoader")
struct LMLoaderTests {
    /// Tiny loading itself is already covered by `BackboneParityTests.loadsTinyWithFullCoverage`;
    /// this exercises `LMWeightLoader.load(directory:config:)` end-to-end instead.
    @Test func loaderLoadsTinyCheckpointFromDirectory() throws {
        let dir = try makeTinyLMDirectory()
        _ = try LMWeightLoader.load(directory: dir, config: tinyLMConfig())
    }

    @Test func loaderRejectsCheckpointMissingLMHead() throws {
        let dir = try makeTinyLMDirectory(dropping: "lm_head.weight")
        #expect(throws: YuE2Error.self) {
            try LMWeightLoader.load(directory: dir, config: tinyLMConfig())
        }
    }
}
