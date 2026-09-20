// SmokeTests.swift - trivial green test proving the package builds and tests run
// Copyright 2026 Vincent Gourbin

import Testing
@testable import YuE2Core

@Suite("Smoke")
struct SmokeTests {
    @Test func versionIsSet() {
        #expect(YuE2.version == "0.1.0")
    }
}
