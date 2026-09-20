// ProfilingTests.swift - profiler wiring sanity checks (tier 1, T-2.11)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLXProfiler
import Testing

@Suite("Profiling")
struct ProfilingTests {
    @Test func phaseSummariesReportBothPhases() {
        let session = ProfilingSession(config: .singleRun)
        session.beginPhase("alpha", category: .custom)
        Thread.sleep(forTimeInterval: 0.01)
        session.endPhase("alpha", category: .custom)
        session.beginPhase("beta", category: .custom)
        Thread.sleep(forTimeInterval: 0.01)
        session.endPhase("beta", category: .custom)
        session.finish()

        let summaries = session.phaseSummaries()
        let names = Set(summaries.map(\.name))
        #expect(names == ["alpha", "beta"])
        #expect(summaries.allSatisfy { $0.durationMs > 0 })
    }

    @Test func chromeTraceExportIsNonEmpty() {
        let session = ProfilingSession(config: .singleRun)
        session.beginPhase("alpha", category: .custom)
        session.endPhase("alpha", category: .custom)
        session.finish()

        let data = ChromeTraceExporter.export(session: session)
        #expect(!data.isEmpty)
    }
}
