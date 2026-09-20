// MetricsPanel.swift - live/last-run measurements, source unique = swift-mlx-profiler (T-4.7, O10)
// Copyright 2026 Vincent Gourbin

import SwiftUI

/// Everything shown here is copied verbatim from `MLXProfiler`/`ProfilingSession` — the GUI
/// never re-times or re-derives a number itself (same rule as `CLAUDE.md` § Performance work
/// and `BENCHMARKS.md`: one source of truth, the profiler).
struct RunMetrics {
    let ttsSummary: String
    let peakProcessMB: Double?
    let peakMLXActiveMB: Double?
    let phaseReport: String
}

struct MetricsPanel: View {
    let metrics: RunMetrics?
    @State private var showFullReport = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Mesures").font(.headline)
                Spacer()
                if metrics != nil {
                    Button(showFullReport ? "Masquer le rapport complet" : "Rapport complet") {
                        showFullReport.toggle()
                    }
                    .font(.caption)
                }
            }

            if let metrics {
                VStack(alignment: .leading, spacing: 6) {
                    Text(metrics.ttsSummary)
                        .font(.system(size: 12, design: .monospaced))
                        .textSelection(.enabled)

                    HStack(spacing: 16) {
                        memoryTile("Pic MLX actif", metrics.peakMLXActiveMB)
                        memoryTile("Pic process", metrics.peakProcessMB)
                    }

                    if showFullReport {
                        ScrollView {
                            Text(metrics.phaseReport)
                                .font(.system(size: 10, design: .monospaced))
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .frame(maxHeight: 240)
                    }
                }
            } else {
                Text("Aucune mesure — lance une génération.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding()
        .background(RoundedRectangle(cornerRadius: 10).fill(.quaternary.opacity(0.3)))
    }

    private func memoryTile(_ label: String, _ mb: Double?) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label).font(.caption2).foregroundStyle(.secondary)
            Text(mb.map { String(format: "%.0f MB", $0) } ?? "—")
                .font(.system(size: 14, weight: .semibold, design: .rounded))
        }
    }
}
