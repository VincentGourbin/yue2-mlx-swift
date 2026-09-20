// HistoryView.swift - past runs table + CSV export (T-4.7, O10)
// Copyright 2026 Vincent Gourbin

import AppKit
import SwiftUI
import UniformTypeIdentifiers

/// One completed `Generate` run, kept in memory only (`GenerationViewModel.history`) — cleared
/// when the app quits, same as the CLI's `.local-runs/` artifacts are gitignored, not persisted
/// as a database. Every number here is copied straight from `SongResult`/`ProfilingSession`,
/// never recomputed (same rule as `BENCHMARKS.md`: source of truth is the profiler).
struct HistoryEntry: Identifiable {
    let id = UUID()
    let date: Date
    let requestID: String
    let quant: String
    let cot: String
    let abcTokens: Int
    let abcTPS: Double
    let semanticTokens: Int
    let semanticTPS: Double
    let narSeconds: Double
    let vaeSeconds: Double
    let audioSeconds: Double
    let e2eSeconds: Double
    let peakProcessMB: Double?
    let peakMLXActiveMB: Double?
    let audioURL: URL?

    var rtf: Double { e2eSeconds > 0 ? audioSeconds / e2eSeconds : 0 }

    static let csvHeader =
        "date,id,quant,cot,abc_tokens,abc_tok_s,semantic_tokens,semantic_tok_s,nar_s,vae_s,audio_s,e2e_s,rtf,peak_process_mb,peak_mlx_active_mb,audio_path"

    var csvRow: String {
        let f = ISO8601DateFormatter()
        func n(_ v: Double) -> String { String(format: "%.3f", v) }
        func opt(_ v: Double?) -> String { v.map { String(format: "%.1f", $0) } ?? "" }
        return [
            f.string(from: date), requestID, quant, cot,
            "\(abcTokens)", n(abcTPS), "\(semanticTokens)", n(semanticTPS),
            n(narSeconds), n(vaeSeconds), n(audioSeconds), n(e2eSeconds), n(rtf),
            opt(peakProcessMB), opt(peakMLXActiveMB), audioURL?.path ?? "",
        ].map { $0.contains(",") ? "\"\($0)\"" : $0 }.joined(separator: ",")
    }
}

struct HistoryView: View {
    let entries: [HistoryEntry]
    @State private var exportError: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Historique (\(entries.count) runs)").font(.headline)
                Spacer()
                Button("Exporter CSV…", action: exportCSV)
                    .disabled(entries.isEmpty)
            }
            .padding([.horizontal, .top])

            if entries.isEmpty {
                ContentUnavailableView(
                    "Aucun run", systemImage: "clock", description: Text("Lance une génération dans l'onglet Génération."))
            } else {
                // At most 9 TableColumns: SwiftUI's TableColumnBuilder overloads top out at 10,
                // so a couple of closely-related fields share one column (label / cot, NAR+VAE).
                Table(entries.sorted(by: { $0.date > $1.date })) {
                    TableColumn("Date") { Text($0.date, style: .time) }
                    TableColumn("ID") { Text($0.requestID) }
                    TableColumn("Quant / cot") { Text("\($0.quant) / \($0.cot)") }
                    TableColumn("ABC tok/s") { Text(String(format: "%.1f", $0.abcTPS)) }
                    TableColumn("Sém. tok/s") { Text(String(format: "%.1f", $0.semanticTPS)) }
                    TableColumn("NAR/VAE s") { entry in
                        Text("\(String(format: "%.1f", entry.narSeconds)) / \(String(format: "%.1f", entry.vaeSeconds))")
                    }
                    TableColumn("Audio s") { Text(String(format: "%.1f", $0.audioSeconds)) }
                    TableColumn("RTF") { Text(String(format: "%.2f", $0.rtf)) }
                    TableColumn("Pic MB") { entry in
                        Text(entry.peakProcessMB.map { mb in String(format: "%.0f", mb) } ?? "—")
                    }
                }
            }
            if let exportError {
                Text(exportError).font(.caption).foregroundStyle(.red).padding(.horizontal)
            }
        }
    }

    private func exportCSV() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.commaSeparatedText]
        panel.nameFieldStringValue = "yue2-bench-history.csv"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        let csv = ([HistoryEntry.csvHeader] + entries.map(\.csvRow)).joined(separator: "\n")
        do {
            try csv.write(to: url, atomically: true, encoding: .utf8)
            exportError = nil
        } catch {
            exportError = "Export échoué : \(error.localizedDescription)"
        }
    }
}
