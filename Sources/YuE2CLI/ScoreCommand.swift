// ScoreCommand.swift - `yue2 score`: a score's layout, and its lead-in / tail shortened (ASK Q16)
// Copyright 2026 Vincent Gourbin

import ArgumentParser
import Foundation
import YuE2Core

/// `yue2 score`: no weights. Prints the layout of a score (sections, first and last sung bar,
/// nominal seconds) as JSON; with `--lead-in` / `--tail`, writes the score with whole groups
/// dropped before the voice / after it, ready for `generate --abc-file`.
struct ScoreCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "score",
        abstract: "Show where a score sings (sections, first and last sung bar, nominal seconds); optionally shorten its instrumental lead-in and tail."
    )

    @Option(name: .long, help: "Path to an ABC score, or to a saved plan directory (its score.abc).")
    var abc: String

    @Option(name: .long, help: "Keep at most this many bars before the voice's first group (whole groups of the plan, usually 4 bars).")
    var leadIn: Int?

    @Option(name: .long, help: "Keep at most this many bars after the voice's last group.")
    var tail: Int?

    @Option(name: .long, help: "Where to write the shortened score (needs --lead-in or --tail).")
    var out: String?

    func run() async throws {
        var url = URL(fileURLWithPath: abc)
        if (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true {
            url = url.appendingPathComponent("score.abc")
        }
        var text = try String(contentsOf: url, encoding: .utf8)
        if leadIn != nil || tail != nil {
            text = try ABCScore.trimming(text, leadIn: leadIn, tail: tail)
            guard let out else { throw ValidationError("--lead-in / --tail need --out") }
            try text.write(to: URL(fileURLWithPath: out), atomically: true, encoding: .utf8)
        }
        guard let layout = ABCScore.layout(of: text) else { throw ValidationError("the score does not parse") }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        print(String(decoding: try encoder.encode(layout), as: UTF8.self))
    }
}
