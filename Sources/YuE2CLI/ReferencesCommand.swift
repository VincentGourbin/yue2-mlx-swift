// ReferencesCommand.swift - `yue2 references`: the six pre-qualified configurations
// Copyright 2026 Vincent Gourbin

import ArgumentParser
import Foundation
import YuE2Core

struct ReferencesCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "references",
        abstract: "List the six reference configurations (4/8/16 bits × fast/lean) accepted by `generate --reference`."
    )

    func run() throws {
        for p in YuE2ReferenceProfile.all {
            print(
                "\(p.id.padding(toLength: 11, withPad: " ", startingAt: 0)) quant=\(p.quant.rawValue)"
                    + (p.quantizeHead ? "+head" : "") + " precision=\(p.precision.rawValue)"
                    + " nar=\(p.narCompute.rawValue) compiled=\(p.compiledDecode) vae=\(p.vaePrecision == .fp16 ? "fp16" : "fp32")/\(p.vaeCoreFrames)"
                    + " residency=\(p.releaseWeightsBetweenStages ? "stage" : "all") memory=\(p.memoryProfile == .mobile ? "mobile" : "mac")"
                    + " ode=\(p.odeSteps.map(String.init) ?? "32")"
                    + (p.mobileLimitsMB.map { " cache=\($0.cache)MB threshold=\($0.limit)MB" } ?? ""))
            print("            \(p.summary)")
            print("            weights: " + (p.pack.map { "yue2 download --model \($0.rawValue)  (\(YuE2Pack.repoID), \(String(format: "%.1f", Double($0.approximateBytes) / 1e9)) GB)" } ?? "yue2 download --model lm  (m-a-p/YuE2-3B, bf16, 7.3 GB)"))
        }
    }
}
