// CatalogTests.swift - YuE2Model / license invariants (T-1.4)
// Copyright 2026 Vincent Gourbin

import Foundation
import Testing
@testable import YuE2Core

@Suite("Catalog")
struct CatalogTests {
    @Test func lmFilesIncludeWeightsAndTokenizer() {
        #expect(YuE2Model.lm.files.contains("model.safetensors"))
        #expect(YuE2Model.lm.files.contains("qwen.tiktoken"))
        #expect(YuE2Model.lm.files.contains("weights_manifest.json"))
    }

    @Test func vaeFilesAreMinimalAndSharedWithLegacy() {
        let expected = ["config.json", "model.safetensors", "weights_manifest.json"]
        #expect(YuE2Model.vae.files == expected)
        #expect(YuE2Model.vaeLegacy.files == expected)
    }

    @Test func repoIDsAreExact() {
        #expect(YuE2Model.lm.repoID == "m-a-p/YuE2-3B")
        #expect(YuE2Model.vae.repoID == "m-a-p/YuE2-Vae")
        #expect(YuE2Model.vaeLegacy.repoID == "m-a-p/YuE2-Vae-legacy")
    }

    @Test func licenseIsNonCommercial() {
        #expect(!YuE2Model.lm.license.allowsCommercialUse)
        #expect(YuE2Model.lm.license.id == "cc-by-nc-4.0")
        for model in YuE2Model.allCases {
            #expect(model.license == YuE2License.ccByNc4)
        }
    }

    @Test func downloadURLIsExact() {
        let url = ModelDownloader.remoteURL(repoID: "m-a-p/YuE2-3B", remotePath: "model.safetensors")
        #expect(url?.absoluteString == "https://huggingface.co/m-a-p/YuE2-3B/resolve/main/model.safetensors")
    }

    @Test func downloadURLEncodesNestedPaths() {
        let url = ModelDownloader.remoteURL(repoID: "m-a-p/YuE2-3B", remotePath: "examples/tonight-awake.json")
        #expect(url?.absoluteString == "https://huggingface.co/m-a-p/YuE2-3B/resolve/main/examples/tonight-awake.json")
    }

    @Test func verifyReportsMissingFilesOnEmptyDirectory() async throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: dir) }
        let downloader = ModelDownloader(modelsDir: dir, token: nil)
        let status = try await downloader.verify(YuE2Model.vae)
        #expect(status["model.safetensors"] == false)
        #expect(status["config.json"] == false)
    }

    @Test func packsMapToPresetsAndLocalDirectories() {
        #expect(YuE2Pack.repoID == "VincentGOURBIN/yue2-mlx-packs")
        #expect(YuE2Pack.matching(.int4Mixed, quantizeHead: true) == .int4MixedHead)
        #expect(YuE2Pack.matching(.qint8All, quantizeHead: true) == .qint8AllHead)
        #expect(YuE2Pack.matching(.int4, quantizeHead: true) == .int4Head)
        #expect(YuE2Pack.matching(.int4Mixed, quantizeHead: false) == nil)
        #expect(YuE2Pack.matching(.none, quantizeHead: true) == nil)
        for pack in YuE2Pack.allCases {
            #expect(pack.files == ["\(pack.rawValue)/model.safetensors", "\(pack.rawValue)/model.safetensors.sha256"])
            #expect(pack.localDirectory == "YuE2-3B/mlx-prequantized/\(pack.rawValue)")
            #expect(!pack.license.allowsCommercialUse)
        }
        // Every quantized reference profile has a published pack; the bf16 ones have none.
        for profile in YuE2ReferenceProfile.all {
            #expect((profile.pack == nil) == (profile.quant == .none), "\(profile.id)")
        }
        #expect(ModelDownloader.remoteURL(repoID: YuE2Pack.repoID, remotePath: YuE2Pack.int4MixedHead.files[0])?.absoluteString
            == "https://huggingface.co/VincentGOURBIN/yue2-mlx-packs/resolve/main/int4-mixed-head/model.safetensors")
    }

    @Test func packVerifyChecksTheSidecarHash() async throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: dir) }
        let packDir = dir.appendingPathComponent(YuE2Pack.int4Head.localDirectory)
        try FileManager.default.createDirectory(at: packDir, withIntermediateDirectories: true)
        let downloader = ModelDownloader(modelsDir: dir, token: nil)
        #expect(try await downloader.verify(pack: .int4Head) == false)  // nothing there

        try Data("not really weights".utf8).write(to: packDir.appendingPathComponent("model.safetensors"))
        // sha256("not really weights")
        try "e1a4d9b3f9a9c6a0e9b1c1d3f1f1b7d9d8c0b2e5b7a0c6d4e2f8a9b1c3d5e7f9\n".write(
            to: packDir.appendingPathComponent("model.safetensors.sha256"), atomically: true, encoding: .utf8)
        #expect(try await downloader.verify(pack: .int4Head) == false)  // wrong hash

        let real = try ModelDownloader.sha256HexForTesting(of: packDir.appendingPathComponent("model.safetensors"))
        try "\(real)  model.safetensors\n".write(
            to: packDir.appendingPathComponent("model.safetensors.sha256"), atomically: true, encoding: .utf8)
        #expect(try await downloader.verify(pack: .int4Head) == true)
    }
}

