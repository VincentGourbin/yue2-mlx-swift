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
}
