// GenerationView.swift - style/lyrics/cot/seed/abc/cfg/quant form + progress + player (T-4.7, O10)
// Copyright 2026 Vincent Gourbin

import AppKit
import SwiftUI
import YuE2Core

struct GenerationView: View {
    @Bindable var vm: GenerationViewModel

    var body: some View {
        HSplitView {
            form.frame(minWidth: 340, idealWidth: 380)
            VStack(alignment: .leading, spacing: 12) {
                progressSection
                MetricsPanel(metrics: vm.lastMetrics)
                if vm.audioPlayer.url != nil {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Lecture").font(.headline)
                        AudioPlayerView(model: vm.audioPlayer)
                    }
                }
                Spacer()
            }
            .padding()
            .frame(minWidth: 360)
        }
        .onAppear { vm.refreshVariants() }
    }

    private var form: some View {
        Form {
            Section("Modèle") {
                HStack {
                    TextField("Répertoire des checkpoints", text: $vm.modelsDir)
                    Button("Choisir…") { chooseModelsDir() }
                }
                Button("Rafraîchir les variantes") { vm.refreshVariants() }

                Picker("Quantification", selection: $vm.selectedVariant) {
                    ForEach(vm.availableVariants) { variant in
                        Text("\(variant.label) — \(variant.sizeLabel)").tag(Optional(variant))
                    }
                }
                Picker("VAE", selection: $vm.selectedVAE) {
                    ForEach(vm.availableVAEs) { Text($0.label).tag($0) }
                }
            }

            Section("Requête") {
                TextField("id", text: $vm.requestID)
                TextField("style", text: $vm.style, axis: .vertical).lineLimit(2...4)
                TextField("lyrics", text: $vm.lyrics, axis: .vertical).lineLimit(4...10)
                Picker("cot", selection: $vm.cot) {
                    ForEach(CoTMode.allCases, id: \.self) { Text($0.rawValue).tag($0) }
                }
                HStack {
                    Stepper("seed: \(vm.seed)", value: $vm.seed, in: 0...Int.max, step: 1)
                    Button("Aléatoire") { vm.randomizeSeed() }
                }
                TextField("cfg_scale (vide = défaut)", text: $vm.cfgScaleText)
                TextField("durée cible en secondes (vide = auto)", text: $vm.targetDurationText)
            }

            Section {
                HStack {
                    Button(vm.isRunning ? "Génération…" : "Générer") { vm.generate() }
                        .disabled(vm.isRunning || vm.selectedVariant == nil)
                        .keyboardShortcut(.return, modifiers: .command)
                    if vm.isRunning {
                        Button("Annuler", role: .destructive) { vm.cancel() }
                    }
                }
            }
        }
        .formStyle(.grouped)
    }

    private var progressSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Progression").font(.headline)
            switch vm.stage {
            case .idle:
                Text("En attente.").foregroundStyle(.secondary)
            case .loadingModel:
                HStack { ProgressView().controlSize(.small); Text("Chargement du modèle…") }
            case .running(let name):
                VStack(alignment: .leading, spacing: 4) {
                    HStack { ProgressView().controlSize(.small); Text("Étape : \(name)") }
                    Text("ABC : \(vm.abcTokenCount) tokens").font(.caption)
                    Text("Sémantique : \(vm.semanticTokenCount) tokens").font(.caption)
                    if vm.narProgress.total > 0 {
                        ProgressView(value: Double(vm.narProgress.completed), total: Double(vm.narProgress.total))
                        Text("NAR : \(vm.narProgress.completed)/\(vm.narProgress.total)").font(.caption)
                    }
                    if vm.vaeProgress.total > 0 {
                        ProgressView(value: Double(vm.vaeProgress.completed), total: Double(vm.vaeProgress.total))
                        Text("VAE : \(vm.vaeProgress.completed)/\(vm.vaeProgress.total)").font(.caption)
                    }
                }
            case .done:
                Label("Terminé", systemImage: "checkmark.circle.fill").foregroundStyle(.green)
            case .failed(let message):
                Label(message, systemImage: "xmark.octagon.fill").foregroundStyle(.red)
            }
        }
    }

    private func chooseModelsDir() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        vm.modelsDir = url.path
        vm.refreshVariants()
    }
}
