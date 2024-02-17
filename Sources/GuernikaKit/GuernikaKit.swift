//
//  GuernikaKit.swift
//  
//
//  Created by Guillermo Cique Fernández on 20/6/23.
//

import Foundation

public enum GuernikaKit {
    public static func loadAll(at baseUrl: URL) -> [any DiffusionPipeline] {
        do {
            return try FileManager.default.contentsOfDirectory(
                at: baseUrl,
                includingPropertiesForKeys: nil,
                options: .skipsHiddenFiles
            ).compactMap { try? load(at: $0) }
                .sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
        } catch {
            return []
        }
    }
    
    public static func load(at baseUrl: URL) throws -> any DiffusionPipeline {
        let unetUrl = baseUrl.appending(path: "Unet.mlmodelc")
        let unetChunk1Url = baseUrl.appending(path: "UnetChunk1.mlmodelc")
        let unetChunk2Url = baseUrl.appending(path: "UnetChunk2.mlmodelc")
        
        let unet: Unet
        if FileManager.default.fileExists(atPath: unetUrl.path(percentEncoded: false)) {
            unet = try Unet(modelAt: unetUrl)
        } else if FileManager.default.fileExists(atPath: unetChunk1Url.path) &&
            FileManager.default.fileExists(atPath: unetChunk2Url.path) {
            unet = try Unet(chunksAt: [unetChunk1Url, unetChunk2Url])
        } else {
            return try loadWuerstchen(at: baseUrl)
        }
        return try loadStableDiffusion(at: baseUrl, unet: unet)
    }
    
    static func loadStableDiffusion(at baseUrl: URL, unet: Unet) throws -> any StableDiffusionPipeline {
        // Device RAM GB
        let physicalMemory: Float64 = Float64(ProcessInfo.processInfo.physicalMemory) / (1024*1024*1024)
        
        let textEncoderUrl = baseUrl.appending(path: "TextEncoder.mlmodelc")
        let encoderUrl = baseUrl.appending(path: "VAEEncoder.mlmodelc")
        let decoderUrl = baseUrl.appending(path: "VAEDecoder.mlmodelc")
        let safetyCheckerUrl = baseUrl.appending(path: "SafetyChecker.mlmodelc")
        
        let decoder = try Decoder(modelAt: decoderUrl, configuration: unet.configuration)
        var safetyChecker: SafetyChecker? = nil
        if FileManager.default.fileExists(atPath: safetyCheckerUrl.path(percentEncoded: false)) {
            safetyChecker = SafetyChecker(modelAt: safetyCheckerUrl, configuration: unet.configuration)
        }
        
        switch unet.function {
        case .refiner:
            let textEncoder2Url = baseUrl.appending(path: "TextEncoder2.mlmodelc")
            let textEncoder2 = try TextEncoder(modelAt: textEncoder2Url, configuration: unet.configuration)
            let encoder = try Encoder(modelAt: encoderUrl, configuration: unet.configuration)
            let reduceMemory = physicalMemory < 9
            return StableDiffusionXLRefinerPipeline(
                baseUrl: baseUrl,
                textEncoder: textEncoder2,
                encoder: encoder,
                unet: unet,
                decoder: decoder,
                safetyChecker: safetyChecker,
                reduceMemory: reduceMemory
            )
        case .instructions:
            let textEncoder = try TextEncoder(modelAt: textEncoderUrl, configuration: unet.configuration)
            let encoder = try Encoder(modelAt: encoderUrl, configuration: unet.configuration)
            let reduceMemory = physicalMemory < 6
            return StableDiffusionPix2PixPipeline(
                baseUrl: baseUrl,
                textEncoder: textEncoder,
                encoder: encoder,
                unet: unet,
                decoder: decoder,
                safetyChecker: safetyChecker,
                reduceMemory: reduceMemory
            )
        default:
            var encoder: Encoder? = nil
            if FileManager.default.fileExists(atPath: encoderUrl.path(percentEncoded: false)) {
                encoder = try Encoder(modelAt: encoderUrl, configuration: unet.configuration)
            }
            let textEncoder = try TextEncoder(modelAt: textEncoderUrl, configuration: unet.configuration)
            var textEncoder2: TextEncoder? = nil
            let textEncoder2Url = baseUrl.appending(path: "TextEncoder2.mlmodelc")
            if FileManager.default.fileExists(atPath: textEncoder2Url.path(percentEncoded: false)) {
                textEncoder2 = try TextEncoder(modelAt: textEncoder2Url, configuration: unet.configuration)
            }
            if let textEncoder2 {
                textEncoder.tokenizer.padToken = "<|endoftext|>"
                let reduceMemory = physicalMemory < 9
                return StableDiffusionXLPipeline(
                    baseUrl: baseUrl,
                    textEncoder: textEncoder,
                    textEncoder2: textEncoder2,
                    encoder: encoder,
                    unet: unet,
                    decoder: decoder,
                    safetyChecker: safetyChecker,
                    reduceMemory: reduceMemory
                )
            }
            let reduceMemory = physicalMemory < 6
            return StableDiffusionMainPipeline(
                baseUrl: baseUrl,
                textEncoder: textEncoder,
                encoder: encoder,
                unet: unet,
                decoder: decoder,
                safetyChecker: safetyChecker,
                reduceMemory: reduceMemory
            )
        }
    }
    
    static func loadWuerstchen(at baseUrl: URL) throws -> WuerstchenPipeline {
        // Device RAM GB
        let physicalMemory: Float64 = Float64(ProcessInfo.processInfo.physicalMemory) / (1024*1024*1024)
        
        let priorUrl = baseUrl.appending(path: "WuerstchenPrior.mlmodelc")
        let priorChunk1Url = baseUrl.appending(path: "WuerstchenPriorChunk1.mlmodelc")
        let priorChunk2Url = baseUrl.appending(path: "WuerstchenPriorChunk2.mlmodelc")
        
        let prior: WuerstchenPrior
        if FileManager.default.fileExists(atPath: priorUrl.path(percentEncoded: false)) {
            prior = try WuerstchenPrior(modelAt: priorUrl)
        } else if FileManager.default.fileExists(atPath: priorChunk1Url.path) &&
                    FileManager.default.fileExists(atPath: priorChunk2Url.path) {
            prior = try WuerstchenPrior(chunksAt: [priorChunk1Url, priorChunk2Url])
        } else {
            throw DiffusionPipelineError.invalidPipeline
        }
        
        let decoderUrl = baseUrl.appending(path: "WuerstchenDecoder.mlmodelc")
        let decoderChunk1Url = baseUrl.appending(path: "WuerstchenDecoderChunk1.mlmodelc")
        let decoderChunk2Url = baseUrl.appending(path: "WuerstchenDecoderChunk2.mlmodelc")
        
        let decoder: WuerstchenDecoder
        if FileManager.default.fileExists(atPath: decoderUrl.path(percentEncoded: false)) {
            decoder = try WuerstchenDecoder(modelAt: decoderUrl)
        } else if FileManager.default.fileExists(atPath: decoderChunk1Url.path) &&
                    FileManager.default.fileExists(atPath: decoderChunk2Url.path) {
            decoder = try WuerstchenDecoder(chunksAt: [decoderChunk1Url, decoderChunk2Url])
        } else {
            throw DiffusionPipelineError.invalidPipeline
        }
        
        let textEncoderUrl = baseUrl.appending(path: "TextEncoder.mlmodelc")
        let textEncoder = try TextEncoder(modelAt: textEncoderUrl, configuration: decoder.configuration)
        let vqganUrl = baseUrl.appending(path: "WuerstchenVQGAN.mlmodelc")
        let vqgan = try WuerstchenVQGAN(modelAt: vqganUrl, configuration: decoder.configuration)
        
        let reduceMemory = physicalMemory < 6
        return WuerstchenPipeline(
            baseUrl: baseUrl,
            textEncoder: textEncoder,
            prior: prior,
            decoder: decoder,
            vqgan: vqgan,
            reduceMemory: reduceMemory
        )
    }
}
