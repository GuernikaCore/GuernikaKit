//
//  StableDiffusionPipeline.swift
//  
//
//  Created by Guillermo Cique Fernández on 25/5/23.
//

import CoreML
import Foundation

public protocol StableDiffusionPipeline: DiffusionPipeline {
    /// Model to generate embeddings for tokenized input text
    var textEncoder: TextEncoder { get }
    
    var overrideTextEncoder: TextEncoder? { get set }
    
    /// Models used to control diffusion models by adding extra conditions
    var controlNets: [ControlNet.Input] { get set }
    
    /// Model used to predict noise residuals given an input, diffusion time step, and conditional embedding
    var unet: Unet { get }
    
    /// Model used to generate final image from latent diffusion process
    var decoder: Decoder { get }
    
    /// Optional model for checking safety of generated image
    var safetyChecker: SafetyChecker? { get }
    
    var disableSafety: Bool { get set }

    /// Option to reduce memory during image generation
    ///
    /// If true, the pipeline will lazily load TextEncoder, Unet, Decoder, and SafetyChecker
    /// when needed and aggressively unload their resources after
    ///
    /// This will increase latency in favor of reducing memory
    var reduceMemory: Bool { get set }
    
    func generateImages(
        input: SampleInput,
        progressHandler: (StableDiffusionProgress) -> Bool
    ) throws -> CGImage?
    
    func decodeToImage(_ latent: MLShapedArray<Float32>) throws -> CGImage?
}

/// Sampling progress details
public struct StableDiffusionProgress {
    public let pipeline: any StableDiffusionPipeline
    public let input: SampleInput
    public let step: Int
    public let stepCount: Int
    public let currentLatentSample: MLShapedArray<Float32>
    public var currentImage: CGImage? {
        do {
            return try pipeline.decodeToImage(currentLatentSample)
        } catch {
            print("Error decoding progress images", error.localizedDescription)
            return nil
        }
    }
}

extension StableDiffusionPipeline {
    /// Expected text encoder max input length
    public var maximumTokensAllowed: Int { textEncoder.maxInputLength }
    /// Unet sample size
    public var sampleSize: CGSize { unet.sampleSize }
    public var supportsControlNet: Bool { unet.supportsControlNet }
    
    /// Reports whether this pipeline can perform safety checks
    public var canSafetyCheck: Bool {
        safetyChecker != nil
    }
    
    public var configuration: MLModelConfiguration { unet.configuration }
    
    public func addControlNet(_ controlNet: ControlNet) throws {
        guard controlNet.hiddenSize == unet.hiddenSize else {
            throw StableDiffusionError.incompatibleControlNet
        }
        controlNets.append(.init(controlNet: controlNet))
    }
    
    public func generateImages(input: SampleInput) throws -> CGImage? {
        try generateImages(input: input, progressHandler: { _ in true })
    }

    func hiddenStates(prompt: String, negativePrompt: String) throws -> MLShapedArray<Float32> {
        // Encode the input prompt as well as a blank unconditioned input
        let promptEmbedding: MLShapedArray<Float32>
        let blankEmbedding: MLShapedArray<Float32>
        if let overrideTextEncoder {
            (_, promptEmbedding) = try overrideTextEncoder.encode(prompt)
            (_, blankEmbedding) = try overrideTextEncoder.encode(negativePrompt)
        } else {
            (_, promptEmbedding) = try textEncoder.encode(prompt)
            (_, blankEmbedding) = try textEncoder.encode(negativePrompt)
        }
        
        if reduceMemory {
            textEncoder.unloadResources()
            overrideTextEncoder?.unloadResources()
        }

        // Convert to Unet hidden state representation
        let concatEmbedding: MLShapedArray<Float32> = MLShapedArray<Float32>(
            concatenating: [blankEmbedding, promptEmbedding],
            alongAxis: 0
        )
        let hiddenStates = toHiddenStates(concatEmbedding)
        guard hiddenStates.shape[1] == unet.hiddenSize else {
            throw StableDiffusionError.incompatibleTextEncoder
        }
        return hiddenStates
    }
    
    func toHiddenStates(_ embedding: MLShapedArray<Float32>) -> MLShapedArray<Float32> {
        // Unoptimized manual transpose [0, 2, None, 1]
        // e.g. From [2, 77, 768] to [2, 768, 1, 77]
        let fromShape = embedding.shape
        let stateShape = [fromShape[0], fromShape[2], 1, fromShape[1]]
        var states = MLShapedArray<Float32>(repeating: 0.0, shape: stateShape)
        for i0 in 0..<fromShape[0] {
            for i1 in 0..<fromShape[1] {
                for i2 in 0..<fromShape[2] {
                    states[scalarAt: i0, i2, 0, i1] = embedding[scalarAt: i0, i1, i2]
                }
            }
        }
        return states
    }
    
    public func decodeToImage(_ latent: MLShapedArray<Float32>) throws -> CGImage? {
        let image = try decoder.decode([latent])[0]
        
        // If there is no safety checker return what was decoded
        guard !disableSafety, let safetyChecker else {
            return image
        }
        
        // Otherwise change images which are not safe to nil
        let safeImage = try safetyChecker.isSafe(image) ? image : nil
        return safeImage
    }
}
