//
//  WuerstchenPipeline.swift
//
//
//  Created by Guillermo Cique Fernández on 15/9/23.
//

import CoreML
import Schedulers
import Accelerate
import CoreGraphics
import RandomGenerator

/// A pipeline used to generate image samples from text input using stable diffusion
///
/// This implementation matches:
/// [Hugging Face Diffusers Pipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py)
public class WuerstchenPipeline: DiffusionPipeline {
    public let baseUrl: URL
    /// Model to generate embeddings for tokenized input text
    public let priorPipeline: WuerstchenPipelinePrior
    public let decoderPipeline: WuerstchenPipelineDecoder
    
    public var sampleSize: CGSize { priorPipeline.sampleSize }
    public var minimumSize: CGSize { priorPipeline.minimumSize }
    public var maximumSize: CGSize { priorPipeline.maximumSize }

    /// Option to reduce memory during image generation
    ///
    /// If true, the pipeline will lazily load TextEncoder, Unet, Decoder, and SafetyChecker
    /// when needed and aggressively unload their resources after
    ///
    /// This will increase latency in favor of reducing memory
    public var reduceMemory: Bool = false {
        didSet { 
            priorPipeline.reduceMemory = reduceMemory
            decoderPipeline.reduceMemory = reduceMemory
        }
    }
    
    public var computeUnits: ComputeUnits {
        didSet {
            guard computeUnits != oldValue else { return }
            priorPipeline.computeUnits = computeUnits
            decoderPipeline.computeUnits = computeUnits
        }
    }

    /// Creates a pipeline using the specified models and tokenizer
    ///
    /// - Parameters:
    ///   - textEncoder: Model for encoding tokenized text
    ///   - unet: Model for noise prediction on latent samples
    ///   - decoder: Model for decoding latent sample to image
    ///   - safetyChecker: Optional model for checking safety of generated images
    ///   - guidanceScale: Influence of the text prompt on generation process
    /// - Returns: Pipeline ready for image generation
    public init(
        baseUrl: URL,
        textEncoder: TextEncoder,
        prior: WuerstchenPrior,
        decoder: WuerstchenDecoder,
        vqgan: WuerstchenVQGAN,
        computeUnits: ComputeUnits = .auto,
        reduceMemory: Bool = false
    ) {
        self.baseUrl = baseUrl
        self.priorPipeline = WuerstchenPipelinePrior(
            baseUrl: baseUrl, 
            textEncoder: textEncoder,
            prior: prior,
            computeUnits: computeUnits,
            reduceMemory: reduceMemory
        )
        self.decoderPipeline = WuerstchenPipelineDecoder(
            baseUrl: baseUrl,
            textEncoder: textEncoder,
            decoder: decoder,
            vqgan: vqgan,
            computeUnits: computeUnits,
            reduceMemory: reduceMemory
        )
        self.computeUnits = computeUnits
        self.reduceMemory = reduceMemory
    }

    /// Unload the underlying resources to free up memory
    public func unloadResources() {
        priorPipeline.unloadResources()
        decoderPipeline.unloadResources()
    }
    
    /// Text to image generation using stable diffusion
    ///
    /// - Parameters:
    ///   - prompt: Text prompt to guide sampling
    ///   - stepCount: Number of inference steps to perform
    ///   - imageCount: Number of samples/images to generate for the input prompt
    ///   - seed: Random seed which
    ///   - disableSafety: Safety checks are only performed if `self.canSafetyCheck && !disableSafety`
    ///   - progressHandler: Callback to perform after each step, stops on receiving false response
    /// - Returns: An array of `imageCount` optional images.
    ///            The images will be nil if safety checks were performed and found the result to be un-safe
    public func generateImages(
        input: SampleInput,
        progressHandler: (DiffusionProgress) -> Bool = { _ in true }
    ) throws -> CGImage? {
#if DEBUG
        let mainTick = CFAbsoluteTimeGetCurrent()
#endif
        guard let imageEmbedding = try priorPipeline.generateEmbedding(input: input, progressHandler: progressHandler) else {
            return nil
        }
        var input = input
        input.stepCount = 10
        let image = try decoderPipeline.generateImages(input: input, imageEmbedding: imageEmbedding, progressHandler: progressHandler)
        
#if DEBUG
        let mainTock = CFAbsoluteTimeGetCurrent()
        let runtime = String(format:"%.2fs", mainTock - mainTick)
        print("Time", runtime)
#endif
        return image
    }
}
