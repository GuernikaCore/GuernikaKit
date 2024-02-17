//
//  WuerstchenPipelineDecoder.swift
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
public class WuerstchenPipelineDecoder: DiffusionPipeline {
    public let baseUrl: URL
    /// Model to generate embeddings for tokenized input text
    public let textEncoder: TextEncoder
    /// Model used to generate final image from latent diffusion process
    public var decoder: WuerstchenDecoder
    /// Model used to generate final image from latent diffusion process
    public var vqgan: WuerstchenVQGAN
    
    public var sampleSize: CGSize { decoder.sampleSize }
    public var minimumSize: CGSize { decoder.minimumSize }
    public var maximumSize: CGSize { decoder.maximumSize }

    /// Option to reduce memory during image generation
    ///
    /// If true, the pipeline will lazily load TextEncoder, Unet, Decoder, and SafetyChecker
    /// when needed and aggressively unload their resources after
    ///
    /// This will increase latency in favor of reducing memory
    public var reduceMemory: Bool = false {
        didSet { decoder.reduceMemory = reduceMemory }
    }
    
    public var computeUnits: ComputeUnits {
        didSet {
            guard computeUnits != oldValue else { return }
            let configuration = MLModelConfiguration()
            configuration.computeUnits = computeUnits.mlComputeUnits(for: decoder.attentionImplementation)
            textEncoder.configuration = configuration
            decoder.configuration = configuration
            vqgan.configuration = configuration
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
        decoder: WuerstchenDecoder,
        vqgan: WuerstchenVQGAN,
        computeUnits: ComputeUnits = .auto,
        reduceMemory: Bool = false
    ) {
        self.baseUrl = baseUrl
        self.textEncoder = textEncoder
        self.decoder = decoder
        decoder.reduceMemory = reduceMemory
        self.vqgan = vqgan
        self.computeUnits = computeUnits
        self.reduceMemory = reduceMemory
    }

    /// Unload the underlying resources to free up memory
    public func unloadResources() {
        textEncoder.unloadResources(clearCache: true)
        decoder.unloadResources()
        vqgan.unloadResources()
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
        imageEmbedding: MLShapedArray<Float32>,
        progressHandler: (DiffusionProgress) -> Bool = { _ in true }
    ) throws -> CGImage? {
#if DEBUG
        let mainTick = CFAbsoluteTimeGetCurrent()
#endif
        let hiddenStates = try hiddenStates(prompt: input.prompt)
        
        let generator: RandomGenerator = TorchRandomGenerator(seed: input.seed)
        let scheduler: Scheduler = DDPMWuerstchenScheduler(strength: input.strength, stepCount: input.stepCount)
//        input.scheduler.create(
//            strength: input.strength, stepCount: input.stepCount, predictionType: .epsilon
//        )

        // Generate random latent sample from specified seed
        var latent = try prepareLatent(imageEmbedding: imageEmbedding, generator: generator, scheduler: scheduler)

        // De-noising loop
        for (step, t) in scheduler.timeSteps.enumerated() {
            // Expand the latents for classifier-free guidance
            // and input to the Unet noise prediction model
            var latentInput = scheduler.scaleModelInput(timeStep: t, sample: latent)

            // Predict noise residuals from latent samples
            // and current time step conditioned on hidden states
            var noisePrediction = try decoder.predictNoise(
                latents: [latentInput],
                effnets: [imageEmbedding],
                timeStep: t,
                clipTextPooled: hiddenStates
            )[0]

//            noisePrediction = performGuidance(noisePrediction, guidanceScale: input.guidanceScale)

            // Have the scheduler compute the previous (t-1) latent
            // sample given the predicted noise and current sample
            latent = scheduler.step(
                output: noisePrediction,
                timeStep: t,
                sample: latent,
                generator: generator
            )

            // Report progress
            let progress = DiffusionProgress(
                pipeline: self,
                input: input,
                step: step,
                stepCount: scheduler.timeSteps.count - 1,
                currentLatentSample: scheduler.modelOutputs.last ?? latent
            )
            if !progressHandler(progress) {
                // Stop if requested by handler
                return nil
            }
        }
        
        if reduceMemory {
            decoder.unloadResources()
        }

        // Decode the latent sample to image
        let image = try vqgan.decode([latent])[0]
        
        if reduceMemory {
            vqgan.unloadResources()
        }
        
#if DEBUG
        let mainTock = CFAbsoluteTimeGetCurrent()
        let runtime = String(format:"%.2fs", mainTock - mainTick)
        print("Time", runtime)
#endif
        return image
    }
    
    func hiddenStates(prompt: String) throws -> MLShapedArray<Float32> {
        // Encode the input prompt as well as a blank unconditioned input
        let poolPromptEmbedding: MLShapedArray<Float32>
        (poolPromptEmbedding, _) = try textEncoder.encode(prompt)
        
        if reduceMemory {
            textEncoder.unloadResources()
        }
        
        let hiddenStates = toHiddenStates(poolPromptEmbedding)
        return hiddenStates
    }
    
    func toHiddenStates(_ embedding: MLShapedArray<Float32>) -> MLShapedArray<Float32> {
        guard embedding.shape != decoder.hiddenStatesShape else { return embedding }
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
    
    func prepareLatent(
        imageEmbedding: MLShapedArray<Float32>,
        generator: RandomGenerator,
        scheduler: Scheduler
    ) throws -> MLShapedArray<Float32> {
        let latentHeight = Int(Double(imageEmbedding.shape[2]) * decoder.latentDimScale)
        let latentWidth = Int(Double(imageEmbedding.shape[3]) * decoder.latentDimScale)
        let latentShape = [imageEmbedding.shape[0], 4, latentHeight, latentWidth]
        
        let stdev = scheduler.initNoiseSigma
        let latent = generator.nextArray(shape: latentShape, mean: 0, stdev: stdev)
        return latent
    }

    func performGuidance(_ noise: [MLShapedArray<Float32>], guidanceScale: Float) -> [MLShapedArray<Float32>] {
        noise.map { performGuidance($0, guidanceScale: guidanceScale) }
    }

    func performGuidance(_ noise: MLShapedArray<Float32>, guidanceScale: Float) -> MLShapedArray<Float32> {
        var shape = noise.shape
        shape[0] = 1
        return MLShapedArray<Float>(unsafeUninitializedShape: shape) { result, _ in
            vDSP.linearInterpolate(noise[1].scalars, noise[0].scalars, using: guidanceScale, result: &result)
        }
    }
}
