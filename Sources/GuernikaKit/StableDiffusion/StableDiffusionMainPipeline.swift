//
//  StableDiffusionPipeline.swift
//  
//
//  Created by Guillermo Cique Fernández on 25/5/23.
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
public class StableDiffusionMainPipeline: StableDiffusionPipeline {
    public let baseUrl: URL
    /// Model to generate embeddings for tokenized input text
    public let textEncoder: TextEncoder
    public var overrideTextEncoder: TextEncoder?
    /// Model used to generate initial image for latent diffusion process
    var encoder: Encoder? = nil
    /// Models used to control diffusion models by adding extra conditions
    public var controlNets: [ControlNet.Input] = [] {
        didSet {
            // Only allow compatible ControlNets
            controlNets = controlNets.filter { $0.controlNet.hiddenSize == unet.hiddenSize }
        }
    }
    /// Model used to predict noise residuals given an input, diffusion time step, and conditional embedding
    public var unet: Unet
    /// Model used to generate final image from latent diffusion process
    public var decoder: Decoder
    /// Optional model for checking safety of generated image
    public var safetyChecker: SafetyChecker? = nil
    
    public var disableSafety: Bool = false {
        didSet {
            if !disableSafety {
                safetyChecker?.unloadResources()
            }
        }
    }

    /// Option to reduce memory during image generation
    ///
    /// If true, the pipeline will lazily load TextEncoder, Unet, Decoder, and SafetyChecker
    /// when needed and aggressively unload their resources after
    ///
    /// This will increase latency in favor of reducing memory
    public var reduceMemory: Bool = false
    
    /// Reports whether this pipeline can perform image to image
    public var canGenerateVariations: Bool {
        encoder != nil
    }
    
    public var computeUnits: ComputeUnits {
        didSet {
            guard computeUnits != oldValue else { return }
            let configuration = MLModelConfiguration()
            configuration.computeUnits = computeUnits.mlComputeUnits(for: unet.attentionImplementation)
            textEncoder.configuration = configuration
            encoder?.configuration = configuration
            unet.configuration = configuration
            decoder.configuration = configuration
            safetyChecker?.configuration = configuration
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
        encoder: Encoder? = nil,
        unet: Unet,
        decoder: Decoder,
        safetyChecker: SafetyChecker? = nil,
        computeUnits: ComputeUnits = .auto,
        reduceMemory: Bool = false
    ) {
        self.baseUrl = baseUrl
        self.textEncoder = textEncoder
        self.encoder = encoder
        self.unet = unet
        self.decoder = decoder
        self.safetyChecker = safetyChecker
        self.computeUnits = computeUnits
        self.reduceMemory = reduceMemory
    }

    /// Unload the underlying resources to free up memory
    public func unloadResources() {
        textEncoder.unloadResources(clearCache: true)
        overrideTextEncoder?.unloadResources(clearCache: true)
        encoder?.unloadResources(clearCache: true)
        unet.unloadResources()
        controlNets.forEach { $0.controlNet.unloadResources() }
        decoder.unloadResources()
        safetyChecker?.unloadResources()
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
        progressHandler: (StableDiffusionProgress) -> Bool = { _ in true }
    ) throws -> CGImage? {
#if DEBUG
        let mainTick = CFAbsoluteTimeGetCurrent()
#endif
        let hiddenStates = try hiddenStates(prompt: input.prompt, negativePrompt: input.negativePrompt)
        
        let generator: RandomGenerator = TorchRandomGenerator(seed: input.seed)
        let scheduler: Scheduler = input.scheduler.create(
            strength: input.strength, stepCount: input.stepCount, predictionType: unet.predictionType
        )

        // Generate random latent sample from specified seed
        var latent = try generateLatentSample(input: input, generator: generator, scheduler: scheduler)

        // De-noising loop
        for (step, t) in scheduler.timeSteps.enumerated() {
            // Expand the latents for classifier-free guidance
            // and input to the Unet noise prediction model
            var latentUnetInput = MLShapedArray<Float32>(concatenating: [latent, latent], alongAxis: 0)
            
            latentUnetInput = scheduler.scaleModelInput(timeStep: t, sample: latentUnetInput)
            
            let additionalResiduals = try controlNets.predictResiduals(
                latent: latentUnetInput,
                timeStep: t,
                hiddenStates: hiddenStates
            )

            // Predict noise residuals from latent samples
            // and current time step conditioned on hidden states
            var noise = try unet.predictNoise(
                latents: [latentUnetInput],
                additionalResiduals: additionalResiduals.map { [$0] },
                timeStep: t,
                hiddenStates: hiddenStates
            )[0]

            noise = performGuidance(noise, guidanceScale: input.guidanceScale)

            // Have the scheduler compute the previous (t-1) latent
            // sample given the predicted noise and current sample
            latent = scheduler.step(
                output: noise,
                timeStep: t,
                sample: latent,
                generator: generator
            )

            // Report progress
            let progress = StableDiffusionProgress(
                pipeline: self,
                input: input,
                step: step,
                stepCount: scheduler.timeSteps.count - 1,
                currentLatentSample: latent
            )
            if !progressHandler(progress) {
                // Stop if requested by handler
                return nil
            }
        }
        
        if reduceMemory {
            controlNets.forEach { $0.controlNet.unloadResources() }
            unet.unloadResources()
        }

        // Decode the latent sample to image
        let image = try decodeToImage(latent)
        
        if reduceMemory {
            decoder.unloadResources()
        }
        
#if DEBUG
        let mainTock = CFAbsoluteTimeGetCurrent()
        let runtime = String(format:"%.2fs", mainTock - mainTick)
        print("Time", runtime)
#endif
        return image
    }
    
    func generateLatentSample(input: SampleInput, generator: RandomGenerator, scheduler: Scheduler) throws -> MLShapedArray<Float32> {
        var sampleShape = unet.latentSampleShape
        sampleShape[0] = 1
        
        let sample: MLShapedArray<Float32>
        if let image = input.initImage, input.strength != nil {
            guard let encoder else {
                throw StableDiffusionError.encoderMissing
            }
            let latent = try encoder.encode(image, generator: generator)
            let noise: MLShapedArray<Float32> = generator.nextArray(shape: sampleShape, mean: 0, stdev: 1)
            sample = scheduler.addNoise(originalSample: latent, noise: [noise])[0]
        } else {
            let stdev = scheduler.initNoiseSigma
            sample = generator.nextArray(shape: sampleShape, mean: 0, stdev: stdev)
        }
        
        if reduceMemory {
            encoder?.unloadResources()
        }
        
        return sample
    }

    func performGuidance(_ noise: [MLShapedArray<Float32>], guidanceScale: Float) -> [MLShapedArray<Float32>] {
        noise.map { performGuidance($0, guidanceScale: guidanceScale) }
    }

    func performGuidance(_ noise: MLShapedArray<Float32>, guidanceScale: Float) -> MLShapedArray<Float32> {
        var shape = noise.shape
        shape[0] = 1
        return MLShapedArray<Float>(unsafeUninitializedShape: shape) { result, _ in
            noise.withUnsafeShapedBufferPointer { scalars, _, strides in
                for i in 0..<result.count {
                    // unconditioned + guidance*(text - unconditioned)
                    result.initializeElement(
                        at: i,
                        to: scalars[i] + guidanceScale * (scalars[strides[0] + i] - scalars[i])
                    )
                }
            }
        }
    }
}
