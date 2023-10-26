//
//  LatentConsistencyModelPipeline.swift
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
public class LatentConsistencyModelPipeline: StableDiffusionPipeline {
    public let baseUrl: URL
    /// Model to generate embeddings for tokenized input text
    public let textEncoder: TextEncoder
    public var overrideTextEncoder: TextEncoder? {
        didSet {
            oldValue?.unloadResources(clearCache: true)
            if overrideTextEncoder != nil {
                textEncoder.unloadResources(clearCache: true)
            }
        }
    }
    /// Model used to generate initial image for latent diffusion process
    var encoder: Encoder? = nil
    /// Models used to control diffusion models by adding extra conditions
    public var conditioningInput: [ConditioningInput] = []
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
    public var reduceMemory: Bool = false {
        didSet { unet.reduceMemory = reduceMemory }
    }
    
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
        unet.reduceMemory = reduceMemory
        self.decoder = decoder
        self.safetyChecker = safetyChecker
        self.computeUnits = computeUnits
        self.reduceMemory = reduceMemory
    }
    
    public func prewarmResources() throws {
        if let overrideTextEncoder {
            try overrideTextEncoder.model.prewarmResources()
        } else {
            try textEncoder.model.prewarmResources()
        }
        try encoder?.model.prewarmResources()
        try unet.models.forEach { try $0.prewarmResources() }
        try decoder.model.prewarmResources()
        try safetyChecker?.model.prewarmResources()
    }

    /// Unload the underlying resources to free up memory
    public func unloadResources() {
        textEncoder.unloadResources(clearCache: true)
        overrideTextEncoder?.unloadResources(clearCache: true)
        encoder?.unloadResources(clearCache: true)
        unet.unloadResources()
        conditioningInput.forEach { $0.module.unloadResources() }
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
        progressHandler: (DiffusionProgress) -> Bool = { _ in true }
    ) throws -> CGImage? {
#if DEBUG
        let mainTick = CFAbsoluteTimeGetCurrent()
#endif
        let input = try checkInput(input: input)
        let hiddenStates = try hiddenStates(prompt: input.prompt)
        
        let generator: RandomGenerator = TorchRandomGenerator(seed: input.seed)
        let scheduler: Scheduler = LCMScheduler(
            strength: input.strength, stepCount: input.stepCount, predictionType: unet.predictionType
        )

        // Generate random latent sample from specified seed
        var (latent, noise, imageLatent) = try prepareLatent(input: input, generator: generator, scheduler: scheduler)
        // Prepare mask only for inpainting
        let mask = try prepareMaskLatents(input: input, generator: generator)
        if reduceMemory {
            encoder?.unloadResources()
        }
        
        guard let timeCondProjDim = unet.timeCondProjDim else {
            fatalError("Incompatible Unet")
        }
        let wEmbedding = getGuidanceScaleEmbedding(w: input.guidanceScale - 1, embeddingDim: timeCondProjDim)
        
        let adapterState = try conditioningInput.predictAdapterResiduals(latent: latent, reduceMemory: reduceMemory)

        // De-noising loop
        for (step, t) in scheduler.timeSteps.enumerated() {
            // Predict noise residuals from latent samples
            // and current time step conditioned on hidden states
            var noisePrediction = try unet.predictNoise(
                latents: [latent],
                additionalResiduals: adapterState,
                timeStep: t,
                timeStepCond: wEmbedding,
                hiddenStates: hiddenStates
            )[0]

            // Have the scheduler compute the previous (t-1) latent
            // sample given the predicted noise and current sample
            latent = scheduler.step(
                output: noisePrediction,
                timeStep: t,
                sample: latent,
                generator: generator
            )
            
            if let mask, let noise, let imageLatent {
                var initLatentProper = imageLatent
                if step < scheduler.timeSteps.count - 1 {
                    let noiseTimeStep = scheduler.timeSteps[step + 1]
                    initLatentProper = scheduler.addNoise(
                        originalSample: initLatentProper,
                        noise: [noise],
                        timeStep: noiseTimeStep
                    )[0]
                }
                latent = MLShapedArray<Float>(unsafeUninitializedShape: latent.shape) { result, _ in
                    mask.withUnsafeShapedBufferPointer { maskScalars, _, _ in
                        latent.withUnsafeShapedBufferPointer { latentScalars, _, _ in
                            initLatentProper.withUnsafeShapedBufferPointer { initScalars, _, _ in
                                for i in 0..<result.count {
                                    // (1 - init_mask) * init_latents_proper + init_mask * latents
                                    let maskScalar = maskScalars[i % maskScalars.count]
                                    result.initializeElement(
                                        at: i,
                                        to: (1 - maskScalar) * initScalars[i] + maskScalar * latentScalars[i]
                                    )
                                }
                            }
                        }
                    }
                }
            }

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
            conditioningInput.controlNets.forEach { $0.unloadResources() }
            unet.unloadResources()
        }

        // Decode the latent sample to image
        let image = try decodeToImage(scheduler.modelOutputs.last ?? latent)
        
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
    
    func prepareLatent(
        input: SampleInput,
        generator: RandomGenerator,
        scheduler: Scheduler
    ) throws -> (MLShapedArray<Float32>, MLShapedArray<Float32>?, MLShapedArray<Float32>?) {
        var sampleShape = unet.latentSampleShape
        sampleShape[0] = 1
        sampleShape[1] = 4
        if let size = input.size {
            var newHeight = Int(size.height / 8)
            var newWidth = Int(size.width / 8)
            // Sample shape size must be divisible by 8
            newHeight -= (newHeight % 8)
            newWidth -= (newWidth % 8)
            sampleShape[2] = newHeight
            sampleShape[3] = newWidth
        }
        
        let latent: MLShapedArray<Float32>
        let noise: MLShapedArray<Float32>?
        var imageLatent: MLShapedArray<Float32>?
        if let image = input.initImage, let strength = input.strength {
            guard let encoder else {
                throw StableDiffusionError.encoderMissing
            }
            imageLatent = try encoder.encode(image, generator: generator)
            if input.inpaintMask != nil && strength >= 1 {
                let stdev = scheduler.initNoiseSigma
                noise = generator.nextArray(shape: sampleShape, mean: 0, stdev: stdev)
                latent = noise!
            } else {
                noise = generator.nextArray(shape: sampleShape, mean: 0, stdev: 1)
                latent = scheduler.addNoise(originalSample: imageLatent!, noise: [noise!])[0]
            }
        } else {
            let stdev = scheduler.initNoiseSigma
            latent = generator.nextArray(shape: sampleShape, mean: 0, stdev: stdev)
            // Not needed for text to image
            noise = nil
            imageLatent = nil
        }
        
        return (latent, noise, imageLatent)
    }
    
    func prepareMaskLatents(input: SampleInput, generator: RandomGenerator) throws -> MLShapedArray<Float32>? {
        guard let image = input.initImage, let mask = input.inpaintMask else {
            return nil
        }
        guard let encoder else {
            throw StableDiffusionError.encoderMissing
        }
        var imageData = image.toShapedArray()
        var maskData = mask.toAlphaShapedArray()
        imageData = MLShapedArray(unsafeUninitializedShape: imageData.shape) { result, _ in
            imageData.withUnsafeShapedBufferPointer { image, _, _ in
                maskData.withUnsafeShapedBufferPointer { mask, _, _ in
                    for i in 0..<result.count {
                        let maskScalar: Float32 = mask[i % mask.count] < 0.5 ? 1 : 0
                        result.initializeElement(at: i, to: image[i] * maskScalar)
                    }
                }
            }
        }
        
        // Encode the mask image into latents space so we can concatenate it to the latents
        var maskedImageLatent = try encoder.encode(imageData, generator: generator)
        let resizedMask = mask.scaledAspectFill(size: CGSize(width: maskedImageLatent.shape[3], height: maskedImageLatent.shape[2]))
        maskData = resizedMask.toAlphaShapedArray()
        
        return maskData
    }
    
    func hiddenStates(prompt: String) throws -> MLShapedArray<Float32> {
        // Encode the input prompt as well as a blank unconditioned input
        let promptEmbedding: MLShapedArray<Float32>
        if let overrideTextEncoder {
            (_, promptEmbedding) = try overrideTextEncoder.encode(prompt)
        } else {
            (_, promptEmbedding) = try textEncoder.encode(prompt)
        }
        
        if reduceMemory {
            textEncoder.unloadResources()
            overrideTextEncoder?.unloadResources()
        }
        
        // Convert to Unet hidden state representation
        let hiddenStates = toHiddenStates(promptEmbedding)
        guard hiddenStates.shape[1] == unet.hiddenSize else {
            throw StableDiffusionError.incompatibleTextEncoder
        }
        return hiddenStates
    }
    
    /**
     See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298
     
     Args:
     timesteps (`torch.Tensor`):
     generate embedding vectors at these timesteps
     embedding_dim (`int`, *optional*, defaults to 512):
     dimension of the embeddings to generate
     dtype:
     data type of the generated embeddings
     
     Returns:
     `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
     */
    func getGuidanceScaleEmbedding(w: Float, embeddingDim: Int) -> MLShapedArray<Float32> {
        let w = Double(w * 1000.0)
        
        let halfDim = Int(embeddingDim / 2)
        let emb = log(10000.0) / Double(halfDim - 1)
        let emb1 = (0..<halfDim).map { exp(Double($0) * -emb) * w }
        return MLShapedArray(unsafeUninitializedShape: [1, embeddingDim, 1, 1]) { pointer, _ in
            for i in 0..<emb1.count {
                pointer.initializeElement(at: i, to: Float(sin(emb1[i])))
            }
            for i in 0..<emb1.count {
                pointer.initializeElement(at: i + emb1.count, to: Float(sin(emb1[i])))
            }
        }
    }
}
