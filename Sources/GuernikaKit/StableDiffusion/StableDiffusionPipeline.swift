//
//  StableDiffusionPipeline.swift
//  
//
//  Created by Guillermo Cique Fernández on 25/5/23.
//

import CoreML
import CoreImage
import Accelerate
import Foundation

public protocol StableDiffusionPipeline: DiffusionPipeline {
    /// Model to generate embeddings for tokenized input text
    var textEncoder: TextEncoder { get }
    
    var overrideTextEncoder: TextEncoder? { get set }
    
    /// Models used to control diffusion models by adding extra conditions
    var conditioningInput: [ConditioningInput] { get set }
    var supportsAdapter: Bool { get }
    var supportsControlNet: Bool { get }
    
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
    
    func prewarmResources() throws
    
    func generateImages(input: SampleInput, progressHandler: (DiffusionProgress) -> Bool) throws -> CGImage?
    
    func decodeToImage(_ latent: MLShapedArray<Float32>) throws -> CGImage?
    
    var latentRGBFactors: [[Float]] { get }
    
    func latentToImage(_ latent: MLShapedArray<Float32>) -> CGImage?
}

extension StableDiffusionPipeline {
    /// Expected text encoder max input length
    public var maximumTokensAllowed: Int { textEncoder.maxInputLength }
    /// Unet sample size
    public var sampleSize: CGSize { unet.sampleSize }
    public var minimumSize: CGSize { unet.minimumSize }
    public var maximumSize: CGSize { unet.maximumSize }
    public var batchSize: Int { unet.batchSize }
    public var doClassifierFreeGuidance: Bool { batchSize > 1 }
    public var supportsAdapter: Bool { unet.supportsAdapter }
    public var supportsControlNet: Bool { unet.supportsControlNet }
    
    /// Reports whether this pipeline can perform safety checks
    public var canSafetyCheck: Bool {
        safetyChecker != nil
    }
    
    public var configuration: MLModelConfiguration { unet.configuration }
    
    @discardableResult
    public func addConditioningInput(_ module: any ConditioningModule) throws -> ConditioningInput {
        if module is T2IAdapter, !supportsAdapter {
            throw StableDiffusionError.incompatibleAdapter
        } else if module is ControlNet, !supportsControlNet {
            throw StableDiffusionError.incompatibleControlNet
        }
        let input = ConditioningInput(module: module)
        conditioningInput.append(input)
        return input
    }
    
    func checkInput(input: SampleInput) throws -> SampleInput {
        var input = input
        if var size = input.size {
            guard size.isBetween(min: unet.minimumSize, max: unet.maximumSize) else {
                throw StableDiffusionError.incompatibleSize
            }
            // Output size must be divisible by 64
            if !Int(size.height).isMultiple(of: 64) {
                size.height = CGFloat(Int(size.height / 64) * 64)
            }
            if !Int(size.width).isMultiple(of: 64) {
                size.width = CGFloat(Int(size.width / 64) * 64)
            }
            input.size = size
            if let image = input.initImage, image.size != size {
                input.initImage = image.scaledAspectFill(size: size)
            }
        }
        if let image = input.initImage, let maskImage = input.inpaintMask, image.size != maskImage.size {
            input.inpaintMask = maskImage.scaledAspectFill(size: image.size)
        }
        return input
    }
    
    public func generateImages(input: SampleInput) throws -> CGImage? {
        try generateImages(input: input, progressHandler: { _ in true })
    }

    func hiddenStates(prompt: String, negativePrompt: String) throws -> MLShapedArray<Float32> {
        // Encode the input prompt as well as a blank unconditioned input
        let (_, promptEmbedding) = try (overrideTextEncoder ?? textEncoder).encode(prompt)
        var blankEmbedding: MLShapedArray<Float32>?
        if doClassifierFreeGuidance {
            (_, blankEmbedding) = try (overrideTextEncoder ?? textEncoder).encode(negativePrompt)
        }
        
        if reduceMemory {
            textEncoder.unloadResources()
            overrideTextEncoder?.unloadResources()
        }

        // Convert to Unet hidden state representation
        var finalEmbedding: MLShapedArray<Float32> = promptEmbedding
        if let blankEmbedding {
            finalEmbedding = MLShapedArray<Float32>(
                concatenating: [blankEmbedding, promptEmbedding],
                alongAxis: 0
            )
        }
        let hiddenStates = toHiddenStates(finalEmbedding)
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
    func getGuidanceScaleEmbedding(w: Float) -> MLShapedArray<Float32>? {
        guard let embeddingDim = unet.timeCondProjDim else { return nil }
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
    
    public func decodeToImage(_ latent: MLShapedArray<Float32>) throws -> CGImage? {
        let image = try decoder.decode(latent)
        
        // If there is no safety checker return what was decoded
        guard !disableSafety, let safetyChecker else {
            return image
        }
        
        // Otherwise change images which are not safe to nil
        let safeImage = try safetyChecker.isSafe(image) ? image : nil
        return safeImage
    }
    
    // https://discuss.huggingface.co/t/decoding-latents-to-rgb-without-upscaling/23204/12
    public var latentRGBFactors: [[Float]] {[
        //  R       G       B
        [ 0.298,  0.207,  0.208],  // L1
        [ 0.187,  0.286,  0.173],  // L2
        [-0.158,  0.189,  0.264],  // L3
        [-0.184, -0.271, -0.473],  // L4
    ]}
    
    typealias PixelBufferPFx1 = vImage.PixelBuffer<vImage.PlanarF>
    typealias PixelBufferP8x3 = vImage.PixelBuffer<vImage.Planar8x3>
    typealias PixelBufferIFx3 = vImage.PixelBuffer<vImage.InterleavedFx3>
    typealias PixelBufferI8x3 = vImage.PixelBuffer<vImage.Interleaved8x3>
    
    public func latentToImage(_ latent: MLShapedArray<Float32>) -> CGImage? {
        // array is [N,C,H,W], where C==4
        let height = latent.shape[2]
        let width = latent.shape[3]
        
        // https://developer.apple.com/documentation/accelerate/finding_the_sharpest_image_in_a_sequence_of_captured_images#4199619
        let storageR = UnsafeMutableBufferPointer<Float>.allocate(capacity: Int(width * height))
        let storageG = UnsafeMutableBufferPointer<Float>.allocate(capacity: Int(width * height))
        let storageB = UnsafeMutableBufferPointer<Float>.allocate(capacity: Int(width * height))
        let red = PixelBufferPFx1(data: storageR.baseAddress!, width: width, height: height, byteCountPerRow: width * MemoryLayout<Float>.stride)
        let green = PixelBufferPFx1(data: storageG.baseAddress!, width: width, height: height, byteCountPerRow: width * MemoryLayout<Float>.stride)
        let blue = PixelBufferPFx1(data: storageB.baseAddress!, width: width, height: height, byteCountPerRow: width * MemoryLayout<Float>.stride)
        defer {
            storageR.deallocate()
            storageG.deallocate()
            storageB.deallocate()
        }
        
        let latentRGBFactors: [[Float]] = latentRGBFactors
        
        // Reference this channel in the array and normalize
        latent[0][0].withUnsafeShapedBufferPointer { channelPtr, _, strides in
            let cIn = PixelBufferPFx1(data: .init(mutating: channelPtr.baseAddress!),
                                      width: width, height: height,
                                      byteCountPerRow: strides[0]*4)
            // Map [-1.0 1.0] -> [0.0 1.0]
            cIn.multiply(by: latentRGBFactors[0][0], preBias: 0.0, postBias: 0.0, destination: red)
            cIn.multiply(by: latentRGBFactors[0][1], preBias: 0.0, postBias: 0.0, destination: green)
            cIn.multiply(by: latentRGBFactors[0][2], preBias: 0.0, postBias: 0.0, destination: blue)
        }
        
        processChannel(0, latent: latent, buffer: red)
        processChannel(1, latent: latent, buffer: green)
        processChannel(2, latent: latent, buffer: blue)
        
        let floatChannels = [red, green, blue]
        // Convert to interleaved and then to UInt8
        let floatImage = PixelBufferIFx3(planarBuffers: floatChannels)
        let uint8Image = PixelBufferI8x3(width: width, height: height)
        floatImage.convert(to: uint8Image) // maps [0.0 1.0] -> [0 255] and clips
        
        // Convert to uint8x3 to RGB CGImage (no alpha)
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
        let cgImage = uint8Image.makeCGImage(cgImageFormat:
                .init(bitsPerComponent: 8,
                      bitsPerPixel: 3*8,
                      colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!,
                      bitmapInfo: bitmapInfo)!)!
        
        guard let filter = CIFilter(name: "CIColorControls") else { return cgImage }
        filter.setValue(CIImage(cgImage: cgImage), forKey: kCIInputImageKey)
        filter.setValue(1.4, forKey: kCIInputSaturationKey)
        filter.setValue(1.1, forKey: kCIInputContrastKey)
        guard let result = filter.value(forKey: kCIOutputImageKey) as? CIImage else { return cgImage }
        guard let newCgImage = CIContext(options: nil).createCGImage(result, from: result.extent) else { return cgImage }
        return newCgImage
    }
    
    fileprivate func processChannel(_ channel: Int, latent: MLShapedArray<Float32>, buffer: PixelBufferPFx1) {
        let latentRGBFactors: [[Float]] = latentRGBFactors
        latent[0][1].withUnsafeShapedBufferPointer { channel1Ptr, _, _ in
            latent[0][2].withUnsafeShapedBufferPointer { channel2Ptr, _, _ in
                latent[0][3].withUnsafeShapedBufferPointer { channel3Ptr, _, _ in
                    buffer.withUnsafeMutableBufferPointer { ptr in
                        for i in 0..<channel3Ptr.count {
                            let channel1: Float = (channel1Ptr[i] * latentRGBFactors[1][channel])
                            let channel2: Float = (channel2Ptr[i] * latentRGBFactors[2][channel])
                            let channel3: Float = (channel3Ptr[i] * latentRGBFactors[3][channel])
                            ptr[i] = ptr[i] + channel1 + channel2 + channel3
                            ptr[i] = min(1, max(0, (ptr[i] + 1) / 2))
                        }
                    }
                }
            }
        }
    }
}
