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
    var controlNets: [ControlNet.Input] { get set }
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
    
    func generateImages(
        input: SampleInput,
        progressHandler: (StableDiffusionProgress) -> Bool
    ) throws -> CGImage?
    
    func decodeToImage(_ latent: MLShapedArray<Float32>) throws -> CGImage?
    
    var latentRGBFactors: [[Float]] { get }
    
    func latentToImage(_ latent: MLShapedArray<Float32>) -> CGImage?
}

/// Sampling progress details
public struct StableDiffusionProgress {
    public let pipeline: any StableDiffusionPipeline
    public let input: SampleInput
    public let step: Int
    public let stepCount: Int
    public let currentLatentSample: MLShapedArray<Float32>
    public var latentImage: CGImage? {
        pipeline.latentToImage(currentLatentSample)
    }
    public var decodedImage: CGImage? {
        do {
            return try pipeline.decodeToImage(currentLatentSample)
        } catch {
            print("Error decoding progress images", error.localizedDescription)
            return latentImage
        }
    }
}

extension StableDiffusionPipeline {
    /// Expected text encoder max input length
    public var maximumTokensAllowed: Int { textEncoder.maxInputLength }
    /// Unet sample size
    public var sampleSize: CGSize { unet.sampleSize }
    public var minimumSize: CGSize { unet.minimumSize }
    public var maximumSize: CGSize { unet.maximumSize }
    public var allowsVariableSize: Bool { unet.minimumSize != unet.maximumSize }
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
        let channelCount = latent.shape[1]
        let height = latent.shape[2]
        let width = latent.shape[3]
        
        let red = PixelBufferPFx1(width: width, height: height)
        let green = PixelBufferPFx1(width: width, height: height)
        let blue = PixelBufferPFx1(width: width, height: height)
        
        let latentRGBFactors: [[Float]] = latentRGBFactors
        
        let stride = vDSP_Stride(1)
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
        latent[0][1].withUnsafeShapedBufferPointer { channel1Ptr, _, _ in
            latent[0][2].withUnsafeShapedBufferPointer { channel2Ptr, _, _ in
                latent[0][3].withUnsafeShapedBufferPointer { channel3Ptr, _, _ in
                    red.withUnsafeMutableBufferPointer { ptr in
                        for i in 0..<ptr.count {
                            ptr[i] = ptr[i] + (channel1Ptr[i] * latentRGBFactors[1][0]) + (channel2Ptr[i] * latentRGBFactors[2][0])
                                + (channel3Ptr[i] * latentRGBFactors[3][0])
                            ptr[i] = min(1, max(0, (ptr[i] + 1) / 2))
                        }
                    }
                }
            }
        }
        latent[0][1].withUnsafeShapedBufferPointer { channel1Ptr, _, _ in
            latent[0][2].withUnsafeShapedBufferPointer { channel2Ptr, _, _ in
                latent[0][3].withUnsafeShapedBufferPointer { channel3Ptr, _, _ in
                    green.withUnsafeMutableBufferPointer { ptr in
                        for i in 0..<ptr.count {
                            ptr[i] = ptr[i] + (channel1Ptr[i] * latentRGBFactors[1][1]) + (channel2Ptr[i] * latentRGBFactors[2][1])
                            + (channel3Ptr[i] * latentRGBFactors[3][1])
                            ptr[i] = min(1, max(0, (ptr[i] + 1) / 2))
                        }
                    }
                }
            }
        }
        latent[0][1].withUnsafeShapedBufferPointer { channel1Ptr, _, _ in
            latent[0][2].withUnsafeShapedBufferPointer { channel2Ptr, _, _ in
                latent[0][3].withUnsafeShapedBufferPointer { channel3Ptr, _, _ in
                    blue.withUnsafeMutableBufferPointer { ptr in
                        for i in 0..<ptr.count {
                            ptr[i] = ptr[i] + (channel1Ptr[i] * latentRGBFactors[1][2]) + (channel2Ptr[i] * latentRGBFactors[2][2])
                                + (channel3Ptr[i] * latentRGBFactors[3][2])
                            ptr[i] = min(1, max(0, (ptr[i] + 1) / 2))
                        }
                    }
                }
            }
        }
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
                      colorSpace: CGColorSpaceCreateDeviceRGB(),
                      bitmapInfo: bitmapInfo)!)!
        
        guard let filter = CIFilter(name: "CIColorControls") else { return cgImage }
        filter.setValue(CIImage(cgImage: cgImage), forKey: kCIInputImageKey)
        filter.setValue(1.4, forKey: kCIInputSaturationKey)
        filter.setValue(1.1, forKey: kCIInputContrastKey)
        guard let result = filter.value(forKey: kCIOutputImageKey) as? CIImage else { return cgImage }
        guard let newCgImage = CIContext(options: nil).createCGImage(result, from: result.extent) else { return cgImage }
        return newCgImage
    }
}
