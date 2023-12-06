//
//  Decoder.swift
//  
//
//  Created by Guillermo Cique Fernández on 25/5/23.
//

import CoreML
import Accelerate
import Foundation

/// A decoder model which produces RGB images from latent samples
public class Decoder {
    /// VAE decoder model
    let model: ManagedMLModel
    public var configuration: MLModelConfiguration {
        get { model.configuration }
        set { model.configuration = newValue }
    }
    let scaleFactor: Float32?
    
    /// Create decoder from Core ML model
    ///
    /// - Parameters:
    ///     - url: Location of compiled VAE decoder Core ML model
    ///     - configuration: configuration to be used when the model is loaded
    /// - Returns: A decoder that will lazily load its required resources when needed or requested
    public convenience init(modelAt url: URL, configuration: MLModelConfiguration) throws {
        let metadata = try CoreMLMetadata.metadataForModel(at: url)
        try self.init(modelAt: url, metadata: metadata, configuration: configuration)
    }
    
    public init(modelAt url: URL, metadata: CoreMLMetadata, configuration: MLModelConfiguration) throws {
        scaleFactor = metadata.userDefinedMetadata?["scaling_factor"].flatMap { Float32($0) }
        self.model = ManagedMLModel(modelAt: url, configuration: configuration)
    }
    
    /// Unload the underlying model to free up memory
    public func unloadResources() {
        model.unloadResources()
    }
    
    /// Batch decode latent samples into images
    ///
    ///  - Parameters:
    ///    - latents: Batch of latent samples to decode
    ///  - Returns: decoded images
    public func decode(_ latent: MLShapedArray<Float32>, scaleFactor: Float32 = 0.18215) throws -> CGImage {
        // Give preference to the default scaling factor of the model
        let scaleFactor = self.scaleFactor ?? scaleFactor
        let result = try model.perform { model in
            let inputName = model.modelDescription.inputDescriptionsByName.first!.key
            // Reference pipeline scales the latent samples before decoding
            let sampleScaled = if scaleFactor == 1 {
                latent
            } else {
                MLShapedArray(unsafeUninitializedShape: latent.shape) { scalars, _ in
                    latent.withUnsafeShapedBufferPointer { sample, _, _ in
                        vDSP.divide(sample, scaleFactor, result: &scalars)
                    }
                }
            }
            
            let dict = [inputName: MLMultiArray(sampleScaled)]
            let input = try MLDictionaryFeatureProvider(dictionary: dict)
            return try model.prediction(from: input)
        }
        
        // Transform the output to CGImage
        let outputName = result.featureNames.first!
        let output = result.featureValue(for: outputName)!.multiArrayValue!
        return toRGBCGImage(MLShapedArray<Float32>(converting: output))
    }
    
    typealias PixelBufferPFx1 = vImage.PixelBuffer<vImage.PlanarF>
    typealias PixelBufferP8x3 = vImage.PixelBuffer<vImage.Planar8x3>
    typealias PixelBufferIFx3 = vImage.PixelBuffer<vImage.InterleavedFx3>
    typealias PixelBufferI8x3 = vImage.PixelBuffer<vImage.Interleaved8x3>
    
    func toRGBCGImage(_ array: MLShapedArray<Float32>) -> CGImage {
        // array is [N,C,H,W], where C==3
        let channelCount = array.shape[1]
        assert(channelCount == 3, "Decoding model output has \(channelCount) channels, expected 3")
        let height = array.shape[2]
        let width = array.shape[3]
        
        // Normalize each channel into a float between 0 and 1.0
        let floatChannels = (0..<channelCount).map { i in
            // Normalized channel output
            let cOut = PixelBufferPFx1(width: width, height:height)
            
            // Reference this channel in the array and normalize
            array[0][i].withUnsafeShapedBufferPointer { ptr, _, strides in
                let cIn = PixelBufferPFx1(data: .init(mutating: ptr.baseAddress!),
                                          width: width, height: height,
                                          byteCountPerRow: strides[0]*4)
                // Map [-1.0 1.0] -> [0.0 1.0]
                cIn.multiply(by: 0.5, preBias: 1.0, postBias: 0.0, destination: cOut)
            }
            return cOut
        }
        
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
        
        return cgImage
    }
}
