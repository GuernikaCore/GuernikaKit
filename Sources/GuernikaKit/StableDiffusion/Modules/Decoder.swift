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
    var configuration: MLModelConfiguration {
        get { model.configuration }
        set { model.configuration = newValue }
    }
    
    /// Create decoder from Core ML model
    ///
    /// - Parameters:
    ///     - url: Location of compiled VAE decoder Core ML model
    ///     - configuration: configuration to be used when the model is loaded
    /// - Returns: A decoder that will lazily load its required resources when needed or requested
    public init(modelAt url: URL, configuration: MLModelConfiguration) {
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
    public func decode(
        _ latents: [MLShapedArray<Float32>],
        scaleFactor: Float32 = 0.18215
    ) throws -> [CGImage] {
        // Batch predict with model
        let results = try model.perform { model in
            let inputName = model.modelDescription.inputDescriptionsByName.first!.key
            
            // Form batch inputs for model
            let inputs: [MLFeatureProvider] = try latents.map { sample in
                // Reference pipeline scales the latent samples before decoding
                let sampleScaled = MLShapedArray(unsafeUninitializedShape: sample.shape) { scalars, _ in
                    sample.withUnsafeShapedBufferPointer { sample, _, _ in
                        vDSP.divide(sample, scaleFactor, result: &scalars)
                    }
                }
                
                let dict = [inputName: MLMultiArray(sampleScaled)]
                return try MLDictionaryFeatureProvider(dictionary: dict)
            }
            let batch = MLArrayBatchProvider(array: inputs)
            
            return try model.predictions(fromBatch: batch)
        }
        
        // Transform the outputs to CGImages
        let images: [CGImage] = (0..<results.count).map { i in
            let result = results.features(at: i)
            let outputName = result.featureNames.first!
            let output = result.featureValue(for: outputName)!.multiArrayValue!
            
            return toRGBCGImage(MLShapedArray<Float32>(converting: output))
        }
        
        return images
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
                      colorSpace: CGColorSpaceCreateDeviceRGB(),
                      bitmapInfo: bitmapInfo)!)!
        
        return cgImage
    }
}
