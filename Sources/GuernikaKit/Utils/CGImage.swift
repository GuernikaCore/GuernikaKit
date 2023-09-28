//
//  CGImage.swift
//  
//
//  Created by Guillermo Cique Fernández on 23/5/23.
//

import CoreML
import Accelerate

public extension CGImage {
    typealias PixelBufferPFx1 = vImage.PixelBuffer<vImage.PlanarF>
    typealias PixelBufferP8x3 = vImage.PixelBuffer<vImage.Planar8x3>
    typealias PixelBufferIFx3 = vImage.PixelBuffer<vImage.InterleavedFx3>
    typealias PixelBufferP8x1 = vImage.PixelBuffer<vImage.Planar8>
    typealias PixelBufferI8x2 = vImage.PixelBuffer<vImage.Interleaved8x2>
    typealias PixelBufferI8x3 = vImage.PixelBuffer<vImage.Interleaved8x3>
    typealias PixelBufferI8x4 = vImage.PixelBuffer<vImage.Interleaved8x4>
    
    /// Convert CGImage to a shaped array normalizing the values to a range of min-max
    ///
    /// - Parameters:
    ///   - min: Min value of the array
    ///   - max: Max value of the array
    ///   - components: Array of 4 booleans indicating what components to include in the result ARGB
    /// - Returns: An array of shape [1, components, height, width]
    func toShapedArray(
        min: Float = -1.0,
        max: Float = 1.0,
        components: [Bool] = [false, true, true, true] // ARGB
    ) -> MLShapedArray<Float32> {
        var sourceFormat = vImage_CGImageFormat(cgImage: self)!
        var mediumFormat = vImage_CGImageFormat(
            bitsPerComponent: 8 * MemoryLayout<UInt8>.size,
            bitsPerPixel: 8 * MemoryLayout<UInt8>.size * 4,
            colorSpace: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.first.rawValue)
        )!
        let width = vImagePixelCount(exactly: width)!
        let height = vImagePixelCount(exactly: height)!
        
        var sourceImageBuffer = try! vImage_Buffer(cgImage: self)
        
        var mediumDesination = try! vImage_Buffer(width: Int(width), height: Int(height), bitsPerPixel: mediumFormat.bitsPerPixel)
        
        let converter = vImageConverter_CreateWithCGImageFormat(
            &sourceFormat,
            &mediumFormat,
            nil,
            vImage_Flags(kvImagePrintDiagnosticsToConsole),
            nil
        )!.takeRetainedValue()
        
        vImageConvert_AnyToAny(converter, &sourceImageBuffer, &mediumDesination, nil, vImage_Flags(kvImagePrintDiagnosticsToConsole))
        
        // https://developer.apple.com/documentation/accelerate/finding_the_sharpest_image_in_a_sequence_of_captured_images#4199619
        let storageA = UnsafeMutableBufferPointer<Float>.allocate(capacity: Int(width * height))
        let storageR = UnsafeMutableBufferPointer<Float>.allocate(capacity: Int(width * height))
        let storageG = UnsafeMutableBufferPointer<Float>.allocate(capacity: Int(width * height))
        let storageB = UnsafeMutableBufferPointer<Float>.allocate(capacity: Int(width * height))
        var destinationA = vImage_Buffer(data: storageA.baseAddress!, height: height, width: width, rowBytes: Int(width) * MemoryLayout<Float>.stride)
        var destinationR = vImage_Buffer(data: storageR.baseAddress!, height: height, width: width, rowBytes: Int(width) * MemoryLayout<Float>.stride)
        var destinationG = vImage_Buffer(data: storageG.baseAddress!, height: height, width: width, rowBytes: Int(width) * MemoryLayout<Float>.stride)
        var destinationB = vImage_Buffer(data: storageB.baseAddress!, height: height, width: width, rowBytes: Int(width) * MemoryLayout<Float>.stride)
        defer {
            storageA.deallocate()
            storageR.deallocate()
            storageG.deallocate()
            storageB.deallocate()
        }
        
        var minFloat: [Float] = Array(repeating: min, count: 4)
        var maxFloat: [Float] = Array(repeating: max, count: 4)
        
        vImageConvert_ARGB8888toPlanarF(&mediumDesination, &destinationA, &destinationR, &destinationG, &destinationB, &maxFloat, &minFloat, .zero)
        
        let componentsData: [Data] = zip(components, [destinationA, destinationR, destinationG, destinationB])
            .compactMap { include, component in
                guard include else { return nil }
                return Data(bytes: component.data, count: Int(width) * Int(height) * MemoryLayout<Float>.size)
            }
        let imageData = componentsData.reduce(Data(), +)
        
        let shapedArray = MLShapedArray<Float32>(data: imageData, shape: [1, componentsData.count, Int(height), Int(width)])
        
        return shapedArray
    }
    
    func toAlphaShapedArray() -> MLShapedArray<Float32> {
        return toShapedArray(min: 0, max: 1, components: [true, false, false, false])
    }
    
    var size: CGSize { CGSize(width: width, height: height) }
    
    func scaledAspectFill(size newSize: CGSize) -> CGImage {
        let size = CGSize(width: width, height: height)
        guard size != newSize else { return self }
        let (offset, scaledSize): (CGPoint, CGSize) = size.scaledAspectFill(size: newSize)
        return scaled(newSize: newSize, scaledSize: scaledSize, offset: offset)
    }
    
    func scaledAspectFit(size newSize: CGSize) -> CGImage {
        let size = CGSize(width: width, height: height)
        guard size != newSize else { return self }
        let (offset, scaledSize): (CGPoint, CGSize) = size.scaledAspectFit(size: newSize)
        return scaled(newSize: newSize, scaledSize: scaledSize, offset: offset)
    }
    
    private func scaled(newSize: CGSize, scaledSize: CGSize, offset: CGPoint) -> CGImage {
        let context = CGContext(
            data: nil,
            width: Int(newSize.width),
            height: Int(newSize.height),
            bitsPerComponent: bitsPerComponent,
            bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )!
        context.interpolationQuality = .high
        context.draw(self, in: CGRect(origin: offset, size: scaledSize))

        return context.makeImage()!
    }
}
