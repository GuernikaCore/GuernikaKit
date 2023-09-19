//
//  ConditioningInput.swift
//  
//
//  Created by Guillermo Cique Fernández on 10/9/23.
//

import CoreML
import Accelerate
import Foundation

public enum ConditioningMethod: String, Hashable, Decodable {
    case canny
    case depth
    case pose
    case mlsd
    case normal
    case scribble
    case hed
    case segmentation
    case unknown
    
    public init(from decoder: Swift.Decoder) throws {
        do {
            let container = try decoder.singleValueContainer()
            let rawValue = try container.decode(String.self)
            if let value = Self(rawValue: rawValue) {
                self = value
            } else {
                self = .unknown
            }
        } catch {
            self = .unknown
        }
    }
}

public protocol ConditioningModule {
    var url: URL { get }
    var method: ConditioningMethod { get }
    var sampleSize: CGSize { get }
    var minimumSize: CGSize { get }
    var maximumSize: CGSize { get }
    var attentionImplementation: AttentionImplementation { get }
    
    func unloadResources()
}

public class ConditioningInput: Identifiable {
    public let id: UUID = UUID()
    public let module: any ConditioningModule
    public var method: ConditioningMethod { module.method }
    public var conditioningScale: Float = 1.0
    public var image: CGImage? {
        didSet {
            // Reset imageData, it will be updated when run with the correct size
            imageData = nil
        }
    }
    var imageData: MLShapedArray<Float32>?
    
    public init(module: any ConditioningModule) {
        self.module = module
    }
}

extension Array where Element == [String: MLShapedArray<Float32>] {
    func addResiduals() -> [String: MLShapedArray<Float32>]? {
        guard count > 1 else { return first }
        var total: [String: MLShapedArray<Float32>] = self[0]
        for res in self.dropFirst() {
            let stride = vDSP_Stride(1)
            // Go through each latent residuals of the total and current ControlNet residuals
            for key in total.keys {
                total[key]?.withUnsafeMutableShapedBufferPointer { total, _, _ in
                    res[key]?.withUnsafeShapedBufferPointer { latent, _, _ in
                        let n = vDSP_Length(total.count)
                        vDSP_vadd(
                            total.baseAddress!, stride,
                            latent.baseAddress!, stride,
                            total.baseAddress!, stride,
                            n
                        )
                    }
                }
            }
        }
        return total
    }
}
