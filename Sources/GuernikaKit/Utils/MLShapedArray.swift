//
//  MLShapedArray.swift
//  
//
//  Created by Guillermo Cique Fernández on 25/5/23.
//

import CoreML
import Accelerate

extension MLShapedArray where ArrayLiteralElement == Float32 {
    mutating func scale(_ scaleFactor: Float32) {
        guard scaleFactor != 1 else { return }
        withUnsafeMutableShapedBufferPointer { pointer, _, _ in
            vDSP.multiply(scaleFactor, pointer, result: &pointer)
        }
    }
}
