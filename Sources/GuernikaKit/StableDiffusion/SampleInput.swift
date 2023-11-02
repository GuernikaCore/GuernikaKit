//
//  SampleInput.swift
//  
//
//  Created by Guillermo Cique Fernández on 25/5/23.
//

import Schedulers
import CoreGraphics

public struct SampleInput: Hashable {
    public var size: CGSize?
    public var prompt: String
    public var negativePrompt: String
    public var initImage: CGImage?
    public var inpaintMask: CGImage?
    public var strength: Float?
    public var seed: UInt32
    public var stepCount: Int
    public var originalStepCount: Int?
    /// Controls the influence of the text prompt on sampling process (0=random images)
    public var guidanceScale: Float
    public var imageGuidanceScale: Float?
    public var scheduler: Schedulers = .pndm
    
    // Text to image
    public init(
        size: CGSize? = nil,
        prompt: String,
        negativePrompt: String = "",
        seed: UInt32 = UInt32.random(in: 0...UInt32.max),
        stepCount: Int = 20,
        originalStepCount: Int? = nil,
        guidanceScale: Float = 7.5,
        scheduler: Schedulers = .pndm
    ) {
        self.size = size
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.initImage = nil
        self.strength = nil
        self.inpaintMask = nil
        self.seed = seed
        self.stepCount = stepCount
        self.originalStepCount = originalStepCount
        self.guidanceScale = guidanceScale
        self.scheduler = scheduler
    }
    
    // Image to image and inpainting
    public init(
        prompt: String,
        negativePrompt: String = "",
        initImage: CGImage,
        inpaintMask: CGImage? = nil,
        strength: Float = 0.75,
        seed: UInt32 = UInt32.random(in: 0...UInt32.max),
        stepCount: Int = 20,
        originalStepCount: Int? = nil,
        guidanceScale: Float = 7.5,
        scheduler: Schedulers = .pndm
    ) {
        self.size = CGSize(width: initImage.width, height: initImage.height)
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.initImage = initImage
        self.inpaintMask = inpaintMask
        self.strength = strength
        self.inpaintMask = inpaintMask
        self.seed = seed
        self.stepCount = stepCount
        self.originalStepCount = originalStepCount
        self.guidanceScale = guidanceScale
        self.scheduler = scheduler
    }
    
    // Instructions
    public init(
        prompt: String,
        negativePrompt: String = "",
        initImage: CGImage,
        seed: UInt32 = UInt32.random(in: 0...UInt32.max),
        stepCount: Int = 20,
        originalStepCount: Int? = nil,
        guidanceScale: Float = 7.5,
        imageGuidanceScale: Float = 1.5,
        scheduler: Schedulers = .pndm
    ) {
        self.size = CGSize(width: initImage.width, height: initImage.height)
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.initImage = initImage
        self.strength = nil
        self.inpaintMask = nil
        self.seed = seed
        self.stepCount = stepCount
        self.originalStepCount = originalStepCount
        self.guidanceScale = guidanceScale
        self.imageGuidanceScale = imageGuidanceScale
        self.scheduler = scheduler
    }
}

public extension SampleInput {
    // Image to image and inpainting
    init(
        prompt: String,
        negativePrompt: String = "",
        initImage: OSImage,
        inpaintMask: OSImage? = nil,
        strength: Float = 0.75,
        seed: UInt32 = UInt32.random(in: 0...UInt32.max),
        stepCount: Int = 20,
        originalStepCount: Int? = nil,
        guidanceScale: Float = 7.5,
        scheduler: Schedulers = .pndm
    ) {
        self.size = initImage.size
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.initImage = initImage.cgImage!
        self.strength = strength
        self.inpaintMask = inpaintMask?.cgImage!
        self.seed = seed
        self.stepCount = stepCount
        self.originalStepCount = originalStepCount
        self.guidanceScale = guidanceScale
        self.scheduler = scheduler
    }
    
    // Instructions
    init(
        prompt: String,
        negativePrompt: String = "",
        initImage: OSImage,
        seed: UInt32 = UInt32.random(in: 0...UInt32.max),
        stepCount: Int = 20,
        guidanceScale: Float = 7.5,
        imageGuidanceScale: Float = 1.5,
        scheduler: Schedulers = .pndm
    ) {
        self.size = initImage.size
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.initImage = initImage.cgImage!
        self.strength = nil
        self.inpaintMask = nil
        self.seed = seed
        self.stepCount = stepCount
        self.guidanceScale = guidanceScale
        self.imageGuidanceScale = imageGuidanceScale
        self.scheduler = scheduler
    }
}
