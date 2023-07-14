//
//  SampleInput.swift
//  
//
//  Created by Guillermo Cique Fernández on 25/5/23.
//

import Schedulers
import CoreGraphics

public struct SampleInput: Hashable {
    public var prompt: String
    public var negativePrompt: String
    public var initImage: CGImage?
    public var strength: Float?
    public var inpaintMask: CGImage?
    public var seed: UInt32
    public var stepCount: Int
    /// Controls the influence of the text prompt on sampling process (0=random images)
    public var guidanceScale: Float
    public var imageGuidanceScale: Float?
    public var scheduler: Schedulers = .pndm
    
    public init(
        prompt: String,
        negativePrompt: String = "",
        seed: UInt32 = UInt32.random(in: 0...UInt32.max),
        stepCount: Int = 20,
        guidanceScale: Float = 7.5,
        scheduler: Schedulers = .pndm
    ) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.initImage = nil
        self.strength = nil
        self.inpaintMask = nil
        self.seed = seed
        self.stepCount = stepCount
        self.guidanceScale = guidanceScale
        self.scheduler = scheduler
    }
    
    public init(
        prompt: String,
        negativePrompt: String = "",
        initImage: CGImage?,
        strength: Float = 0.75,
        seed: UInt32 = UInt32.random(in: 0...UInt32.max),
        stepCount: Int = 20,
        guidanceScale: Float = 7.5,
        scheduler: Schedulers = .pndm
    ) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.initImage = initImage
        self.strength = strength
        self.inpaintMask = nil
        self.seed = seed
        self.stepCount = stepCount
        self.guidanceScale = guidanceScale
        self.scheduler = scheduler
    }
    
    public init(
        prompt: String,
        negativePrompt: String = "",
        initImage: CGImage?,
        seed: UInt32 = UInt32.random(in: 0...UInt32.max),
        stepCount: Int = 20,
        guidanceScale: Float = 7.5,
        imageGuidanceScale: Float = 1.5,
        scheduler: Schedulers = .pndm
    ) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.initImage = initImage
        self.strength = nil
        self.inpaintMask = nil
        self.seed = seed
        self.stepCount = stepCount
        self.guidanceScale = guidanceScale
        self.imageGuidanceScale = imageGuidanceScale
        self.scheduler = scheduler
    }
    
    public init(
        prompt: String,
        negativePrompt: String = "",
        initImage: CGImage?,
        inpaintMask: CGImage,
        seed: UInt32 = UInt32.random(in: 0...UInt32.max),
        stepCount: Int = 20,
        guidanceScale: Float = 7.5,
        scheduler: Schedulers = .pndm
    ) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.initImage = initImage
        self.strength = nil
        self.inpaintMask = inpaintMask
        self.seed = seed
        self.stepCount = stepCount
        self.guidanceScale = guidanceScale
        self.scheduler = scheduler
    }
}

public extension SampleInput {
    init(
        prompt: String,
        negativePrompt: String = "",
        initImage: OSImage,
        strength: Float = 0.75,
        seed: UInt32 = UInt32.random(in: 0...UInt32.max),
        stepCount: Int = 50,
        guidanceScale: Float = 7.5,
        scheduler: Schedulers = .pndm
    ) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.initImage = initImage.cgImage!
        self.strength = strength
        self.inpaintMask = nil
        self.seed = seed
        self.stepCount = stepCount
        self.guidanceScale = guidanceScale
        self.scheduler = scheduler
    }
    
    init(
        prompt: String,
        negativePrompt: String = "",
        initImage: OSImage,
        seed: UInt32 = UInt32.random(in: 0...UInt32.max),
        stepCount: Int = 50,
        guidanceScale: Float = 7.5,
        imageGuidanceScale: Float = 1.5,
        scheduler: Schedulers = .pndm
    ) {
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
    
    init(
        prompt: String,
        negativePrompt: String = "",
        initImage: OSImage,
        inpaintMask: CGImage,
        seed: UInt32 = UInt32.random(in: 0...UInt32.max),
        stepCount: Int = 50,
        guidanceScale: Float = 7.5,
        scheduler: Schedulers = .pndm
    ) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.initImage = initImage.cgImage!
        self.strength = nil
        self.inpaintMask = inpaintMask
        self.seed = seed
        self.stepCount = stepCount
        self.guidanceScale = guidanceScale
        self.scheduler = scheduler
    }
    
    init(
        prompt: String,
        negativePrompt: String = "",
        initImage: OSImage,
        inpaintMask: OSImage,
        seed: UInt32 = UInt32.random(in: 0...UInt32.max),
        stepCount: Int = 50,
        guidanceScale: Float = 7.5,
        scheduler: Schedulers = .pndm
    ) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.initImage = initImage.cgImage!
        self.strength = nil
        self.inpaintMask = inpaintMask.cgImage!
        self.seed = seed
        self.stepCount = stepCount
        self.guidanceScale = guidanceScale
        self.scheduler = scheduler
    }
}
