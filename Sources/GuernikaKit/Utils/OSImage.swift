//
//  OSImage.swift
//  
//
//  Created by Guillermo Cique Fernández on 23/5/23.
//

import SwiftUI

#if os(macOS)
public typealias OSImage = NSImage

public extension NSBitmapImageRep {
    var png: Data? { representation(using: .png, properties: [:]) }
}

public extension Data {
    var bitmap: NSBitmapImageRep? { NSBitmapImageRep(data: self) }
}

public extension OSImage {
    convenience init(cgImage: CGImage) {
        self.init(cgImage: cgImage, size: NSSize(width: cgImage.width, height: cgImage.height))
    }
    
    func pngData() -> Data? {
        tiffRepresentation?.bitmap?.png
    }
    
    func scaledAspectFill(size newSize: CGSize) -> OSImage {
        guard size != newSize else { return self }
        let (offset, scaledSize): (CGPoint, CGSize) = size.scaledAspectFill(size: newSize)
        return scaled(newSize: newSize, scaledSize: scaledSize, offset: offset)
    }
    
    func scaledAspectFit(size newSize: CGSize) -> OSImage {
        guard size != newSize else { return self }
        let (offset, scaledSize): (CGPoint, CGSize) = size.scaledAspectFit(size: newSize)
        return scaled(newSize: newSize, scaledSize: scaledSize, offset: offset)
    }
    
    private func scaled(newSize: CGSize, scaledSize: CGSize, offset: CGPoint) -> OSImage {
        let resizedImage = NSImage(size: newSize)
        resizedImage.lockFocus()
        defer {
            resizedImage.unlockFocus()
        }
        if let context = NSGraphicsContext.current {
            context.imageInterpolation = .high
            draw(
                in: NSRect(origin: offset, size: scaledSize),
                from: NSRect(origin: .zero, size: size),
                operation: .copy,
                fraction: 1
            )
        }
        return resizedImage
    }
    
    var cgImage: CGImage? {
        var imageRect = CGRect(x: 0, y: 0, width: size.width, height: size.height)
        return cgImage(forProposedRect: &imageRect, context: nil, hints: nil)
    }
}

public extension Image {
    init(osImage: OSImage) {
        self.init(nsImage: osImage)
    }
}
#else
public typealias OSImage = UIImage

public extension Image {
    init(osImage: OSImage) {
        self.init(uiImage: osImage)
    }
}

public extension OSImage {
    convenience init?(contentsOf url: URL) {
        guard let data = try? Data(contentsOf: url) else { return nil }
        self.init(data: data)
    }
    
    func scaledAspectFill(size newSize: CGSize) -> OSImage {
        guard size != newSize else { return self }
        let (offset, scaledSize): (CGPoint, CGSize) = size.scaledAspectFill(size: newSize)
        return scaled(newSize: newSize, scaledSize: scaledSize, offset: offset)
    }
    
    func scaledAspectFit(size newSize: CGSize) -> OSImage {
        guard size != newSize else { return self }
        let (offset, scaledSize): (CGPoint, CGSize) = size.scaledAspectFit(size: newSize)
        return scaled(newSize: newSize, scaledSize: scaledSize, offset: offset)
    }
    
    private func scaled(newSize: CGSize, scaledSize: CGSize, offset: CGPoint) -> OSImage {
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        let image = UIGraphicsImageRenderer(size: newSize, format: format).image { _ in
            draw(in: CGRect(origin: offset, size: scaledSize))
        }
        return image.withRenderingMode(renderingMode)
    }
}
#endif
