//
//  CGSize.swift
//  
//
//  Created by Guillermo Cique Fernández on 23/5/23.
//

import CoreGraphics

public extension CGSize {
    var aspectRatio: Double { width / height }
    
    func scaledAspectFill(size: CGSize) -> (CGPoint, CGSize) {
        var offset: CGPoint = .zero
        var scaledSize: CGSize = self
        if size.width / size.height < width / height {
            scaledSize.height = size.height
            scaledSize.width = size.height * (width / height)
            offset.x = (size.width - scaledSize.width) / 2
        } else {
            scaledSize.width = size.width
            scaledSize.height = size.width * (height / width)
            offset.y = (size.height - scaledSize.height) / 2
        }
        return (offset, scaledSize)
    }
    
    func scaledAspectFit(size: CGSize) -> (CGPoint, CGSize) {
        var offset: CGPoint = .zero
        var scaledSize: CGSize = self
        if width < height {
            scaledSize.height = size.height
            scaledSize.width = size.height * (width / height)
            offset.x = (size.width - scaledSize.width) / 2
        } else {
            scaledSize.width = size.width
            scaledSize.height = size.width * (height / width)
            offset.y = (size.height - scaledSize.height) / 2
        }
        return (offset, scaledSize)
    }
    
    func isBetween(min: CGSize, max: CGSize) -> Bool {
        guard height >= min.height, height <= max.height else { return false }
        guard width >= min.width, width <= max.width else { return false }
        return true
    }
}

extension CGSize: Hashable {
    public func hash(into hasher: inout Hasher) {
        hasher.combine(width)
        hasher.combine(height)
    }
}
