//
//  Cache.swift
//  
//
//  Created by Guillermo Cique Fernández on 24/5/23.
//

import Foundation

public class Cache<Key: Hashable, Value> {
    let maxItems: Int
    private var cacheOrder: [Key] = []
    private var cache: [Key: Value] = [:]
    
    init(maxItems: Int) {
        self.maxItems = maxItems
    }
    
    subscript(key: Key) -> Value? {
        get {
            guard let value = cache[key] else { return nil }
            cacheOrder.removeAll(where: { $0 == key })
            cacheOrder.insert(key, at: 0)
            return value
        }
        set {
            cacheOrder.removeAll(where: { $0 == key })
            if newValue != nil {
                cacheOrder.insert(key, at: 0)
                if cacheOrder.count > maxItems, let oldestKey = cacheOrder.popLast() {
                    cache.removeValue(forKey: oldestKey)
                }
            }
            cache[key] = newValue
        }
    }
    
    func clear() {
        cacheOrder.removeAll()
        cache.removeAll()
    }
}
