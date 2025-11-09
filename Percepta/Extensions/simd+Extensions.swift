import simd

extension simd_float4x4 {
    var translation: SIMD3<Float> {
        let column = columns.3
        return SIMD3<Float>(column.x, column.y, column.z)
    }
}
