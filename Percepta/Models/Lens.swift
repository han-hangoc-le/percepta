import SwiftUI

// MARK: - Model
struct LensMode: Identifiable, Codable, Equatable {
    let id: String
    let name: String
    let icon: String
    let color: String
    let description: String
}

// MARK: - Data
let LENS_MODES: [LensMode] = [
    LensMode(
        id: "mathematician",
        name: "Mathematician",
        icon: "🔢",
        color: "#3b82f6",
        description: "Translate scenes into formulas, ratios, and elegant proofs hidden in plain sight."
    ),
    LensMode(
        id: "physicist",
        name: "Physicist",
        icon: "⚛️",
        color: "#8b5cf6",
        description: "Surface forces, trajectories, and thought experiments that govern every motion."
    ),
    LensMode(
        id: "biologist",
        name: "Biologist",
        icon: "🧬",
        color: "#10b981",
        description: "Reveal living systems, evolutionary quirks, and ecological stories in the environment."
    ),
    LensMode(
        id: "artist",
        name: "Artist",
        icon: "🎨",
        color: "#f59e0b",
        description: "Highlight palettes, composition tricks, and creative prompts to inspire your next masterpiece."
    ),
    LensMode(
        id: "ecologist",
        name: "Ecologist",
        icon: "🌿",
        color: "#22c55e",
        description: "Observe interactions between species, energy flows, and balance in natural ecosystems."
    ),
    LensMode(
        id: "cultural",
        name: "Cultural Analyst",
        icon: "🏺",
        color: "#ef4444",
        description: "Decode traditions, symbols, and social meanings embedded in everyday human activity."
    )
]
