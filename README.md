# Percepta

![Percepta](https://github.com/han-hangoc-le/percepta/blob/feat/hanle-thunguyen-swiftui/Percepta_thumbnail.png)

Developed during HackPrinceton Fall'25, **Percepta** is an iOS camera application that transforms your view of the world through different knowledge-based "lenses" - seeing everyday scenes through the eyes of a mathematician, physicist, biologist, or artist.

## Overview

Percepta uses your device's camera combined with AI-powered object detection to provide unique perspectives on the world around you. Select a lens mode, point your camera, and receive interpretations and insights tailored to that specific viewpoint.

## Features

### 🔢 Mathematician Lens

Translate scenes into formulas, ratios, and elegant proofs hidden in plain sight.

### ⚛️ Physicist Lens

Surface forces, trajectories, and thought experiments that govern every motion.

### 🧬 Biologist Lens

Reveal living systems, evolutionary quirks, and ecological stories in the environment.

### 🎨 Artist Lens

Highlight palettes, composition tricks, and creative insights to inspire your next masterpiece.

## Tech Stack

- **Frontend**: SwiftUI (iOS)
- **Language**: Swift
- **Architecture**: MVVM with modern Swift concurrency (async/await)
- **Backend Integration**: RESTful API communication with Flask backend
- **Camera**: AVFoundation framework

## Project Structure

```
Percepta/
├── PerceptaApp.swift          # App entry point
├── ContentView.swift          # Navigation root
├── Camera/
│   ├── CameraModel.swift      # Camera logic and capture handling
│   └── CameraPreview.swift    # Camera preview UI component
├── Components/
│   ├── CameraView.swift       # Main camera interface
│   ├── LenSelectorView.swift  # Lens mode selector
│   ├── PermissionDeniedView.swift
│   └── ShutterButton.swift    # Camera capture button
├── Extensions/
│   └── Color+Hex.swift        # Hex color support
├── Models/
│   └── Lens.swift             # Lens mode data models
├── Screens/
│   ├── HomeScreen.swift       # Lens selection screen
│   └── CameraScreen.swift     # Camera capture screen
└── Services/
    └── APIService.swift       # Backend API communication
```

## Installation

### Prerequisites

- Xcode 14.0 or later
- iOS 15.0 or later
- Swift 5.7 or later
- Active iOS device or simulator with camera support

### Setup

1. Clone the repository:

```bash
git clone https://github.com/han-hangoc-le/percepta.git
cd percepta
```

2. Open the project in Xcode:

```bash
open Percepta.xcodeproj
```

3. Configure backend URL in `APIService.swift` if needed (default: `http://127.0.0.1:5000/api`)

4. Build and run the project on your device or simulator.

## Backend API

The app expects a Flask backend running with the following endpoints:

### `/api/health` (GET)

Health check endpoint to verify backend connectivity.

**Response:**

```json
{
  "status": "ok"
}
```

### `/api/detect` (POST)

Object detection endpoint that processes camera frames.

**Request:**

```json
{
  "imageBase64": "base64_encoded_image_data",
  "lensMode": "mathematician|physicist|biologist|artist"
}
```

**Response:**

```json
{
  "objects": [
    {
      "id": "unique-id",
      "label": "Object Name",
      "confidence": 0.95,
      "boundingBox": {
        "x": 0.1,
        "y": 0.2,
        "width": 0.3,
        "height": 0.4
      }
    }
  ],
  "lensMode": "mathematician",
  "message": "Interpretation message from the lens perspective"
}
```

## Usage

1. **Launch the app** - You'll see the WorldLens home screen
2. **Select a lens mode** - Choose from Mathematician, Physicist, Biologist, or Artist
3. **Open Camera** - Tap the "Open Camera" button
4. **Grant permissions** - Allow camera access when prompted
5. **Capture frames** - Tap the shutter button to analyze what you're seeing
6. **View insights** - Receive AI-powered interpretations based on your selected lens

## Features & Highlights

- **Offline Mock Mode**: Automatically falls back to mock data if backend is unavailable
- **Real-time Camera**: Native camera integration with AVFoundation
- **Dark Mode UI**: Sleek, modern interface optimized for low-light viewing
- **Async/Await**: Modern Swift concurrency for smooth performance
- **Smart Error Handling**: User-friendly error messages and automatic retry logic
- **Flexible Backend**: Configurable API endpoints with multiple fallback options

## Error Handling

The app includes robust error handling:

- Network connectivity issues
- Backend timeouts
- Invalid API responses
- Camera permission denials

## Development

### Running Tests

```bash
# Unit Tests
xcodebuild test -scheme Percepta -destination 'platform=iOS Simulator,name=iPhone 14'

# UI Tests
xcodebuild test -scheme PerceptaUITests -destination 'platform=iOS Simulator,name=iPhone 14'
```

### Building for Release

1. Select your development team in Xcode project settings
2. Update bundle identifier if needed
3. Archive the project: Product → Archive
4. Distribute through App Store Connect or ad-hoc distribution

## Configuration

### API Configuration

Modify `APIService.swift` to configure backend endpoints:

```swift
private let defaultPort = "5000"
private let defaultPath = "/api"
private let defaultTimeout: TimeInterval = 10
```

### Camera Settings

Adjust camera settings in `CameraModel.swift` for different quality or performance requirements.

## License

This project is available under the MIT License.

## Team

- Thu Nguyen on the App dev and CV Phase
- Alex Tran on the GenAI Overlay and CV/AR Phase
- Ethan Do on the LLM and GenAI Overlay
- Han Le on the App dev and CV/AR Phase

## Acknowledgments

- Built with SwiftUI and modern iOS development best practices
- Inspired by the desire to see the world through different perspectives
- Thanks to the HackPrinceton organizers and mentors
