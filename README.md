# AI-Powered Playlist Generator

An intelligent music playlist generator that analyzes audio files to create mood and energy-based playlists using PyTorch machine learning.

## Features

- **Audio Analysis**: Extract tempo, loudness, spectral centroid, and other audio features
- **Mood Classification**: Use PyTorch models to classify songs by mood and energy
- **Pattern Recognition**: Identify patterns across songs to form cohesive playlists
- **Automatic Playlist Generation**: Create playlists based on detected patterns and similarities

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
├── src/
│   ├── audio_analysis/     # Core audio analysis functions
│   ├── feature_extraction/ # Feature extraction pipeline
│   ├── models/            # PyTorch models for classification
│   ├── playlist_gen/      # Playlist generation algorithms
│   └── data/              # Data management and metadata handling
├── tests/                 # Unit tests
├── examples/              # Example usage scripts
└── requirements.txt       # Python dependencies
```

## Usage

```python
from src.playlist_gen import PlaylistGenerator

# Initialize the generator
generator = PlaylistGenerator()

# Add songs to analyze
generator.add_songs("path/to/music/folder")

# Generate playlists
playlists = generator.generate_playlists()
```

## Development Status

🚧 **Under Development** - Core audio analysis and ML model implementation in progress.
