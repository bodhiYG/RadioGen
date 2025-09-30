"""
Basic usage example for the AI-Powered Playlist Generator.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from playlist_gen import PlaylistGenerator


def main():
    """Demonstrate basic usage of the playlist generator."""
    
    # Initialize the generator
    print("Initializing AI-Powered Playlist Generator...")
    generator = PlaylistGenerator(
        model_dir="models",
        data_dir="data"
    )
    
    # Example: Add songs from a directory
    music_directory = "path/to/your/music/folder"  # Replace with actual path
    
    if os.path.exists(music_directory):
        print(f"Adding songs from {music_directory}...")
        added_files = generator.add_songs_from_directory(music_directory)
        print(f"Added {len(added_files)} songs")
    else:
        print(f"Music directory {music_directory} not found. Please update the path.")
        print("For demonstration, we'll show how to use the system...")
    
    # Analyze songs (this would normally be done after adding songs)
    print("\nAnalyzing songs...")
    try:
        analysis_results = generator.analyze_songs()
        print(f"Analysis complete: {analysis_results['processed_count']} songs processed")
    except ValueError as e:
        print(f"Analysis skipped: {e}")
    
    # Extract features
    print("\nExtracting features...")
    try:
        features = generator.extract_features()
        print(f"Extracted features shape: {features.shape}")
    except ValueError as e:
        print(f"Feature extraction skipped: {e}")
    
    # Example: Train models (requires labeled data)
    print("\nTraining models...")
    print("Note: This requires labeled training data (mood and energy labels)")
    print("For demonstration purposes, we'll show the training interface:")
    
    # Example labels (in practice, these would come from user input or existing datasets)
    example_mood_labels = [0, 1, 2, 3, 4, 5] * 10  # Example mood labels
    example_energy_labels = [0.2, 0.4, 0.6, 0.8, 1.0] * 12  # Example energy labels
    
    try:
        # This would work if we had enough songs and matching labels
        if len(generator.analysis_cache) >= len(example_mood_labels):
            training_results = generator.train_models(
                mood_labels=example_mood_labels[:len(generator.analysis_cache)],
                energy_labels=example_energy_labels[:len(generator.analysis_cache)]
            )
            print("Training completed successfully!")
        else:
            print("Not enough songs for training. Need at least 6 songs.")
    except ValueError as e:
        print(f"Training skipped: {e}")
    
    # Example: Generate playlists
    print("\nGenerating playlists...")
    try:
        # Mood-based playlist
        mood_playlist = generator.generate_playlist(
            playlist_type="mood_based",
            target_mood="Happy/Upbeat",
            playlist_length=10
        )
        print(f"Generated mood-based playlist with {len(mood_playlist)} songs")
        
        # Energy-based playlist
        energy_playlist = generator.generate_playlist(
            playlist_type="energy_based",
            target_energy=0.7,
            playlist_length=10
        )
        print(f"Generated energy-based playlist with {len(energy_playlist)} songs")
        
        # Cluster-based playlist
        cluster_playlist = generator.generate_playlist(
            playlist_type="cluster_based",
            playlist_length=10
        )
        print(f"Generated cluster-based playlist with {len(cluster_playlist)} songs")
        
        # Transition playlist
        transition_playlist = generator.generate_playlist(
            playlist_type="transition",
            playlist_length=10
        )
        print(f"Generated transition playlist with {len(transition_playlist)} songs")
        
    except ValueError as e:
        print(f"Playlist generation skipped: {e}")
    
    # Example: Get song information
    print("\nGetting song information...")
    try:
        song_ids = list(generator.analysis_cache.keys())
        if song_ids:
            song_info = generator.get_song_info(song_ids[0])
            print(f"Song info for {song_ids[0]}:")
            print(f"  Title: {song_info['metadata'].get('title', 'Unknown')}")
            print(f"  Duration: {song_info['analysis'].get('duration', 0):.2f} seconds")
            print(f"  Tempo: {song_info['analysis'].get('tempo', 0):.2f} BPM")
    except Exception as e:
        print(f"Song info retrieval skipped: {e}")
    
    print("\nDemo completed! Check the 'models' and 'data' directories for saved files.")


if __name__ == "__main__":
    main()
