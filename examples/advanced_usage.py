"""
Advanced usage example showing custom training and playlist generation.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from playlist_gen import PlaylistGenerator
from audio_analysis import AudioAnalyzer, FeatureExtractor
from models import MoodClassifier, EnergyClassifier, PatternAnalyzer


def create_sample_data():
    """Create sample audio features for demonstration."""
    np.random.seed(42)
    
    # Create sample features (60 songs, 50 features each)
    n_songs = 60
    n_features = 50
    
    features = np.random.randn(n_songs, n_features)
    
    # Create realistic mood labels (0-5)
    mood_labels = np.random.randint(0, 6, n_songs)
    
    # Create realistic energy labels (0.0-1.0)
    energy_labels = np.random.uniform(0.0, 1.0, n_songs)
    
    # Create sample metadata
    metadata = []
    for i in range(n_songs):
        metadata.append({
            'song_id': f'song_{i:03d}',
            'title': f'Sample Song {i+1}',
            'artist': f'Artist {(i % 10) + 1}',
            'genre': ['Rock', 'Pop', 'Jazz', 'Classical', 'Electronic'][i % 5],
            'duration': np.random.uniform(120, 300),  # 2-5 minutes
            'file_path': f'/path/to/song_{i:03d}.mp3'
        })
    
    return features, mood_labels, energy_labels, metadata


def demonstrate_audio_analysis():
    """Demonstrate audio analysis capabilities."""
    print("=== Audio Analysis Demo ===")
    
    analyzer = AudioAnalyzer()
    
    # Create sample audio data (sine wave)
    sample_rate = 22050
    duration = 5  # seconds
    frequency = 440  # A4 note
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * frequency * t)
    
    # Add some variation to make it more realistic
    audio += 0.1 * np.sin(2 * np.pi * frequency * 2 * t)  # Harmonic
    audio += 0.05 * np.random.randn(len(audio))  # Noise
    
    print(f"Sample audio: {len(audio)} samples, {duration}s duration")
    
    # Extract tempo features
    tempo_features = analyzer.extract_tempo(audio)
    print(f"Tempo features: {tempo_features}")
    
    # Extract loudness features
    loudness_features = analyzer.extract_loudness(audio)
    print(f"Loudness features: {loudness_features}")
    
    # Extract spectral features
    spectral_features = analyzer.extract_spectral_features(audio)
    print(f"Spectral features: {spectral_features}")
    
    # Extract rhythm features
    rhythm_features = analyzer.extract_rhythm_features(audio)
    print(f"Rhythm features: {rhythm_features}")


def demonstrate_feature_extraction():
    """Demonstrate feature extraction pipeline."""
    print("\n=== Feature Extraction Demo ===")
    
    # Create sample analysis results
    analysis_results = []
    for i in range(5):
        result = {
            'song_id': f'song_{i}',
            'duration': np.random.uniform(120, 300),
            'tempo': np.random.uniform(80, 160),
            'tempo_stability': np.random.uniform(0.5, 1.0),
            'rms_mean': np.random.uniform(0.1, 0.5),
            'rms_std': np.random.uniform(0.01, 0.1),
            'spectral_centroid_mean': np.random.uniform(1000, 4000),
            'spectral_centroid_std': np.random.uniform(100, 500),
            'mfcc_mean': np.random.uniform(-10, 10, 13).tolist(),
            'mfcc_std': np.random.uniform(0, 5, 13).tolist(),
            'chroma_mean': np.random.uniform(0, 1, 12).tolist(),
            'chroma_std': np.random.uniform(0, 0.5, 12).tolist(),
        }
        analysis_results.append(result)
    
    # Extract features
    extractor = FeatureExtractor(normalize=True, use_pca=False)
    features = extractor.extract_features(analysis_results)
    
    print(f"Extracted features shape: {features.shape}")
    print(f"Feature names: {extractor.get_feature_names()[:10]}...")  # Show first 10
    
    return features


def demonstrate_model_training():
    """Demonstrate model training."""
    print("\n=== Model Training Demo ===")
    
    # Create sample data
    features, mood_labels, energy_labels, metadata = create_sample_data()
    
    # Split data
    train_size = int(0.8 * len(features))
    train_features = features[:train_size]
    val_features = features[train_size:]
    train_mood_labels = mood_labels[:train_size]
    val_mood_labels = mood_labels[train_size:]
    train_energy_labels = energy_labels[:train_size]
    val_energy_labels = energy_labels[train_size:]
    
    print(f"Training data: {len(train_features)} samples")
    print(f"Validation data: {len(val_features)} samples")
    
    # Train mood classifier
    print("\nTraining mood classifier...")
    mood_classifier = MoodClassifier(input_size=features.shape[1])
    mood_trainer = MoodClassifierTrainer(mood_classifier)
    
    # Create datasets
    from torch.utils.data import DataLoader
    train_dataset = MoodClassifier.MoodDataset(train_features, train_mood_labels)
    val_dataset = MoodClassifier.MoodDataset(val_features, val_mood_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Train (short training for demo)
    history = mood_trainer.train(train_loader, val_loader, epochs=10, patience=5)
    
    print(f"Mood classifier training completed!")
    print(f"Final validation accuracy: {history['val_accuracy'][-1]:.2f}%")
    
    # Train energy classifier
    print("\nTraining energy classifier...")
    energy_classifier = EnergyClassifier(input_size=features.shape[1])
    energy_trainer = EnergyClassifierTrainer(energy_classifier)
    
    train_dataset = EnergyClassifier.EnergyDataset(train_features, train_energy_labels)
    val_dataset = EnergyClassifier.EnergyDataset(val_features, val_energy_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Train (short training for demo)
    history = energy_trainer.train(train_loader, val_loader, epochs=10, patience=5)
    
    print(f"Energy classifier training completed!")
    print(f"Final validation MAE: {history['val_mae'][-1]:.4f}")
    
    return mood_trainer, energy_trainer


def demonstrate_pattern_analysis():
    """Demonstrate pattern analysis capabilities."""
    print("\n=== Pattern Analysis Demo ===")
    
    # Create sample data
    features, mood_labels, energy_labels, metadata = create_sample_data()
    
    # Initialize pattern analyzer
    analyzer = PatternAnalyzer(similarity_metric='cosine', clustering_method='kmeans')
    
    # Calculate similarity matrix
    similarity_matrix = analyzer.calculate_similarity_matrix(features)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    
    # Find similar songs
    similar_songs = analyzer.find_similar_songs(0, top_k=5)
    print(f"Songs similar to song 0: {similar_songs}")
    
    # Cluster songs
    cluster_labels = analyzer.cluster_songs(features, n_clusters=5)
    unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
    print(f"Clusters found: {len(unique_clusters)}")
    print(f"Cluster sizes: {dict(zip(unique_clusters, counts))}")
    
    # Analyze cluster characteristics
    feature_names = [f'feature_{i}' for i in range(features.shape[1])]
    cluster_analysis = analyzer.analyze_cluster_characteristics(
        features, cluster_labels, feature_names
    )
    
    print(f"Cluster analysis completed for {len(cluster_analysis)} clusters")
    
    # Create playlist from cluster
    playlist = analyzer.create_playlist_from_cluster(
        0, cluster_labels, metadata, max_songs=10
    )
    print(f"Created playlist with {len(playlist)} songs from cluster 0")
    
    # Create transition playlist
    transition_playlist = analyzer.create_transition_playlist(
        0, features, metadata, playlist_length=10
    )
    print(f"Created transition playlist with {len(transition_playlist)} songs")
    
    return analyzer


def demonstrate_full_pipeline():
    """Demonstrate the complete pipeline."""
    print("\n=== Full Pipeline Demo ===")
    
    # Initialize generator
    generator = PlaylistGenerator(model_dir="demo_models", data_dir="demo_data")
    
    # Create sample data
    features, mood_labels, energy_labels, metadata = create_sample_data()
    
    # Simulate adding songs to database
    print("Adding sample songs to database...")
    for i, meta in enumerate(metadata):
        song_id = generator.song_database.add_song(
            file_path=meta['file_path'],
            title=meta['title'],
            artist=meta['artist'],
            genre=meta['genre'],
            duration=meta['duration']
        )
        
        generator.metadata_manager.add_song_metadata(song_id, meta)
        
        # Add to analysis cache (simulating analysis)
        generator.analysis_cache[song_id] = {
            'song_id': song_id,
            'duration': meta['duration'],
            'tempo': np.random.uniform(80, 160),
            'rms_mean': np.random.uniform(0.1, 0.5),
            'spectral_centroid_mean': np.random.uniform(1000, 4000),
            # ... other features would be here
        }
    
    print(f"Added {len(metadata)} songs to database")
    
    # Extract features
    print("Extracting features...")
    extracted_features = generator.extract_features()
    print(f"Extracted features shape: {extracted_features.shape}")
    
    # Train models
    print("Training models...")
    training_results = generator.train_models(
        mood_labels=mood_labels.tolist(),
        energy_labels=energy_labels.tolist(),
        validation_split=0.2
    )
    
    print("Models trained successfully!")
    
    # Generate playlists
    print("Generating playlists...")
    
    # Mood-based playlist
    mood_playlist = generator.generate_playlist(
        playlist_type="mood_based",
        target_mood="Happy/Upbeat",
        playlist_length=10
    )
    print(f"Mood-based playlist: {len(mood_playlist)} songs")
    
    # Energy-based playlist
    energy_playlist = generator.generate_playlist(
        playlist_type="energy_based",
        target_energy=0.7,
        playlist_length=10
    )
    print(f"Energy-based playlist: {len(energy_playlist)} songs")
    
    # Cluster-based playlist
    cluster_playlist = generator.generate_playlist(
        playlist_type="cluster_based",
        playlist_length=10
    )
    print(f"Cluster-based playlist: {len(cluster_playlist)} songs")
    
    # Transition playlist
    transition_playlist = generator.generate_playlist(
        playlist_type="transition",
        playlist_length=10
    )
    print(f"Transition playlist: {len(transition_playlist)} songs")
    
    print("\nFull pipeline demonstration completed!")


def main():
    """Run all demonstrations."""
    print("AI-Powered Playlist Generator - Advanced Usage Demo")
    print("=" * 60)
    
    try:
        # Individual component demos
        demonstrate_audio_analysis()
        demonstrate_feature_extraction()
        demonstrate_model_training()
        demonstrate_pattern_analysis()
        
        # Full pipeline demo
        demonstrate_full_pipeline()
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nDemo completed! Check the 'demo_models' and 'demo_data' directories.")


if __name__ == "__main__":
    main()



