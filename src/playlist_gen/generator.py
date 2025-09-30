"""
Main playlist generation system that orchestrates all components.
"""

import os
import glob
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import pickle

from ..audio_analysis import AudioAnalyzer, FeatureExtractor
from ..models import MoodClassifier, EnergyClassifier, PatternAnalyzer
from ..data import SongDatabase, MetadataManager


class PlaylistGenerator:
    """
    Main class for generating AI-powered playlists based on mood and energy analysis.
    """
    
    def __init__(self, model_dir: str = "models", data_dir: str = "data"):
        """
        Initialize the playlist generator.
        
        Args:
            model_dir: Directory to save/load trained models
            data_dir: Directory to save/load data and features
        """
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        
        # Create directories if they don't exist
        self.model_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.audio_analyzer = AudioAnalyzer()
        self.feature_extractor = FeatureExtractor()
        self.pattern_analyzer = PatternAnalyzer()
        
        # Initialize models (will be loaded when needed)
        self.mood_classifier = None
        self.energy_classifier = None
        self.mood_trainer = None
        self.energy_trainer = None
        
        # Data management
        self.song_database = SongDatabase(str(self.data_dir / "songs.db"))
        self.metadata_manager = MetadataManager(str(self.data_dir / "metadata.json"))
        
        # Cache for features
        self.features_cache = {}
        self.analysis_cache = {}
        
    def add_songs_from_directory(self, directory: str, 
                                supported_formats: List[str] = None) -> List[str]:
        """
        Add all songs from a directory to the database.
        
        Args:
            directory: Path to directory containing audio files
            supported_formats: List of supported audio formats
            
        Returns:
            List of file paths that were added
        """
        if supported_formats is None:
            supported_formats = ['.mp3', '.wav', '.flac', '.m4a', '.ogg']
        
        directory = Path(directory)
        if not directory.exists():
            raise ValueError(f"Directory does not exist: {directory}")
        
        added_files = []
        
        for format_ext in supported_formats:
            pattern = f"**/*{format_ext}"
            files = list(directory.glob(pattern))
            
            for file_path in files:
                try:
                    # Add to database
                    song_id = self.song_database.add_song(
                        file_path=str(file_path),
                        title=file_path.stem,
                        artist="Unknown",
                        genre="Unknown"
                    )
                    
                    # Add to metadata
                    self.metadata_manager.add_song_metadata(song_id, {
                        'file_path': str(file_path),
                        'title': file_path.stem,
                        'artist': "Unknown",
                        'genre': "Unknown",
                        'duration': 0,  # Will be updated after analysis
                        'added_date': pd.Timestamp.now().isoformat()
                    })
                    
                    added_files.append(str(file_path))
                    
                except Exception as e:
                    print(f"Error adding song {file_path}: {str(e)}")
                    continue
        
        print(f"Added {len(added_files)} songs to database")
        return added_files
    
    def analyze_songs(self, song_ids: Optional[List[str]] = None, 
                     force_reanalyze: bool = False) -> Dict[str, any]:
        """
        Analyze songs to extract audio features.
        
        Args:
            song_ids: List of song IDs to analyze (None for all songs)
            force_reanalyze: Whether to reanalyze already processed songs
            
        Returns:
            Dictionary with analysis results
        """
        if song_ids is None:
            song_ids = self.song_database.get_all_song_ids()
        
        analysis_results = []
        processed_count = 0
        error_count = 0
        
        for song_id in song_ids:
            try:
                # Check if already analyzed
                if not force_reanalyze and song_id in self.analysis_cache:
                    analysis_results.append(self.analysis_cache[song_id])
                    continue
                
                # Get song metadata
                metadata = self.metadata_manager.get_song_metadata(song_id)
                file_path = metadata.get('file_path')
                
                if not file_path or not os.path.exists(file_path):
                    print(f"File not found for song {song_id}: {file_path}")
                    error_count += 1
                    continue
                
                # Analyze audio
                print(f"Analyzing: {metadata.get('title', song_id)}")
                analysis_result = self.audio_analyzer.analyze_audio_file(file_path)
                analysis_result['song_id'] = song_id
                
                # Cache the result
                self.analysis_cache[song_id] = analysis_result
                
                # Update metadata with duration
                self.metadata_manager.update_song_metadata(song_id, {
                    'duration': analysis_result['duration']
                })
                
                analysis_results.append(analysis_result)
                processed_count += 1
                
            except Exception as e:
                print(f"Error analyzing song {song_id}: {str(e)}")
                error_count += 1
                continue
        
        print(f"Analysis complete: {processed_count} processed, {error_count} errors")
        
        # Save analysis cache
        self._save_analysis_cache()
        
        return {
            'analysis_results': analysis_results,
            'processed_count': processed_count,
            'error_count': error_count
        }
    
    def extract_features(self, analysis_results: Optional[List[Dict]] = None) -> np.ndarray:
        """
        Extract ML-ready features from analysis results.
        
        Args:
            analysis_results: List of analysis results (None to use cached)
            
        Returns:
            Feature matrix
        """
        if analysis_results is None:
            analysis_results = list(self.analysis_cache.values())
        
        if not analysis_results:
            raise ValueError("No analysis results available. Run analyze_songs() first.")
        
        features = self.feature_extractor.extract_features(analysis_results)
        
        # Cache features
        for i, result in enumerate(analysis_results):
            song_id = result['song_id']
            self.features_cache[song_id] = features[i]
        
        # Save features cache
        self._save_features_cache()
        
        return features
    
    def train_models(self, mood_labels: Optional[List[int]] = None,
                    energy_labels: Optional[List[float]] = None,
                    validation_split: float = 0.2) -> Dict[str, any]:
        """
        Train mood and energy classification models.
        
        Args:
            mood_labels: List of mood labels for training (0-5)
            energy_labels: List of energy labels for training (0.0-1.0)
            validation_split: Fraction of data to use for validation
            
        Returns:
            Dictionary with training results
        """
        # Get features
        features = self.extract_features()
        
        if features is None or len(features) == 0:
            raise ValueError("No features available. Run analyze_songs() first.")
        
        results = {}
        
        # Train mood classifier if labels provided
        if mood_labels is not None:
            if len(mood_labels) != len(features):
                raise ValueError("Number of mood labels must match number of songs")
            
            mood_results = self._train_mood_classifier(features, mood_labels, validation_split)
            results['mood_classifier'] = mood_results
        
        # Train energy classifier if labels provided
        if energy_labels is not None:
            if len(energy_labels) != len(features):
                raise ValueError("Number of energy labels must match number of songs")
            
            energy_results = self._train_energy_classifier(features, energy_labels, validation_split)
            results['energy_classifier'] = energy_results
        
        return results
    
    def _train_mood_classifier(self, features: np.ndarray, labels: List[int],
                              validation_split: float) -> Dict[str, any]:
        """Train the mood classifier."""
        from torch.utils.data import DataLoader, random_split
        
        # Create dataset
        dataset = MoodClassifier.MoodDataset(features, np.array(labels))
        
        # Split into train/validation
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Initialize model and trainer
        self.mood_classifier = MoodClassifier(input_size=features.shape[1])
        self.mood_trainer = MoodClassifierTrainer(self.mood_classifier)
        
        # Train
        history = self.mood_trainer.train(train_loader, val_loader)
        
        # Save model
        model_path = self.model_dir / "mood_classifier.pth"
        self.mood_trainer.save_model(str(model_path))
        
        return {
            'history': history,
            'model_path': str(model_path)
        }
    
    def _train_energy_classifier(self, features: np.ndarray, labels: List[float],
                                validation_split: float) -> Dict[str, any]:
        """Train the energy classifier."""
        from torch.utils.data import DataLoader, random_split
        
        # Create dataset
        dataset = EnergyClassifier.EnergyDataset(features, np.array(labels))
        
        # Split into train/validation
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Initialize model and trainer
        self.energy_classifier = EnergyClassifier(input_size=features.shape[1])
        self.energy_trainer = EnergyClassifierTrainer(self.energy_classifier)
        
        # Train
        history = self.energy_trainer.train(train_loader, val_loader)
        
        # Save model
        model_path = self.model_dir / "energy_classifier.pth"
        self.energy_trainer.save_model(str(model_path))
        
        return {
            'history': history,
            'model_path': str(model_path)
        }
    
    def load_models(self):
        """Load pre-trained models."""
        mood_model_path = self.model_dir / "mood_classifier.pth"
        energy_model_path = self.model_dir / "energy_classifier.pth"
        
        if mood_model_path.exists():
            self.mood_classifier = MoodClassifier(input_size=100)  # Will be updated
            self.mood_trainer = MoodClassifierTrainer(self.mood_classifier)
            self.mood_trainer.load_model(str(mood_model_path))
            print("Loaded mood classifier")
        
        if energy_model_path.exists():
            self.energy_classifier = EnergyClassifier(input_size=100)  # Will be updated
            self.energy_trainer = EnergyClassifierTrainer(self.energy_classifier)
            self.energy_trainer.load_model(str(energy_model_path))
            print("Loaded energy classifier")
    
    def generate_playlist(self, playlist_type: str = "mood_based", 
                         target_mood: Optional[str] = None,
                         target_energy: Optional[float] = None,
                         playlist_length: int = 20,
                         **kwargs) -> List[Dict]:
        """
        Generate a playlist based on specified criteria.
        
        Args:
            playlist_type: Type of playlist ('mood_based', 'energy_based', 'cluster_based', 'transition')
            target_mood: Target mood for mood-based playlists
            target_energy: Target energy level for energy-based playlists
            playlist_length: Desired playlist length
            **kwargs: Additional parameters for specific playlist types
            
        Returns:
            List of song metadata dictionaries
        """
        if not self.analysis_cache:
            raise ValueError("No songs analyzed. Run analyze_songs() first.")
        
        # Get features
        features = self.extract_features()
        song_ids = list(self.analysis_cache.keys())
        
        if playlist_type == "mood_based":
            return self._generate_mood_playlist(
                features, song_ids, target_mood, playlist_length
            )
        elif playlist_type == "energy_based":
            return self._generate_energy_playlist(
                features, song_ids, target_energy, playlist_length
            )
        elif playlist_type == "cluster_based":
            return self._generate_cluster_playlist(
                features, song_ids, playlist_length, **kwargs
            )
        elif playlist_type == "transition":
            return self._generate_transition_playlist(
                features, song_ids, playlist_length, **kwargs
            )
        else:
            raise ValueError(f"Unknown playlist type: {playlist_type}")
    
    def _generate_mood_playlist(self, features: np.ndarray, song_ids: List[str],
                               target_mood: str, playlist_length: int) -> List[Dict]:
        """Generate mood-based playlist."""
        if not self.mood_trainer:
            raise ValueError("Mood classifier not trained. Train models first.")
        
        # Predict moods for all songs
        mood_predictions = []
        for i, song_id in enumerate(song_ids):
            prediction = self.mood_trainer.predict_mood(features[i:i+1])
            mood_predictions.append((song_id, prediction['predicted_mood']))
        
        # Filter songs by target mood
        matching_songs = [
            song_id for song_id, mood in mood_predictions 
            if mood.lower() == target_mood.lower()
        ]
        
        if not matching_songs:
            print(f"No songs found with mood '{target_mood}'. Using all songs.")
            matching_songs = song_ids
        
        # Select random songs up to playlist length
        selected_songs = np.random.choice(
            matching_songs, 
            min(playlist_length, len(matching_songs)), 
            replace=False
        )
        
        # Get metadata for selected songs
        playlist = []
        for song_id in selected_songs:
            metadata = self.metadata_manager.get_song_metadata(song_id)
            playlist.append(metadata)
        
        return playlist
    
    def _generate_energy_playlist(self, features: np.ndarray, song_ids: List[str],
                                 target_energy: float, playlist_length: int) -> List[Dict]:
        """Generate energy-based playlist."""
        if not self.energy_trainer:
            raise ValueError("Energy classifier not trained. Train models first.")
        
        # Predict energy levels for all songs
        energy_predictions = []
        for i, song_id in enumerate(song_ids):
            prediction = self.energy_trainer.predict_energy(features[i:i+1])
            energy_predictions.append((song_id, prediction['energy_level']))
        
        # Filter songs by target energy (with tolerance)
        tolerance = 0.1
        matching_songs = [
            song_id for song_id, energy in energy_predictions 
            if abs(energy - target_energy) <= tolerance
        ]
        
        if not matching_songs:
            print(f"No songs found with energy level {target_energy}. Using all songs.")
            matching_songs = song_ids
        
        # Select random songs up to playlist length
        selected_songs = np.random.choice(
            matching_songs, 
            min(playlist_length, len(matching_songs)), 
            replace=False
        )
        
        # Get metadata for selected songs
        playlist = []
        for song_id in selected_songs:
            metadata = self.metadata_manager.get_song_metadata(song_id)
            playlist.append(metadata)
        
        return playlist
    
    def _generate_cluster_playlist(self, features: np.ndarray, song_ids: List[str],
                                  playlist_length: int, **kwargs) -> List[Dict]:
        """Generate cluster-based playlist."""
        # Cluster songs
        cluster_labels = self.pattern_analyzer.cluster_songs(features)
        
        # Create playlist from largest cluster
        unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
        largest_cluster = unique_clusters[np.argmax(counts)]
        
        playlist = self.pattern_analyzer.create_playlist_from_cluster(
            largest_cluster, cluster_labels, 
            [self.metadata_manager.get_song_metadata(song_id) for song_id in song_ids],
            playlist_length
        )
        
        return playlist
    
    def _generate_transition_playlist(self, features: np.ndarray, song_ids: List[str],
                                     playlist_length: int, **kwargs) -> List[Dict]:
        """Generate transition-based playlist."""
        # Calculate similarity matrix
        self.pattern_analyzer.calculate_similarity_matrix(features)
        
        # Start with random song
        start_song_index = np.random.randint(0, len(song_ids))
        
        playlist = self.pattern_analyzer.create_transition_playlist(
            start_song_index, features,
            [self.metadata_manager.get_song_metadata(song_id) for song_id in song_ids],
            playlist_length
        )
        
        return playlist
    
    def _save_analysis_cache(self):
        """Save analysis cache to disk."""
        cache_path = self.data_dir / "analysis_cache.pkl"
        with open(cache_path, 'wb') as f:
            pickle.dump(self.analysis_cache, f)
    
    def _load_analysis_cache(self):
        """Load analysis cache from disk."""
        cache_path = self.data_dir / "analysis_cache.pkl"
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                self.analysis_cache = pickle.load(f)
    
    def _save_features_cache(self):
        """Save features cache to disk."""
        cache_path = self.data_dir / "features_cache.pkl"
        with open(cache_path, 'wb') as f:
            pickle.dump(self.features_cache, f)
    
    def _load_features_cache(self):
        """Load features cache from disk."""
        cache_path = self.data_dir / "features_cache.pkl"
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                self.features_cache = pickle.load(f)
    
    def get_song_info(self, song_id: str) -> Dict:
        """Get comprehensive information about a song."""
        metadata = self.metadata_manager.get_song_metadata(song_id)
        analysis = self.analysis_cache.get(song_id, {})
        features = self.features_cache.get(song_id, None)
        
        info = {
            'metadata': metadata,
            'analysis': analysis,
            'features': features.tolist() if features is not None else None
        }
        
        # Add predictions if models are available
        if features is not None:
            if self.mood_trainer:
                mood_pred = self.mood_trainer.predict_mood(features.reshape(1, -1))
                info['mood_prediction'] = mood_pred
            
            if self.energy_trainer:
                energy_pred = self.energy_trainer.predict_energy(features.reshape(1, -1))
                info['energy_prediction'] = energy_pred
        
        return info
