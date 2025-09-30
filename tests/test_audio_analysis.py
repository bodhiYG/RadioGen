"""
Tests for audio analysis functionality.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from audio_analysis import AudioAnalyzer, FeatureExtractor


class TestAudioAnalyzer(unittest.TestCase):
    """Test cases for AudioAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = AudioAnalyzer()
        
        # Create sample audio data
        self.sample_rate = 22050
        self.duration = 2  # seconds
        self.frequency = 440  # A4 note
        
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        self.audio = np.sin(2 * np.pi * self.frequency * t)
        
        # Add some variation
        self.audio += 0.1 * np.sin(2 * np.pi * self.frequency * 2 * t)
        self.audio += 0.05 * np.random.randn(len(self.audio))
    
    def test_audio_analyzer_initialization(self):
        """Test AudioAnalyzer initialization."""
        self.assertEqual(self.analyzer.sample_rate, 22050)
        self.assertEqual(self.analyzer.hop_length, 512)
    
    def test_extract_tempo(self):
        """Test tempo extraction."""
        tempo_features = self.analyzer.extract_tempo(self.audio)
        
        self.assertIn('tempo', tempo_features)
        self.assertIn('tempo_stability', tempo_features)
        self.assertIn('beat_count', tempo_features)
        
        self.assertIsInstance(tempo_features['tempo'], float)
        self.assertIsInstance(tempo_features['tempo_stability'], float)
        self.assertIsInstance(tempo_features['beat_count'], int)
        
        self.assertGreater(tempo_features['tempo'], 0)
        self.assertGreaterEqual(tempo_features['tempo_stability'], 0)
        self.assertLessEqual(tempo_features['tempo_stability'], 1)
    
    def test_extract_loudness(self):
        """Test loudness extraction."""
        loudness_features = self.analyzer.extract_loudness(self.audio)
        
        required_keys = [
            'rms_mean', 'rms_std', 'rms_max', 'rms_min', 'dynamic_range',
            'spectral_centroid_mean', 'spectral_centroid_std',
            'zero_crossing_rate_mean', 'zero_crossing_rate_std'
        ]
        
        for key in required_keys:
            self.assertIn(key, loudness_features)
            self.assertIsInstance(loudness_features[key], float)
            self.assertGreaterEqual(loudness_features[key], 0)
    
    def test_extract_spectral_features(self):
        """Test spectral feature extraction."""
        spectral_features = self.analyzer.extract_spectral_features(self.audio)
        
        required_keys = [
            'spectral_centroid_mean', 'spectral_centroid_std',
            'spectral_rolloff_mean', 'spectral_rolloff_std',
            'spectral_bandwidth_mean', 'spectral_bandwidth_std'
        ]
        
        for key in required_keys:
            self.assertIn(key, spectral_features)
            self.assertIsInstance(spectral_features[key], float)
            self.assertGreaterEqual(spectral_features[key], 0)
        
        # Check MFCC features
        self.assertIn('mfcc_mean', spectral_features)
        self.assertIn('mfcc_std', spectral_features)
        self.assertEqual(len(spectral_features['mfcc_mean']), 13)
        self.assertEqual(len(spectral_features['mfcc_std']), 13)
        
        # Check chroma features
        self.assertIn('chroma_mean', spectral_features)
        self.assertIn('chroma_std', spectral_features)
        self.assertEqual(len(spectral_features['chroma_mean']), 12)
        self.assertEqual(len(spectral_features['chroma_std']), 12)
    
    def test_extract_rhythm_features(self):
        """Test rhythm feature extraction."""
        rhythm_features = self.analyzer.extract_rhythm_features(self.audio)
        
        required_keys = [
            'onset_density', 'onset_strength_mean', 
            'onset_strength_std', 'onset_count'
        ]
        
        for key in required_keys:
            self.assertIn(key, rhythm_features)
            self.assertIsInstance(rhythm_features[key], float)
            self.assertGreaterEqual(rhythm_features[key], 0)
    
    def test_analyze_audio_file(self):
        """Test complete audio file analysis."""
        # Create a temporary audio file
        import tempfile
        import soundfile as sf
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sf.write(tmp_file.name, self.audio, self.sample_rate)
            
            try:
                analysis_result = self.analyzer.analyze_audio_file(tmp_file.name)
                
                # Check that all expected keys are present
                expected_keys = [
                    'file_path', 'duration', 'sample_rate',
                    'tempo', 'tempo_stability', 'beat_count',
                    'rms_mean', 'rms_std', 'rms_max', 'rms_min', 'dynamic_range',
                    'spectral_centroid_mean', 'spectral_centroid_std',
                    'zero_crossing_rate_mean', 'zero_crossing_rate_std',
                    'spectral_rolloff_mean', 'spectral_rolloff_std',
                    'spectral_bandwidth_mean', 'spectral_bandwidth_std',
                    'mfcc_mean', 'mfcc_std', 'chroma_mean', 'chroma_std',
                    'onset_density', 'onset_strength_mean', 'onset_strength_std', 'onset_count'
                ]
                
                for key in expected_keys:
                    self.assertIn(key, analysis_result)
                
                self.assertEqual(analysis_result['file_path'], tmp_file.name)
                self.assertAlmostEqual(analysis_result['duration'], self.duration, places=1)
                self.assertEqual(analysis_result['sample_rate'], self.sample_rate)
                
            finally:
                # Clean up temporary file
                import os
                os.unlink(tmp_file.name)


class TestFeatureExtractor(unittest.TestCase):
    """Test cases for FeatureExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = FeatureExtractor(normalize=True, use_pca=False)
        
        # Create sample analysis results
        self.analysis_results = []
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
            self.analysis_results.append(result)
    
    def test_feature_extractor_initialization(self):
        """Test FeatureExtractor initialization."""
        self.assertTrue(self.extractor.normalize)
        self.assertFalse(self.extractor.use_pca)
        self.assertIsNone(self.extractor.feature_names)
        self.assertFalse(self.extractor.is_fitted)
    
    def test_extract_features(self):
        """Test feature extraction."""
        features = self.extractor.extract_features(self.analysis_results)
        
        # Check shape
        self.assertEqual(features.shape[0], len(self.analysis_results))
        self.assertGreater(features.shape[1], 0)
        
        # Check that features are normalized (mean should be close to 0)
        self.assertAlmostEqual(np.mean(features), 0, places=1)
        
        # Check that extractor is now fitted
        self.assertTrue(self.extractor.is_fitted)
        self.assertIsNotNone(self.extractor.feature_names)
    
    def test_extract_single_features(self):
        """Test single feature extraction."""
        single_result = self.analysis_results[0]
        features = self.extractor.extract_single_features(single_result)
        
        # Check shape
        self.assertEqual(features.shape[0], 1)
        self.assertGreater(features.shape[1], 0)
    
    def test_get_feature_names(self):
        """Test getting feature names."""
        # Extract features first to initialize feature names
        self.extractor.extract_features(self.analysis_results)
        
        feature_names = self.extractor.get_feature_names()
        self.assertIsInstance(feature_names, list)
        self.assertGreater(len(feature_names), 0)
    
    def test_save_and_load(self):
        """Test saving and loading feature extractor."""
        import tempfile
        import os
        
        # Extract features first
        self.extractor.extract_features(self.analysis_results)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            try:
                # Save extractor
                self.extractor.save(tmp_file.name)
                
                # Create new extractor and load
                new_extractor = FeatureExtractor()
                new_extractor.load(tmp_file.name)
                
                # Check that loaded extractor has same properties
                self.assertEqual(new_extractor.normalize, self.extractor.normalize)
                self.assertEqual(new_extractor.use_pca, self.extractor.use_pca)
                self.assertEqual(new_extractor.n_components, self.extractor.n_components)
                self.assertTrue(new_extractor.is_fitted)
                
            finally:
                os.unlink(tmp_file.name)


if __name__ == '__main__':
    unittest.main()
