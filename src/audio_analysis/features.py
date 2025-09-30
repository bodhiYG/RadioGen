"""
Feature extraction pipeline for converting audio analysis results into ML-ready features.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import joblib
import os


class FeatureExtractor:
    """
    Feature extraction pipeline for converting raw audio analysis into
    machine learning ready features.
    """
    
    def __init__(self, normalize: bool = True, use_pca: bool = False, n_components: int = 20):
        """
        Initialize the feature extractor.
        
        Args:
            normalize: Whether to normalize features
            use_pca: Whether to apply PCA dimensionality reduction
            n_components: Number of PCA components if use_pca is True
        """
        self.normalize = normalize
        self.use_pca = use_pca
        self.n_components = n_components
        
        self.scaler = StandardScaler() if normalize else None
        self.pca = PCA(n_components=n_components) if use_pca else None
        self.feature_names = None
        self.is_fitted = False
    
    def _extract_numerical_features(self, analysis_result: Dict[str, Any]) -> np.ndarray:
        """
        Extract numerical features from analysis result.
        
        Args:
            analysis_result: Result from AudioAnalyzer.analyze_audio_file()
            
        Returns:
            Array of numerical features
        """
        features = []
        feature_names = []
        
        # Basic audio properties
        features.extend([
            analysis_result.get('duration', 0),
            analysis_result.get('sample_rate', 0)
        ])
        feature_names.extend(['duration', 'sample_rate'])
        
        # Tempo features
        features.extend([
            analysis_result.get('tempo', 0),
            analysis_result.get('tempo_stability', 0),
            analysis_result.get('beat_count', 0)
        ])
        feature_names.extend(['tempo', 'tempo_stability', 'beat_count'])
        
        # Loudness features
        features.extend([
            analysis_result.get('rms_mean', 0),
            analysis_result.get('rms_std', 0),
            analysis_result.get('rms_max', 0),
            analysis_result.get('rms_min', 0),
            analysis_result.get('dynamic_range', 0),
            analysis_result.get('spectral_centroid_mean', 0),
            analysis_result.get('spectral_centroid_std', 0),
            analysis_result.get('zero_crossing_rate_mean', 0),
            analysis_result.get('zero_crossing_rate_std', 0)
        ])
        feature_names.extend([
            'rms_mean', 'rms_std', 'rms_max', 'rms_min', 'dynamic_range',
            'spectral_centroid_mean', 'spectral_centroid_std',
            'zero_crossing_rate_mean', 'zero_crossing_rate_std'
        ])
        
        # Spectral features
        features.extend([
            analysis_result.get('spectral_rolloff_mean', 0),
            analysis_result.get('spectral_rolloff_std', 0),
            analysis_result.get('spectral_bandwidth_mean', 0),
            analysis_result.get('spectral_bandwidth_std', 0)
        ])
        feature_names.extend([
            'spectral_rolloff_mean', 'spectral_rolloff_std',
            'spectral_bandwidth_mean', 'spectral_bandwidth_std'
        ])
        
        # MFCC features (mean and std of first 13 coefficients)
        mfcc_mean = analysis_result.get('mfcc_mean', [0] * 13)
        mfcc_std = analysis_result.get('mfcc_std', [0] * 13)
        features.extend(mfcc_mean)
        features.extend(mfcc_std)
        feature_names.extend([f'mfcc_mean_{i}' for i in range(13)])
        feature_names.extend([f'mfcc_std_{i}' for i in range(13)])
        
        # Chroma features (mean and std of 12 pitch classes)
        chroma_mean = analysis_result.get('chroma_mean', [0] * 12)
        chroma_std = analysis_result.get('chroma_std', [0] * 12)
        features.extend(chroma_mean)
        features.extend(chroma_std)
        feature_names.extend([f'chroma_mean_{i}' for i in range(12)])
        feature_names.extend([f'chroma_std_{i}' for i in range(12)])
        
        # Rhythm features
        features.extend([
            analysis_result.get('onset_density', 0),
            analysis_result.get('onset_strength_mean', 0),
            analysis_result.get('onset_strength_std', 0),
            analysis_result.get('onset_count', 0)
        ])
        feature_names.extend([
            'onset_density', 'onset_strength_mean', 'onset_strength_std', 'onset_count'
        ])
        
        self.feature_names = feature_names
        return np.array(features, dtype=np.float32)
    
    def extract_features(self, analysis_results: List[Dict[str, Any]]) -> np.ndarray:
        """
        Extract features from multiple analysis results.
        
        Args:
            analysis_results: List of analysis results from AudioAnalyzer
            
        Returns:
            Array of shape (n_samples, n_features)
        """
        features_list = []
        
        for result in analysis_results:
            features = self._extract_numerical_features(result)
            features_list.append(features)
        
        features_array = np.array(features_list)
        
        # Fit scaler and PCA if not already fitted
        if not self.is_fitted:
            if self.scaler is not None:
                features_array = self.scaler.fit_transform(features_array)
            
            if self.pca is not None:
                features_array = self.pca.fit_transform(features_array)
            
            self.is_fitted = True
        else:
            # Apply fitted transformations
            if self.scaler is not None:
                features_array = self.scaler.transform(features_array)
            
            if self.pca is not None:
                features_array = self.pca.transform(features_array)
        
        return features_array
    
    def extract_single_features(self, analysis_result: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from a single analysis result.
        
        Args:
            analysis_result: Single analysis result from AudioAnalyzer
            
        Returns:
            Array of shape (1, n_features)
        """
        return self.extract_features([analysis_result])
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of extracted features.
        
        Returns:
            List of feature names
        """
        if self.feature_names is None:
            raise ValueError("Feature extractor not initialized. Call extract_features() first.")
        
        if self.use_pca:
            return [f'pca_component_{i}' for i in range(self.n_components)]
        else:
            return self.feature_names
    
    def save(self, filepath: str):
        """
        Save the fitted feature extractor to disk.
        
        Args:
            filepath: Path to save the extractor
        """
        if not self.is_fitted:
            raise ValueError("Feature extractor must be fitted before saving.")
        
        data = {
            'scaler': self.scaler,
            'pca': self.pca,
            'feature_names': self.feature_names,
            'normalize': self.normalize,
            'use_pca': self.use_pca,
            'n_components': self.n_components,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(data, filepath)
    
    def load(self, filepath: str):
        """
        Load a fitted feature extractor from disk.
        
        Args:
            filepath: Path to load the extractor from
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Feature extractor file not found: {filepath}")
        
        data = joblib.load(filepath)
        
        self.scaler = data['scaler']
        self.pca = data['pca']
        self.feature_names = data['feature_names']
        self.normalize = data['normalize']
        self.use_pca = data['use_pca']
        self.n_components = data['n_components']
        self.is_fitted = data['is_fitted']
