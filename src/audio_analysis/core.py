"""
Core audio analysis functionality for extracting musical features.
"""

import librosa
import numpy as np
import soundfile as sf
from typing import Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class AudioAnalyzer:
    """
    Core audio analyzer for extracting tempo, loudness, spectral centroid,
    and other musical features from audio files.
    """
    
    def __init__(self, sample_rate: int = 22050, hop_length: int = 512):
        """
        Initialize the audio analyzer.
        
        Args:
            sample_rate: Target sample rate for audio analysis
            hop_length: Number of samples between successive frames
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and return audio data and sample rate.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            return audio, sr
        except Exception as e:
            raise ValueError(f"Error loading audio file {file_path}: {str(e)}")
    
    def extract_tempo(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract tempo and beat information from audio.
        
        Args:
            audio: Audio signal
            
        Returns:
            Dictionary containing tempo information
        """
        # Extract tempo using librosa's beat tracking
        tempo, beats = librosa.beat.beat_track(
            y=audio, 
            sr=self.sample_rate, 
            hop_length=self.hop_length
        )
        
        # Calculate tempo stability (variance in beat intervals)
        if len(beats) > 1:
            beat_intervals = np.diff(beats)
            tempo_stability = 1.0 / (1.0 + np.var(beat_intervals))
        else:
            tempo_stability = 0.0
            
        return {
            'tempo': float(tempo),
            'tempo_stability': float(tempo_stability),
            'beat_count': len(beats)
        }
    
    def extract_loudness(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract loudness and dynamic range information.
        
        Args:
            audio: Audio signal
            
        Returns:
            Dictionary containing loudness information
        """
        # Calculate RMS energy
        rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
        
        # Calculate spectral centroid (brightness)
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )[0]
        
        # Calculate zero crossing rate (roughness/noisiness)
        zcr = librosa.feature.zero_crossing_rate(audio, hop_length=self.hop_length)[0]
        
        # Calculate dynamic range
        dynamic_range = np.max(rms) - np.min(rms)
        
        return {
            'rms_mean': float(np.mean(rms)),
            'rms_std': float(np.std(rms)),
            'rms_max': float(np.max(rms)),
            'rms_min': float(np.min(rms)),
            'dynamic_range': float(dynamic_range),
            'spectral_centroid_mean': float(np.mean(spectral_centroid)),
            'spectral_centroid_std': float(np.std(spectral_centroid)),
            'zero_crossing_rate_mean': float(np.mean(zcr)),
            'zero_crossing_rate_std': float(np.std(zcr))
        }
    
    def extract_spectral_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract spectral features including spectral centroid, rolloff, and bandwidth.
        
        Args:
            audio: Audio signal
            
        Returns:
            Dictionary containing spectral features
        """
        # Spectral centroid (brightness)
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )[0]
        
        # Spectral rolloff (frequency below which 85% of energy is contained)
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )[0]
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )[0]
        
        # MFCC features (first 13 coefficients)
        mfccs = librosa.feature.mfcc(
            y=audio, sr=self.sample_rate, n_mfcc=13, hop_length=self.hop_length
        )
        
        # Chroma features (pitch class profile)
        chroma = librosa.feature.chroma_stft(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )
        
        return {
            'spectral_centroid_mean': float(np.mean(spectral_centroid)),
            'spectral_centroid_std': float(np.std(spectral_centroid)),
            'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
            'spectral_rolloff_std': float(np.std(spectral_rolloff)),
            'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
            'spectral_bandwidth_std': float(np.std(spectral_bandwidth)),
            'mfcc_mean': [float(np.mean(mfcc)) for mfcc in mfccs],
            'mfcc_std': [float(np.std(mfcc)) for mfcc in mfccs],
            'chroma_mean': [float(np.mean(ch)) for ch in chroma],
            'chroma_std': [float(np.std(ch)) for ch in chroma]
        }
    
    def extract_rhythm_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract rhythm and timing features.
        
        Args:
            audio: Audio signal
            
        Returns:
            Dictionary containing rhythm features
        """
        # Onset strength
        onset_strength = librosa.onset.onset_strength(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )
        
        # Onset detection
        onsets = librosa.onset.onset_detect(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )
        
        # Calculate onset density
        onset_density = len(onsets) / (len(audio) / self.sample_rate)
        
        # Calculate onset strength statistics
        onset_strength_mean = np.mean(onset_strength)
        onset_strength_std = np.std(onset_strength)
        
        return {
            'onset_density': float(onset_density),
            'onset_strength_mean': float(onset_strength_mean),
            'onset_strength_std': float(onset_strength_std),
            'onset_count': len(onsets)
        }
    
    def analyze_audio_file(self, file_path: str) -> Dict[str, any]:
        """
        Perform complete audio analysis on a file.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dictionary containing all extracted features
        """
        # Load audio
        audio, sr = self.load_audio(file_path)
        
        # Extract all features
        features = {
            'file_path': file_path,
            'duration': len(audio) / sr,
            'sample_rate': sr
        }
        
        # Add tempo features
        features.update(self.extract_tempo(audio))
        
        # Add loudness features
        features.update(self.extract_loudness(audio))
        
        # Add spectral features
        features.update(self.extract_spectral_features(audio))
        
        # Add rhythm features
        features.update(self.extract_rhythm_features(audio))
        
        return features
