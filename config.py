"""
Configuration module for TTS Dataset Generator.

This module provides a centralized configuration class that can be used
by both the main.py script and the Colab notebook to ensure consistent
settings across different execution environments.
"""

import os
import json
from typing import Optional, Dict, Any


class Config:
    """Centralized configuration management for TTS Dataset Generator."""
    
    def __init__(
        self,
        project_name: str = "MyProject",
        base_directory: str = "MyTTSDataset",
        input_file_path: str = "",
        min_duration: float = 3.0,
        max_duration: float = 10.0,
        silence_threshold: int = -40,
        min_silence_len: int = 250,
        keep_silence: int = 150,
        sample_rate: int = 22050,
        whisper_model: str = "large",
        language: str = "en",
        ljspeech: bool = True,
        log_level: str = "INFO",
        merge_short_segments: bool = False,
        merge_threshold: float = 2.0
    ):
        """
        Initialize configuration with default or custom values.
        
        Args:
            project_name: Name of the TTS project
            base_directory: Base directory for output files
            input_file_path: Path to input audio/video file
            min_duration: Minimum segment duration in seconds
            max_duration: Maximum segment duration in seconds
            silence_threshold: Audio level (dBFS) below which is considered silence
            min_silence_len: Minimum silence duration (ms) to mark a split point
            keep_silence: Padding silence (ms) to keep at segment boundaries
            sample_rate: Audio sample rate in Hz
            whisper_model: Whisper model size for transcription
            language: Language code for transcription
            ljspeech: Whether to use LJSpeech format
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            merge_short_segments: Whether to merge segments shorter than threshold
            merge_threshold: Duration threshold (seconds) for merging short segments
        """
        self.project_name = project_name
        self.base_directory = base_directory
        self.input_file_path = input_file_path
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.silence_threshold = silence_threshold
        self.min_silence_len = min_silence_len
        self.keep_silence = keep_silence
        self.sample_rate = sample_rate
        self.whisper_model = whisper_model
        self.language = language
        self.ljspeech = ljspeech
        self.log_level = log_level
        self.merge_short_segments = merge_short_segments
        self.merge_threshold = merge_threshold
    
    @classmethod
    def from_colab_step0(cls) -> 'Config':
        """
        Load configuration from Colab STEP 0 cell variables.
        
        This method expects the following variables to be defined in the
        Colab environment (typically from STEP 0):
            - project_name
            - base_directory
            - input_file_path
            - min_duration
            - max_duration
            - silence_threshold
            - min_silence_len
            - keep_silence
            - sample_rate
            - whisper_model
            - language
            - ljspeech
            - log_level
            - merge_short_segments
            - merge_threshold
        
        Returns:
            Config: Configuration instance with Colab variables
        """
        try:
            # These variables should be defined in Colab STEP 0
            return cls(
                project_name=project_name,
                base_directory=base_directory,
                input_file_path=input_file_path,
                min_duration=min_duration,
                max_duration=max_duration,
                silence_threshold=silence_threshold,
                min_silence_len=min_silence_len,
                keep_silence=keep_silence,
                sample_rate=sample_rate,
                whisper_model=whisper_model,
                language=language,
                ljspeech=ljspeech,
                log_level=log_level,
                merge_short_segments=merge_short_segments if 'merge_short_segments' in locals() else False,
                merge_threshold=merge_threshold if 'merge_threshold' in locals() else 2.0
            )
        except NameError as e:
            raise RuntimeError(
                f"Colab variables not found. Ensure STEP 0 has been executed. Missing: {e}"
            )
    
    @classmethod
    def from_json(cls, config_path: str) -> 'Config':
        """
        Load configuration from a JSON file.
        
        Args:
            config_path: Path to the JSON configuration file
            
        Returns:
            Config: Configuration instance loaded from JSON
        """
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)
    
    @classmethod
    def from_argparse(cls, args) -> 'Config':
        """
        Load configuration from argparse namespace.
        
        Args:
            args: Argparse namespace with configuration values
            
        Returns:
            Config: Configuration instance from argparse
        """
        return cls(
            project_name=args.project,
            base_directory=args.base_dir,
            input_file_path=args.file,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            silence_threshold=args.silence_threshold,
            min_silence_len=args.min_silence_len,
            keep_silence=args.keep_silence,
            sample_rate=args.sample_rate,
            whisper_model=args.model,
            language=args.language,
            ljspeech=args.ljspeech,
            log_level=args.log_level,
            merge_short_segments=args.merge_short_segments,
            merge_threshold=args.merge_threshold
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            dict: Configuration as dictionary
        """
        return {
            "project_name": self.project_name,
            "base_directory": self.base_directory,
            "input_file_path": self.input_file_path,
            "min_duration": self.min_duration,
            "max_duration": self.max_duration,
            "silence_threshold": self.silence_threshold,
            "min_silence_len": self.min_silence_len,
            "keep_silence": self.keep_silence,
            "sample_rate": self.sample_rate,
            "whisper_model": self.whisper_model,
            "language": self.language,
            "ljspeech": self.ljspeech,
            "log_level": self.log_level,
            "merge_short_segments": self.merge_short_segments,
            "merge_threshold": self.merge_threshold
        }
    
    def save_to_json(self, config_path: str) -> None:
        """
        Save configuration to a JSON file.
        
        Args:
            config_path: Path to save the JSON configuration file
        """
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def get_output_dirs(self) -> tuple:
        """
        Get the output directories for audio segments and metadata.
        
        Returns:
            tuple: (audio_output_dir, metadata_output_path)
        """
        audio_dir = os.path.join(self.base_directory, self.project_name, "wavs")
        metadata_path = os.path.join(self.base_directory, self.project_name, "metadata.csv")
        return audio_dir, metadata_path
    
    def print_summary(self) -> None:
        """Print a formatted summary of the current configuration."""
        print("=" * 50)
        print("✅ CONFIGURATION SUMMARY")
        print("=" * 50)
        print(f"Project Name: {self.project_name}")
        print(f"Base Directory: {self.base_directory}")
        print(f"Input File: {self.input_file_path}")
        print(f"Whisper Model: {self.whisper_model}")
        print(f"Language: {self.language}")
        print(f"Sample Rate: {self.sample_rate}")
        print(f"LJSpeech Format: {self.ljspeech}")
        print("=" * 50)


# Default configuration instance
default_config = Config()
