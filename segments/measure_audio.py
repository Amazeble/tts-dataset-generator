import os
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tts_dataset_generator.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("tts_dataset_generator")


def measure_audio_and_silence(input_path, silence_thresh_dbfs=-40):
    """
    Measure the total length of an audio file and the total duration of silence periods.
    
    Args:
        input_path (str): Path to the input audio or video file.
        silence_thresh_dbfs (int): Audio level below which is considered silence (dBFS).
                                   Default is -40 dBFS.
    
    Returns:
        dict: A dictionary containing:
            - 'total_duration': Total duration of the audio in seconds (float)
            - 'total_silence': Total duration of silence periods in seconds (float)
            - 'speech_duration': Duration of non-silent audio in seconds (float)
            - 'silence_percentage': Percentage of silence in the audio (float)
    """
    from pydub import AudioSegment
    from pydub.silence import detect_silence
    
    logger.info(f"Analyzing audio file: {input_path}")
    
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return None
    
    try:
        # Load the audio file
        logger.info("Loading audio file...")
        audio = AudioSegment.from_file(input_path)
        total_duration_ms = len(audio)
        total_duration_s = total_duration_ms / 1000.0
        
        logger.info(f"Total audio duration: {total_duration_s:.2f} seconds")
        
        # Detect silence periods
        logger.info(f"Detecting silence periods (threshold: {silence_thresh_dbfs} dBFS)...")
        silence_ranges = detect_silence(audio, silence_thresh=silence_thresh_dbfs)
        
        # Calculate total silence duration
        total_silence_ms = sum(end - start for start, end in silence_ranges)
        total_silence_s = total_silence_ms / 1000.0
        
        # Calculate speech duration (non-silent audio)
        speech_duration_s = total_duration_s - total_silence_s
        
        # Calculate silence percentage
        silence_percentage = (total_silence_s / total_duration_s * 100) if total_duration_s > 0 else 0
        
        logger.info(f"Total silence duration: {total_silence_s:.2f} seconds ({silence_percentage:.2f}%)")
        logger.info(f"Speech/non-silent duration: {speech_duration_s:.2f} seconds")
        logger.info(f"Number of silence periods detected: {len(silence_ranges)}")
        
        return {
            'total_duration': total_duration_s,
            'total_silence': total_silence_s,
            'speech_duration': speech_duration_s,
            'silence_percentage': silence_percentage,
            'silence_periods_count': len(silence_ranges),
            'silence_ranges': silence_ranges  # List of [start_ms, end_ms] tuples
        }
        
    except Exception as e:
        logger.error(f"Error analyzing audio file: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None
