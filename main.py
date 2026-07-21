import sys
import argparse
import logging
import traceback
import os
from pathlib import Path
from segments.segment_audio import segment_audio_flexible
from segments.measure_audio import measure_audio_and_silence
from transcribe.transcribe_audio import transcribe_audio_files
from config import Config

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


def setup_argparse():
    """
    Set up command-line argument parsing.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Audio/Video Processor for TTS Dataset Creation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Single argument that accepts both file or directory
    parser.add_argument('--file', '-f', type=str, 
                          help="Input audio/video file path or directory containing audio/video files")
    
    parser.add_argument('--project', '-p', type=str,
                              help="Project name (e.g., Elise). Creates ./MyTTSDataset/{project}/ structure")
    parser.add_argument('--base-dir', '-b', type=str, default="MyTTSDataset",
                              help="Base directory for output. Default is MyTTSDataset")
    parser.add_argument("--min-duration", type=float, default=3.0,
                              help="Minimum segment duration in seconds")
    parser.add_argument("--max-duration", type=float, default=10.0,
                              help="Maximum segment duration in seconds")
    parser.add_argument("--silence-threshold", type=int, default=-40,
                              help="Audio level (dBFS) below which is considered silence")
    parser.add_argument("--min-silence-len", type=int, default=250,
                              help="Minimum silence duration (ms) to mark a split point")
    parser.add_argument("--keep-silence", type=int, default=150,
                              help="Padding silence (ms) to keep at segment boundaries")
    parser.add_argument("--model", '-m', type=str, default="deepdml/faster-whisper-large-v3-turbo-ct2",
                              help="Whisper model size or path (larger = more accurate but slower)")
    parser.add_argument("--language", "-l", type=str, default="en",
                              help="Language code for transcription and number conversion")
    parser.add_argument("--ljspeech", type=bool, default=True,
                              help="Dataset format for coqui-ai/TTS")
    parser.add_argument("--sample_rate", type=int, default=22050,
                              help="Must be the same as the sampling rate of the sounds in the dataset")
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                              default="INFO", help="Set logging level")
    parser.add_argument("--merge-short-segments", action="store_true", default=False,
                              help="Merge audio segments shorter than threshold with the next segment")
    parser.add_argument("--merge-threshold", type=float, default=2.0,
                              help="Duration threshold in seconds for merging short segments (default: 2.0)")
    parser.add_argument("--measure", type=str, default=None,
                              help="Measure total length and silence period of audio file(s). Accepts a single file path or directory containing audio/video files")
    

    args = parser.parse_args()
    
    # Validate required arguments based on mode
    if not args.measure:
        if not args.file:
            parser.error("the following arguments are required: --file/-f (or use --measure for measurement mode)")
        if not args.project:
            parser.error("the following arguments are required: --project/-p (or use --measure for measurement mode)")
    
    return args



def main():
    
    """
    Main function that runs the audio segmentation and transcription process
    """
    
    args = setup_argparse()
    
    # Handle --measure mode separately
    if args.measure:
        measure_path = Path(args.measure)
        
        if measure_path.is_dir():
            # Process directory of audio/video files
            supported_extensions = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', 
                                   '.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
            
            input_files = [f for f in measure_path.iterdir() 
                           if f.is_file() and f.suffix.lower() in supported_extensions]
            
            if not input_files:
                logger.error(f"No supported audio/video files found in {measure_path}")
                sys.exit(1)
            
            logger.info(f"Found {len(input_files)} files to measure in directory")
            logger.info("\n--- Audio Measurement Results ---\n")
            
            total_duration_all = 0
            total_silence_all = 0
            
            for input_file in sorted(input_files):
                result = measure_audio_and_silence(str(input_file))
                if result:
                    total_duration_all += result['total_duration']
                    total_silence_all += result['total_silence']
                    logger.info(f"\nFile: {input_file.name}")
                    logger.info(f"  Total Duration: {result['total_duration']:.2f}s")
                    logger.info(f"  Silence Duration: {result['total_silence']:.2f}s ({result['silence_percentage']:.2f}%)")
                    logger.info(f"  Speech Duration: {result['speech_duration']:.2f}s")
                    logger.info(f"  Silence Periods: {result['silence_periods_count']}")
            
            logger.info(f"\n--- Summary ---")
            logger.info(f"Total files processed: {len(input_files)}")
            logger.info(f"Combined duration: {total_duration_all:.2f}s")
            logger.info(f"Combined silence: {total_silence_all:.2f}s ({(total_silence_all/total_duration_all*100) if total_duration_all > 0 else 0:.2f}%)")
            
        elif measure_path.is_file():
            # Single file mode
            result = measure_audio_and_silence(str(measure_path))
            if result:
                logger.info("\n--- Audio Measurement Results ---")
                logger.info(f"File: {measure_path.name}")
                logger.info(f"  Total Duration: {result['total_duration']:.2f}s")
                logger.info(f"  Silence Duration: {result['total_silence']:.2f}s ({result['silence_percentage']:.2f}%)")
                logger.info(f"  Speech Duration: {result['speech_duration']:.2f}s")
                logger.info(f"  Silence Periods: {result['silence_periods_count']}")
            else:
                logger.error("Failed to measure audio file.")
                sys.exit(1)
        else:
            logger.error(f"Path does not exist: {measure_path}")
            sys.exit(1)
        
        return  # Exit after measurement mode
    
    # Create config from argparse
    config = Config.from_argparse(args)
    
    # Update logging level based on config
    logger.setLevel(getattr(logging, config.log_level))
        
    # Display banner
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║                                                            ║
    ║        Audio/Video Segmentation & Transcription Tool       ║
    ║                                                            ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    logger.info("Running in PROCESS")
    logger.info("Running with default configuration")
    logger.info(f"Input file: {config.input_file_path}")
    logger.info(f"Project: {config.project_name}")
    logger.info(f"Base Directory: {config.base_directory}")
    logger.info(f"Language: {config.language}")
    logger.info(f"Whisper model: {config.whisper_model}")
    
    # Get output directories from config
    audio_output_dir, metadata_output_path = config.get_output_dirs()
    
    # Determine if input is a directory or single file
    input_path = Path(args.file)
    
    if input_path.is_dir():
        # Process directory of audio/video files
        input_dir = input_path
        if not input_dir.is_dir():
            logger.error(f"Input directory not found: {input_dir}")
            sys.exit(1)
        
        # Supported audio/video extensions
        supported_extensions = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', 
                               '.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        
        # Find all audio/video files in directory
        input_files = [f for f in input_dir.iterdir() 
                       if f.is_file() and f.suffix.lower() in supported_extensions]
        
        if not input_files:
            logger.error(f"No supported audio/video files found in {input_dir}")
            sys.exit(1)
        
        logger.info(f"Found {len(input_files)} files to process in directory")
        
        # List all files that will be processed
        logger.info("\n--- Files to Process ---")
        for i, f in enumerate(sorted(input_files), 1):
            logger.info(f"{i}. {f.name}")
        logger.info("------------------------\n")
        
        # Process each file
        for input_file in sorted(input_files):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing file: {input_file}")
            logger.info(f"{'='*60}")
            
            result = segment_audio_flexible(
                input_path=str(input_file),
                output_dir=audio_output_dir,
                project_name=config.project_name,
                sample_rate=config.sample_rate,
                min_duration_s=config.min_duration,
                max_duration_s=config.max_duration,
                silence_thresh_dbfs=config.silence_threshold,
                min_silence_len_ms=config.min_silence_len,
                keep_silence_ms=config.keep_silence
            )
            
            if not result:
                logger.warning(f"Segmentation failed for {input_file}, continuing with next file...")
        
        # Merge short segments if enabled
        if config.merge_short_segments:
            from segments.segment_audio import merge_short_segments
            logger.info("\nMerging short segments before transcription...")
            merge_result = merge_short_segments(
                audio_dir=audio_output_dir,
                project_name=config.project_name,
                min_duration_threshold=config.merge_threshold
            )
            if not merge_result:
                logger.warning("Merging short segments encountered issues, continuing with transcription...")
        
        # After processing all files, transcribe all segments
        logger.info("\nAll files processed. Starting transcription...")
        result = transcribe_audio_files(
            audio_dir=audio_output_dir,
            output_csv_path=metadata_output_path,
            ljspeech=config.ljspeech,
            model_name=config.whisper_model,
            language_=config.language
        )
        
        if not result:
            logger.error("Transcription failed.")
            sys.exit(1)
    else:
        # Single file mode
        if not input_path.is_file():
            logger.error(f"Input path does not exist: {input_path}")
            sys.exit(1)
        
        # First segment
        result = segment_audio_flexible(
            input_path=str(input_path),
            output_dir=audio_output_dir,
            project_name=config.project_name,
            sample_rate=config.sample_rate,
            min_duration_s=config.min_duration,
            max_duration_s=config.max_duration,
            silence_thresh_dbfs=config.silence_threshold,
            min_silence_len_ms=config.min_silence_len,
            keep_silence_ms=config.keep_silence
        )
        
        if not result:
            logger.error("Segmentation failed. Stopping process.")
            sys.exit(1)
        
        # Merge short segments if enabled
        if config.merge_short_segments:
            from segments.segment_audio import merge_short_segments
            logger.info("\nMerging short segments before transcription...")
            merge_result = merge_short_segments(
                audio_dir=audio_output_dir,
                project_name=config.project_name,
                min_duration_threshold=config.merge_threshold
            )
            if not merge_result:
                logger.warning("Merging short segments encountered issues, continuing with transcription...")
                
        # Then transcribe
        result = transcribe_audio_files(
            audio_dir=audio_output_dir,
            output_csv_path=metadata_output_path,
            ljspeech=config.ljspeech,
            model_name=config.whisper_model,
            language_=config.language
        )
        
        if not result:
            logger.error("Transcription failed.")
            sys.exit(1)
        
    # Print some help information
    logger.info("\n--- IMPORTANT NOTES ---")
    logger.info("- Review the generated CSV file for accuracy.")
    logger.info(f"- Larger Whisper models generally yield better results but are slower.")
    logger.info("- Transcription speed depends heavily on your hardware (GPU highly recommended).")
    logger.info("- Check audio_processor.log for detailed processing information.")
    
    logger.info("\nProcess completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}")
        logger.critical(traceback.format_exc())
        sys.exit(1)
