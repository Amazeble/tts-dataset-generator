import os
import sys
import logging
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_silence
from moviepy import VideoFileClip, AudioFileClip
import traceback
import ffmpeg
import re

# Optional import for torchaudio (only needed if CUDA is available)
try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except (ImportError, OSError):
    TORCHAUDIO_AVAILABLE = False

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




def extract_audio_ffmpeg_py(video_path, audio_path, sample_rate = 22050):
    """
    Extracts audio from a video file and saves it as a mono, 16-bit PCM WAV
    at 22050 Hz using ffmpeg-python.

    Args:
        video_path (str): Path to the input video file.
        audio_path (str): Path to save the output WAV audio file.
    """
    print(f"Processing '{video_path}' with ffmpeg-python...")
    try:
        # Check if video file exists
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at '{video_path}'")
            return

        print(f"Extracting audio to '{audio_path}'...")
        print("Parameters: Mono, 22050 Hz, 16-bit PCM (WAV)")

        # Ensure the output filename ends with .wav for clarity
        # Although ffmpeg usually infers format, being explicit is good.
        if not audio_path.lower().endswith('.wav'):
             print("Warning: Output file does not end with .wav. Forcing WAV format.")
             # audio_path += ".wav" # Or rely on format spec below

        # Build the ffmpeg command pipeline
        stream = ffmpeg.input(video_path)

        # Select only the audio stream and apply filters/encoding options
        stream = ffmpeg.output(
            stream.audio,             # Use only the audio part of the input
            audio_path,
            acodec='pcm_s16le',       # Audio codec: 16-bit signed little-endian PCM
            ac=1,                     # Audio channels: 1 (mono)
            ar=str(sample_rate),               # Audio sample rate: 22050 Hz
            format='wav'            # Explicitly set format (optional if extension is .wav)
        )

        # Run the command, overwriting output file if it exists (-y flag)
        # Capture stdout/stderr for potential debugging
        stdout, stderr = ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)

        print("-" * 20)
        print("Audio extraction successful!")
        print(f"Output saved to: {audio_path}")
        print("-" * 20)

    except ffmpeg.Error as e:
        print('FFmpeg execution failed!', file=sys.stderr)
        print('stdout:', stdout.decode('utf8', errors='ignore'), file=sys.stderr)
        print('stderr:', stderr.decode('utf8', errors='ignore'), file=sys.stderr)
        print(f"An ffmpeg error occurred: {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)



def segment_audio_flexible(input_path, output_dir, project_name, sample_rate= 22050,
                           min_duration_s=3.0, max_duration_s=10.0,
                           silence_thresh_dbfs=-40, min_silence_len_ms=250,
                           keep_silence_ms=150, keep_silent=False,
                           temp_audio_filename="_temp_extracted_audio.wav"):
    """
    Segments an audio or video file into clips of flexible duration (min_duration_s to max_duration_s),
    prioritizing natural speech boundaries based on silence.

    If the input is a video, audio is extracted first.

    Args:
        input_path (str): Path to the input audio or video file.
        output_dir (str): Directory where the segmented WAV files will be saved.
        project_name (str): Name of the project for prefixing output files (e.g., Elise).
        sample_rate (float): Audio sample rate in Hz.
        min_duration_s (float): Minimum desired length of a segment in seconds.
        max_duration_s (float): Maximum desired length of a segment in seconds.
        silence_thresh_dbfs (int): Audio level below which is considered silence (dBFS).
                                   Adjust based on your recording's noise floor.
        min_silence_len_ms (int): Minimum duration of silence (in ms) to mark a split point.
                                  Adjust based on pauses between sentences/phrases.
        keep_silence_ms (int): Amount of original silence (in ms) to keep at the
                               beginning/end of each chunk for natural padding.
        keep_silent (bool): If True, do not remove any silent audio. The sum of durations
                            of all split files will equal the original file duration.
        temp_audio_filename (str): Filename for temporarily storing extracted audio.
    """
    logger.info(f"Processing input file: {input_path}")

    # --- Input Validation and Audio Extraction ---
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return False

    audio_path_to_process = None
    is_temporary_audio = False

    # Determine file type and handle video
    file_extension = os.path.splitext(input_path)[1].lower()
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']  # Add more if needed

    if file_extension in video_extensions:
        logger.info("Input is a video file. Extracting audio...")
        try:
            video = VideoFileClip(input_path)
            # Use the provided temp filename in the same dir as the script for simplicity
            audio_path_to_process = temp_audio_filename
            video.audio.write_audiofile(
                audio_path_to_process,
                fps=sample_rate,          # Sample rate
                nbytes=2,           # Bytes per sample (2 for 16-bit)
                codec='pcm_s16le',  # PCM signed 16-bit little-endian codec
                ffmpeg_params=["-ac", "1"] # Force mono audio (1 channel)
            )
            video.close()  # Release video file handle
            is_temporary_audio = True
            logger.info(f"Audio extracted successfully to: {audio_path_to_process}")
        except Exception as e:
            logger.error(f"Failed to extract audio from video: {e}")
            logger.debug(traceback.format_exc())
            if os.path.exists(temp_audio_filename):  # Clean up partial file if it exists
                os.remove(temp_audio_filename)
            return False
    else:
        # Assume it's an audio file pydub can handle
        audio = AudioFileClip(input_path)
        audio_path_to_process = temp_audio_filename
        audio.write_audiofile(
            audio_path_to_process,
            fps=sample_rate,          # Sample rate
            nbytes=2,           # Bytes per sample (2 for 16-bit)
            codec='pcm_s16le',  # PCM signed 16-bit little-endian codec
            ffmpeg_params=["-ac", "1"] # Force mono audio (1 channel)
        )


    # --- Audio Processing ---
    try:
        # Load the audio (either original or extracted)
        logger.info(f"Loading audio from: {audio_path_to_process}")
        try:
            # Attempt loading (pydub uses ffmpeg/libav)
            audio = AudioSegment.from_file(audio_path_to_process)
            logger.info(f"Audio loaded successfully. Duration: {len(audio) / 1000:.2f} seconds")
        except Exception as e:
            logger.error(f"Failed to load audio file: {e}")
            logger.error("Ensure the file is a valid audio format supported by pydub/ffmpeg.")
            logger.error("Also ensure 'ffmpeg' or 'libav' is installed and accessible.")
            logger.debug(traceback.format_exc())
            return False

        # Create the output directory
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

        logger.info(f"Splitting audio based on silence...")
        logger.info(f"  Parameters:")
        logger.info(f"    Min Silence Duration: {min_silence_len_ms} ms")
        logger.info(f"    Silence Threshold: {silence_thresh_dbfs} dBFS")
        logger.info(f"    Padding Silence: {keep_silence_ms} ms")
        logger.info(f"    Keep Silent Mode: {keep_silent}")

        # Split audio based on silence
        if keep_silent:
            # When keep_silent is True, we don't remove any silence
            # We split the audio into fixed-size chunks without silence detection
            original_duration_ms = len(audio)
            logger.info(f"Original audio duration: {original_duration_ms / 1000:.2f} seconds")
            logger.info("Keeping all silent audio - total duration will be preserved")
            
            # Create a single chunk containing the entire audio
            chunks = [audio]
        else:
            chunks = split_on_silence(
                audio,
                min_silence_len=min_silence_len_ms,
                silence_thresh=silence_thresh_dbfs,
                keep_silence=keep_silence_ms
            )

        if not chunks:
            logger.warning("No segments found based on silence detection.")
            logger.warning("Consider adjusting 'silence_thresh_dbfs' or 'min_silence_len_ms'.")
            return False

        logger.info(f"Found {len(chunks)} potential segments based on silence.")
        
        # When keep_silent is True, we don't filter by duration or add padding
        # We split the audio into equal parts while preserving total duration
        if keep_silent:
            original_duration_ms = len(audio)
            logger.info(f"Original audio duration: {original_duration_ms / 1000:.2f} seconds")
            logger.info("Keeping all silent audio - splitting without removing any content")
            
            # Get existing segment count to continue numbering
            saved_count = get_existing_segment_count(output_dir, project_name)
            
            # Split the audio into segments based on max_duration_s
            # This ensures we don't exceed the max duration while keeping all audio
            current_pos = 0
            segment_index = 0
            
            while current_pos < original_duration_ms:
                # Calculate the end position for this segment
                segment_end = min(current_pos + int(max_duration_s * 1000), original_duration_ms)
                segment_duration_ms = segment_end - current_pos
                
                # Extract the segment
                chunk = audio[current_pos:segment_end]
                
                # Save the segment without adding any padding
                saved_count += 1
                segment_index += 1
                output_filename = f"{project_name}_{saved_count:04d}.wav"
                output_path = os.path.join(output_dir, output_filename)
                
                chunk_duration_s = segment_duration_ms / 1000.0
                logger.info(f"  Saving segment {segment_index} ({chunk_duration_s:.2f}s): {output_path}")
                
                try:
                    # Export chunk as WAV file (standard format for TTS)
                    chunk.export(output_path, format="wav")
                except Exception as e:
                    logger.error(f"Failed to save segment ({output_path}): {e}")
                    logger.debug(traceback.format_exc())
                
                current_pos = segment_end
            
            logger.info("\nProcessing Complete!")
            logger.info(f"  Saved {saved_count} segments total.")
            logger.info(f"  Original duration: {original_duration_ms / 1000:.2f} seconds")
            logger.info(f"  Total segmented duration preserved: {original_duration_ms / 1000:.2f} seconds")
            
            return saved_count > 0  # Return True if we saved any segments
        
        logger.info(f"Filtering segments by duration ({min_duration_s:.1f}s - {max_duration_s:.1f}s)...")

        # Get existing segment count to continue numbering
        saved_count = get_existing_segment_count(output_dir, project_name)
        skipped_too_short = 0
        skipped_too_long = 0

        # Filter and save the chunks
        for i, chunk in enumerate(chunks):
            chunk_duration_s = len(chunk) / 1000.0

            # Check duration
            if chunk_duration_s < min_duration_s:
                skipped_too_short += 1
                continue
            if chunk_duration_s > max_duration_s:
                skipped_too_long += 1
                continue

            padding_needed = 250
            silence = AudioSegment.silent(duration=padding_needed)
            final_segment = chunk + silence

            # Save the valid segment
            saved_count += 1
            output_filename = f"{project_name}_{saved_count:04d}.wav"
            output_path = os.path.join(output_dir, output_filename)
            logger.debug(f"  Segment {saved_count} added {padding_needed/1000:.2f}s silence")
            logger.info(f"  Saving segment {saved_count} ({chunk_duration_s + (padding_needed/1000):.2f}s): {output_path}")
            try:
                # Export chunk as WAV file (standard format for TTS)
                final_segment.export(output_path, format="wav")
            except Exception as e:
                logger.error(f"Failed to save segment ({output_path}): {e}")
                logger.debug(traceback.format_exc())

        logger.info("\nProcessing Complete!")
        logger.info(f"  Saved {saved_count} segments total.")
        logger.info(f"  Skipped {skipped_too_short} segments (duration < {min_duration_s:.1f}s).")
        logger.info(f"  Skipped {skipped_too_long} segments (duration > {max_duration_s:.1f}s).")

        return saved_count > 0  # Return True if we saved any segments

    except Exception as e:
        logger.error(f"Error during audio segmentation: {e}")
        logger.debug(traceback.format_exc())
        return False
    finally:
        # --- Cleanup ---
        if is_temporary_audio and os.path.exists(audio_path_to_process):
            logger.info(f"Cleaning up temporary audio file: {audio_path_to_process}")
            try:
                os.remove(audio_path_to_process)
            except Exception as e:
                logger.warning(f"Could not delete temporary file {audio_path_to_process}: {e}")




def merge_short_segments(audio_dir, project_name, min_duration_threshold=2.0):
    """
    Merge any audio file shorter than min_duration_threshold with the next available file in chronological order 
    (File 1 + File 2 -> Saved as File 2, File 1 is deleted). Repeats seamlessly if the newly merged file is still under the threshold.
    """
    # 1. Rename the wavs directory to wavs_before_merge
    # (Assuming audio_dir is passed as the path to the 'wavs' folder)
    src_dir = os.path.normpath(audio_dir)
    parent_dir = os.path.dirname(src_dir)
    
    if os.path.basename(src_dir) == "wavs":
        backup_dir = os.path.join(parent_dir, "wavs_before_merge")
    else:
        backup_dir = f"{src_dir}_before_merge"

    if os.path.exists(src_dir) and not os.path.exists(backup_dir):
        logger.info(f"Renaming {src_dir} to {backup_dir}...")
        os.rename(src_dir, backup_dir)

    # Re-create the clean target 'wavs' directory to write into
    os.makedirs(src_dir, exist_ok=True)

    logger.info(f"Checking for segments shorter than {min_duration_threshold}s to merge...")
    # The source of our files to check is now the renamed directory
    if not os.path.exists(backup_dir):
        logger.error(f"Audio directory not found: {backup_dir}")
        return False

    # Regex to grab the 4-digit tracking number from the file names
    pattern = re.compile(rf'^{re.escape(project_name)}_(\d{{4}})\.wav$')

    def get_sorted_numbers():
        nums = []
        # Read files from the backup directory
        for filename in os.listdir(backup_dir):
            match = pattern.match(filename)
            if match:
                nums.append(int(match.group(1)))
        nums.sort()
        return nums

    i = 0
    while True:
        # Dynamically fetch the current files from the source folder at the start of every check
        nums = get_sorted_numbers()

        # Break condition: if we are at or past the final file, no forward merge is possible
        if i >= len(nums) - 1:
            # Write the final remaining file into wavs sequentially if it's left over
            if nums:
                final_num = nums[i]
                final_file = f"{project_name}_{final_num:04d}.wav"
                final_src = os.path.join(backup_dir, final_file)
                final_dest = os.path.join(src_dir, final_file)
                if os.path.exists(final_src) and not os.path.exists(final_dest):
                    os.rename(final_src, final_dest)
            break

        current_num = nums[i]
        current_file = f"{project_name}_{current_num:04d}.wav"
        
        # Read from target folder if already written there, otherwise read from the source folder
        current_path = os.path.join(src_dir, current_file)
        if not os.path.exists(current_path):
            current_path = os.path.join(backup_dir, current_file)

        try:
            current_audio = AudioSegment.from_file(current_path)
            duration_s = len(current_audio) / 1000.0

            # Action step: The file is too short, let's merge it forward
            if duration_s < min_duration_threshold:
                next_num = nums[i + 1]
                next_file = f"{project_name}_{next_num:04d}.wav"
                
                # Check source folder first for the next sequential file
                next_path = os.path.join(backup_dir, next_file)
                if not os.path.exists(next_path):
                    next_path = os.path.join(src_dir, next_file)

                logger.info(f"Segment {current_file} ({duration_s:.2f}s) is too short. Merging into {next_file}...")
                next_audio = AudioSegment.from_file(next_path)

                # CRITICAL: Order is explicitly kept as File 1 + File 2
                merged_audio = current_audio + next_audio

                # Write the merge file into wavs in sequence
                dest_next_path = os.path.join(src_dir, next_file)
                merged_audio.export(dest_next_path, format="wav")

                # Remove the processed source files sequentially
                if os.path.exists(current_path):
                    os.remove(current_path)
                if os.path.exists(next_path) and next_path.startswith(backup_dir):
                    os.remove(next_path)

                logger.info(f"Successfully deleted original short file: {current_file}")

                # CRITICAL: Do NOT increase 'i'. The loop spins again, treating the updated
                # next_file (File 2) as the new 'current_file' to check if it's still under 2 seconds.
                continue
            else:
                # Write valid sequence file into wavs
                dest_path = os.path.join(src_dir, current_file)
                if current_path.startswith(backup_dir) and os.path.exists(current_path):
                    os.rename(current_path, dest_path)

        except Exception as e:
            logger.error(f"Error handling sequential merge for file {current_file}: {e}")
            logger.debug(traceback.format_exc())
            return False

        # Only advance to the next index if the current file met the minimum duration
        i += 1

    logger.info("Sequential short segment merge completed successfully.")
    return True




def get_existing_segment_count(output_dir, project_name):
    """
    Count existing segments with the given project prefix to continue numbering.

    Args:
        output_dir (str): Directory to check for existing segments.
        project_name (str): Project name prefix to look for.

    Returns:
        int: The highest existing segment number, or 0 if none exist.
    """
    if not os.path.exists(output_dir):
        return 0

    import re
    pattern = re.compile(rf'^{re.escape(project_name)}_(\d{{4}})\.wav$')
    max_num = 0

    for filename in os.listdir(output_dir):
        match = pattern.match(filename)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num

    return max_num
