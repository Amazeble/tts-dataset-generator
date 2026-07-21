import os
import sys
import time
import logging
import traceback
from natsort import natsorted
from faster_whisper import WhisperModel
import torch
import numpy as np
from scipy.io import wavfile
import mediapipe as mp

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

if torch.cuda.is_available():
    GPU_AVAILABLE = True
else:
    GPU_AVAILABLE = False

def detect_snaps_with_google(file_path, model_path="yamnet.tflite", score_threshold=0.35):
    """
    Uses Google's MediaPipe AI framework to pinpoint finger-snapping sounds.
    Returns a list of timestamps (in seconds) where a finger snap was detected.
    """
    snap_timestamps = []
    try:
        # Load WAV file sample data
        sample_rate, data = wavfile.read(file_path)
        
        # Google MediaPipe requires a float32 array
        if data.dtype != np.float32:
            data = data.astype(np.float32) / np.iinfo(data.dtype).max
            
        # Convert stereo channel tracks into a single mono track
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        # Initialize Google's MediaPipe Audio Classifier
        BaseOptions = mp.tasks.BaseOptions
        AudioClassifier = mp.tasks.audio.AudioClassifier
        AudioClassifierOptions = mp.tasks.audio.AudioClassifierOptions
        
        # Restrict the model category filter to find only "Finger snapping" events
        options = AudioClassifierOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.audio.RunningMode.AUDIO_CLIPS,
            category_allowlist=["Finger snapping"],
            score_threshold=score_threshold
        )
        
        with AudioClassifier.create_from_options(options) as classifier:
            # Wrap standard array data into a dedicated MediaPipe audio block
            media_pipe_audio = mp.tasks.audio.AudioData.create_from_array(data, sample_rate)
            classification_result = classifier.classify(media_pipe_audio)
            
            # Extract timestamp tags from valid classification frame structures
            for classification in classification_result:
                timestamp_ms = classification.timestamp_ms
                for category in classification.categories:
                    if category.category_name == "Finger snapping":
                        timestamp_sec = timestamp_ms / 1000.0
                        snap_timestamps.append(timestamp_sec)
                        
        return sorted(list(set(snap_timestamps)))
    except Exception as e:
        logger.warning(f"Google MediaPipe snap detection bypassed for {os.path.basename(file_path)}. Detail: {e}")
        return []

def transcribe_audio_files(audio_dir, output_csv_path="metadata.csv", ljspeech=False, model_name="deepdml/faster-whisper-large-v3-turbo-ct2", language_="en"):
    """
    Transcribes all audio files, keeps filler words using initial prompts, 
    and checks for finger snaps with Google AI to inject <snap> text flags.
    """
    logger.info(f"Looking for .wav files in: {audio_dir}")
    if not os.path.isdir(audio_dir):
        logger.error(f"Directory not found: {audio_dir}")
        return False

    try:
        wav_files = natsorted([f for f in os.listdir(audio_dir) if f.lower().endswith('.wav')])
    except Exception as e:
        logger.error(f"Error accessing audio directory: {e}")
        logger.debug(traceback.format_exc())
        return False

    if not wav_files:
        logger.error(f"No .wav files found in {audio_dir}")
        return False

    logger.info(f"Found {len(wav_files)} .wav files to process.")

    # --- Load Faster-Whisper Model ---
    logger.info(f"Loading Faster-Whisper model: '{model_name}'...")
    device = "cuda" if GPU_AVAILABLE else "cpu"
    compute_type = "float16" if GPU_AVAILABLE else "float32"

    try:
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        logger.info(f"Faster-Whisper model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load Faster-Whisper model: {e}")
        return False

    # --- Process Files ---
    metadata_text = []
    metadata_audio_path = []
    total_files = len(wav_files)
    start_time_total = time.time()

    # The initial prompt forces Whisper to transcribe spoken filler words
    filler_prompt = "Umm, let me think, uh, yeah, like, ah, okay. So, um, what I mean is..."

    for i, filename in enumerate(wav_files):
        file_path = os.path.join(audio_dir, filename)
        logger.info(f"Processing file {i+1}/{total_files}: {filename}...")
        start_time_file = time.time()
        
        # 1. Use Google AI pipeline to extract the finger snap timestamps
        snap_times = detect_snaps_with_google(file_path)
        
        try:
            # 2. Transcribe text with individual word timestamps active
            segments, info = model.transcribe(
                file_path, 
                language=language_, 
                beam_size=5,
                initial_prompt=filler_prompt,
                word_timestamps=True
            )
            
            final_text_pieces = []
            
            # 3. Interweave words and <snap> tags chronologically
            for segment in segments:
                if segment.words:
                    for word in segment.words:
                        remaining_snaps = []
                        for snap_t in snap_times:
                            if snap_t < word.start:
                                final_text_pieces.append("<snap>")
                            else:
                                remaining_snaps.append(snap_t)
                        snap_times = remaining_snaps
                        
                        final_text_pieces.append(word.word.strip())
                else:
                    final_text_pieces.append(segment.text.strip())
            
            # Append trailing finger snaps that happen after the final word
            for _ in snap_times:
                final_text_pieces.append("<snap>")
                
            text = " ".join(final_text_pieces).strip()
            text = " ".join(text.split())  # Sanitize formatting spaces
            
            end_time_file = time.time()
            print(f"Done ({end_time_file - start_time_file:.2f}s). Result: '{text}'")
            
        except Exception as e:
            text = "[WHISPER_ERROR]"
            logger.error(f"Error processing {filename}: {e}")
            logger.debug(traceback.format_exc())

        metadata_text.append(text)
        metadata_audio_path.append(filename[:-4])

    end_time_total = time.time()
    logger.info(f"Finished processing in {end_time_total - start_time_total:.2f} seconds.")

    # --- Save To CSV ---
    try:
        with open(output_csv_path, 'w', encoding='utf-8') as f:
            for i in range(len(metadata_audio_path)):
                if ljspeech:
                    f.writelines(f'{metadata_audio_path[i]}|{metadata_text[i]}|{metadata_text[i]}\n')
                else:
                    f.writelines(f'wavs/{metadata_audio_path[i]}.wav|{metadata_text[i]}\n')
        logger.info("Metadata CSV file created successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to write CSV file: {e}")
        return False

if __name__ == "__main__":
    input_folder = "your_audio_folder_path" 
    
    if os.path.exists(input_folder):
        transcribe_audio_files(
            audio_dir=input_folder,
            output_csv_path="metadata.csv",
            ljspeech=True,
            language_="en"
        )
    else:
        logger.warning(f"Please replace 'your_audio_folder_path' with your actual WAV file directory.")
