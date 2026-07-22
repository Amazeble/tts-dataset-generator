import os
import sys
import time
import logging
from natsort import natsorted
from faster_whisper import WhisperModel
import torch
import numpy as np
from scipy.io import wavfile
import mediapipe as mp

# Suppress Google MediaPipe warning screen clutter
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[sys.stdout] if hasattr(sys, 'stdout') else [logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("dataset_generator")

def transcribe_audio_files(audio_dir, output_csv_path="metadata.csv", ljspeech=True, model_name="deepdml/faster-whisper-large-v3-turbo-ct2", language_="en"):
    """
    Transcribes all .wav files in a directory using a single pass layout.
    Enforces int8_float16 precision and chronological word-timestamp interweaving.
    """
    if not os.path.isdir(audio_dir):
        logger.error(f"Directory not found: {audio_dir}")
        return False

    wav_files = natsorted([f for f in os.listdir(audio_dir) if f.lower().endswith('.wav')])
    if not wav_files:
        logger.error(f"No .wav files found in directory: {audio_dir}")
        return False

    total_files = len(wav_files)
    logger.info(f"--- RUNNING ONE-PASS EVALUATION FOR {total_files} FILES ---")

    # -------------------------------------------------------------------------
    # INITIALIZE GOOGLE MEDIAPIPE SOUND CLASSIFIER
    # -------------------------------------------------------------------------
    import transcribe
    script_dir = os.path.dirname(os.path.abspath(transcribe.__file__))
    model_absolute_path = os.path.join(script_dir, "yamnet.tflite")
        
    try:
        BaseOptions = mp.tasks.BaseOptions
        AudioClassifier = mp.tasks.audio.AudioClassifier
        AudioClassifierOptions = mp.tasks.audio.AudioClassifierOptions
        
        options = AudioClassifierOptions(
            base_options=BaseOptions(model_asset_path=model_absolute_path),
            running_mode=mp.tasks.audio.RunningMode.AUDIO_CLIPS,
            category_allowlist=["Finger snapping"],
            score_threshold=0.35
        )
        classifier = AudioClassifier.create_from_options(options)
    except Exception as e:
        logger.error(f"Failed to load Google MediaPipe: {e}")
        return False

    try:
        model = WhisperModel(
            model_name, 
            device="cuda" if torch.cuda.is_available() else "cpu", 
            device_index=0, 
            compute_type="int8_float16" if torch.cuda.is_available() else "float32"
        )
        logger.info("Faster-Whisper engine mapped successfully at int8_float16 precision.")
    except Exception as e:
        logger.error(f"GPU engine failure: {e}")
        classifier.close()
        return False

    final_metadata = {}
    filler_prompt = "Umm, let me think, uh, yeah, like, ah, okay. So, um, what I mean is..."
    start_total_time = time.time()

    # -------------------------------------------------------------------------
    # UNIFIED SEQUENTIAL ONE-PASS FILE EVALUATION LOOP
    # -------------------------------------------------------------------------
    for i, filename in enumerate(wav_files):
        file_path = os.path.join(audio_dir, filename)
        start_time_file = time.time()
        snap_timestamps = []
        
        try:
            sample_rate, data = wavfile.read(file_path)
            if data.dtype != np.float32:
                data = data.astype(np.float32) / np.iinfo(data.dtype).max
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)

            # Execution Step A: Run MediaPipe Classification natively from RAM
            from mediapipe.tasks.python.components.containers import audio_data as mp_audio
            media_pipe_audio = mp_audio.AudioData.create_from_array(data, sample_rate)
            classification_result = classifier.classify(media_pipe_audio)
            
            if classification_result and hasattr(classification_result, 'classifications'):
                for entry in classification_result.classifications:
                    for category in entry.categories:
                        if category.category_name == "Finger snapping":
                            snap_timestamps.append(entry.timestamp_ms / 1000.0)
            
            snap_timestamps = sorted(list(set(snap_timestamps)))
            
            # Print explicit MediaPipe detection outputs for this exact file
            print(f"[{i+1}/{total_files}] MediaPipe Snaps Detected at: {snap_timestamps}")

            # Execution Step B: Run Faster-Whisper Inference with word timestamps enabled
            segments, info = model.transcribe(
                data, language=language_, beam_size=1, 
                initial_prompt=filler_prompt, word_timestamps=True
            )
            
            final_text_pieces = []
            
            # Execution Step C: Interweave text structures chronologically using word metrics
            for segment in segments:
                if segment.words:
                    for word_obj in segment.words:
                        remaining_snaps = []
                        for snap_t in snap_timestamps:
                            if snap_t < word_obj.start:
                                final_text_pieces.append("<snap>")
                            else:
                                remaining_snaps.append(snap_t)
                        snap_timestamps = remaining_snaps
                        final_text_pieces.append(word_obj.word.strip())
                else:
                    final_text_pieces.extend(segment.text.strip().split())
            
            # Catch trailing items at the end of the timeline
            for _ in snap_timestamps:
                final_text_pieces.append("<snap>")
                
            text = " ".join(final_text_pieces).strip()
            text = " ".join(text.split())
            final_metadata[filename] = text
            
            print(f"[{i+1}/{total_files}] ({time.time() - start_time_file:.2f}s) Result: '{text}'")
        except Exception as e:
            final_metadata[filename] = "[ERROR]"

    classifier.close()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"Unified pass complete in {time.time() - start_total_time:.2f}s.")

    # -------------------------------------------------------------------------
    # WRITE DATASET CSV
    # -------------------------------------------------------------------------
    try:
        with open(output_csv_path, 'w', encoding='utf-8') as f:
            for filename in wav_files:
                clean_name = filename[:-4]
                text_content = final_metadata.get(filename, "")
                if ljspeech:
                    f.writelines(f'{clean_name}|{text_content}|{text_content}\n')
                else:
                    f.writelines(f'wavs/{clean_name}.wav|{text_content}\n')
        logger.info(f"Success! Output metadata saved at: {output_csv_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save CSV: {e}")
        return False
