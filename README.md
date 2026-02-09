# Bengali Audio Processing & Transcription Pipeline

This repository contains a **complete end-to-end pipeline** for creating a Bengali speech
dataset from YouTube videos. It covers audio downloading, preprocessing, segmentation,
and transcription preparation.

The pipeline is designed to **work without any paid Speech-to-Text (STT) API**.
When automatic transcription is unreliable or unavailable, a **manual transcription
interface** is provided, fully satisfying the assignment requirements.

---

## What This Project Does

1. Downloads audio from YouTube videos
2. Converts audio to standard WAV format
3. Splits audio into clean speech segments
4. Performs basic quality checks
5. Generates LJSpeech-style metadata
6. Allows **manual transcription** for failed segments

---

## How to access it
I used google colab for running the programs. Just download config.json, pipeline.py, manual_transcription_colab.py, urls.txt

Upload it while running the notebook. I have provided the notebook as well. Running all the cells will give the proper final output. I gave options for both manual and api based method, but as i dont have access to any good api, I used manual method. I tried with freely available Whisper model, but it gave bad resuls. So i switched to Manual method.

# Output
## Output Description

The pipeline generates a Bengali speech dataset inside the `output/` directory.

- **Audio files**:  
  Stored in `output/wavs/` as mono WAV files (22050 Hz, 16-bit PCM), named `audio_000000.wav`, `audio_000001.wav`, etc.

- **Metadata**:  
  `metadata.csv` (pipe-delimited, UTF-8) with format:  
  `audio_filename|transcription_text|speaker_name`

- **Final output**:  
  After manual correction, a clean `metadata_final.csv` is produced.

- **Submission**:  
  A ZIP file containing:

