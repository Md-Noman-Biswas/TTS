Bengali Speech Dataset – Manual Transcription

Dataset Source:
- Audio extracted from publicly available YouTube video(s)

Processing Pipeline:
- Audio downloaded and converted to WAV (22050 Hz, mono, 16-bit PCM)
- Silence-based segmentation (2–12 seconds)
- Automatic transcription attempted
- Low-quality segments manually transcribed

Transcription Method:
- Manual transcription performed by listening to each audio segment
- Pretrained model like Whisper was giving bad results without finetuning. So manual one is preferred.
- Bengali text typed by human annotator
- Segments with poor ASR output corrected manually

Dataset Structure:
- wavs/ : audio segments
- metadata_final.csv : LJSpeech-style metadata

Metadata Format:
audio_filename|transcription_text|speaker_name

Language:
- Bengali (Bangla)

Speaker:
- Single speaker (speaker1)

Notes:
- Some segments were skipped if unclear or noisy
