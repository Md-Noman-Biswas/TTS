#!/usr/bin/env python3
"""
YouTube Audio Processing Pipeline - IMPROVED VERSION
Downloads audio from YouTube videos, splits into sentence-level segments,
applies noise reduction, and creates a structured dataset.

IMPROVEMENTS:
- Better language detection and handling
- Improved text normalization
- Quality checks for transcriptions
- Better error handling
"""

import os
import sys
import csv
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import logging
import re

# Audio processing
import yt_dlp
import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import noisereduce as nr
import soundfile as sf

# Transcription
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioPipeline:
    """Main pipeline for processing YouTube audio into sentence-level segments."""
    
    def __init__(self, config: dict):
        """Initialize pipeline with configuration."""
        self.config = config
        self.output_dir = Path(config.get('output_dir', 'output'))
        self.wavs_dir = self.output_dir / 'wavs'
        self.raw_audio_dir = self.output_dir / 'raw_audio'
        self.metadata_file = self.output_dir / 'metadata.csv'
        self.progress_file = self.output_dir / 'progress.json'
        self.quality_report_file = self.output_dir / 'quality_report.txt'
        
        # Audio parameters
        self.sample_rate = config.get('sample_rate', 22050)
        self.min_segment_duration = config.get('min_segment_duration', 2.0)
        self.max_segment_duration = config.get('max_segment_duration', 15.0)
        self.min_silence_len = config.get('min_silence_len', 400)
        self.silence_thresh = config.get('silence_thresh', -38)
        
        # Create directories
        self.wavs_dir.mkdir(parents=True, exist_ok=True)
        self.raw_audio_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or initialize progress
        self.progress = self._load_progress()
        
        # Quality tracking
        self.quality_stats = {
            'total_segments': 0,
            'manual_needed': 0,
            'low_confidence': 0,
            'good_quality': 0,
            'language_issues': 0
        }
        
        # Load whisper model if available
        self.whisper_model = None
        if WHISPER_AVAILABLE and config.get('use_whisper', True):
            try:
                model_size = config.get('whisper_model', 'medium')
                logger.info(f"Loading Whisper model: {model_size}")
                self.whisper_model = whisper.load_model(model_size)
            except Exception as e:
                logger.warning(f"Failed to load Whisper model: {e}")
    
    def _load_progress(self) -> dict:
        """Load processing progress from file."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {'downloaded': [], 'processed': []}
    
    def _save_progress(self):
        """Save processing progress to file."""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def _save_quality_report(self):
        """Save quality statistics report."""
        with open(self.quality_report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("TRANSCRIPTION QUALITY REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            total = self.quality_stats['total_segments']
            if total == 0:
                f.write("No segments processed yet.\n")
                return
            
            f.write(f"Total segments: {total}\n\n")
            f.write(f"✓ Good quality: {self.quality_stats['good_quality']} "
                   f"({self.quality_stats['good_quality']/total*100:.1f}%)\n")
            f.write(f"⚠ Low confidence: {self.quality_stats['low_confidence']} "
                   f"({self.quality_stats['low_confidence']/total*100:.1f}%)\n")
            f.write(f"⚠ Language issues: {self.quality_stats['language_issues']} "
                   f"({self.quality_stats['language_issues']/total*100:.1f}%)\n")
            f.write(f"✋ Manual needed: {self.quality_stats['manual_needed']} "
                   f"({self.quality_stats['manual_needed']/total*100:.1f}%)\n\n")
            
            needs_review = (self.quality_stats['manual_needed'] + 
                          self.quality_stats['low_confidence'] + 
                          self.quality_stats['language_issues'])
            
            f.write(f"Total needing review: {needs_review} "
                   f"({needs_review/total*100:.1f}%)\n")
    
    def download_audio(self, url: str) -> Optional[Path]:
        """Download audio from YouTube URL."""
        # Check if already downloaded
        if url in self.progress['downloaded']:
            logger.info(f"Skipping already downloaded: {url}")
            video_id = self._extract_video_id(url)
            return self.raw_audio_dir / f"{video_id}.wav"
        
        try:
            logger.info(f"Downloading audio from: {url}")
            
            # Configure yt-dlp options
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': str(self.raw_audio_dir / '%(id)s.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                }],
                'quiet': False,
                'no_warnings': False,
                'cookiefile': self.config.get('cookie_file'),
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_id = info['id']
                output_file = self.raw_audio_dir / f"{video_id}.wav"
                
                # Convert to required format
                logger.info("Converting to required audio format...")
                audio = AudioSegment.from_file(output_file)
                audio = audio.set_frame_rate(self.sample_rate)
                audio = audio.set_channels(1)  # Mono
                audio = audio.set_sample_width(2)  # 16-bit
                
                audio.export(output_file, format='wav')
                
                # Update progress
                self.progress['downloaded'].append(url)
                self._save_progress()
                
                logger.info(f"Successfully downloaded: {output_file}")
                return output_file
                
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return None
    
    def _extract_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL."""
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:embed\/)([0-9A-Za-z_-]{11})',
            r'(?:watch\?v=)([0-9A-Za-z_-]{11})'
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return url.replace('/', '_').replace(':', '_')
    
    def split_on_silence(self, audio_path: Path) -> List[Tuple[AudioSegment, float, float]]:
        """Split audio into segments based on silence detection."""
        logger.info(f"Splitting audio: {audio_path}")
        
        audio = AudioSegment.from_wav(audio_path)
        
        # Detect non-silent chunks
        nonsilent_ranges = detect_nonsilent(
            audio,
            min_silence_len=self.min_silence_len,
            silence_thresh=self.silence_thresh,
            seek_step=10
        )
        
        logger.info(f"Found {len(nonsilent_ranges)} non-silent regions")
        
        segments = []
        for start_ms, end_ms in nonsilent_ranges:
            duration_sec = (end_ms - start_ms) / 1000.0
            
            if duration_sec < self.min_segment_duration:
                continue
            
            if duration_sec > self.max_segment_duration:
                num_splits = int(np.ceil(duration_sec / self.max_segment_duration))
                split_duration_ms = (end_ms - start_ms) / num_splits
                
                for i in range(num_splits):
                    split_start = start_ms + i * split_duration_ms
                    split_end = min(start_ms + (i + 1) * split_duration_ms, end_ms)
                    segment = audio[split_start:split_end]
                    segment = self._trim_silence(segment)
                    
                    if len(segment) >= self.min_segment_duration * 1000:
                        segments.append((segment, split_start / 1000.0, split_end / 1000.0))
            else:
                segment = audio[start_ms:end_ms]
                segment = self._trim_silence(segment)
                
                if len(segment) >= self.min_segment_duration * 1000:
                    segments.append((segment, start_ms / 1000.0, end_ms / 1000.0))
        
        segments = self._merge_short_segments(segments)
        
        logger.info(f"Created {len(segments)} segments")
        return segments
    
    def _trim_silence(self, segment: AudioSegment, silence_thresh: int = -40) -> AudioSegment:
        """Trim leading and trailing silence from a segment."""
        nonsilent = detect_nonsilent(
            segment,
            min_silence_len=50,
            silence_thresh=silence_thresh,
            seek_step=10
        )
        
        if not nonsilent:
            return segment
        
        start_trim = nonsilent[0][0]
        end_trim = nonsilent[-1][1]
        
        return segment[start_trim:end_trim]
    
    def _merge_short_segments(self, segments: List[Tuple[AudioSegment, float, float]]) -> List[Tuple[AudioSegment, float, float]]:
        """Merge consecutive short segments."""
        if not segments:
            return segments
        
        merged = []
        current_segment, current_start, current_end = segments[0]
        
        for i in range(1, len(segments)):
            next_segment, next_start, next_end = segments[i]
            current_duration = len(current_segment) / 1000.0
            
            if (current_duration < self.min_segment_duration and 
                next_start - current_end < 1.0):
                current_segment = current_segment + next_segment
                current_end = next_end
            else:
                merged.append((current_segment, current_start, current_end))
                current_segment, current_start, current_end = next_segment, next_start, next_end
        
        merged.append((current_segment, current_start, current_end))
        return merged
    
    def clean_audio(self, segment: AudioSegment) -> AudioSegment:
        """Apply noise reduction and normalization to audio segment."""
        samples = np.array(segment.get_array_of_samples())
        
        if segment.sample_width == 2:  # 16-bit
            samples = samples.astype(np.float32) / 32768.0
        
        # Apply gentle noise reduction
        try:
            reduced = nr.reduce_noise(
                y=samples,
                sr=self.sample_rate,
                stationary=True,
                prop_decrease=0.7,  # Gentler than before
                n_fft=2048
            )
        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}, using original")
            reduced = samples
        
        # Normalize loudness
        rms = np.sqrt(np.mean(reduced ** 2))
        target_rms = 0.1
        
        if rms > 0:
            reduced = reduced * (target_rms / rms)
        
        reduced = np.clip(reduced, -1.0, 1.0)
        samples_int = (reduced * 32767).astype(np.int16)
        
        cleaned = AudioSegment(
            samples_int.tobytes(),
            frame_rate=self.sample_rate,
            sample_width=2,
            channels=1
        )
        
        return cleaned
    
    def transcribe_segment(self, audio_path: Path) -> Tuple[Optional[str], dict]:
        """
        Transcribe audio segment using Whisper with quality metrics.
        
        Returns:
            (transcription_text, quality_info)
        """
        if not self.whisper_model:
            return None, {'status': 'no_model'}

        try:
            # First pass: auto-detect language
            result = self.whisper_model.transcribe(
                str(audio_path),
                language=None,  # Auto-detect
                task="transcribe",
                temperature=0.0,
                beam_size=5,
                best_of=5,
                no_speech_threshold=0.6,
                logprob_threshold=-1.0,
                condition_on_previous_text=False,
                fp16=False  # Better compatibility
            )
            
            detected_lang = result.get("language", "unknown")
            text = result.get("text", "").strip()
            segments = result.get("segments", [])
            
            # Calculate average confidence
            avg_logprob = np.mean([s.get("avg_logprob", -1.0) for s in segments]) if segments else -1.0
            no_speech_prob = np.mean([s.get("no_speech_prob", 1.0) for s in segments]) if segments else 1.0
            
            quality_info = {
                'detected_language': detected_lang,
                'avg_logprob': avg_logprob,
                'no_speech_prob': no_speech_prob,
                'status': 'success'
            }
            
            # If not Bengali, retry with explicit Bengali
            if detected_lang not in ['bn', 'hi', 'mr', 'unknown']:
                logger.warning(f"Detected '{detected_lang}', retrying with Bengali...")
                quality_info['status'] = 'language_retry'
                
                result = self.whisper_model.transcribe(
                    str(audio_path),
                    language="bn",
                    task="transcribe",
                    temperature=0.0,
                    beam_size=5,
                    best_of=5,
                    no_speech_threshold=0.6,
                    logprob_threshold=-1.0,
                    condition_on_previous_text=False,
                    fp16=False
                )
                text = result.get("text", "").strip()
                segments = result.get("segments", [])
                avg_logprob = np.mean([s.get("avg_logprob", -1.0) for s in segments]) if segments else -1.0
                quality_info['avg_logprob'] = avg_logprob
            
            # Quality assessment
            if avg_logprob < -0.9 or no_speech_prob > 0.8:
                quality_info['status'] = 'low_confidence'
            elif detected_lang not in ['bn', 'hi', 'mr', 'unknown']:
                quality_info['status'] = 'language_issue'
            
            return text if text else None, quality_info
            
        except Exception as e:
            logger.error(f"Transcription failed for {audio_path}: {e}")
            return None, {'status': 'error', 'error': str(e)}
    
    def is_valid_bangla_text(self, text: str) -> bool:
        """Check if text contains valid Bengali characters."""
        if not text or len(text) < 2:
            return False
        
        # Count Bengali characters
        bangla_chars = sum(1 for ch in text if '\u0980' <= ch <= '\u09FF')
        total_chars = sum(1 for ch in text if ch.strip() and ch not in '.,?!। ')
        
        if total_chars == 0:
            return False
        
        # At least 50% should be Bengali
        return (bangla_chars / total_chars) >= 0.5
    
    def normalize_bangla_text(self, text: str, quality_info: dict) -> str:
        """
        Normalize transcription text with better handling.
        """
        if not text:
            return "[MANUAL TRANSCRIPTION NEEDED]"
        
        # Remove any mojibake attempts
        cleaned = text.strip()
        
        # Check if it's valid Bengali
        if not self.is_valid_bangla_text(cleaned):
            # Mark for review but keep original
            return f"[NEEDS_REVIEW] {cleaned}"
        
        # Only keep Bengali + common punctuation
        result = []
        for ch in cleaned:
            # Bengali Unicode range
            if '\u0980' <= ch <= '\u09FF':
                result.append(ch)
            # Common punctuation
            elif ch in ' .,?!।;:""\'()-':
                result.append(ch)
        
        final_text = ''.join(result).strip()
        
        # Clean up extra spaces
        final_text = re.sub(r'\s+', ' ', final_text)
        
        if len(final_text) < 2:
            return "[MANUAL TRANSCRIPTION NEEDED]"
        
        # Add quality marker if needed
        if quality_info.get('status') == 'low_confidence':
            return f"[LOW_CONF] {final_text}"
        elif quality_info.get('status') == 'language_issue':
            return f"[LANG_ISSUE] {final_text}"
        
        return final_text
    
    def process_video(self, url: str, speaker_name: str = "speaker1") -> int:
        """Process a single YouTube video through the entire pipeline."""
        logger.info(f"Processing video: {url}")
        
        # Step 1: Download audio
        audio_path = self.download_audio(url)
        if not audio_path or not audio_path.exists():
            logger.error(f"Failed to download: {url}")
            return 0
        
        # Step 2: Split on silence
        segments = self.split_on_silence(audio_path)
        
        if not segments:
            logger.warning(f"No segments found for: {url}")
            return 0
        
        # Process each segment
        metadata_rows = []
        segment_count = 0
        
        # Get starting index
        existing_files = list(self.wavs_dir.glob("audio_*.wav"))
        start_idx = len(existing_files)
        
        for i, (segment, start_time, end_time) in enumerate(segments):
            segment_idx = start_idx + i
            filename = f"audio_{segment_idx:06d}.wav"
            output_path = self.wavs_dir / filename
            
            logger.info(f"Processing segment {i+1}/{len(segments)}: {filename}")
            
            # Step 3: Clean audio
            try:
                cleaned_segment = self.clean_audio(segment)
            except Exception as e:
                logger.error(f"Failed to clean segment {i}: {e}")
                cleaned_segment = segment
            
            # Save cleaned segment
            cleaned_segment.export(
                output_path,
                format='wav',
                parameters=[
                    "-ar", str(self.sample_rate),
                    "-ac", "1",
                    "-sample_fmt", "s16"
                ]
            )
            
            # Step 4: Transcribe
            duration = len(cleaned_segment) / 1000.0
            self.quality_stats['total_segments'] += 1
            
            if duration < 1.5:  # Too short
                transcription = "[TOO_SHORT_MANUAL_NEEDED]"
                quality_info = {'status': 'too_short'}
                self.quality_stats['manual_needed'] += 1
            else:
                transcription, quality_info = self.transcribe_segment(output_path)
                
                if transcription is None:
                    transcription = "[MANUAL TRANSCRIPTION NEEDED]"
                    self.quality_stats['manual_needed'] += 1
                else:
                    transcription = self.normalize_bangla_text(transcription, quality_info)
                    
                    # Update stats based on markers
                    if "[MANUAL" in transcription or "[TOO_SHORT" in transcription:
                        self.quality_stats['manual_needed'] += 1
                    elif "[LOW_CONF]" in transcription:
                        self.quality_stats['low_confidence'] += 1
                    elif "[LANG_ISSUE]" in transcription or "[NEEDS_REVIEW]" in transcription:
                        self.quality_stats['language_issues'] += 1
                    else:
                        self.quality_stats['good_quality'] += 1
            
            # Log quality info
            logger.info(f"  Quality: {quality_info.get('status', 'unknown')}")
            if 'detected_language' in quality_info:
                logger.info(f"  Detected language: {quality_info['detected_language']}")
            
            # Add to metadata
            metadata_rows.append({
                'audio_filename': filename,
                'transcription_text': transcription,
                'speaker_name': speaker_name,
                'start_time': f"{start_time:.2f}",
                'end_time': f"{end_time:.2f}"
            })
            
            segment_count += 1
        
        # Step 5: Update metadata.csv
        self._append_metadata(metadata_rows)
        
        # Update progress
        self.progress['processed'].append(url)
        self._save_progress()
        
        # Save quality report
        self._save_quality_report()
        
        logger.info(f"Successfully processed {segment_count} segments from {url}")
        return segment_count
    
    def _append_metadata(self, rows: List[dict]):
        """Append rows to metadata CSV file."""
        file_exists = self.metadata_file.exists()
        
        with open(self.metadata_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['audio_filename', 'transcription_text', 'speaker_name'],
                delimiter='|'
            )
            
            if not file_exists:
                writer.writeheader()
            
            for row in rows:
                writer.writerow({
                    'audio_filename': row['audio_filename'],
                    'transcription_text': row['transcription_text'],
                    'speaker_name': row['speaker_name']
                })
    
    def process_urls_from_file(self, urls_file: Path, speaker_name: str = "speaker1"):
        """Process multiple YouTube URLs from a file."""
        with open(urls_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        logger.info(f"Processing {len(urls)} videos")
        
        total_segments = 0
        for i, url in enumerate(urls, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Video {i}/{len(urls)}")
            logger.info(f"{'='*60}\n")
            
            segments = self.process_video(url, speaker_name)
            total_segments += segments
        
        # Final report
        logger.info(f"\n{'='*60}")
        logger.info(f"Pipeline complete!")
        logger.info(f"Total segments created: {total_segments}")
        logger.info(f"Quality statistics:")
        logger.info(f"  ✓ Good quality: {self.quality_stats['good_quality']}")
        logger.info(f"  ⚠ Low confidence: {self.quality_stats['low_confidence']}")
        logger.info(f"  ⚠ Language issues: {self.quality_stats['language_issues']}")
        logger.info(f"  ✋ Manual needed: {self.quality_stats['manual_needed']}")
        logger.info(f"\nQuality report saved to: {self.quality_report_file}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"{'='*60}\n")


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description='YouTube Audio Processing Pipeline (Improved)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process URLs from a file
  python pipeline_improved.py --urls urls.txt --speaker speaker1
  
  # Process a single URL
  python pipeline_improved.py --url "https://youtube.com/watch?v=..." --speaker speaker1
  
  # Use custom configuration
  python pipeline_improved.py --urls urls.txt --config config.json
        """
    )
    
    parser.add_argument(
        '--url',
        type=str,
        help='Single YouTube URL to process'
    )
    
    parser.add_argument(
        '--urls',
        type=str,
        help='File containing YouTube URLs (one per line)'
    )
    
    parser.add_argument(
        '--speaker',
        type=str,
        default='speaker1',
        help='Speaker name/ID (default: speaker1)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='output',
        help='Output directory (default: output)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='JSON configuration file'
    )
    
    parser.add_argument(
        '--no-whisper',
        action='store_true',
        help='Disable Whisper transcription (manual transcription only)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Override config with command-line arguments
    config['output_dir'] = args.output
    config['use_whisper'] = not args.no_whisper
    
    # Initialize pipeline
    pipeline = AudioPipeline(config)
    
    # Process videos
    if args.url:
        pipeline.process_video(args.url, args.speaker)
    elif args.urls:
        pipeline.process_urls_from_file(Path(args.urls), args.speaker)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
