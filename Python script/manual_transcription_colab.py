#!/usr/bin/env python3
"""
Manual Transcription Interface
A simple web interface for manually transcribing audio segments.
"""

import os
import csv
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify, send_file

app = Flask(__name__)

# Configuration
OUTPUT_DIR = Path('output')
WAVS_DIR = OUTPUT_DIR / 'wavs'
METADATA_FILE = OUTPUT_DIR / 'metadata.csv'

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="bn">
<head>
<meta charset="utf-8">
<title>Manual Bengali Transcription</title>

<style>
:root {
    --bg: #0f172a;
    --panel: #111827;
    --card: #1f2933;
    --border: #334155;
    --text: #e5e7eb;
    --muted: #94a3b8;
    --accent: #22c55e;
    --accent-2: #38bdf8;
    --danger: #ef4444;
}

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: system-ui, -apple-system, Segoe UI, Roboto;
    background: linear-gradient(135deg, #020617, #0f172a);
    color: var(--text);
}

.container {
    max-width: 900px;
    margin: 40px auto;
    background: var(--panel);
    border-radius: 14px;
    box-shadow: 0 30px 80px rgba(0,0,0,0.6);
    overflow: hidden;
}

.header {
    padding: 28px;
    background: linear-gradient(135deg, #020617, #020617);
    border-bottom: 1px solid var(--border);
}

.header h1 {
    margin: 0;
    font-size: 26px;
    font-weight: 700;
}

.header p {
    margin-top: 6px;
    color: var(--muted);
}

.content {
    padding: 28px;
}

.progress {
    margin-bottom: 26px;
}

.progress-bar {
    background: #020617;
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
    width: 0%;
    transition: width 0.3s;
}

.progress-text {
    margin-top: 8px;
    text-align: right;
    font-size: 13px;
    color: var(--muted);
}

.segment-card {
    background: var(--card);
    border-radius: 14px;
    padding: 22px;
    border: 1px solid var(--border);
}

.segment-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 14px;
}

.segment-title {
    font-size: 18px;
    font-weight: 600;
}

.badge {
    font-size: 12px;
    padding: 4px 10px;
    border-radius: 999px;
    font-weight: 600;
}

.badge-pending {
    background: rgba(251, 191, 36, 0.15);
    color: #fbbf24;
}

.badge-done {
    background: rgba(34, 197, 94, 0.15);
    color: #22c55e;
}

audio {
    width: 100%;
    margin: 14px 0 18px 0;
}

.current-text {
    background: #020617;
    border-radius: 10px;
    padding: 12px;
    font-family: monospace;
    color: #cbd5f5;
    margin-bottom: 14px;
    border: 1px dashed var(--border);
}

textarea {
    width: 100%;
    min-height: 120px;
    background: #020617;
    color: var(--text);
    border-radius: 10px;
    border: 1px solid var(--border);
    padding: 14px;
    font-size: 16px;
    resize: vertical;
    font-family: 'Noto Sans Bengali', system-ui;
}

textarea::placeholder {
    color: #64748b;
}

textarea:focus {
    outline: none;
    border-color: var(--accent);
}

.actions {
    display: flex;
    gap: 12px;
    margin-top: 18px;
    justify-content: center;
}

button {
    border: none;
    padding: 10px 18px;
    border-radius: 10px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
}

.btn-save {
    background: var(--accent);
    color: #052e16;
}

.btn-save:hover {
    filter: brightness(1.1);
}

.btn-skip {
    background: #475569;
    color: white;
}

.btn-skip:hover {
    filter: brightness(1.1);
}

.btn-prev {
    background: #1e40af;
    color: white;
}

.btn-prev:hover {
    filter: brightness(1.1);
}

.empty {
    text-align: center;
    padding: 60px 20px;
    color: var(--muted);
}
</style>
</head>

<body>
<div class="container">
    <div class="header">
        <h1>🎧 Bengali Manual Transcription</h1>
        <p>Listen carefully and type clean Bangla text</p>
    </div>

    <div class="content">
        <div class="progress">
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill"></div>
            </div>
            <div class="progress-text" id="progressText">Loading…</div>
        </div>

        <div id="segments"></div>
    </div>
</div>

<script>
let segments = [];
let current = 0;

async function loadSegments() {
    const res = await fetch('/api/segments');
    const data = await res.json();
    segments = data.segments;
    render();
}

function render() {
    const el = document.getElementById('segments');
    if (segments.length === 0) {
        el.innerHTML = '<div class="empty">✅ All segments completed</div>';
        return;
    }

    const s = segments[current];
    const done = s.transcription !== '[MANUAL TRANSCRIPTION NEEDED]';

    el.innerHTML = `
    <div class="segment-card">
        <div class="segment-header">
            <div class="segment-title">${s.filename}</div>
            <div class="badge ${done ? 'badge-done' : 'badge-pending'}">
                ${done ? 'Done' : 'Needs Transcription'}
            </div>
        </div>

        <audio controls>
            <source src="/audio/${s.filename}" type="audio/wav">
        </audio>

        <div class="current-text">${s.transcription}</div>

        <textarea id="text" placeholder="এখানে পরিষ্কার বাংলা লিখুন…">${done ? s.transcription : ''}</textarea>

        <div class="actions">
            <button class="btn-save" onclick="save()">💾 Save</button>
            <button class="btn-skip" onclick="next()">⏭ Skip</button>
            ${current > 0 ? '<button class="btn-prev" onclick="prev()">⏮ Previous</button>' : ''}
        </div>
    </div>
    `;

    updateProgress();
}

async function save() {
    const text = document.getElementById('text').value.trim();
    if (!text) return;

    await fetch('/api/transcribe', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({
            filename: segments[current].filename,
            transcription: text
        })
    });

    segments[current].transcription = text;
    next();
}

function next() {
    if (current < segments.length - 1) {
        current++;
        render();
    }
}

function prev() {
    if (current > 0) {
        current--;
        render();
    }
}

function updateProgress() {
    const done = segments.filter(s => s.transcription !== '[MANUAL TRANSCRIPTION NEEDED]').length;
    const pct = Math.round((done / segments.length) * 100);
    document.getElementById('progressFill').style.width = pct + '%';
    document.getElementById('progressText').textContent = `${done}/${segments.length} completed (${pct}%)`;
}

loadSegments();
</script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/segments')
def get_segments():
    """Get all segments that need transcription."""
    segments = []
    
    if METADATA_FILE.exists():
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='|')
            for row in reader:
                segments.append({
                    'filename': row['audio_filename'],
                    'transcription': row['transcription_text'],
                    'speaker': row['speaker_name']
                })
    
    return jsonify({'segments': segments})

@app.route('/api/transcribe', methods=['POST'])
def save_transcription():
    """Save a transcription for a segment."""
    data = request.json
    filename = data.get('filename')
    transcription = data.get('transcription')
    
    if not filename or not transcription:
        return jsonify({'error': 'Missing filename or transcription'}), 400
    
    # Read all rows
    rows = []
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='|')
        for row in reader:
            if row['audio_filename'] == filename:
                row['transcription_text'] = transcription
            rows.append(row)
    
    # Write back
    with open(METADATA_FILE, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['audio_filename', 'transcription_text', 'speaker_name'], delimiter='|')
        writer.writeheader()
        writer.writerows(rows)
    
    return jsonify({'success': True})

@app.route('/audio/<filename>')
def serve_audio(filename):
    """Serve audio files."""
    audio_path = WAVS_DIR / filename
    if audio_path.exists():
        return send_file(audio_path, mimetype='audio/wav')
    return 'File not found', 404

if __name__ == '__main__':
    if not METADATA_FILE.exists():
        print("Error: metadata.csv not found. Please run the pipeline first.")
        exit(1)
    
    print("\n" + "="*60)
    print("Manual Transcription Interface")
    print("="*60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Metadata file: {METADATA_FILE}")
    print(f"\nOpen your browser and go to: http://localhost:5000")
    print("\nKeyboard shortcuts:")
    print("  Ctrl+Enter: Save and next")
    print("  Ctrl+→: Next segment")
    print("  Ctrl+←: Previous segment")
    print("\nPress Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
#!/usr/bin/env python3
"""
Manual Transcription Tool
Allows users to listen to audio segments and manually transcribe them.
Works in Jupyter/Colab notebooks with interactive audio playback.
"""

import csv
import json
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from IPython.display import Audio, display, clear_output
from ipywidgets import widgets, Layout, VBox, HBox
import warnings
warnings.filterwarnings('ignore')


class ManualTranscriptionTool:
    """Interactive tool for manual transcription with audio playback."""
    
    def __init__(self, wavs_dir: str = "output/wavs", metadata_file: str = "output/metadata.csv"):
        self.wavs_dir = Path(wavs_dir)
        self.metadata_file = Path(metadata_file)
        self.current_index = 0
        self.segments = []
        self.load_metadata()
        
    def load_metadata(self):
        """Load metadata and identify segments needing transcription."""
        if not self.metadata_file.exists():
            print(f"❌ Metadata file not found: {self.metadata_file}")
            return
        
        # Read metadata
        df = pd.read_csv(self.metadata_file, sep='|', encoding='utf-8')
        
        # Find segments needing manual transcription
        needs_manual = df[
            df['transcription_text'].str.contains(
                r'\[MANUAL|\[LOW_CONF\]|\[NEEDS_REVIEW\]|\[LANG_ISSUE\]|^ব+$',
                regex=True,
                na=False
            )
        ]
        
        self.segments = needs_manual.to_dict('records')
        print(f"✅ Loaded {len(self.segments)} segments needing transcription")
        
    def save_transcription(self, index: int, new_transcription: str):
        """Save updated transcription to metadata file."""
        if not new_transcription.strip():
            print("⚠️ Empty transcription not saved")
            return False
        
        # Read full metadata
        df = pd.read_csv(self.metadata_file, sep='|', encoding='utf-8')
        
        # Find and update the row
        filename = self.segments[index]['audio_filename']
        df.loc[df['audio_filename'] == filename, 'transcription_text'] = new_transcription.strip()
        
        # Save back to file
        df.to_csv(self.metadata_file, sep='|', index=False, encoding='utf-8')
        
        # Update local copy
        self.segments[index]['transcription_text'] = new_transcription.strip()
        
        print(f"✅ Saved transcription for {filename}")
        return True
    
    def export_corrected_metadata(self, output_file: str = "output/metadata_corrected.csv"):
        """Export the corrected metadata to a new file."""
        df = pd.read_csv(self.metadata_file, sep='|', encoding='utf-8')
        df.to_csv(output_file, sep='|', index=False, encoding='utf-8')
        print(f"✅ Exported corrected metadata to: {output_file}")
        
    def show_progress(self):
        """Show transcription progress statistics."""
        df = pd.read_csv(self.metadata_file, sep='|', encoding='utf-8')
        
        total = len(df)
        needs_work = len(df[
            df['transcription_text'].str.contains(
                r'\[MANUAL|\[LOW_CONF\]|\[NEEDS_REVIEW\]|\[LANG_ISSUE\]|^ব+$',
                regex=True,
                na=False
            )
        ])
        completed = total - needs_work
        
        print("=" * 60)
        print("TRANSCRIPTION PROGRESS")
        print("=" * 60)
        print(f"Total segments: {total}")
        print(f"✅ Completed: {completed} ({completed/total*100:.1f}%)")
        print(f"⏳ Remaining: {needs_work} ({needs_work/total*100:.1f}%)")
        print("=" * 60)


class InteractiveTranscriber:
    """Interactive widget-based transcription interface for Jupyter/Colab."""
    
    def __init__(self, tool: ManualTranscriptionTool):
        self.tool = tool
        self.current_idx = 0
        
        # Create widgets
        self.audio_widget = None
        self.info_widget = widgets.HTML()
        self.transcription_input = widgets.Textarea(
            placeholder='Type Bengali transcription here...',
            layout=Layout(width='100%', height='100px')
        )
        self.save_button = widgets.Button(
            description='💾 Save',
            button_style='success',
            layout=Layout(width='120px')
        )
        self.skip_button = widgets.Button(
            description='⏭️ Skip',
            button_style='warning',
            layout=Layout(width='120px')
        )
        self.prev_button = widgets.Button(
            description='⏮️ Previous',
            button_style='info',
            layout=Layout(width='120px')
        )
        self.progress_widget = widgets.HTML()
        
        # Button handlers
        self.save_button.on_click(self.on_save)
        self.skip_button.on_click(self.on_skip)
        self.prev_button.on_click(self.on_prev)
        
    def on_save(self, b):
        """Save button handler."""
        transcription = self.transcription_input.value
        if self.tool.save_transcription(self.current_idx, transcription):
            self.current_idx += 1
            self.show_current_segment()
    
    def on_skip(self, b):
        """Skip button handler."""
        self.current_idx += 1
        self.show_current_segment()
    
    def on_prev(self, b):
        """Previous button handler."""
        if self.current_idx > 0:
            self.current_idx -= 1
            self.show_current_segment()
    
    def show_current_segment(self):
        """Display current segment for transcription."""
        if self.current_idx >= len(self.tool.segments):
            self.show_completion()
            return

        segment = self.tool.segments[self.current_idx]
        audio_path = self.tool.wavs_dir / segment['audio_filename']
        current_text = segment['transcription_text']

        progress_percent = (self.current_idx / len(self.tool.segments)) * 100

        progress_html = f"""
        <div style="
            background: #0f172a;
            color: #e5e7eb;
            padding: 14px;
            border-radius: 10px;
            margin-bottom: 16px;
            font-family: system-ui;
        ">
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                <strong>Progress</strong>
                <span>{self.current_idx + 1} / {len(self.tool.segments)}</span>
            </div>
            <div style="background:#1e293b; height:10px; border-radius:6px;">
                <div style="
                    width:{progress_percent:.1f}%;
                    background:#22c55e;
                    height:100%;
                    border-radius:6px;
                "></div>
            </div>
        </div>
        """

        info_html = f"""
        <div style="
            background:#ffffff;
            border-radius:12px;
            padding:18px;
            margin-bottom:16px;
            box-shadow:0 4px 12px rgba(0,0,0,0.08);
            font-family: system-ui;
        ">
            <h3 style="margin:0 0 10px 0; color:#0f172a;">
                🎧 {segment['audio_filename']}
            </h3>

            <p style="margin:0; color:#475569;">
                <strong>Current transcription:</strong>
            </p>

            <div style="
                margin-top:8px;
                background:#f8fafc;
                padding:10px;
                border-radius:6px;
                font-family: monospace;
                color:#334155;
                white-space: pre-wrap;
            ">
                {current_text}
            </div>
        </div>
        """

        clear_output(wait=True)

        display(widgets.HTML(progress_html))
        display(widgets.HTML(info_html))

        display(Audio(str(audio_path), autoplay=False))

        self.transcription_input.layout.height = "120px"
        display(self.transcription_input)

        display(HBox(
            [self.save_button, self.skip_button, self.prev_button],
            layout=Layout(justify_content="center", gap="12px")
        ))

        # Pre-fill logic
        if not any(m in current_text for m in ['[MANUAL', '[LOW_CONF', '[NEEDS_REVIEW', '[LANG_ISSUE']):
            self.transcription_input.value = current_text
        else:
            self.transcription_input.value = ""

    
    def show_completion(self):
        clear_output(wait=True)

        completion_html = """
        <div style="
            background:#ecfdf5;
            border-left:6px solid #22c55e;
            padding:24px;
            border-radius:12px;
            font-family: system-ui;
            text-align:center;
        ">
            <h2 style="color:#065f46; margin-top:0;">
                🎉 Transcription Complete
            </h2>
            <p style="color:#064e3b; font-size:16px;">
                All segments requiring manual transcription are finished.
            </p>
            <p style="color:#047857;">
                You can now export the final metadata.
            </p>
            <code style="
                background:#d1fae5;
                padding:6px 10px;
                border-radius:6px;
                display:inline-block;
            ">
                tool.export_corrected_metadata()
            </code>
        </div>
        """

        display(widgets.HTML(completion_html))
        self.tool.show_progress()

    
    def start(self):
        """Start the interactive transcription session."""
        if not self.tool.segments:
            print("✅ No segments need transcription!")
            return
        
        print(f"🚀 Starting transcription session with {len(self.tool.segments)} segments\n")
        print("Instructions:")
        print("  1. Listen to the audio")
        print("  2. Type the Bengali transcription")
        print("  3. Click 'Save' to save and move to next")
        print("  4. Click 'Skip' to skip this segment")
        print("  5. Click 'Previous' to go back\n")
        
        self.show_current_segment()


# Simple CLI-based transcription (for non-notebook environments)
class CLITranscriber:
    """Command-line interface for manual transcription."""
    
    def __init__(self, tool: ManualTranscriptionTool):
        self.tool = tool
    
    def start(self):
        """Start CLI transcription session."""
        if not self.tool.segments:
            print("✅ No segments need transcription!")
            return
        
        print(f"\n🚀 Starting transcription of {len(self.tool.segments)} segments")
        print("=" * 60)
        print("Commands: [s]ave, [p]lay again, [sk]ip, [q]uit")
        print("=" * 60 + "\n")
        
        for idx, segment in enumerate(self.tool.segments):
            audio_path = self.tool.wavs_dir / segment['audio_filename']
            
            print(f"\n📁 Segment {idx + 1}/{len(self.tool.segments)}")
            print(f"   File: {segment['audio_filename']}")
            print(f"   Current: {segment['transcription_text']}")
            print(f"   Audio file: {audio_path}")
            print("\n   🎧 Play this file in your audio player, then type transcription below:")
            
            while True:
                cmd = input("\n   [s]ave / [sk]ip / [q]uit: ").strip().lower()
                
                if cmd == 'q':
                    print("\n👋 Quitting...")
                    return
                elif cmd == 'sk':
                    print("   ⏭️ Skipped")
                    break
                elif cmd == 's':
                    transcription = input("   Enter transcription: ").strip()
                    if transcription:
                        self.tool.save_transcription(idx, transcription)
                        break
                    else:
                        print("   ⚠️ Empty transcription, try again")
                else:
                    print("   ❌ Invalid command")
        
        print("\n" + "=" * 60)
        print("✅ Session complete!")
        self.tool.show_progress()


def create_simple_review_html(wavs_dir: str = "output/wavs", 
                              metadata_file: str = "output/metadata.csv",
                              output_html: str = "output/review.html"):
    """
    Create a simple HTML file for reviewing/transcribing audio segments.
    Works offline, can be opened in any browser.
    """
    import os
    
    wavs_path = Path(wavs_dir)
    meta_path = Path(metadata_file)
    
    if not meta_path.exists():
        print(f"❌ Metadata file not found: {meta_path}")
        return
    
    df = pd.read_csv(meta_path, sep='|', encoding='utf-8')
    
    # Filter segments needing work
    needs_work = df[
        df['transcription_text'].str.contains(
            r'\[MANUAL|\[LOW_CONF\]|\[NEEDS_REVIEW\]|\[LANG_ISSUE\]|^ব+$',
            regex=True,
            na=False
        )
    ]
    
    html_content = f"""
<!DOCTYPE html>
<html lang="bn">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Manual Transcription Review</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            margin-bottom: 10px;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .stats {{
            background: #e8f5e9;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
            border-left: 5px solid #4CAF50;
        }}
        .segment {{
            background: #fafafa;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            border-left: 5px solid #ff9800;
        }}
        .segment h3 {{
            color: #333;
            margin-bottom: 15px;
        }}
        .current-text {{
            background: #fff3cd;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            font-family: 'Kalpurush', 'Noto Sans Bengali', sans-serif;
            border-left: 3px solid #ffc107;
        }}
        audio {{
            width: 100%;
            margin: 15px 0;
        }}
        textarea {{
            width: 100%;
            min-height: 80px;
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
            font-family: 'Kalpurush', 'Noto Sans Bengali', sans-serif;
            resize: vertical;
        }}
        textarea:focus {{
            outline: none;
            border-color: #4CAF50;
        }}
        .btn {{
            background: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            margin: 5px;
        }}
        .btn:hover {{ background: #45a049; }}
        .btn-export {{
            background: #2196F3;
            margin-top: 20px;
            display: block;
            width: 100%;
        }}
        .btn-export:hover {{ background: #0b7dda; }}
        .progress {{
            background: #e0e0e0;
            height: 25px;
            border-radius: 12px;
            overflow: hidden;
            margin: 20px 0;
        }}
        .progress-bar {{
            background: linear-gradient(90deg, #4CAF50, #8BC34A);
            height: 100%;
            transition: width 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }}
        .instructions {{
            background: #e3f2fd;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
            border-left: 5px solid #2196F3;
        }}
        .instructions ol {{
            margin-left: 20px;
            margin-top: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎧 Manual Transcription Review</h1>
        
        <div class="stats">
            <strong>Total segments needing review:</strong> {len(needs_work)}<br>
            <strong>Date:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
        </div>
        
        <div class="instructions">
            <strong>📝 Instructions:</strong>
            <ol>
                <li>Listen to each audio segment</li>
                <li>Type the correct Bengali transcription in the text box</li>
                <li>Copy the corrected text back to your metadata.csv file</li>
                <li>Or use the Python tool for easier workflow</li>
            </ol>
        </div>
        
        <div class="progress">
            <div class="progress-bar" id="progress-bar" style="width: 0%">0%</div>
        </div>
"""
    
    for idx, row in needs_work.iterrows():
        audio_rel_path = f"../{wavs_dir}/{row['audio_filename']}"
        
        html_content += f"""
        <div class="segment" id="segment-{idx}">
            <h3>📁 {row['audio_filename']}</h3>
            <div class="current-text">
                <strong>Current transcription:</strong><br/>
                {row['transcription_text']}
            </div>
            <audio controls preload="metadata">
                <source src="{audio_rel_path}" type="audio/wav">
                Your browser does not support audio playback.
            </audio>
            <textarea id="text-{idx}" placeholder="Type Bengali transcription here...">{'' if '[' in str(row['transcription_text']) else row['transcription_text']}</textarea>
            <button class="btn" onclick="markComplete({idx})">✓ Mark Complete</button>
            <button class="btn" style="background: #ff9800;" onclick="copyText({idx})">📋 Copy Text</button>
        </div>
"""
    
    html_content += """
        <button class="btn btn-export" onclick="exportResults()">💾 Export All Transcriptions (JSON)</button>
    </div>
    
    <script>
        let completedCount = 0;
        const totalSegments = """ + str(len(needs_work)) + """;
        
        function updateProgress() {
            const percent = (completedCount / totalSegments) * 100;
            document.getElementById('progress-bar').style.width = percent + '%';
            document.getElementById('progress-bar').textContent = Math.round(percent) + '%';
        }
        
        function markComplete(id) {
            const segment = document.getElementById('segment-' + id);
            segment.style.opacity = '0.5';
            segment.style.borderLeftColor = '#4CAF50';
            completedCount++;
            updateProgress();
        }
        
        function copyText(id) {
            const textarea = document.getElementById('text-' + id);
            textarea.select();
            document.execCommand('copy');
            alert('✓ Copied to clipboard!');
        }
        
        function exportResults() {
            const results = {};
""" + "\n".join([
        f"            results['{row['audio_filename']}'] = document.getElementById('text-{idx}').value;"
        for idx, row in needs_work.iterrows()
    ]) + """
            
            const dataStr = JSON.stringify(results, null, 2);
            const dataBlob = new Blob([dataStr], {type: 'application/json'});
            const url = URL.createObjectURL(dataBlob);
            const link = document.createElement('a');
            link.href = url;
            link.download = 'transcriptions.json';
            link.click();
        }
    </script>
</body>
</html>
"""
    
    output_path = Path(output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Created review HTML: {output_path}")
    print(f"📂 Open this file in your browser to review {len(needs_work)} segments")
    return output_path


# Example usage
if __name__ == "__main__":
    print("Manual Transcription Tool")
    print("=" * 60)
    print("\nUsage in Jupyter/Colab notebook:")
    print("""
    from manual_transcription import ManualTranscriptionTool, InteractiveTranscriber
    
    # Initialize tool
    tool = ManualTranscriptionTool()
    
    # Start interactive session
    transcriber = InteractiveTranscriber(tool)
    transcriber.start()
    
    # Or create HTML review page
    create_simple_review_html()
    """)