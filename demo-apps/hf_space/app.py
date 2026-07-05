import os
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import numpy as np
import gradio as gr
from torchvision import models

# ── Constants ────────────────────────────────────────────────────────────────
CLASSES       = ["Negative", "Sad", "Positive", "Neutral"]
EMOJIS        = ["😠", "😢", "😊", "😐"]
COLORS_HEX    = ["#E05C5C", "#6B8FD4", "#5DBF7A", "#9B8EC4"]
EMOTION_DESC  = {
    "Negative": "angry / frustrated",
    "Sad":      "sad / dejected",
    "Positive": "happy / excited",
    "Neutral":  "calm / neutral",
}
SAMPLE_RATE   = 16000
CHUNK_SEC     = 3.0
HOP_LENGTH    = 160
N_FFT         = 1024
WIN_LENGTH    = 400
N_MELS        = 80
CHUNK_FRAMES  = 481
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_TOKEN      = os.environ.get("HF_TOKEN", None)

# ── Model ────────────────────────────────────────────────────────────────────
class EfficientNetSER(nn.Module):
    def __init__(self, num_classes=4, dropout=0.5, freeze_until=0):
        super().__init__()
        base = models.efficientnet_b0(weights=None)
        old_conv = base.features[0][0]
        new_conv = nn.Conv2d(1, old_conv.out_channels,
                             kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride,
                             padding=old_conv.padding,
                             bias=False)
        new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
        base.features[0][0] = new_conv
        self.features   = base.features
        self.avgpool    = base.avgpool
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(1280, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# load SER model once
ser_model = EfficientNetSER(num_classes=4, dropout=0.5, freeze_until=0).to(DEVICE)
ser_model.load_state_dict(torch.load("best_exp10.pt", map_location=DEVICE))
ser_model.eval()

mel_transform    = T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=N_FFT,
                                    hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
                                    n_mels=N_MELS).to(DEVICE)
amplitude_to_db  = T.AmplitudeToDB(top_db=80).to(DEVICE)

# diarization pipeline (lazy load)
_diarization_pipeline = None

def get_diarization_pipeline():
    global _diarization_pipeline
    if _diarization_pipeline is None:
        from pyannote.audio import Pipeline
        _diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=HF_TOKEN,
        )
    return _diarization_pipeline

# ── Audio helpers ─────────────────────────────────────────────────────────────
def load_audio(path):
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    return waveform.squeeze(0)

def waveform_to_chunks(waveform):
    chunk_samples = int(CHUNK_SEC * SAMPLE_RATE)
    chunks, timestamps = [], []
    start = 0
    while start < waveform.shape[0]:
        end   = start + chunk_samples
        chunk = waveform[start:end]
        if len(chunk) < chunk_samples:
            chunk = torch.nn.functional.pad(chunk, (0, chunk_samples - len(chunk)))
        chunks.append(chunk)
        timestamps.append((start / SAMPLE_RATE, min(end, waveform.shape[0]) / SAMPLE_RATE))
        start += chunk_samples
    return chunks, timestamps

def chunk_to_spec(chunk):
    chunk = chunk.to(DEVICE).unsqueeze(0)
    spec  = mel_transform(chunk)
    spec  = amplitude_to_db(spec)
    if spec.shape[-1] != CHUNK_FRAMES:
        spec = torch.nn.functional.interpolate(
            spec.unsqueeze(0), size=(N_MELS, CHUNK_FRAMES),
            mode="bilinear", align_corners=False
        ).squeeze(0)
    mean, std = spec.mean(), spec.std() + 1e-6
    spec = (spec - mean) / std
    return spec.unsqueeze(0)

@torch.no_grad()
def predict_chunks(chunks):
    probs_list = []
    for chunk in chunks:
        spec   = chunk_to_spec(chunk)
        logits = ser_model(spec)
        probs  = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        probs_list.append(probs)
    return np.array(probs_list)

# ── HTML builders ─────────────────────────────────────────────────────────────
def build_hero_html(probs, duration, n_chunks):
    mean_probs = probs.mean(axis=0)
    dom_idx    = int(mean_probs.argmax())
    dom_cls    = CLASSES[dom_idx]
    dom_color  = COLORS_HEX[dom_idx]
    dom_conf   = mean_probs[dom_idx]
    dom_desc   = EMOTION_DESC[dom_cls]

    bars = ""
    for i, (cls, col, emoji) in enumerate(zip(CLASSES, COLORS_HEX, EMOJIS)):
        pct = mean_probs[i] * 100
        bars += f"""
        <div class="bar-row">
          <span class="bar-label">{emoji} {cls}</span>
          <div class="bar-track">
            <div class="bar-fill" style="width:{pct:.1f}%;background:{col}"></div>
          </div>
          <span class="bar-pct">{pct:.1f}%</span>
        </div>"""

    html = f"""
    <div class="card hero-card">
      <div class="hero-left">
        <div class="hero-emoji">{EMOJIS[dom_idx]}</div>
        <div class="hero-emotion" style="color:{dom_color}">{dom_cls}</div>
        <div class="hero-desc">({dom_desc})</div>
        <div class="hero-conf">{dom_conf*100:.1f}% confidence</div>
        <div class="hero-meta">{duration:.1f}s of audio · {n_chunks} chunk(s) analysed</div>
      </div>
      <div class="hero-right">
        <div class="bars-title">Probability per emotion</div>
        {bars}
      </div>
    </div>"""
    return html

def build_timeline_html(timestamps, probs):
    duration = timestamps[-1][1]
    preds    = probs.argmax(axis=1)
    segments = ""
    for i, (s, e) in enumerate(timestamps):
        left  = s / duration * 100
        width = (e - s) / duration * 100
        col   = COLORS_HEX[preds[i]]
        cls   = CLASSES[preds[i]]
        segments += f'<div class="seg" style="left:{left:.2f}%;width:{width:.2f}%;background:{col}" title="{cls} {s:.1f}s–{e:.1f}s"></div>'

    legend = ""
    for cls, col, emoji in zip(CLASSES, COLORS_HEX, EMOJIS):
        legend += f'<span class="leg-item"><span class="leg-dot" style="background:{col}"></span>{emoji} {cls}</span>'

    # build SVG line chart
    n      = len(timestamps)
    W, H   = 800, 200
    pad    = 40
    chart_w = W - pad * 2
    chart_h = H - pad * 2

    paths = ""
    for ci, (cls, col) in enumerate(zip(CLASSES, COLORS_HEX)):
        pts = []
        for i, (s, e) in enumerate(timestamps):
            mid = (s + e) / 2
            x   = pad + (mid / duration) * chart_w
            y   = pad + (1 - probs[i, ci]) * chart_h
            pts.append(f"{x:.1f},{y:.1f}")
        paths += f'<polyline points="{" ".join(pts)}" fill="none" stroke="{col}" stroke-width="2.5" stroke-linejoin="round"/>'

    # y-axis labels
    y_labels = ""
    for v in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = pad + (1 - v) * chart_h
        y_labels += f'<text x="{pad-6}" y="{y+4}" text-anchor="end" fill="#aaa" font-size="11">{v:.2f}</text>'
        y_labels += f'<line x1="{pad}" y1="{y}" x2="{W-pad}" y2="{y}" stroke="#e8e0d0" stroke-width="0.8"/>'

    # x-axis labels
    x_labels = ""
    n_ticks  = min(8, n)
    for ti in range(n_ticks + 1):
        t = duration * ti / n_ticks
        x = pad + (t / duration) * chart_w
        x_labels += f'<text x="{x}" y="{H-6}" text-anchor="middle" fill="#aaa" font-size="11">{t:.1f}s</text>'

    svg = f"""
    <svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:220px">
      {y_labels}{x_labels}{paths}
    </svg>"""

    html = f"""
    <div class="card">
      <div class="section-title">Emotion over time</div>
      <div class="timeline-bar">{segments}</div>
      <div class="legend">{legend}</div>
      {svg}
    </div>"""
    return html

def build_diarization_html(diarization, timestamps, probs):
    if diarization is None:
        return ""

    duration = timestamps[-1][1]
    preds    = probs.argmax(axis=1)

    # map each diarization segment to dominant emotion
    rows = ""
    speaker_colors = ["#F0A500", "#00B4D8", "#FF6B9D", "#43AA8B", "#9B5DE5"]
    speaker_map    = {}

    segments = list(diarization.itertracks(yield_label=True))
    timeline_segs  = ""

    for turn, _, speaker in segments:
        if speaker not in speaker_map:
            idx = len(speaker_map) % len(speaker_colors)
            speaker_map[speaker] = speaker_colors[idx]

        s, e   = turn.start, turn.end
        col    = speaker_map[speaker]
        left   = s / duration * 100
        width  = max((e - s) / duration * 100, 0.3)
        label  = speaker.replace("SPEAKER_", "S")
        timeline_segs += f'<div class="seg spk-seg" style="left:{left:.2f}%;width:{width:.2f}%;background:{col}" title="{speaker} {s:.1f}s–{e:.1f}s"><span class="spk-label">{label}</span></div>'

        # find dominant emotion during this segment
        chunk_emotions = []
        for i, (ts, te) in enumerate(timestamps):
            mid = (ts + te) / 2
            if ts >= s - 0.5 and te <= e + 0.5:
                chunk_emotions.append(preds[i])
        if chunk_emotions:
            from collections import Counter
            dom = Counter(chunk_emotions).most_common(1)[0][0]
            emo_str = f"{EMOJIS[dom]} {CLASSES[dom]}"
            emo_col = COLORS_HEX[dom]
        else:
            emo_str = "—"
            emo_col = "#888"

        dur_str = f"{e - s:.1f}s"
        rows += f"""
        <tr>
          <td><span class="spk-badge" style="background:{col}">{speaker.replace('SPEAKER_', 'Speaker ')}</span></td>
          <td class="td-time">{s:.1f}s – {e:.1f}s</td>
          <td class="td-dur">{dur_str}</td>
          <td style="color:{emo_col};font-weight:500">{emo_str}</td>
        </tr>"""

    legend = "".join(
        f'<span class="leg-item"><span class="leg-dot" style="background:{col}"></span>{spk.replace("SPEAKER_", "Speaker ")}</span>'
        for spk, col in speaker_map.items()
    )

    html = f"""
    <div class="card">
      <div class="section-title">Speaker diarization</div>
      <div class="timeline-bar" style="margin-bottom:8px">{timeline_segs}</div>
      <div class="legend" style="margin-bottom:16px">{legend}</div>
      <table class="diar-table">
        <thead><tr><th>Speaker</th><th>Segment</th><th>Duration</th><th>Emotion</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>"""
    return html

# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body, .gradio-container { background: #F5F0E8 !important; color: #1a1a1a; font-family: 'Inter', system-ui, sans-serif; }
.main-wrap { max-width: 860px; margin: 0 auto; padding: 32px 20px 60px; }
h1.app-title { font-size: 1.6rem; font-weight: 700; color: #1a1a1a; margin-bottom: 4px; letter-spacing: -0.5px; }
p.app-sub { color: #888; font-size: 0.9rem; margin-bottom: 28px; }

.card { background:#fff; border:1px solid #e8e0d0; border-radius:12px; padding:24px; margin-bottom:20px; }
.section-title { font-size:0.78rem; font-weight:600; text-transform:uppercase; letter-spacing:1px; color:#aaa; margin-bottom:16px; }

/* hero */
.hero-card { display:flex; gap:32px; flex-wrap:wrap; }
.hero-left { min-width:160px; display:flex; flex-direction:column; align-items:center; justify-content:center; }
.hero-emoji { font-size:3rem; margin-bottom:8px; }
.hero-emotion { font-size:1.5rem; font-weight:700; }
.hero-desc { font-size:0.85rem; color:#999; margin-top:2px; }
.hero-conf { font-size:0.9rem; font-weight:600; color:#444; margin-top:10px; }
.hero-meta { font-size:0.78rem; color:#bbb; margin-top:4px; text-align:center; }
.hero-right { flex:1; min-width:260px; }
.bars-title { font-size:0.78rem; font-weight:600; text-transform:uppercase; letter-spacing:1px; color:#aaa; margin-bottom:14px; }
.bar-row { display:flex; align-items:center; gap:10px; margin-bottom:10px; }
.bar-label { width:110px; font-size:0.88rem; color:#444; flex-shrink:0; }
.bar-track { flex:1; height:10px; background:#f0ebe0; border-radius:6px; overflow:hidden; }
.bar-fill { height:100%; border-radius:6px; }
.bar-pct { width:42px; text-align:right; font-size:0.82rem; color:#999; flex-shrink:0; }

/* timeline bar */
.timeline-bar { position:relative; height:32px; background:#f0ebe0; border-radius:8px; overflow:hidden; margin-bottom:12px; }
.seg { position:absolute; top:0; height:100%; transition:opacity 0.2s; }
.seg:hover { opacity:0.8; }

/* legend */
.legend { display:flex; flex-wrap:wrap; gap:12px; margin-bottom:16px; }
.leg-item { display:flex; align-items:center; gap:6px; font-size:0.82rem; color:#666; }
.leg-dot { width:10px; height:10px; border-radius:50%; flex-shrink:0; }

/* diarization */
.spk-seg { display:flex; align-items:center; justify-content:center; overflow:hidden; }
.spk-label { font-size:0.65rem; font-weight:700; color:#fff; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; padding:0 3px; }
.spk-badge { display:inline-block; padding:2px 8px; border-radius:20px; font-size:0.75rem; font-weight:600; color:#fff; }
.diar-table { width:100%; border-collapse:collapse; font-size:0.85rem; }
.diar-table th { text-align:left; color:#bbb; font-weight:600; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.8px; padding:6px 10px 10px; border-bottom:1px solid #f0ebe0; }
.diar-table td { padding:10px 10px; border-bottom:1px solid #f7f3ec; color:#444; }
.td-time { font-variant-numeric: tabular-nums; color:#999; }
.td-dur  { color:#bbb; }
.diar-table tr:last-child td { border-bottom:none; }

/* gradio overrides */
.gradio-container { background:#F5F0E8 !important; }
footer { display:none !important; }
"""

# ── Main function ─────────────────────────────────────────────────────────────
def analyze(audio_path, use_diarization):
    if audio_path is None:
        return "<p style='color:#888;padding:20px'>Upload an audio file to begin.</p>"

    try:
        waveform            = load_audio(audio_path)
        duration            = waveform.shape[0] / SAMPLE_RATE
        chunks, timestamps  = waveform_to_chunks(waveform)
        probs               = predict_chunks(chunks)

        hero_html       = build_hero_html(probs, duration, len(chunks))
        timeline_html   = build_timeline_html(timestamps, probs)
        diar_html       = ""

        # Diarization temporarily disabled in the public demo
        diar_html = """
        <div class="card">
            <div class="section-title">Speaker diarization</div>
            <p style="color:#888;">
                Speaker diarization is currently under development and is not available in this demo.
            </p>
        </div>
        """

        """if use_diarization:
            try:
                pipeline    = get_diarization_pipeline()
                diarization = pipeline(audio_path)
                diar_html   = build_diarization_html(diarization, timestamps, probs)
            except Exception as de:
                diar_html = f"<div class='card' style='color:#E05C5C'>Diarization error: {str(de)}</div>"""

        full_html = f"""
        <div class='main-wrap'>
          {hero_html}
          {timeline_html}
          {diar_html}
        </div>"""
        return full_html

    except Exception as e:
        return f"<div style='color:#E05C5C;padding:20px'>Error: {str(e)}</div>"

# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(css=CSS, title="Speech Emotion Recognition") as demo:
    gr.HTML("""
    <div class='main-wrap'>
      <h1 class='app-title'>🎙️ Speech Emotion Recognition</h1>
      <p class='app-sub'>Upload audio to classify emotions across time. Trained on IEMOCAP · EfficientNet-B0 · Speaker-independent splits.</p>
    </div>
    """)

    with gr.Row(elem_classes="main-wrap"):
        with gr.Column():
            audio_input = gr.Audio(
                type="filepath",
                label="Upload audio (wav / mp3 / m4a / flac / ogg)",
            )
            diar_toggle = gr.Checkbox(
                label="Enable speaker diarization (slower — adds ~30–60s on CPU)",
                value=False,
            )
            submit_btn = gr.Button("Analyse", variant="primary")

    output_html = gr.HTML()

    submit_btn.click(
        fn=analyze,
        inputs=[audio_input, diar_toggle],
        outputs=output_html,
    )

if __name__ == "__main__":
    demo.launch()
