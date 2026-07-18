"""
HTML preview gallery generator.
Produces a single self-contained index.html that opens in any browser,
including from a phone on the local network.

Displays:
  - 8 standalone clips
  - 3 story series × 3 episodes each
  - Full JSON report
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def generate_gallery(
    output_dir: Path,
    clips: List[Path],
    report: dict,
    video_title: str = "",
) -> Path:
    """
    Generate an index.html gallery page showing all clips with metadata.
    """
    # Build clip cards data
    clip_cards = []
    for i, clip_path in enumerate(clips):
        seg_info = {}
        for seg in report.get("candidates", []):
            seg_start = seg.get("start", 0)
            seg_end = seg.get("end", 0)
            if _segment_matches_clip(seg, clip_path.name, i):
                seg_info = seg
                break

        clip_cards.append({
            "index": i + 1,
            "filename": clip_path.name,
            "path": f"clips/{clip_path.name}",
            "type": seg_info.get("type", "standalone"),
            "series_index": seg_info.get("series_index", 0),
            "episode_index": seg_info.get("episode_index", 0),
            "score": seg_info.get("score", 0),
            "hook_line": seg_info.get("hook_line", ""),
            "title": seg_info.get("title", ""),
            "reasoning": seg_info.get("reasoning", ""),
            "suggested_title": seg_info.get("suggested_title", ""),
            "hashtags": seg_info.get("hashtags", []),
            "start": seg_info.get("start", 0),
            "end": seg_info.get("end", 0),
            "content_type": seg_info.get("content_type", ""),
            "retention": seg_info.get("retention", {}),
            "sub_scores": seg_info.get("sub_scores", {}),
        })

    cards_json = json.dumps(clip_cards, ensure_ascii=False)
    story_series_json = json.dumps(report.get("story_series", []), ensure_ascii=False)
    report_json = json.dumps(report, ensure_ascii=False)
    title = video_title or "AI Clipper Results"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} — AI Clipper</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; background:#0f0f0f; color:#e0e0e0; padding:20px; }}
h1 {{ text-align:center; margin:20px 0 10px; font-size:1.8em; color:#fff; }}
h2 {{ color:#fff; margin:30px 0 16px; font-size:1.4em; }}
h3 {{ color:#ccc; margin:16px 0 8px; font-size:1.1em; }}
.subtitle {{ text-align:center; color:#888; margin-bottom:30px; font-size:0.95em; }}
.section {{ max-width:1400px; margin:0 auto 40px; }}
.clip-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); gap:24px; }}
.card {{ background:#1a1a1a; border-radius:16px; overflow:hidden; border:1px solid #2a2a2a; transition:transform .15s,border-color .15s; }}
.card:hover {{ transform:translateY(-2px); border-color:#444; }}
.card video {{ width:100%; display:block; aspect-ratio:9/16; object-fit:contain; background:#000; }}
.card-body {{ padding:16px; }}
.type-badge {{ display:inline-block; background:#333; color:#aaa; font-weight:600; padding:3px 10px; border-radius:12px; font-size:0.75em; margin-bottom:8px; text-transform:uppercase; }}
.type-badge.series {{ background:#4a2c5a; color:#e0b0ff; }}
.score-badge {{ display:inline-block; background:linear-gradient(135deg,#FFD700,#FF8C00); color:#000; font-weight:700; padding:4px 12px; border-radius:20px; font-size:0.9em; margin-bottom:10px; }}
.title-line {{ font-size:1.05em; font-weight:600; color:#fff; margin-bottom:6px; line-height:1.3; }}
.hook-line {{ font-size:1.0em; font-weight:500; color:#ddd; margin-bottom:6px; line-height:1.3; }}
.reasoning {{ font-size:0.85em; color:#999; line-height:1.4; margin-bottom:10px; }}
.hashtags {{ display:flex; flex-wrap:wrap; gap:6px; margin-top:8px; }}
.hashtags span {{ background:#222; color:#4fc3f7; padding:2px 8px; border-radius:10px; font-size:0.75em; }}
.sub-scores {{ margin-top:10px; }}
.sub-row {{ display:flex; justify-content:space-between; align-items:center; margin:3px 0; font-size:0.78em; color:#888; }}
.sub-bar-bg {{ flex:1; height:4px; background:#333; border-radius:2px; margin:0 8px; }}
.sub-bar-fill {{ height:100%; border-radius:2px; }}
.retention-row {{ display:flex; gap:12px; margin-top:8px; font-size:0.78em; color:#aaa; flex-wrap:wrap; }}
.retention-row span {{ background:#1f1f1f; padding:3px 8px; border-radius:6px; }}
.series-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:16px; }}
.series-card {{ background:#1a1a1a; border:1px solid #2a2a2a; border-radius:16px; padding:16px; }}
.series-header {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; }}
.series-score {{ background:linear-gradient(135deg,#667eea,#764ba2); color:#fff; font-weight:700; padding:4px 12px; border-radius:20px; font-size:0.85em; }}
.episode-list {{ display:flex; flex-direction:column; gap:8px; }}
.episode-item {{ background:#252525; border-radius:10px; padding:10px 12px; display:flex; justify-content:space-between; align-items:flex-start; }}
.episode-info {{ flex:1; }}
.episode-title {{ color:#fff; font-weight:500; font-size:0.9em; margin-bottom:2px; }}
.episode-meta {{ color:#777; font-size:0.75em; }}
.episode-teaser {{ color:#999; font-size:0.8em; margin-top:4px; font-style:italic; }}
.report-section {{ max-width:1400px; margin:40px auto 0; }}
.report-section h2 {{ color:#fff; margin-bottom:12px; }}
.report-section pre {{ background:#1a1a1a; border:1px solid #2a2a2a; border-radius:12px; padding:20px; overflow-x:auto; font-size:0.8em; color:#aaa; max-height:500px; overflow-y:auto; }}
.download-btn {{ display:inline-block; background:#333; color:#fff; padding:6px 14px; border-radius:8px; text-decoration:none; font-size:0.8em; margin-top:10px; transition:background .15s; }}
.download-btn:hover {{ background:#555; }}
.footer {{ text-align:center; margin-top:40px; color:#555; font-size:0.75em; }}
</style>
</head>
<body>

<h1>✂️ {title}</h1>
<p class="subtitle">{len(clip_cards)} clip{'' if len(clip_cards)==1 else 's'} extracted • scored by AI • ready to post</p>

<div class="section">
  <h2>🎬 Standalone Clips</h2>
  <div class="clip-grid" id="standaloneGrid"></div>
</div>

<div class="section">
  <h2>📖 Story Series</h2>
  <div class="series-grid" id="seriesGrid"></div>
</div>

<div class="report-section">
  <h2>📊 Full Report</h2>
  <pre id="reportJson"></pre>
</div>

<p class="footer">Generated by AI Clipper • Review before posting • Local files only</p>

<script>
const clips = {cards_json};
const storySeries = {story_series_json};

function scoreColor(s) {{
  if (s >= 80) return '#4caf50';
  if (s >= 60) return '#FFD700';
  if (s >= 40) return '#ff9800';
  return '#f44336';
}}

function subScoreColor(s) {{
  if (s >= 70) return '#4caf50';
  if (s >= 50) return '#FFD700';
  if (s >= 30) return '#ff9800';
  return '#f44336';
}}

function escapeHtml(s) {{ return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }}
function formatTime(s) {{ const m=Math.floor(s/60); const sec=Math.floor(s%60); return m+':'+String(sec).padStart(2,'0'); }}

function renderCard(c, container) {{
  const card = document.createElement('div');
  card.className = 'card';

  const typeLabel = c.type === 'story_series' ? 'Series ' + c.series_index + ' · Ep ' + c.episode_index : 'Standalone';
  const typeClass = c.type === 'story_series' ? 'type-badge series' : 'type-badge';

  let hashtagHtml = '';
  if (c.hashtags && c.hashtags.length) {{
    hashtagHtml = '<div class="hashtags">' +
      c.hashtags.map(t => '<span>#' + t + '</span>').join('') +
      '</div>';
  }}

  let subScoresHtml = '';
  const subs = c.sub_scores || {{}};
  const subLabels = [
    ['Retention', 'viewer_retention'], ['Hook', 'hook_strength'],
    ['Story', 'story_quality'], ['Emotion', 'emotional_intensity'],
    ['Education', 'educational_value'], ['Visual', 'visual_quality'],
    ['Audio', 'audio_quality'], ['Virality', 'virality']
  ];
  subLabels.forEach(([label, key]) => {{
    const v = subs[key] || 0;
    const pct = Math.min(100, Math.max(0, v));
    subScoresHtml +=
      '<div class="sub-row"><span>' + label + '</span>' +
      '<div class="sub-bar-bg"><div class="sub-bar-fill" style="width:' + pct + '%;background:' + subScoreColor(v) + '"></div></div>' +
      '<span>' + Math.round(v) + '</span></div>';
  }});

  let retentionHtml = '';
  if (c.retention) {{
    const r = c.retention;
    retentionHtml = '<div class="retention-row">' +
      (r.completion_rate ? '<span>▶ ' + Math.round(r.completion_rate*100) + '% completion</span>' : '') +
      (r.share_likelihood ? '<span>↗ ' + Math.round(r.share_likelihood*100) + '% share</span>' : '') +
      (r.comment_likelihood ? '<span>💬 ' + Math.round(r.comment_likelihood*100) + '% comment</span>' : '') +
      (r.engagement_rate ? '<span>⚡ ' + Math.round(r.engagement_rate*100) + '% engagement</span>' : '') +
      '</div>';
  }}

  const displayTitle = c.title || c.suggested_title || c.hook_line || '';

  card.innerHTML =
    '<video controls preload="metadata" playsinline>' +
    '<source src="' + c.path + '" type="video/mp4">' +
    '</video>' +
    '<div class="card-body">' +
    '<span class="' + typeClass + '">' + typeLabel + '</span>' +
    '<div class="score-badge">⭐ ' + Math.round(c.score) + '/100</div>' +
    (displayTitle ? '<div class="title-line">' + escapeHtml(displayTitle) + '</div>' : '') +
    (c.hook_line && !displayTitle ? '<div class="hook-line">' + escapeHtml(c.hook_line) + '</div>' : '') +
    (c.reasoning ? '<div class="reasoning">' + escapeHtml(c.reasoning) + '</div>' : '') +
    '<div class="sub-scores">' + subScoresHtml + '</div>' +
    retentionHtml +
    hashtagHtml +
    '<a class="download-btn" href="' + c.path + '" download>⬇ Download</a>' +
    '<span style="color:#666;font-size:0.75em;margin-left:8px;">' +
    formatTime(c.start) + ' – ' + formatTime(c.end) + '</span>' +
    '</div>';

  container.appendChild(card);
}}

// Render standalone clips
const standaloneGrid = document.getElementById('standaloneGrid');
const standaloneClips = clips.filter(c => c.type !== 'story_series');
if (standaloneClips.length === 0) {{
  standaloneGrid.innerHTML = '<p style="color:#666;">No standalone clips produced.</p>';
}} else {{
  standaloneClips.forEach(c => renderCard(c, standaloneGrid));
}}

// Render story series
const seriesGrid = document.getElementById('seriesGrid');
if (storySeries.length === 0) {{
  seriesGrid.innerHTML = '<p style="color:#666;">No story series produced.</p>';
}} else {{
  storySeries.forEach((series, idx) => {{
    const seriesCard = document.createElement('div');
    seriesCard.className = 'series-card';

    let episodesHtml = '';
    ['episode_1', 'episode_2', 'episode_3'].forEach((key, epIdx) => {{
      const ep = series[key];
      const clip = clips.find(c => c.type === 'story_series' && c.series_index === (idx+1) && c.episode_index === (epIdx+1));
      const teaser = epIdx < 2 ? ep.cliffhanger : ep.ending;
      episodesHtml +=
        '<div class="episode-item">' +
        '<div class="episode-info">' +
        '<div class="episode-title">' + escapeHtml(ep.title) + '</div>' +
        '<div class="episode-meta">' + formatTime(ep.start) + ' – ' + formatTime(ep.end) + ' · ' + formatTime(ep.duration || (ep.end-ep.start)) + '</div>' +
        (teaser ? '<div class="episode-teaser">"' + escapeHtml(teaser) + '"</div>' : '') +
        '</div>' +
        (clip ? '<a class="download-btn" href="' + clip.path + '" download>⬇</a>' : '') +
        '</div>';
    }});

    seriesCard.innerHTML =
      '<div class="series-header">' +
      '<h3>' + escapeHtml(series.title) + '</h3>' +
      '<span class="series-score">' + Math.round(series.overall_story_score) + '/100</span>' +
      '</div>' +
      '<div class="episode-list">' + episodesHtml + '</div>';

    seriesGrid.appendChild(seriesCard);
  }});
}}

document.getElementById('reportJson').textContent = JSON.stringify({report_json}, null, 2);
</script>
</body>
</html>"""

    gallery_path = output_dir / "index.html"
    gallery_path.write_text(html, encoding="utf-8")
    logger.info(f"Gallery written: {gallery_path} ({len(html):,} bytes)")
    return gallery_path


def _segment_matches_clip(seg: dict, clip_filename: str, clip_index: int) -> bool:
    """Heuristic: does this report segment correspond to this clip file?"""
    seg_idx = seg.get("_clip_index", -1)
    if seg_idx == clip_index:
        return True

    seg_start = seg.get("start", -1)
    if seg_start >= 0:
        start_str = f"{seg_start:.1f}"
        if start_str in clip_filename:
            return True

    return False
