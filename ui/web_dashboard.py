import os
import json

class WebDashboardGenerator:
    """Generates a comprehensive, modern HTML5 forensic audio steganography report."""

    @staticmethod
    def generate(audio_container, all_snapshots, anomalies, lsb_results, carved_results, morse_result, pam_result, top_highlights, output_dir, dtmf_result=None):
        report_path = os.path.join(output_dir, "index.html")
        json_report_path = os.path.join(output_dir, "forensic_report.json")

        # Collect all decoded secrets
        all_secrets = []
        for s in anomalies:
            for dc in s.get("decoded_candidates", []):
                all_secrets.append({
                    "time": s["time_str"],
                    "source": f"Spectrogram ({s['band_name']})",
                    "type": dc["type"],
                    "content": dc["decoded"],
                    "is_flag": dc.get("is_flag", False)
                })

        for l in lsb_results:
            for dc in l.get("decoded_messages", []):
                all_secrets.append({
                    "time": "Full Track",
                    "source": f"LSB Bit {l['bit_index']}",
                    "type": dc["type"],
                    "content": dc["decoded"],
                    "is_flag": dc.get("is_flag", False)
                })

        for c in carved_results:
            for dc in c.get("decoded_messages", []):
                all_secrets.append({
                    "time": "Slack / EOF",
                    "source": f"Carved {c['type']}",
                    "type": dc["type"],
                    "content": dc["decoded"],
                    "is_flag": dc.get("is_flag", False)
                })

        if morse_result:
            all_secrets.append({
                "time": "Audio Signal",
                "source": "Tone Envelope",
                "type": morse_result["type"],
                "content": morse_result["decoded_text"],
                "is_flag": True
            })

        if pam_result:
            for dc in pam_result.get("messages", []):
                all_secrets.append({
                    "time": "Signal Samples",
                    "source": "Discrete Amplitude (PAM)",
                    "type": dc["type"],
                    "content": dc["decoded"],
                    "is_flag": True
                })

        if dtmf_result:
            for dc in dtmf_result.get("messages", []):
                all_secrets.append({
                    "time": "Audio Signal",
                    "source": "DTMF Keypad Tone",
                    "type": dc["type"],
                    "content": dc["decoded"],
                    "is_flag": dc.get("is_flag", True)
                })
            if not dtmf_result.get("messages") and dtmf_result.get("sequence"):
                all_secrets.append({
                    "time": "Audio Signal",
                    "source": "DTMF Keypad Tone",
                    "type": "Sequence",
                    "content": dtmf_result["sequence"],
                    "is_flag": False
                })

        # Save structured JSON report
        full_json_data = {
            "target": {
                "filename": audio_container.filename,
                "filesize": audio_container.filesize,
                "duration": audio_container.duration,
                "sample_rate": audio_container.sample_rate,
                "hashes": audio_container.hashes
            },
            "secrets_found": all_secrets,
            "top_highlights": [
                {
                    "rank": i + 1,
                    "clarity_score": h["clarity_score"],
                    "time_str": h["time_str"],
                    "channel": h["channel"],
                    "band_name": h["band_name"],
                    "heatmap_file": h["heatmap_file"],
                    "banner_file": h.get("banner_file"),
                    "word_width": h.get("word_width", 0)
                } for i, h in enumerate(top_highlights)
            ],
            "lsb_analysis": lsb_results,
            "carved_files": carved_results,
            "snapshot_count": len(all_snapshots),
            "anomaly_count": len(anomalies)
        }
        with open(json_report_path, "w", encoding="utf-8") as jf:
            json.dump(full_json_data, jf, indent=2)

        # Build HTML
        has_anomalies = len(anomalies) > 0 or len(all_secrets) > 0
        has_bombs = any(c.get("security_alert") for c in carved_results)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stego Hunter Report - {audio_container.filename}</title>
    <style>
        :root {{
            --bg-main: #0b0f19;
            --card-bg: #131b2e;
            --border-color: #23314d;
            --accent-cyan: #00d2ff;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --accent-amber: #f59e0b;
            --accent-purple: #8b5cf6;
            --text-main: #cbd5e1;
            --text-white: #f8fafc;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: var(--bg-main);
            color: var(--text-main);
            padding: 24px;
            line-height: 1.5;
        }}
        .header {{
            background: linear-gradient(135deg, #131b2e 0%, #1a2540 100%);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.4);
        }}
        .header-title-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
        }}
        h1 {{
            color: var(--text-white);
            font-size: 24px;
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 13px;
            font-weight: 600;
        }}
        .badge-alert {{ background: rgba(239, 68, 68, 0.2); color: #fca5a5; border: 1px solid var(--accent-red); }}
        .badge-safe {{ background: rgba(16, 185, 129, 0.2); color: #6ee7b7; border: 1px solid var(--accent-green); }}
        .badge-warning {{ background: rgba(245, 158, 11, 0.2); color: #fcd34d; border: 1px solid var(--accent-amber); }}
        .badge-info {{ background: rgba(0, 210, 255, 0.15); color: #7dd3fc; border: 1px solid var(--accent-cyan); }}
        .badge-clarity {{ background: linear-gradient(135deg, #059669 0%, #10b981 100%); color: #fff; border: 1px solid #34d399; font-weight: bold; }}

        .meta-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-top: 18px;
            padding-top: 18px;
            border-top: 1px solid var(--border-color);
            font-size: 13px;
        }}
        .meta-item strong {{ color: var(--text-white); display: block; margin-bottom: 4px; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }}

        .section-title {{
            color: var(--text-white);
            font-size: 18px;
            margin: 28px 0 12px 0;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .section-desc {{
            color: #94a3b8;
            font-size: 13px;
            margin-bottom: 16px;
        }}

        /* Decoded Secrets Box */
        .secrets-container {{
            display: grid;
            gap: 12px;
            margin-bottom: 24px;
        }}
        .secret-card {{
            background: #1e1b2e;
            border: 2px solid #8b5cf6;
            border-radius: 8px;
            padding: 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
            box-shadow: 0 0 20px rgba(139, 92, 246, 0.25);
        }}
        .secret-content {{
            font-family: monospace;
            font-size: 16px;
            color: #a7f3d0;
            word-break: break-all;
        }}
        .btn-copy {{
            background: #4c1d95;
            color: #ede9fe;
            border: 1px solid #7c3aed;
            padding: 6px 14px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.2s;
        }}
        .btn-copy:hover {{ background: #6d28d9; }}

        /* VIP Highlights Section */
        .vip-container {{
            display: flex;
            flex-direction: column;
            gap: 20px;
            margin-bottom: 30px;
        }}
        .vip-card {{
            background: #101726;
            border: 2px solid #10b981;
            border-radius: 12px;
            padding: 18px;
            box-shadow: 0 0 24px rgba(16, 185, 129, 0.2);
            transition: transform 0.2s;
        }}
        .vip-card:hover {{ transform: translateY(-2px); }}
        .vip-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 14px;
            padding-bottom: 10px;
            border-bottom: 1px solid #23314d;
            font-size: 15px;
            font-weight: 600;
            color: var(--text-white);
        }}

        /* Snapshot Trio */
        .img-trio {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 12px;
            background: #000;
            padding: 10px;
            border-radius: 8px;
        }}
        .img-trio figure {{ text-align: center; }}
        .img-trio img {{
            width: 100%;
            height: auto;
            border-radius: 4px;
            border: 1px solid #222;
            cursor: zoom-in;
            transition: border-color 0.2s;
        }}
        .img-trio img:hover {{ border-color: var(--accent-cyan); }}
        .img-trio figcaption {{
            font-size: 11px;
            color: #94a3b8;
            margin-top: 6px;
        }}

        /* Archive Details Box */
        details.archive-box {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            margin-top: 30px;
            padding: 12px;
        }}
        details.archive-box summary {{
            cursor: pointer;
            font-weight: 600;
            color: var(--accent-cyan);
            padding: 8px 12px;
            user-select: none;
            outline: none;
        }}
        .controls {{
            display: flex;
            gap: 10px;
            margin: 16px 0 20px 0;
            flex-wrap: wrap;
        }}
        .filter-btn {{
            background: #1e293b;
            border: 1px solid var(--border-color);
            color: var(--text-main);
            padding: 6px 14px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.2s;
        }}
        .filter-btn.active, .filter-btn:hover {{
            background: #334155;
            border-color: var(--accent-cyan);
            color: var(--text-white);
        }}
        .snapshot-grid {{
            display: flex;
            flex-direction: column;
            gap: 16px;
            margin-top: 16px;
        }}
        .snapshot-card {{
            background: #0d1322;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 14px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="header-title-row">
            <h1>
                <span>🛡️ Forensic Audio Steganography Scanner</span>
                {'<span class="badge badge-alert">⚠️ Secrets / Anomalies Detected</span>' if has_anomalies else '<span class="badge badge-safe">✓ Clean Spectrum</span>'}
                {'<span class="badge badge-warning">🛡️ Zip Bomb Defended</span>' if has_bombs else ''}
            </h1>
        </div>
        <div class="meta-grid">
            <div class="meta-item"><strong>Target File</strong>{audio_container.filename}</div>
            <div class="meta-item"><strong>Duration & Rate</strong>{audio_container.duration:.2f}s @ {audio_container.sample_rate} Hz</div>
            <div class="meta-item"><strong>File Size</strong>{audio_container.filesize / 1024:.1f} KB ({audio_container.filesize:,} bytes)</div>
            <div class="meta-item"><strong>MD5 Hash</strong><code>{audio_container.hashes['md5']}</code></div>
            <div class="meta-item"><strong>SHA-256</strong><code>{audio_container.hashes['sha256'][:28]}...</code></div>
        </div>
    </div>
"""

        # Add Decoded Secrets Showcase if any
        if all_secrets:
            html += """
    <h2 class="section-title">🔑 Decoded Messages & Discovered Secrets</h2>
    <div class="secrets-container">
"""
            for sec in all_secrets:
                html += f"""
        <div class="secret-card">
            <div>
                <span class="badge badge-info">{sec['time']} &bull; {sec['source']}</span>
                <span class="badge badge-safe">{sec['type']}</span>
                <div class="secret-content" style="margin-top: 8px;">{sec['content']}</div>
            </div>
            <button class="btn-copy" onclick="navigator.clipboard.writeText('{sec['content']}'); alert('Copied to clipboard!')">Copy</button>
        </div>
"""
            html += """
    </div>
"""

        # Security Defenses Box
        for c in carved_results:
            if c.get("security_alert"):
                html += f"""
    <div style="background: rgba(239,68,68,0.1); border: 1px solid var(--accent-red); border-radius: 8px; padding: 16px; margin-bottom: 24px; color: #fca5a5;">
        <h3>🛡️ Decompression Shield Triggered</h3>
        <p><strong>Source:</strong> {c['label']}</p>
        <p><strong>Alert:</strong> {c['security_alert']}</p>
        <p><em>Extraction was safely halted before memory or disk exhaustion occurred.</em></p>
    </div>
"""

        # 🌟 TOP HIGHLIGHTS (VIP Best Shots)
        if top_highlights:
            html += f"""
    <h2 class="section-title">🌟 En Belirgin Şüpheli Görüntüler (Top Priority Highlights)</h2>
    <p class="section-desc">
        Tüm kesitler taranarak taban çizgisi uyumu (baseline alignment), harf boyutu tutarlılığı ve kontrast metriklerine göre 
        <strong>yazının en belirgin olduğu ilk {len(top_highlights)} kesit</strong> öncelikli olarak seçilmiştir.
    </p>

    <div class="vip-container">
"""
            for rank, h in enumerate(top_highlights, 1):
                html += f"""
        <div class="vip-card">
            <div class="vip-header">
                <span>🏆 #{rank} En Belirgin Aday &bull; ⏱️ {h['time_str']} &bull; {h['channel']} &bull; {h['band_name']} ({h['f_min']:,} - {h['f_max']:,} Hz, FFT {h['n_fft']})</span>
                <span class="badge badge-clarity">🔥 %{h['clarity_score']:.1f} Netlik &bull; 📏 {h.get('word_width', 0)}px Kelime Bandı</span>
            </div>
            {f"""
            <div style="margin-bottom: 14px; text-align: center; background: #000; padding: 12px; border-radius: 8px; border: 1px solid #10b981;">
                <div style="font-size: 13px; font-weight: 700; color: #10b981; margin-bottom: 8px; text-transform: uppercase;">
                    🔍 OTOMATİK BÜYÜTEÇ KESİTİ (AUTO-CROP BANNER — GÖZLE OKUMA PANKARTI)
                </div>
                <img src="{h['banner_file']}" alt="Zoomed Banner Crop" style="max-width: 100%; height: auto; border-radius: 4px; box-shadow: 0 0 16px rgba(16,185,129,0.3); cursor: zoom-in;" onclick="window.open(this.src)">
            </div>
""" if h.get('banner_file') else ""}
            <div class="img-trio">
                <figure>
                    <img src="{h['heatmap_file']}" alt="Heatmap" onclick="window.open(this.src)">
                    <figcaption>Heatmap (Inferno + Tespit Kutuları)</figcaption>
                </figure>
                <figure>
                    <img src="{h['inverted_file']}" alt="Inverted" onclick="window.open(this.src)">
                    <figcaption>Ters Gri Ton (Yüksek Kontrast)</figcaption>
                </figure>
                <figure>
                    <img src="{h['thresh_file']}" alt="Otsu" onclick="window.open(this.src)">
                    <figcaption>Otsu Binarized (Gürültüsüz Maske)</figcaption>
                </figure>
            </div>
        </div>
"""
            html += """
    </div>
"""

        # Collapsible Archive for remaining snapshots
        html += f"""
    <details class="archive-box">
        <summary>📁 Tüm Diğer Kesitler (Arşiv - {len(all_snapshots)} Adet Görseli İncele)</summary>
        
        <div class="controls">
            <button class="filter-btn active" onclick="filterGallery('all')">Tüm Kesitler ({len(all_snapshots)})</button>
            <button class="filter-btn" onclick="filterGallery('anomaly')">Yalnızca Anomaliler ({len(anomalies)})</button>
        </div>

        <div class="snapshot-grid">
"""
        sorted_snapshots = sorted(all_snapshots, key=lambda x: (not x["anomaly_detected"], x["time_start"]))
        for s in sorted_snapshots:
            is_anom = s["anomaly_detected"]
            data_type = "anomaly" if is_anom else "normal"
            badge = f'<span class="badge badge-alert">🚨 {s["num_clusters"]} Kutu</span>' if is_anom else '<span class="badge badge-safe">Clean</span>'

            html += f"""
            <div class="snapshot-card" data-type="{data_type}">
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 12px; color: #94a3b8;">
                    <span>⏱️ {s['time_str']} &bull; {s['channel']} &bull; {s['band_name']} &bull; FFT {s['n_fft']}</span>
                    <div>{badge} <span style="margin-left: 8px; color: #64748b;">Skor: %{s['clarity_score']:.1f}</span></div>
                </div>
                <div class="img-trio">
                    <figure><img src="{s['heatmap_file']}" alt="Heatmap" onclick="window.open(this.src)"></figure>
                    <figure><img src="{s['inverted_file']}" alt="Inverted" onclick="window.open(this.src)"></figure>
                    <figure><img src="{s['thresh_file']}" alt="Otsu" onclick="window.open(this.src)"></figure>
                </div>
            </div>
"""

        html += """
        </div>
    </details>

    <script>
        function filterGallery(type) {
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');
            
            document.querySelectorAll('.snapshot-card').forEach(card => {
                if (type === 'all') {
                    card.style.display = 'block';
                } else if (type === 'anomaly') {
                    card.style.display = card.getAttribute('data-type') === 'anomaly' ? 'block' : 'none';
                }
            });
        }
    </script>
</body>
</html>
"""
        with open(report_path, "w", encoding="utf-8") as rf:
            rf.write(html)

        return report_path
