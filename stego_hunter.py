"""
=============================================================================
 AudioStegoHunter & Forensics Suite
=============================================================================
Advanced Audio Steganalysis, Continuous Sliding-Window Radar,
Multimodal Decoders (Base64/Hex/Morse/ROT), LSB Analysis,
and Decompression Bomb Defenses.
=============================================================================
"""

import os
import sys
import argparse
import webbrowser

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from core.audio_loader import AudioContainer, AudioLoadError
from core.safety import SafeArchiveInspector, SecurityAnomaly
from detectors.file_carver import FileCarver
from detectors.lsb_analyzer import LSBAnalyzer
from detectors.signal_decoders import SignalDecoders
from detectors.sliding_spectrogram import SlidingSpectrogramScanner
from ui.console_hud import ConsoleHUD, console
from ui.web_dashboard import WebDashboardGenerator
from demo_generator import generate_multi_stego_demo

class StegoHunterEngine:
    """Orchestrates forensic audio steganography analysis."""
    def __init__(self, audio_path, output_dir="stego_output", window_sec=1.5, overlap=0.5):
        self.audio_path = os.path.abspath(audio_path)
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.window_sec = window_sec
        self.overlap = overlap

    def run(self):
        ConsoleHUD.render_banner()

        # 1. Load Audio and Fingerprint
        try:
            container = AudioContainer(self.audio_path)
        except AudioLoadError as e:
            console.print(f"[bold red]Error loading audio:[/bold red] {e}")
            sys.exit(1)

        ConsoleHUD.render_file_meta(container)

        # 2. File Carving & Slack Space Inspection (with Zip Bomb Defense)
        console.print("\n[bold cyan][*] Stage 1:[/bold cyan] Scanning Slack Space & File Carving (with Zip Bomb Protection)...")
        carver = FileCarver(self.audio_path, appended_payload=container.appended_payload, output_dir=self.output_dir)
        carved_results = carver.scan_and_carve()
        if carved_results:
            console.print(f"    [!] Carved [yellow]{len(carved_results)}[/yellow] embedded payload(s)/chunk(s).")
            for c in carved_results:
                if c.get("security_alert"):
                    console.print(f"    [bold red]⚠️ DEFENSE ALERT:[/bold red] {c['security_alert']}")
                else:
                    console.print(f"    [green]✓[/green] Carved {c['type']} ({c['size_bytes']:,} bytes) - Status: {c['safety_status']}")
        else:
            console.print("    [.] No trailing slack payload found.")

        # 3. LSB Bitplane & Entropy Analysis
        console.print("\n[bold cyan][*] Stage 2:[/bold cyan] Analyzing LSB (Least Significant Bits) & Shannon Entropy...")
        lsb_analyzer = LSBAnalyzer(self.audio_path, output_dir=self.output_dir)
        lsb_results = lsb_analyzer.analyze()
        for l in lsb_results:
            ent_status = "[red]SUSPICIOUS / HIGH RANDOMNESS[/red]" if l["is_suspicious_entropy"] else "[green]Normal[/green]"
            console.print(f"    - LSB Bit {l['bit_index']}: Entropy = [yellow]{l['entropy']}[/yellow] / 8.00 ({ent_status})")
            if l.get("has_flag"):
                console.print(f"      [bold red]🎯 Flag signature detected in LSB bitstream![/bold red]")

        # 4. Audio Signal Decoders (Morse / DTMF / PAM)
        console.print("\n[bold cyan][*] Stage 3:[/bold cyan] Testing Audio Signal Envelopes & Modulation (Morse, DTMF, PAM)...")
        primary_signal = next(iter(container.channels.values()))
        morse_res = SignalDecoders.detect_morse_tones(primary_signal, container.sample_rate)
        if morse_res:
            console.print(f"    [bold green]✓ Morse Code Tone Detected:[/bold green] [bold white]{morse_res['decoded_text']}[/bold white] (Raw: {morse_res['raw_morse']})")
        else:
            console.print("    [.] No prominent Morse tone envelope detected.")

        dtmf_res = SignalDecoders.detect_dtmf(primary_signal, container.sample_rate)
        if dtmf_res:
            console.print(f"    [bold green]✓ DTMF Keypad Tones Detected:[/bold green] Sequence = {dtmf_res['sequence']}")
            for msg in dtmf_res.get("messages", []):
                console.print(f"      [bold red]🎯 Flag Decoded from DTMF:[/bold red] {msg['decoded']}")

        pam_res = SignalDecoders.detect_pam_discrete_levels(primary_signal)
        if pam_res:
            console.print(f"    [bold green]✓ Discrete Amplitude Modulation (PAM) Detected:[/bold green] {pam_res['type']}")
            for msg in pam_res.get("messages", []):
                console.print(f"      [bold red]🎯 Flag Detected in Amplitude Stream:[/bold red] {msg['decoded']}")

        # 5. Continuous Sliding-Window Spectrogram Radar (Frequent Snapshots & OCR)
        console.print(f"\n[bold cyan][*] Stage 4:[/bold cyan] Continuous Sliding-Window Radar (Window: {self.window_sec}s, Overlap: {int(self.overlap * 100)}%)...")
        scanner = SlidingSpectrogramScanner(
            container.channels,
            container.sample_rate,
            container.duration,
            output_dir=self.output_dir,
            window_sec=self.window_sec,
            overlap=self.overlap
        )
        all_snapshots, anomalies, top_highlights = scanner.scan()
        console.print(f"    [✓] Processed [cyan]{len(all_snapshots)}[/cyan] micro-snapshots across all bands. Detected [bold red]{len(anomalies)}[/bold red] anomalous region(s).")

        # 6. Terminal Summary & Top Priority Highlights Table
        ConsoleHUD.render_anomalies_summary(len(all_snapshots), anomalies, lsb_results, carved_results, morse_res, pam_res, dtmf_res)
        ConsoleHUD.render_top_highlights_table(top_highlights)
        ConsoleHUD.render_ascii_top_candidate(top_highlights, anomalies)

        # 7. Generate Interactive HTML5 & JSON Reports
        report_path = WebDashboardGenerator.generate(
            container,
            all_snapshots,
            anomalies,
            lsb_results,
            carved_results,
            morse_res,
            pam_res,
            top_highlights,
            self.output_dir,
            dtmf_result=dtmf_res
        )
        console.print(f"\n[bold green][+] Forensic Report Ready:[/bold green] [underline]file:///{report_path.replace(os.sep, '/')}[/underline]\n")
        return report_path

def main():
    parser = argparse.ArgumentParser(description="Audio Stego Hunter & Forensics Suite")
    parser.add_argument("audio", nargs="?", default="sample_challenge.wav", help="Path to WAV audio file")
    parser.add_argument("--demo", action="store_true", help="Generate a synthetic multi-vector challenge and analyze it")
    parser.add_argument("--flag", default="FLAG{4UD10_5T3G0_15_FUN}", help="Custom flag string for --demo")
    parser.add_argument("--out", default="stego_output", help="Output directory for reports and snapshot images")
    parser.add_argument("--open", action="store_true", help="Automatically open HTML report in web browser")
    parser.add_argument("--window", type=float, default=3.5, help="Sliding window size in seconds (default: 3.5)")
    parser.add_argument("--overlap", type=float, default=0.5, help="Window overlap fraction (default: 0.5)")
    args = parser.parse_args()

    target_audio = args.audio
    if args.demo:
        os.makedirs("samples", exist_ok=True)
        target_audio = generate_multi_stego_demo(filename=os.path.join("samples", "sample_challenge.wav"), message=args.flag)
    elif not os.path.exists(target_audio):
        candidate_sample = os.path.join("samples", target_audio)
        if os.path.exists(candidate_sample):
            target_audio = candidate_sample

    if not os.path.exists(target_audio):
        console.print(f"[bold red]Error:[/bold red] Audio file '{args.audio}' not found.")
        console.print("[dim yellow]Tip: Run with --demo to create and analyze a synthetic challenge![/dim yellow]")
        sys.exit(1)

    hunter = StegoHunterEngine(target_audio, output_dir=args.out, window_sec=args.window, overlap=args.overlap)
    report_file = hunter.run()

    if args.open:
        webbrowser.open(f"file://{os.path.abspath(report_file)}")

if __name__ == "__main__":
    main()
