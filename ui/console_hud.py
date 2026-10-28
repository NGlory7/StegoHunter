import sys
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.markup import escape

console = Console()

class ConsoleHUD:
    """Rich-powered terminal HUD for professional steganography and forensics audit."""

    @staticmethod
    def render_banner():
        banner_text = Text()
        banner_text.append("AUDIO STEGO HUNTER & FORENSIC SUITE\n", style="bold cyan")
        banner_text.append("Advanced Audio Steganalysis, Sliding-Window Radar, LSB & Archive Defenses", style="dim white")
        console.print(Panel(banner_text, border_style="cyan", expand=False))

    @staticmethod
    def render_file_meta(audio_container):
        table = Table(title="Target Audio Fingerprint", border_style="bright_blue", show_header=True, header_style="bold magenta")
        table.add_column("Property", style="bold yellow")
        table.add_column("Value", style="white")

        table.add_row("Filename", escape(audio_container.filename))
        table.add_row("File Size", f"{audio_container.filesize:,} bytes ({audio_container.filesize / 1024:.1f} KB)")
        table.add_row("Duration", f"{audio_container.duration:.2f} seconds")
        table.add_row("Sample Rate", f"{audio_container.sample_rate} Hz")
        table.add_row("Channels", f"{audio_container.num_channels} ({', '.join(audio_container.channels.keys())})")
        table.add_row("MD5 Hash", audio_container.hashes["md5"])
        table.add_row("SHA-256", audio_container.hashes["sha256"][:32] + "...")
        if audio_container.appended_payload:
            table.add_row("Slack Space (EOF)", f"[bold red]DETECTED: {len(audio_container.appended_payload)} trailing bytes![/bold red]")
        else:
            table.add_row("Slack Space (EOF)", "[green]Clean (0 extra bytes)[/green]")

        console.print(table)

    @staticmethod
    def render_top_highlights_table(top_highlights):
        """Displays a prioritized table of only the clearest images containing distinct text."""
        if not top_highlights:
            return

        table = Table(title="🌟 EN BELİRGİN YAZI GÖRÜNTÜLERİ (TOP PRIORITY HIGHLIGHTS)", border_style="bold green", show_header=True, header_style="bold yellow")
        table.add_column("Sıra", style="bold cyan", justify="center")
        table.add_column("Netlik Skoru", style="bold magenta", justify="center")
        table.add_column("Zaman Aralığı", style="bold white")
        table.add_column("Kanal & Frekans Bandı", style="bright_blue")
        table.add_column("Karakter Kümesi", style="yellow", justify="center")

        for rank, h in enumerate(top_highlights, 1):
            score_color = "bold green" if h["clarity_score"] >= 70 else ("bold yellow" if h["clarity_score"] >= 45 else "white")
            score_str = f"[{score_color}]%{h['clarity_score']:.1f}[/{score_color}]"
            table.add_row(
                f"#{rank}",
                score_str,
                f"⏱️ {h['time_str']}",
                f"{h['channel']} • {h['band_name']} ({h['f_min']:,} - {h['f_max']:,} Hz, FFT {h['n_fft']})",
                f"{h['num_clusters']} harf"
            )

        console.print(table)

    @staticmethod
    def render_anomalies_summary(snapshots_count, anomalies, lsb_results, carved_results, morse_result, pam_result=None, dtmf_result=None):
        console.print(f"\n[bold green]Scan Summary:[/bold green] Generated and analyzed [cyan]{snapshots_count}[/cyan] micro-spectrogram snapshots.")

        # Flag / Decoded Secrets Table
        secrets_found = []
        for s in anomalies:
            for dc in s.get("decoded_candidates", []):
                secrets_found.append((s["time_str"], f"Spectrogram ({s['band_name']})", dc["type"], dc["decoded"]))

        for l in lsb_results:
            for dc in l.get("decoded_messages", []):
                content = str(dc["decoded"])
                if len(content) > 120:
                    content = content[:117] + "..."
                secrets_found.append(("-", f"LSB Bit {l['bit_index']}", dc["type"], content))

        for c in carved_results:
            for dc in c.get("decoded_messages", []):
                content = str(dc["decoded"])
                if len(content) > 120:
                    content = content[:117] + "..."
                secrets_found.append(("-", f"Carved File ({c['type']})", dc["type"], content))

        if morse_result:
            secrets_found.append(("-", "Audio Tone Envelope", morse_result["type"], morse_result["decoded_text"]))

        if pam_result:
            for dc in pam_result.get("messages", []):
                secrets_found.append(("-", "Discrete Amplitude (PAM)", dc["type"], dc["decoded"]))

        if dtmf_result:
            for dc in dtmf_result.get("messages", []):
                secrets_found.append(("-", "DTMF Keypad Tones", dc["type"], dc["decoded"]))
            if not dtmf_result.get("messages") and dtmf_result.get("sequence"):
                secrets_found.append(("-", "DTMF Keypad Tones", dtmf_result["type"], dtmf_result["sequence"]))

        if secrets_found:
            sec_table = Table(title="Decoded Hidden Messages & Secrets", border_style="bold red", header_style="bold yellow")
            sec_table.add_column("Timestamp", style="cyan")
            sec_table.add_column("Source / Vector", style="magenta")
            sec_table.add_column("Decoder Type", style="green")
            sec_table.add_column("Decoded Content", style="bold white")

            for t, src, dtype, content in secrets_found:
                sec_table.add_row(escape(str(t)), escape(str(src)), escape(str(dtype)), escape(str(content)))
            console.print(sec_table)
        else:
            console.print("[dim yellow][*] No definitive text pattern or flag matched by automatic decoders.[/dim yellow]")

        # Security and Zip Bomb Check
        for c in carved_results:
            if c.get("security_alert"):
                alert_text = Text()
                alert_text.append("DEFENSIVE SHIELD TRIGGERED!\n", style="bold red")
                alert_text.append(f"Source: {c['label']}\nAlert: {c['security_alert']}\nAction: Malicious expansion blocked to prevent resource exhaustion.", style="white")
                console.print(Panel(alert_text, title="Zip / Decompression Bomb Guard", border_style="red"))

    @staticmethod
    def render_ascii_top_candidate(top_highlights, fallback_anomalies):
        candidates = top_highlights if top_highlights else fallback_anomalies
        if not candidates:
            return
        best = candidates[0]
        if best.get("ascii_art"):
            console.print(f"\n[bold cyan]🌟 EN BELİRGİN ADAYIN OKUNUR ASCII ÖNİZLEMESİ (Score: %{best.get('clarity_score', 0):.1f}):[/bold cyan] {best['channel']} | {best['time_str']} | {best['band_name']} (FFT: {best['n_fft']})")
            console.print(best['ascii_art'], markup=False)
