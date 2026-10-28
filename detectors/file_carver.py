import os
import re
from core.safety import SafeArchiveInspector
from decoders.text_decoders import TextDecoderHub, FLAG_REGEX

# 4+ bytes signatures to prevent false positives in audio noise
MAGIC_SIGNATURES = [
    (b"PK\x03\x04", "ZIP Archive", ".zip"),
    (b"Rar!\x1a\x07\x00", "RAR 4.x Archive", ".rar"),
    (b"Rar!\x1a\x07\x01\x00", "RAR 5.x Archive", ".rar"),
    (b"7z\xbc\xaf\x27\x1c", "7-Zip Archive", ".7z"),
    (b"\x89PNG\r\n\x1a\n", "PNG Image", ".png"),
    (b"\xff\xd8\xff\xe0", "JPEG JFIF Image", ".jpg"),
    (b"\xff\xd8\xff\xe1", "JPEG Exif Image", ".jpg"),
    (b"%PDF-1.", "PDF Document", ".pdf"),
    (b"\x1f\x8b\x08", "GZIP File", ".gz")
]

class FileCarver:
    """
    Forensic file carving engine:
    Scans trailing slack space and non-audio chunks for hidden archives, images, or documents,
    enforcing strict Zip Bomb and memory exhaustion defenses.
    """
    def __init__(self, raw_audio_path, appended_payload=b"", output_dir="stego_output"):
        self.raw_audio_path = raw_audio_path
        self.appended_payload = appended_payload
        self.carved_dir = os.path.join(output_dir, "carved_payloads")
        os.makedirs(self.carved_dir, exist_ok=True)
        self.results = []

    def scan_and_carve(self):
        """Executes deep file carving on slack space and trailing bytes."""
        carve_targets = []
        if self.appended_payload and len(self.appended_payload) > 16:
            carve_targets.append(("Trailing Slack Space (EOF)", self.appended_payload, 0))

        # Avoid redundant carving
        seen_offsets = set()
        for label, payload_data, offset in carve_targets:
            if offset in seen_offsets:
                continue
            seen_offsets.add(offset)

            entry = {
                "label": label,
                "offset": offset,
                "size_bytes": len(payload_data),
                "type": "Unknown",
                "extracted_path": None,
                "safety_status": "Clean",
                "security_alert": None,
                "decoded_messages": []
            }

            # Check if it starts with ZIP
            if payload_data.startswith(b"PK\x03\x04"):
                entry["type"] = "ZIP Archive"
                # Inspect with SafeArchiveInspector
                sub_dir = os.path.join(self.carved_dir, f"zip_carve_0x{offset:X}")
                safe_report = SafeArchiveInspector.inspect_and_extract_zip(payload_data, sub_dir)

                if safe_report["anomaly_detected"]:
                    entry["safety_status"] = "BLOCKED / BOMB DETECTED"
                    entry["security_alert"] = safe_report["reason"]
                else:
                    entry["safety_status"] = "Safely Extracted"
                    entry["extracted_path"] = sub_dir
                    # Scan extracted files for text / flags
                    for extracted in safe_report["extracted_files"]:
                        try:
                            with open(extracted["path"], "r", encoding="utf-8", errors="ignore") as ef:
                                content = ef.read(10000)
                                msgs = TextDecoderHub.analyze_string(content)
                                if msgs:
                                    entry["decoded_messages"].extend(msgs)
                        except Exception:
                            pass

                entry["details"] = safe_report

            else:
                matched_type = "Binary / Raw Data"
                target_ext = ".bin"
                for magic, mtype, ext in MAGIC_SIGNATURES:
                    if payload_data.startswith(magic):
                        matched_type = mtype
                        target_ext = ext
                        break

                entry["type"] = matched_type
                save_len = min(len(payload_data), 10 * 1024 * 1024)
                out_filename = f"carved_0x{offset:X}{target_ext}"
                out_path = os.path.join(self.carved_dir, out_filename)
                with open(out_path, "wb") as cf:
                    cf.write(payload_data[:save_len])
                entry["extracted_path"] = out_path

                # Scan for plain strings/flags
                printable = bytes([b for b in payload_data[:20000] if 32 <= b <= 126 or b in (10, 13)]).decode('ascii', errors='ignore')
                msgs = TextDecoderHub.analyze_string(printable)
                if msgs:
                    entry["decoded_messages"].extend(msgs)

            self.results.append(entry)

        return self.results
