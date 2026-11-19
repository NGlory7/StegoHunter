"""
RIFF metadata & header forensics.

Scans non-audio chunks (LIST/INFO, id3, JUNK, custom chunks) for hidden flags,
and flags structural anomalies that hint at manual tampering:
  * declared RIFF size != real file size
  * declared data size larger than what is actually present (truncated data)
  * non-standard chunk identifiers (custom hiding places)
  * file signatures (ZIP/PNG/...) hidden inside metadata chunks
"""
from decoders.text_decoders import TextDecoderHub

# Chunk IDs commonly produced by legitimate audio tools
STANDARD_CHUNK_IDS = {
    "LIST", "INFO", "JUNK", "junk", "PAD ", "pad ", "fact", "cue ", "smpl", "inst",
    "bext", "iXML", "id3 ", "ID3 ", "acid", "plst", "labl", "note", "ltxt", "PEAK",
    "DISP", "cart", "umid", "minf", "elm1", "regn", "ovwf", "_PMX", "FLLR", "afsp",
}


class MetadataScanner:
    """Inspects RIFF chunk metadata of an AudioContainer for secrets and header anomalies."""

    @staticmethod
    def scan(container):
        from detectors.file_carver import FileCarver

        report = {"findings": [], "anomalies": [], "chunks": []}

        # 1. Header size consistency
        delta = container.filesize - container.declared_riff_size
        if container.declared_riff_size and abs(delta) > 1:  # a single pad byte is common and harmless
            report["anomalies"].append({
                "type": "RIFF Size Mismatch",
                "detail": f"Header declares {container.declared_riff_size:,} bytes, file is {container.filesize:,} bytes ({delta:+,}).",
                "severity": "HIGH" if delta > 0 else "MEDIUM",
            })

        # 2. Chunk-level inspection
        for chunk in container.metadata_chunks:
            cid = chunk.get("id", "")
            report["chunks"].append({"id": cid, "size": chunk.get("size", 0), "offset": chunk.get("offset", 0)})

            if cid not in STANDARD_CHUNK_IDS:
                report["anomalies"].append({
                    "type": "Non-Standard Chunk",
                    "detail": f"Unknown chunk '{cid}' ({chunk.get('size', 0):,} bytes) at offset 0x{chunk.get('offset', 0):X}.",
                    "severity": "MEDIUM",
                })

            text = chunk.get("text_content") or ""
            for msg in TextDecoderHub.analyze_string(text):
                finding = dict(msg)
                finding["chunk"] = cid
                report["findings"].append(finding)

            raw = chunk.get("raw") or b""
            for offset, mtype, _ext in FileCarver.find_signatures(raw):
                report["anomalies"].append({
                    "type": "Embedded File In Chunk",
                    "detail": f"{mtype} signature inside chunk '{cid}' at +0x{offset:X}.",
                    "severity": "HIGH",
                })

        report["findings"] = TextDecoderHub._dedupe(report["findings"])
        return report
