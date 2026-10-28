import os
import io
import zipfile
import tarfile

class SecurityAnomaly(Exception):
    """Custom exception raised when a malicious archive or bomb is detected."""
    pass

class SafeArchiveInspector:
    """
    Guards against Zip Bombs, Decompression Bombs, Recursive Archives,
    and Path Traversal attacks during audio slack/carved data extraction.
    """
    MAX_UNCOMPRESSED_BYTES = 50 * 1024 * 1024  # 50 MB total decompressed limit
    MAX_INDIVIDUAL_FILE_BYTES = 25 * 1024 * 1024  # 25 MB max per file
    MAX_COMPRESSION_RATIO = 25.0  # Max uncompressed / compressed ratio
    MAX_FILE_COUNT = 200  # Max number of entries
    MAX_RECURSION_DEPTH = 2  # Max depth of nested archives

    @classmethod
    def sanitize_path(cls, target_dir, filename):
        """Prevents path traversal attacks (e.g., ../../evil.bat or absolute paths)."""
        clean_name = os.path.basename(filename)
        # Avoid reserved names and directory traversal
        clean_name = clean_name.replace("..", "").replace("/", "").replace("\\", "")
        if not clean_name:
            clean_name = "carved_file.bin"
        destination = os.path.abspath(os.path.join(target_dir, clean_name))
        if not destination.startswith(os.path.abspath(target_dir)):
            raise SecurityAnomaly(f"Path traversal detected: {filename}")
        return destination

    @classmethod
    def inspect_and_extract_zip(cls, zip_data_or_path, output_dir, depth=0):
        """
        Safely inspects a ZIP buffer or file for compression bombs and extracts safely.
        Returns a dictionary with status, file list, anomalies, and metrics.
        """
        if depth > cls.MAX_RECURSION_DEPTH:
            raise SecurityAnomaly(f"Recursive zip bomb detected: max depth {cls.MAX_RECURSION_DEPTH} exceeded!")

        os.makedirs(output_dir, exist_ok=True)
        report = {
            "is_safe": True,
            "anomaly_detected": False,
            "reason": None,
            "extracted_files": [],
            "total_compressed": 0,
            "total_uncompressed": 0,
            "compression_ratio": 1.0,
            "file_count": 0
        }

        try:
            if isinstance(zip_data_or_path, (bytes, bytearray)):
                zip_file = zipfile.ZipFile(io.BytesIO(zip_data_or_path))
                report["total_compressed"] = len(zip_data_or_path)
            else:
                zip_file = zipfile.ZipFile(zip_data_or_path)
                report["total_compressed"] = os.path.getsize(zip_data_or_path)
        except Exception as e:
            report["is_safe"] = False
            report["reason"] = f"Corrupted or invalid zip: {str(e)}"
            return report

        with zip_file:
            infolist = zip_file.infolist()
            report["file_count"] = len(infolist)

            # 1. Check entry count
            if len(infolist) > cls.MAX_FILE_COUNT:
                report["is_safe"] = False
                report["anomaly_detected"] = True
                report["reason"] = f"Excessive file count ({len(infolist)} > {cls.MAX_FILE_COUNT}) - Potential Zip Bomb!"
                return report

            # 2. Check total uncompressed size & ratio before extracting
            total_uncompressed = sum(info.file_size for info in infolist)
            report["total_uncompressed"] = total_uncompressed

            if report["total_compressed"] > 0:
                report["compression_ratio"] = total_uncompressed / float(report["total_compressed"])

            if total_uncompressed > cls.MAX_UNCOMPRESSED_BYTES:
                report["is_safe"] = False
                report["anomaly_detected"] = True
                report["reason"] = f"Uncompressed payload too large ({total_uncompressed / 1024 / 1024:.2f} MB > {cls.MAX_UNCOMPRESSED_BYTES / 1024 / 1024} MB) - Decompression Bomb blocked!"
                return report

            if report["compression_ratio"] > cls.MAX_COMPRESSION_RATIO and total_uncompressed > 1024 * 1024:
                report["is_safe"] = False
                report["anomaly_detected"] = True
                report["reason"] = f"Suspicious compression ratio ({report['compression_ratio']:.1f}:1) - Zip Bomb heuristic triggered!"
                return report

            # 3. Streamed Extraction with Byte Accounting
            bytes_written_so_far = 0
            for info in infolist:
                if info.is_dir():
                    continue

                if info.file_size > cls.MAX_INDIVIDUAL_FILE_BYTES:
                    report["anomaly_detected"] = True
                    report["reason"] = f"File {info.filename} exceeds single file size limit."
                    continue

                safe_dest = cls.sanitize_path(output_dir, info.filename)

                with zip_file.open(info) as src, open(safe_dest, "wb") as dst:
                    while True:
                        chunk = src.read(64 * 1024)
                        if not chunk:
                            break
                        bytes_written_so_far += len(chunk)
                        if bytes_written_so_far > cls.MAX_UNCOMPRESSED_BYTES:
                            dst.close()
                            os.remove(safe_dest)
                            report["is_safe"] = False
                            report["anomaly_detected"] = True
                            report["reason"] = "Decompression quota exceeded during streaming extraction!"
                            return report
                        dst.write(chunk)

                report["extracted_files"].append({
                    "filename": os.path.basename(safe_dest),
                    "size": info.file_size,
                    "path": safe_dest
                })

        return report
