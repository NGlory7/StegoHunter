import hashlib

def calculate_file_hashes(filepath):
    """Computes MD5, SHA-1, and SHA-256 hashes of an audio file using streaming chunks."""
    md5 = hashlib.md5()
    sha1 = hashlib.sha1()
    sha256 = hashlib.sha256()

    with open(filepath, "rb") as f:
        while chunk := f.read(64 * 1024):
            md5.update(chunk)
            sha1.update(chunk)
            sha256.update(chunk)

    return {
        "md5": md5.hexdigest(),
        "sha1": sha1.hexdigest(),
        "sha256": sha256.hexdigest()
    }
