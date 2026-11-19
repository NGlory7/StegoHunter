import cv2
import numpy as np
from decoders.text_decoders import TextDecoderHub, FLAG_REGEX

import os
import re
import shutil
from collections import defaultdict

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False


def _find_tesseract():
    """Locates the tesseract binary even when the installer did not add it to PATH."""
    candidates = [
        os.environ.get("TESSERACT_CMD"),
        shutil.which("tesseract"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Tesseract-OCR", "tesseract.exe"),
        "/usr/bin/tesseract", "/usr/local/bin/tesseract", "/opt/homebrew/bin/tesseract",
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return None


TESSERACT_CMD = _find_tesseract() if PYTESSERACT_AVAILABLE else None
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
TESSERACT_AVAILABLE = bool(TESSERACT_CMD)
MIN_OCR_SUPPORT = 2  # readings with a bracket needed before a flag is reconstructed
MIN_OCR_VOTES = 3  # readings close to the consensus needed before it is reported
NEAR_MATCH_RATIO = 0.85
MIN_PARTIAL_BODY = 6  # shorter bracket-only readings are usually speckle noise

# prefix { body }  - OCR often reads braces as ( [ and drops the closing one
_BRACE_TEXT_REGEX = re.compile(r'([A-Za-z0-9_\-]{0,24})\s*[\{\(\[]\s*(.+?)\s*(?:[\}\)\]][\}\)\]\s_\-|.,]*|$)')


def normalize_flag_text(text):
    """
    Repairs typical OCR damage on flag banners and returns (prefix, body) or None:
      'TE(Music is Fr33d0m)}'  -> ('TE', 'Music_is_Fr33d0m')
      'TE{Music . is Fr3d0m}'  -> ('TE', 'Music_is_Fr3d0m')
    Spaces become underscores (thin underscores vanish in spectrograms) and
    punctuation-only tokens are dropped as speckle noise.
    """
    if not text:
        return None
    line = " ".join(text.split())
    # Known flag keywords followed by a misread brace: 'CT Fi', 'CTE(', 'FLAG[' -> 'CTF{', 'FLAG{'
    line = re.sub(r'C\s?T\s?[FE]\s?[\{\(\[iIl|!]\s?(?=[A-Za-z0-9])', 'CTF{', line, count=1)
    # 'fiagG...' -> 'flag{...': l/i/1 swap and '{' read as G or C are common on wide glyphs
    line = re.sub(r'(F)\s?([LIi1|])\s?(A)\s?(G)\s?[\{\(\[iIl|!GC]\s?(?=[A-Za-z0-9])',
                  lambda m: m.group(1) + ("L" if m.group(1).isupper() else "l") + m.group(3) + m.group(4) + "{",
                  line, count=1, flags=re.IGNORECASE)
    m = _BRACE_TEXT_REGEX.search(line)
    if not m:
        return None
    prefix = re.sub(r'[^A-Za-z0-9_\-]', '', m.group(1))
    prefix = repair_keyword(prefix)
    tokens = [re.sub(r'[^A-Za-z0-9_\-!@#$%^&*+=.,]', '', t) for t in m.group(2).split()]
    tokens = [t.strip("._-,") for t in tokens if re.search(r'[A-Za-z0-9]', t)]
    body = "_".join(t for t in tokens if t)
    body = re.sub(r'_+', '_', body).strip("_")
    if len(body) < 3:
        return None
    return prefix, body


def repair_keyword(prefix):
    """Keyword spelled with a look-alike glyph: 'fiag', 'f1ag', 'FIAG' -> 'flag' / 'FLAG'."""
    return re.sub(r'(f)[iIl1|](ag)$', lambda k: k.group(1) + ("L" if k.group(1).isupper() else "l") + k.group(2),
                  prefix or "", flags=re.IGNORECASE)


# Glyphs OCR engines (and Tesseract versions) swap freely; they count as one vote
_GLYPH_CLASS = {c: "l" for c in "lIi1|"}
_GLYPH_CLASS.update({c: "o" for c in "oO0"})


def consensus_prefix(votes, min_agreement=0.75, min_support=3):
    """
    Right-aligned per-character vote over prefix readings ('HRCTF', 'nftCTF', 'RCTF' -> 'CTF').
    Characters are accepted from the brace outwards only while at least `min_agreement`
    of the readings that reach that position agree, so an uncertain start is dropped
    instead of being invented.
    """
    # Accepts {prefix: weight} or a list of (prefix, weight) - one pair per OCR reading,
    # so identical readings still count as separate support.
    pairs = votes.items() if isinstance(votes, dict) else votes
    items = [(p, w) for p, w in pairs if p]
    if not items:
        return ""
    out = []
    for pos in range(1, max(len(p) for p, _ in items) + 1):
        tally, spelled = defaultdict(float), defaultdict(lambda: defaultdict(float))
        reach = 0.0
        for p, w in items:
            if len(p) >= pos:
                cls = _GLYPH_CLASS.get(p[-pos], p[-pos].lower())  # 'F'/'f', 'l'/'I'/'1' vote together
                tally[cls] += w
                spelled[cls][p[-pos]] += w
                reach += w
        if reach == 0 or sum(1 for p, _ in items if len(p) >= pos) < min_support:
            break
        ch, w_ch = max(tally.items(), key=lambda kv: kv[1])
        if w_ch / reach < min_agreement:
            break
        out.append(max(spelled[ch].items(), key=lambda kv: kv[1])[0])
    return "".join(reversed(out))


def consensus_body(votes):
    """
    Body consensus in two steps:
      1. medoid over the separator-free letter sequence (OCR splits words at random:
         'Fr3 3d0m' vs 'Fr33d0m'),
      2. an underscore is kept at a position only if more than half of the readings
         with exactly that letter sequence put one there.
    """
    keyed = defaultdict(float)
    for body, w in votes.items():
        keyed[body.replace("_", "")] += w
    key = weighted_medoid(keyed)

    same = [(b, w) for b, w in votes.items() if b.replace("_", "") == key]
    total = sum(w for _, w in same)
    sep_votes = defaultdict(float)
    for b, w in same:
        idx = 0
        for ch in b:
            if ch == "_":
                sep_votes[idx] += w
            else:
                idx += 1
    out = []
    for i, ch in enumerate(key):
        if i and sep_votes.get(i, 0.0) > total / 2.0:
            out.append("_")
        out.append(ch)
    return "".join(out)


def weighted_medoid(votes):
    """
    Picks the reading that is most similar to all other readings (weighted by their votes).
    Different preprocessing variants make *different* small mistakes, so the most
    'central' string is far more reliable than the single most frequent one.
    """
    from difflib import SequenceMatcher

    items = list(votes.items())
    if len(items) == 1:
        return items[0][0]
    best, best_score = None, None
    for cand, w_c in items:
        dist = sum(w_o * (1.0 - SequenceMatcher(None, cand, other).ratio()) for other, w_o in items if other != cand)
        score = (dist, -w_c)
        if best_score is None or score < best_score:
            best, best_score = cand, score
    return best

class OCREngine:
    """
    Intelligent OCR and visual symbol detector for spectrogram snapshots.
    Supports QR Code decoding, Tesseract OCR (if installed), and OpenCV character heuristics.
    """
    def __init__(self):
        self.qr_detector = cv2.QRCodeDetector()

    @staticmethod
    def _squeeze_factors(gray, target_h=60, target_aspect=0.7):
        """
        Horizontal scale factors to try. One is measured: the median width/height of
        letter-sized components after a quick binarisation, scaled to `target_aspect`.
        1.0 and 0.5 stay as fallbacks in case the measurement is fooled by noise.
        """
        h, w = gray.shape[:2]
        small = cv2.resize(gray, (max(40, int(w * target_h / h)), target_h), interpolation=cv2.INTER_AREA)
        smooth = cv2.GaussianBlur(small, (5, 5), 0)
        _, binary = cv2.threshold(smooth, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        n, _, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8))
        sw = small.shape[1]
        ratios = [stats[i, cv2.CC_STAT_WIDTH] / float(stats[i, cv2.CC_STAT_HEIGHT])
                  for i in range(1, n)
                  if stats[i, cv2.CC_STAT_HEIGHT] >= 0.35 * target_h
                  and stats[i, cv2.CC_STAT_WIDTH] < 0.3 * sw]  # merged runs say nothing about one glyph
        factors = [1.0, 0.5]
        if ratios:
            auto = float(np.clip(target_aspect / float(np.median(ratios)), 0.08, 1.0))
            if all(abs(auto - f) > 0.1 for f in factors):
                factors.append(auto)
        return factors

    @staticmethod
    def _banner_variants(gray, target_h=60):
        """
        Yields binarised versions of a noisy spectrogram banner (black text on white).
        Spectrogram text is bright and smooth while music/noise is fine speckle, so the
        image is shrunk to ~30 px letters, lightly smoothed and cut at a high percentile.
        Several settings are produced because no single one survives every recording.
        """
        h, w = gray.shape[:2]
        if h < 8 or w < 20:
            return
        if float(np.mean(gray)) > 140:  # dark text on bright background -> flip
            gray = 255 - gray
        # Time axis stretch is arbitrary (window length / hop), so letters can be far wider
        # than normal type. Measure the glyphs and squeeze them back to a normal aspect.
        for squeeze in OCREngine._squeeze_factors(gray, target_h):
            new_w = max(40, int(w * target_h / h * squeeze))
            small = cv2.resize(gray, (new_w, target_h), interpolation=cv2.INTER_AREA)
            for blur_k in (3, 5):
                smooth = cv2.GaussianBlur(small, (blur_k, blur_k), 0)
                # Percentile cut works when text fills the crop; Otsu when there is spare background
                for pct in (78, 84, "otsu"):
                    if pct == "otsu":
                        _, binary = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    else:
                        binary = (smooth > np.percentile(smooth, pct)).astype(np.uint8) * 255
                    for open_k in (1, 2):
                        b = binary
                        if open_k > 1:
                            b = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (open_k, open_k)))
                        yield cv2.copyMakeBorder(255 - b, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)

    @staticmethod
    def _ocr_line(img):
        """Single-line Tesseract pass. Returns (text, mean word confidence 0-100)."""
        data = pytesseract.image_to_data(img, config="--psm 7", output_type=pytesseract.Output.DICT)
        words, confs = [], []
        for word, conf in zip(data["text"], data["conf"]):
            if word.strip():
                words.append(word)
                try:
                    confs.append(max(0.0, float(conf)))
                except (TypeError, ValueError):
                    pass
        return " ".join(words), (sum(confs) / len(confs) if confs else 0.0)

    def read_banner(self, banner_gray):
        """
        Multi-variant OCR with voting for spectrogram text banners.
        Returns {"available", "best_text", "raw_texts", "flag", "decoded_candidates"}.
        """
        result = {"available": TESSERACT_AVAILABLE, "best_text": None, "raw_texts": [], "flag": None, "decoded_candidates": []}
        if not TESSERACT_AVAILABLE or banner_gray is None or banner_gray.size == 0:
            return result
        if len(banner_gray.shape) == 3:
            banner_gray = cv2.cvtColor(banner_gray, cv2.COLOR_BGR2GRAY)

        prefix_votes, body_votes, text_votes = [], defaultdict(float), defaultdict(float)
        for variant in self._banner_variants(banner_gray):
            try:
                text, conf = self._ocr_line(variant)
            except Exception:
                continue
            if not text:
                continue
            weight = 1.0 + conf / 100.0
            result["raw_texts"].append(text)
            text_votes[text] += weight
            norm = normalize_flag_text(text)
            if norm:
                prefix, body = norm
                body_votes[body] += weight
                if prefix:
                    prefix_votes.append((prefix, weight))

        if text_votes:
            result["best_text"] = max(text_votes, key=text_votes.get)

        candidates = []
        # A real banner reads (almost) the same under several preprocessings; a body that
        # only one variant produced is noise that happened to contain a bracket.
        support = sum(1 for t in result["raw_texts"] if normalize_flag_text(t))
        if body_votes and support >= MIN_OCR_SUPPORT:
            body = consensus_body(body_votes)
            prefix = repair_keyword(consensus_prefix(prefix_votes))
            flag = f"{prefix}{{{body}}}"
            result["flag"] = flag
            key = body.replace("_", "")
            # Support = readings close to the consensus. Real text gives many near-identical
            # readings; noise gives readings that differ from each other and from the medoid.
            from difflib import SequenceMatcher
            votes = sum(1 for t in result["raw_texts"]
                        if SequenceMatcher(None, ((normalize_flag_text(t) or ("", ""))[1]).replace("_", "").lower(),
                                           key.lower()).ratio() >= NEAR_MATCH_RATIO)
            # OCR is never certain: always MEDIUM, the banner image is the ground truth
            if votes < MIN_OCR_VOTES:
                pass  # readings disagree letter-by-letter: noise, not text
            elif FLAG_REGEX.fullmatch(flag):
                candidates.append({"type": "Spectrogram OCR Flag (verify on banner)", "decoded": flag,
                                   "confidence": "MEDIUM", "is_flag": True, "ocr_votes": votes})
            elif len(key) >= MIN_PARTIAL_BODY:
                candidates.append({"type": "Spectrogram OCR Partial Flag", "decoded": flag, "confidence": "MEDIUM",
                                   "is_flag": True, "ocr_votes": votes})
        # Non-brace secrets (Base64 / hex written as spectrogram text)
        if result["best_text"]:
            for msg in TextDecoderHub.analyze_string(result["best_text"].replace(" ", "")):
                candidates.append(msg)
        # A flag read verbatim by one variant beats the reconstructed one if bodies agree
        def body_key(flag):
            return flag.split("{", 1)[-1].rstrip("}").replace("_", "").lower()
        direct = [c for c in candidates if "ocr_votes" not in c and "{" in c["decoded"]]
        rebuilt = [c for c in candidates if "ocr_votes" in c]
        for c in rebuilt:
            if any(body_key(d["decoded"]) == body_key(c["decoded"]) for d in direct):
                for d in direct:
                    if body_key(d["decoded"]) == body_key(c["decoded"]):
                        d["ocr_votes"] = c["ocr_votes"]
                        d["type"] = "Spectrogram OCR Flag (verify on banner)"
                        d["confidence"] = "MEDIUM"
                candidates.remove(c)
        result["decoded_candidates"] = TextDecoderHub._dedupe(candidates)
        return result

    def scan_image(self, img_bgr_or_gray, use_tesseract=True):
        """
        Scans an image for QR codes, text, and character clusters.
        Returns a dict of findings. `use_tesseract=False` keeps the cheap checks only
        (used for the thousands of sliding-window snapshots).
        """
        findings = {
            "qr_detected": False,
            "qr_payload": None,
            "raw_text": None,
            "decoded_candidates": [],
            "confidence": "NONE"
        }

        if img_bgr_or_gray is None or img_bgr_or_gray.size == 0:
            return findings

        if len(img_bgr_or_gray.shape) == 2:
            gray = img_bgr_or_gray
            bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            bgr = img_bgr_or_gray
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # 1. Try QR Code Detection only if geometry is approximately square (QR code standard)
        h, w = gray.shape[:2]
        aspect = w / float(max(1, h))
        if 0.5 <= aspect <= 2.0 and min(h, w) >= 64:
            try:
                qr_data, points, _ = self.qr_detector.detectAndDecode(bgr)
                if qr_data:
                    findings["qr_detected"] = True
                    findings["qr_payload"] = qr_data
                    findings["confidence"] = "HIGH"
                    findings["decoded_candidates"].append({
                        "type": "Spectrogram QR Code",
                        "decoded": qr_data,
                        "confidence": "VERY HIGH",
                        "is_flag": bool(FLAG_REGEX.search(qr_data))
                    })
            except Exception:
                pass

        # 2. Try Pytesseract if available
        if use_tesseract and TESSERACT_AVAILABLE:
            try:
                # Preprocess with adaptive threshold
                prep = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                tess_text = pytesseract.image_to_string(prep, config='--psm 6').strip()
                if len(tess_text) >= 3:
                    findings["raw_text"] = tess_text
                    msgs = TextDecoderHub.analyze_string(tess_text)
                    if msgs:
                        findings["decoded_candidates"].extend(msgs)
                        findings["confidence"] = "HIGH"
            except Exception:
                pass

        # 3. Contour cluster text analysis
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        letter_boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if 80 < area < (gray.shape[0] * gray.shape[1] * 0.35):
                x, y, w, h = cv2.boundingRect(c)
                aspect = w / float(h)
                if 0.15 < aspect < 7.5:
                    letter_boxes.append((x, y, w, h))

        if len(letter_boxes) >= 4:
            if findings["confidence"] == "NONE":
                findings["confidence"] = "MEDIUM"
            findings["num_clusters"] = len(letter_boxes)

        return findings
