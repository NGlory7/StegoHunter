import cv2
import numpy as np
from decoders.text_decoders import TextDecoderHub, FLAG_REGEX

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

class OCREngine:
    """
    Intelligent OCR and visual symbol detector for spectrogram snapshots.
    Supports QR Code decoding, Tesseract OCR (if installed), and OpenCV character heuristics.
    """
    def __init__(self):
        self.qr_detector = cv2.QRCodeDetector()

    def scan_image(self, img_bgr_or_gray):
        """
        Scans an image for QR codes, text, and character clusters.
        Returns a dict of findings.
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
        if PYTESSERACT_AVAILABLE:
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
