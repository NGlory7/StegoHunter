import os
import cv2
import numpy as np
from detectors.ocr_engine import OCREngine
from decoders.text_decoders import TextDecoderHub, FLAG_REGEX

def safe_imwrite(filepath, img):
    """Safely saves an OpenCV image supporting unicode paths."""
    success, encoded = cv2.imencode(".png", img)
    if success:
        with open(filepath, "wb") as f:
            f.write(encoded.tobytes())
        return True
    return False

def find_best_text_line(boxes, img_width=1000):
    """
    Finds the largest horizontally aligned cluster of character bounding boxes.
    Filters out isolated background audio noise dots and narrow harmonic beat ticks.
    """
    if len(boxes) < 2:
        return boxes
    # Filter out tiny 1-pixel noise dots or single-tone harmonics (height must be >= 10, area >= 35)
    valid_boxes = [b for b in boxes if b[3] >= 10 and b[2] >= 5 and (b[2] * b[3]) >= 35]
    if len(valid_boxes) < 2:
        valid_boxes = boxes

    heights = [h for x, y, w, h in valid_boxes]
    median_h = float(np.median(heights)) + 1e-5
    best_cluster = []
    for b_target in valid_boxes:
        target_base = b_target[1] + b_target[3]
        cluster = [b for b in valid_boxes if abs((b[1] + b[3]) - target_base) <= median_h * 0.45]
        if len(cluster) > len(best_cluster):
            best_cluster = cluster
    return best_cluster if len(best_cluster) >= 2 else valid_boxes

def detect_opencv_word_banner(img_contrast, img_thresh):
    """
    OpenCV MSER and Multi-Character Word Banner Detector.
    Distinguishes genuine CTF flag banners from drum blasts and background harmonics:
    - Genuine text: horizontal continuity (w >= 120px, aspect >= 2.8, 15 <= h <= 120px),
      does NOT span the entire vertical frequency height, and has clean baseline alignment.
    - Drum / cymbal blasts: span > 65% of the total image height across multiple columns (disqualified).
    - Continuous music tones: span 100% of the image horizontally with tiny height (h < 10px).
    Returns: (word_banner_bbox, word_width, letter_boxes, is_drum_blast)
    """
    if img_thresh is None or img_thresh.size == 0:
        return None, 0, [], False

    h, w = img_contrast.shape

    # 1. Percussion / Drum Blast Filter
    vert_coverage = np.sum(img_thresh > 0, axis=0) / float(max(1, h))
    if np.mean(vert_coverage > 0.65) > 0.12:
        return None, 0, [], True

    # 2. OpenCV MSER Letter Region Detection
    letter_boxes = []
    try:
        mser = cv2.MSER_create(min_area=25, max_area=int(h * w * 0.05))
        _, bboxes = mser.detectRegions(img_contrast)
        for (bx, by, bw, bh) in bboxes:
            aspect = bw / float(max(1, bh))
            if 0.20 <= aspect <= 3.0 and 10 <= bh <= min(110, int(h * 0.70)) and bw >= 5:
                if bw < w * 0.40 and by > 5 and (by + bh) < (h - 5):
                    letter_boxes.append((bx, by, bw, bh))
    except Exception:
        pass

    # 3. Multi-Letter Word Banner Detection in Binarized Thresh
    cnts, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_banner = None
    max_w = 0
    for c in cnts:
        bx, by, bw, bh = cv2.boundingRect(c)
        aspect = bw / float(max(1, bh))
        if bw >= 120 and 15 <= bh <= min(120, int(h * 0.70)) and aspect >= 2.8:
            if by > 8 and (by + bh) < (h - 6) and bw < (w * 0.96):
                if bw > max_w:
                    max_w = bw
                    best_banner = (bx, by, bw, bh)

    return best_banner, max_w, letter_boxes, False


def _render_binary_card(bin_crop, title, width=84, height=13):
    """Renders a single bordered high-contrast ASCII card."""
    resized = cv2.resize(bin_crop, (width, height), interpolation=cv2.INTER_AREA)
    border_len = max(2, width - len(title) - 8)
    lines = [f"+--- [ {title} ] " + "-" * border_len + "+"]
    for row in resized:
        line_str = "".join("#" if v > 110 else " " for v in row)
        lines.append("| " + line_str + " |")
    lines.append("+" + "-" * (width + 2) + "+")
    return "\n".join(lines)

def render_ascii_crop(img_gray, text_boxes=None, target_width=110, target_height=14):
    """
    Renders high-contrast, crystal-clear ASCII text from spectrogram crops.
    For wide text clusters (like CTF flags), provides split panoramic cards
    so every single character has generous horizontal width (~7 columns/character)
    and is immediately legible to the human eye.
    """
    if img_gray is None or img_gray.size == 0:
        return None

    # Focus on the primary horizontal text line cluster
    if text_boxes and len(text_boxes) >= 2:
        line_boxes = find_best_text_line(text_boxes, img_gray.shape[1])
        if len(line_boxes) >= 2:
            min_x = max(0, min(b[0] for b in line_boxes) - 10)
            max_x = min(img_gray.shape[1], max(b[0] + b[2] for b in line_boxes) + 10)
            min_y = max(0, min(b[1] for b in line_boxes) - 10)
            max_y = min(img_gray.shape[0], max(b[1] + b[3] for b in line_boxes) + 10)
            crop = img_gray[min_y:max_y, min_x:max_x]
        else:
            crop = img_gray
    else:
        crop = img_gray

    if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 10:
        return None

    # High-contrast thresholding with noise suppression
    if np.mean(crop) > 128:
        p30 = np.percentile(crop, 30)
        _, binary = cv2.threshold(crop, p30, 255, cv2.THRESH_BINARY_INV)
    else:
        p76 = np.percentile(crop, 76)
        _, binary = cv2.threshold(crop, p76, 255, cv2.THRESH_BINARY)

    binary = cv2.medianBlur(binary, 3)

    # Tight crop to active ink rows
    row_sums = np.sum(binary > 0, axis=1)
    ink_rows = np.where(row_sums > np.percentile(row_sums, 25))[0]
    if len(ink_rows) >= 5:
        r0 = max(0, ink_rows[0] - 2)
        r1 = min(binary.shape[0], ink_rows[-1] + 3)
        binary = binary[r0:r1, :]

    # If the text spans a wide aspect ratio (> 3.5) and many pixels (> 200),
    # produce a high-resolution split representation (Part 1 and Part 2)
    aspect = binary.shape[1] / float(max(1, binary.shape[0]))
    if aspect > 3.5 and binary.shape[1] > 200:
        mid_x = binary.shape[1] // 2
        overlap_x = int(binary.shape[1] * 0.03)
        part1 = binary[:, :mid_x + overlap_x]
        part2 = binary[:, max(0, mid_x - overlap_x):]

        card1 = _render_binary_card(part1, "PART 1: Flag Başlangıcı", width=84, height=13)
        card2 = _render_binary_card(part2, "PART 2: Flag Devamı", width=84, height=13)
        return card1 + "\n\n" + card2
    else:
        tw = min(target_width, max(50, int(target_height * aspect * 1.8)))
        return _render_binary_card(binary, "Spectrogram Text Preview", width=tw, height=target_height)

def compute_text_clarity_score(img_contrast, img_thresh, text_boxes):
    """
    Evaluates text clarity by clustering horizontal character lines and computing
    occupancy ratio, horizontal span ratio, and baseline alignment.
    Real CTF text words occupy 40% to 80% of the horizontal window, whereas
    isolated drum pulses occupy less than 15%.
    """
    if len(text_boxes) < 3:
        return 0.0

    img_w = img_contrast.shape[1]
    line_boxes = find_best_text_line(text_boxes, img_w)
    if len(line_boxes) < 3 or len(line_boxes) > 38:
        return 10.0

    min_x = min(b[0] for b in line_boxes)
    max_x = max(b[0] + b[2] for b in line_boxes)
    span_ratio = (max_x - min_x) / float(max(1, img_w))

    sorted_b = sorted(line_boxes, key=lambda b: b[0])
    covered = 0
    cur_x = -1
    for bx, by, bw, bh in sorted_b:
        start = max(bx, cur_x)
        end = bx + bw
        if end > start:
            covered += (end - start)
            cur_x = end
    occupancy = covered / float(max(1, img_w))

    if span_ratio < 0.25 or occupancy < 0.12:
        return 15.0

    y_baselines = [y + h for x, y, w, h in line_boxes]
    heights = [h for x, y, w, h in line_boxes]
    mean_h = float(np.median(heights)) + 1e-5

    std_baseline = float(np.std(y_baselines)) / mean_h
    baseline_score = max(0.0, 1.0 - min(1.0, std_baseline * 2.0))

    occ_score = min(1.0, occupancy / 0.45)
    span_score = min(1.0, span_ratio / 0.55)

    min_y = max(0, min(b[1] for b in line_boxes))
    max_y = min(img_contrast.shape[0], max(b[1] + b[3] for b in line_boxes))
    crop = img_contrast[min_y:max_y, min_x:max_x]
    if crop.size > 0:
        p90 = float(np.percentile(crop, 90))
        p20 = float(np.percentile(crop, 20))
        contrast_score = min(1.0, max(0.0, (p90 - p20) / 100.0))
    else:
        contrast_score = 0.5

    composite = (
        occ_score * 0.35 +
        span_score * 0.30 +
        baseline_score * 0.20 +
        contrast_score * 0.15
    ) * 100.0

    return round(float(np.clip(composite, 0.0, 100.0)), 1)

class SlidingSpectrogramScanner:
    """
    Continuous sliding-window spectrogram scanner:
    Optimized for CTF audio forensics with multi-scale windows, generous band overlaps,
    fine time-resolution hops (hop=32/64), Audacity-tuned dynamic range (55-60 dB),
    and horizontal text line clustering.
    """
    def __init__(self, channels, sample_rate, duration, output_dir="stego_output", window_sec=3.5, overlap=0.60):
        self.channels = channels
        self.sample_rate = sample_rate
        self.duration = duration
        self.output_dir = output_dir
        self.snapshots_dir = os.path.join(output_dir, "snapshots")
        os.makedirs(self.snapshots_dir, exist_ok=True)

        self.window_sec = window_sec
        self.overlap = overlap
        self.ocr = OCREngine()

        self.all_snapshots = []
        self.detected_anomalies = []

    def _stft(self, signal, n_fft=512, hop_length=32):
        """Pure NumPy Short-Time Fourier Transform."""
        window = np.hanning(n_fft)
        n_frames = 1 + int((len(signal) - n_fft) / hop_length)
        if n_frames <= 0:
            signal = np.pad(signal, (0, max(0, n_fft - len(signal))))
            n_frames = 1

        frames = np.lib.stride_tricks.as_strided(
            signal,
            shape=(n_frames, n_fft),
            strides=(signal.strides[0] * hop_length, signal.strides[0]),
            writeable=False
        )
        windowed_frames = frames * window
        spectrogram = np.fft.rfft(windowed_frames, n=n_fft, axis=-1)
        return np.abs(spectrogram).T

    @staticmethod
    def _hop_for(n_fft, duration_s, n_samples, is_full_track=False):
        """Fine-grain hop length for readable horizontal letter width."""
        if is_full_track and duration_s > 8.0:
            return max(128, int(n_samples / 1400))
        return max(32, n_fft // 16)

    def _db_image(self, signal_slice, n_fft, hop_length):
        """STFT -> 0..255 dB image anchored to the loudest channel (55 dB dynamic range)."""
        magnitude = self._stft(signal_slice, n_fft=n_fft, hop_length=hop_length)
        db = 20 * np.log10(magnitude + 1e-9)
        max_db = np.max(db)
        dyn_range = 55.0
        track_ref_db = 20 * np.log10(self.global_peak_mag * (n_fft / 2.0) + 1e-9)
        effective_max_db = max(max_db, track_ref_db - 15.0)
        min_db = effective_max_db - dyn_range
        db_norm = np.clip((db - min_db) / (dyn_range + 1e-9) * 255.0, 0, 255).astype(np.uint8)
        return db_norm, np.fft.rfftfreq(n_fft, 1.0 / self.sample_rate)

    @staticmethod
    def _render_band(db_norm, freqs, f_min, f_max):
        """
        Cuts one frequency band out of the dB image and produces the contrast, heatmap,
        inverted and threshold views plus character boxes. Returns None for silent bands.
        """
        idx_min = np.searchsorted(freqs, f_min)
        idx_max = np.searchsorted(freqs, f_max)
        idx_max = max(idx_max, idx_min + 5)
        band_slice = db_norm[idx_min:idx_max, :]

        # Signal gating: skip near-silent background dither/noise floor
        if band_slice.size == 0 or np.max(band_slice) < 90 or (np.mean(band_slice) < 0.5 and np.max(band_slice) < 110):
            return None
        img_gray = np.flipud(band_slice)

        h, w = img_gray.shape
        scale_y = max(1, 240 // max(1, h))
        scale_x = max(1, 750 // max(1, w))
        img_display = cv2.resize(img_gray, (w * scale_x, h * scale_y), interpolation=cv2.INTER_LINEAR)

        # Global contrast stretching (avoids CLAHE tile noise in silence regions)
        p10 = float(np.percentile(img_display, 10))
        p99 = float(np.percentile(img_display, 99))
        stretched = np.clip((img_display.astype(np.float32) - p10) / max(1.0, p99 - p10) * 255.0, 0, 255).astype(np.uint8)
        img_contrast = cv2.bilateralFilter(stretched, d=5, sigmaColor=35, sigmaSpace=35)

        img_heatmap = cv2.applyColorMap(img_contrast, cv2.COLORMAP_INFERNO)
        img_inverted = 255 - img_contrast

        # Adaptive thresholding for clean character segmentation
        p78 = float(np.percentile(img_contrast, 78))
        _, img_thresh = cv2.threshold(img_contrast, p78, 255, cv2.THRESH_BINARY)
        img_thresh = cv2.medianBlur(img_thresh, 3)

        # Contour detection for characters
        contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        text_boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if 35 < area < (img_thresh.shape[0] * img_thresh.shape[1] * 0.20):
                bx, by, bw, bh = cv2.boundingRect(c)
                aspect = bw / float(bh)
                if 0.20 < aspect < 6.0 and bh >= 10 and bw >= 5:
                    text_boxes.append((bx, by, bw, bh))
        return img_contrast, img_heatmap, img_inverted, img_thresh, text_boxes

    def scan(self, n_fft_list=[512, 1024]):
        """Runs sliding window analysis and captures frequent snapshots with clarity ranking."""
        nyquist = self.sample_rate // 2

        # Generously overlapping frequency bands to guarantee NO text is ever chopped at borders
        bands = [
            ("Mid_Voice_Music_Band", int(nyquist * 0.12), int(nyquist * 0.58)),  # 1,300 - 6,400 Hz (e.g. riftCTF)
            ("Ultra_High_Band", int(nyquist * 0.60), nyquist),                    # 13,000 - 22,050 Hz (Inaudible CTF text)
            ("High_Band", int(nyquist * 0.35), int(nyquist * 0.75)),              # 3,800 - 8,200 Hz
            ("Full_Spectrum", 0, nyquist)
        ]

        step_sec = max(0.5, self.window_sec * (1.0 - self.overlap))
        time_slices = []
        time_slices.append((0.0, self.duration, "Full_Track"))

        t_start = 0.0
        idx = 1
        while t_start < self.duration:
            t_end = min(t_start + self.window_sec, self.duration)
            time_slices.append((t_start, t_end, f"Window_{idx:02d}"))
            t_start += step_sec
            idx += 1
            if t_end >= self.duration:
                break

        # Global peak reference across channels for anchored dynamic range
        self.global_peak_mag = 1e-5
        for ch_sig in self.channels.values():
            if len(ch_sig):
                self.global_peak_mag = max(self.global_peak_mag, float(np.max(np.abs(ch_sig))))

        for ch_name, channel_signal in self.channels.items():
            for t_start, t_end, w_name in time_slices:
                start_sample = int(t_start * self.sample_rate)
                end_sample = int(t_end * self.sample_rate)
                signal_slice = channel_signal[start_sample:end_sample]

                if len(signal_slice) < 512:
                    continue

                for n_fft in n_fft_list:
                    hop_length = self._hop_for(n_fft, t_end - t_start, len(signal_slice), w_name == "Full_Track")
                    db_norm, freqs = self._db_image(signal_slice, n_fft, hop_length)

                    for band_name, f_min, f_max in bands:
                        rendered = self._render_band(db_norm, freqs, f_min, f_max)
                        if rendered is None:
                            continue
                        img_contrast, img_heatmap, img_inverted, img_thresh, text_boxes = rendered

                        if len(text_boxes) < 4:
                            # Not an anomaly, skip OCR and disk writes for massive speedup
                            continue

                        # Run OpenCV Word Banner & Letter Detection
                        best_banner, word_w, letter_boxes, is_drum_blast = detect_opencv_word_banner(img_contrast, img_thresh)
                        if is_drum_blast:
                            continue

                        clarity_score = compute_text_clarity_score(img_contrast, img_thresh, text_boxes)
                        if best_banner:
                            # Genuine horizontal word banner detected!
                            clarity_score = max(clarity_score, 98.0 + min(2.0, word_w / 500.0))
                        elif clarity_score < 30.0:
                            continue

                        # Run targeted OCR on promising candidates
                        ocr_res = self.ocr.scan_image(img_contrast, use_tesseract=False)
                        if ocr_res["qr_detected"] or bool(ocr_res["decoded_candidates"]):
                            clarity_score = max(clarity_score, 95.0)

                        anomaly_detected = True

                        # Save snapshot images only for verified anomalies
                        ss_base = f"ss_{ch_name}_{w_name}_{band_name}_fft{n_fft}"
                        img_heat_path = os.path.join(self.snapshots_dir, f"{ss_base}_heatmap.png")
                        img_inv_path = os.path.join(self.snapshots_dir, f"{ss_base}_inverted.png")
                        img_thresh_path = os.path.join(self.snapshots_dir, f"{ss_base}_thresh.png")
                        img_banner_path = os.path.join(self.snapshots_dir, f"{ss_base}_banner.png")

                        img_annotated = img_heatmap.copy()
                        for bx, by, bw, bh in text_boxes:
                            cv2.rectangle(img_annotated, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                        if best_banner:
                            cv2.rectangle(img_annotated, (best_banner[0], best_banner[1]), 
                                          (best_banner[0] + best_banner[2], best_banner[1] + best_banner[3]), (0, 210, 255), 3)

                        safe_imwrite(img_heat_path, img_annotated)
                        safe_imwrite(img_inv_path, img_inverted)
                        safe_imwrite(img_thresh_path, img_thresh)

                        # Generate zoomed high-contrast cropped banner for instant naked-eye human reading
                        banner_rel = None
                        banner_crop = None
                        if best_banner:
                            min_x = max(0, best_banner[0] - 15)
                            max_x = min(img_contrast.shape[1], best_banner[0] + best_banner[2] + 15)
                            min_y = max(0, best_banner[1] - 12)
                            max_y = min(img_contrast.shape[0], best_banner[1] + best_banner[3] + 12)
                            banner_crop = img_contrast[min_y:max_y, min_x:max_x]
                        else:
                            best_cl = find_best_text_line(text_boxes)
                            if best_cl:
                                min_x = max(0, min(b[0] for b in best_cl) - 15)
                                max_x = min(img_contrast.shape[1], max(b[0] + b[2] for b in best_cl) + 15)
                                min_y = max(0, min(b[1] for b in best_cl) - 12)
                                max_y = min(img_contrast.shape[0], max(b[1] + b[3] for b in best_cl) + 12)
                                banner_crop = img_contrast[min_y:max_y, min_x:max_x]

                        if banner_crop is not None and banner_crop.size > 0 and banner_crop.shape[0] >= 8 and banner_crop.shape[1] >= 20:
                            h_c, w_c = banner_crop.shape
                            scale_f = max(2.0, 150.0 / h_c)
                            zoomed_banner = cv2.resize(banner_crop, (int(w_c * scale_f), int(h_c * scale_f)), interpolation=cv2.INTER_CUBIC)
                            safe_imwrite(img_banner_path, zoomed_banner)
                            banner_rel = os.path.relpath(img_banner_path, self.output_dir).replace("\\", "/")

                        # Render smooth, high-contrast, human-readable ASCII preview
                        ascii_src = banner_crop if banner_crop is not None else img_contrast
                        ascii_preview = render_ascii_crop(ascii_src, text_boxes=text_boxes)

                        snapshot_meta = {
                            "channel": ch_name,
                            "window_label": w_name,
                            "time_start": round(t_start, 2),
                            "time_end": round(t_end, 2),
                            "time_str": f"{t_start:.2f}s - {t_end:.2f}s",
                            "band_name": band_name,
                            "f_min": int(f_min),
                            "f_max": int(f_max),
                            "n_fft": n_fft,
                            "num_clusters": len(text_boxes),
                            "banner_bbox": best_banner,
                            "image_width": int(img_contrast.shape[1]),
                            "word_width": word_w,
                            "clarity_score": clarity_score,
                            "anomaly_detected": True,
                            "ocr_result": ocr_res,
                            "decoded_candidates": ocr_res["decoded_candidates"],
                            "ascii_art": ascii_preview,
                            "banner_file": banner_rel,
                            "heatmap_file": os.path.relpath(img_heat_path, self.output_dir).replace("\\", "/"),
                            "inverted_file": os.path.relpath(img_inv_path, self.output_dir).replace("\\", "/"),
                            "thresh_file": os.path.relpath(img_thresh_path, self.output_dir).replace("\\", "/")
                        }

                        self.all_snapshots.append(snapshot_meta)
                        self.detected_anomalies.append(snapshot_meta)

        # Compute Top Highlights with Non-Maximum Suppression (NMS)
        top_highlights = self._extract_top_highlights(self.detected_anomalies, top_n=5)

        # Re-render the best candidates with full text in frame, then read them with OCR
        seen_ocr = set()
        for rank, h in enumerate(top_highlights):
            try:
                self._refine_highlight(h, rank, seen_ocr)
            except Exception as exc:  # refinement must never break the scan
                h["refine_error"] = str(exc)
        self._merge_ocr_across_highlights(top_highlights)

        return self.all_snapshots, self.detected_anomalies, top_highlights

    @staticmethod
    def _merge_ocr_across_highlights(highlights):
        """
        Several highlights often show the same banner (other channel / FFT size).
        Readings with the same letter sequence (ignoring '_' and case) are merged and only
        the one backed by the most OCR votes is kept.
        """
        from difflib import SequenceMatcher

        def key(c):
            return c["decoded"].split("{", 1)[-1].rstrip("}").replace("_", "").lower()

        groups = []  # each: [representative_key, best_candidate]
        for h in highlights:
            for c in h.get("decoded_candidates", []):
                if "ocr_votes" not in c:
                    continue
                k = key(c)
                for g in groups:
                    if SequenceMatcher(None, g[0], k).ratio() >= 0.8:
                        rank_c = (FLAG_REGEX.fullmatch(c["decoded"]) is not None, c["ocr_votes"], len(c["decoded"]))
                        rank_b = (FLAG_REGEX.fullmatch(g[1]["decoded"]) is not None, g[1]["ocr_votes"], len(g[1]["decoded"]))
                        if rank_c > rank_b:
                            g[1] = c
                        break
                else:
                    groups.append([k, c])
        winners = {id(g[1]) for g in groups}
        for h in highlights:
            h["decoded_candidates"] = [c for c in h.get("decoded_candidates", []) if "ocr_votes" not in c or id(c) in winners]

    @staticmethod
    def _expand_to_text_line(banner, img_thresh, max_gap_factor=1.2, min_v_overlap=0.6):
        """
        The word-banner detector returns the single widest contour, which is only one
        word of a multi-word flag ('spectro' out of 'flag{ocr_spectro}'). Grow it with
        every contour that sits on the same text line and is chained to it by gaps
        no wider than `max_gap_factor` x line height.
        """
        if banner is None:
            return None
        cnts, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) > 20]
        x0, y0, x1, y1 = banner[0], banner[1], banner[0] + banner[2], banner[1] + banner[3]
        # The seed can be a flat '_' (widest single contour when wide letters stay separate).
        # Taller neighbours overlapping it vertically define the real letter band.
        seed_w, seed_h = x1 - x0, y1 - y0
        near = [b for b in boxes
                if seed_h < b[3] <= 4 * seed_h
                and min(y1, b[1] + b[3]) > max(y0, b[1])
                and max(b[0] - x1, x0 - (b[0] + b[2]), 0) <= 2 * seed_w]
        if len(near) >= 2:
            y0 = min(y0, int(np.median([b[1] for b in near])))
            y1 = max(y1, int(np.median([b[1] + b[3] for b in near])))
        line_h = max(1, y1 - y0)
        # The text line's vertical band is fixed by the seed banner so the line cannot
        # drift into neighbouring music partials while it grows sideways.
        band_lo, band_hi = y0 - 0.25 * line_h, y1 + 0.25 * line_h
        glyph_widths = [banner[2]]
        used = set()
        changed = True
        while changed:
            changed = False
            # Time-axis stretch is arbitrary, so glyph spacing scales with glyph width, not height
            gap_limit = max(max_gap_factor * line_h, 0.6 * float(np.median(glyph_widths)))
            for i, (bx, by, bw, bh) in enumerate(boxes):
                if i in used:
                    continue
                if bx >= x0 and bx + bw <= x1 and by >= y0 and by + bh <= y1:
                    used.add(i)
                    continue  # already inside
                full_glyph = (0.5 * line_h <= bh <= 1.8 * line_h and
                              min(y1, by + bh) - max(y0, by) >= min_v_overlap * min(bh, line_h))
                punctuation = bh < 0.5 * line_h and by >= band_lo and by + bh <= band_hi  # '_' '.' ','
                if not (full_glyph or punctuation):
                    continue
                gap = max(bx - x1, x0 - (bx + bw), 0)
                if gap <= gap_limit:
                    x0, x1 = min(x0, bx), max(x1, bx + bw)
                    if full_glyph:
                        y0, y1 = max(min(y0, by), int(band_lo)), min(max(y1, by + bh), int(band_hi))
                        glyph_widths.append(bw)
                    used.add(i)
                    changed = True
        return (x0, y0, x1 - x0, y1 - y0)

    @staticmethod
    def _text_column_span(img_contrast, line, min_contrast=12.0):
        """
        Trims a text line to the columns that really contain text.
        Text is brighter than the rows just above and below it, whereas music/noise
        texture is roughly uniform vertically. The per-column (band - context)
        brightness difference is smoothed, thresholded and the run that overlaps the
        seed line is returned as (x0, x1). Returns None when no such run exists.
        """
        x, y, w, h = line
        H, W = img_contrast.shape[:2]
        img = img_contrast.astype(np.float32)
        band = img[y:y + h]
        above = img[max(0, y - h):y]
        below = img[y + h:min(H, y + 2 * h)]
        ctx = np.vstack([a for a in (above, below) if a.size]) if (above.size or below.size) else None
        if band.size == 0 or ctx is None:
            return None
        # 85th percentile, not mean: a thin '_' lights only a few rows of the band strongly,
        # while music texture raises band and context alike.
        diff = np.percentile(band, 85, axis=0) - np.percentile(ctx, 85, axis=0)
        k = max(3, int(h) | 1)
        diff = np.convolve(diff, np.ones(k) / k, mode="same")
        thr = max(min_contrast, 0.35 * float(np.percentile(diff, 98)))
        mask = (diff > thr).astype(np.uint8)

        def find_runs(m):
            out, start = [], None
            for i, v in enumerate(np.append(m, 0)):
                if v and start is None:
                    start = i
                elif not v and start is not None:
                    out.append((start, i))
                    start = None
            return out

        # Bridge letter / word spacing. Glyph width depends on the arbitrary time stretch,
        # so the bridge follows the measured glyph runs as well as the line height.
        raw_runs = [r for r in find_runs(mask) if r[1] - r[0] >= 3]
        glyph_w = float(np.median([r[1] - r[0] for r in raw_runs])) if raw_runs else 0.0
        bridge = max(int(1.5 * h), int(0.8 * glyph_w), 3)
        mask = cv2.morphologyEx(mask[None, :], cv2.MORPH_CLOSE, np.ones((1, bridge), np.uint8))[0]
        runs = find_runs(mask)
        if not runs:
            return None
        seed_mid = x + w / 2.0
        overlapping = [r for r in runs if r[0] <= x + w and r[1] >= x]
        pick = max(overlapping or runs, key=lambda r: (r[0] <= seed_mid <= r[1], r[1] - r[0]))
        return pick

    @staticmethod
    def _banner_crop(img_contrast, best_banner, text_boxes):
        """Crops the banner (or best aligned text line) with a small margin."""
        H, W = img_contrast.shape[:2]
        if best_banner:
            x0, x1 = best_banner[0] - 15, best_banner[0] + best_banner[2] + 15
            y0, y1 = best_banner[1] - 12, best_banner[1] + best_banner[3] + 12
        else:
            line = find_best_text_line(text_boxes)
            if not line:
                return None
            x0, x1 = min(b[0] for b in line) - 15, max(b[0] + b[2] for b in line) + 15
            y0, y1 = min(b[1] for b in line) - 12, max(b[1] + b[3] for b in line) + 12
        crop = img_contrast[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
        return crop if crop.size and crop.shape[0] >= 8 and crop.shape[1] >= 20 else None

    def _render_window(self, channel, t0, t1, n_fft, f_min, f_max, is_full_track=False):
        signal = self.channels[channel]
        sl = signal[int(t0 * self.sample_rate):int(t1 * self.sample_rate)]
        if len(sl) < max(512, n_fft):
            return None
        hop = self._hop_for(n_fft, t1 - t0, len(sl), is_full_track)
        db_norm, freqs = self._db_image(sl, n_fft, hop)
        return self._render_band(db_norm, freqs, f_min, f_max)

    def _refine_highlight(self, h, rank, seen_ocr, max_extensions=2, edge_px=20):
        """
        1. If the word banner touches the left/right edge of its window, the text continues
           outside it: widen the window by half a window on that side and re-render
           (up to `max_extensions` times) so the whole flag is in one image.
        2. Run multi-variant OCR on the final banner crop and attach the result.
        """
        if h.get("channel") not in self.channels:
            return
        t0, t1 = float(h["time_start"]), float(h["time_end"])
        is_full = h.get("window_label") == "Full_Track"
        args = (h["n_fft"], h["f_min"], h["f_max"])
        final = None
        extended = False
        for _ in range(max_extensions + 1):
            rendered = self._render_window(h["channel"], t0, t1, *args, is_full_track=is_full)
            if rendered is None:
                break
            img_contrast, _, _, img_thresh, text_boxes = rendered
            banner, _, _, drum = detect_opencv_word_banner(img_contrast, img_thresh)
            if drum or (banner is None and final is not None):
                break  # widening lost the banner: keep the last good render
            banner = self._expand_to_text_line(banner, img_thresh)
            if banner is not None:
                span = self._text_column_span(img_contrast, banner)
                if span is not None:  # cut away music texture the contour chaining picked up
                    banner = (span[0], banner[1], span[1] - span[0], banner[3])
            final = (t0, t1, img_contrast, banner, text_boxes)
            if banner is None:
                break
            W = img_contrast.shape[1]
            grow_left = banner[0] <= edge_px and t0 > 0.0
            grow_right = banner[0] + banner[2] >= W - edge_px and t1 < self.duration
            if not (grow_left or grow_right):
                break
            step = self.window_sec / 2.0
            t0 = max(0.0, t0 - step) if grow_left else t0
            t1 = min(self.duration, t1 + step) if grow_right else t1
            extended = True

        if final is None:
            return
        t0, t1, img_contrast, banner, text_boxes = final
        crop = self._banner_crop(img_contrast, banner, text_boxes)
        if crop is None:
            return

        if extended:
            h["original_time_str"] = h["time_str"]
            h["time_start"], h["time_end"] = round(t0, 2), round(t1, 2)
            h["time_str"] = f"{t0:.2f}s - {t1:.2f}s"
            h["banner_extended"] = True

        # Replace the banner image with the (possibly widened) refined crop
        h_c, w_c = crop.shape
        scale_f = max(2.0, 150.0 / h_c)
        zoomed = cv2.resize(crop, (int(w_c * scale_f), int(h_c * scale_f)), interpolation=cv2.INTER_CUBIC)
        path = os.path.join(self.snapshots_dir, f"top{rank + 1}_refined_banner.png")
        if safe_imwrite(path, zoomed):
            h["banner_file"] = os.path.relpath(path, self.output_dir).replace("\\", "/")
        h["ascii_art"] = render_ascii_crop(crop)

        ocr = self.ocr.read_banner(crop)
        h["banner_ocr"] = {k: ocr[k] for k in ("available", "best_text", "flag")}
        new = [c for c in ocr["decoded_candidates"] if c["decoded"] not in seen_ocr]
        seen_ocr.update(c["decoded"] for c in new)
        if new:
            h["decoded_candidates"] = TextDecoderHub._dedupe(list(h.get("decoded_candidates", [])) + new)
            h["clarity_score"] = max(h.get("clarity_score", 0), 99.0)

    @staticmethod
    def _extract_top_highlights(anomalies, top_n=5):
        """Extracts the top clearest snapshots using Non-Maximum Suppression (NMS)."""
        sorted_candidates = sorted(anomalies, key=lambda x: (x.get("word_width", 0) >= 150, x.get("word_width", 0), x["clarity_score"], 1 if "Diff" in x["channel"] else 0), reverse=True)

        selected = []
        for cand in sorted_candidates:
            if cand["clarity_score"] < 35.0 and len(selected) >= 3:
                break

            overlap_found = False
            for s in selected:
                if s["channel"] == cand["channel"] and s["band_name"] == cand["band_name"]:
                    overlap_start = max(s["time_start"], cand["time_start"])
                    overlap_end = min(s["time_end"], cand["time_end"])
                    if overlap_end > overlap_start:
                        overlap_dur = overlap_end - overlap_start
                        min_dur = min(s["time_end"] - s["time_start"], cand["time_end"] - cand["time_start"])
                        if overlap_dur / max(0.1, min_dur) > 0.4:
                            overlap_found = True
                            break

            if not overlap_found:
                selected.append(cand)
                if len(selected) >= top_n:
                    break

        return selected
