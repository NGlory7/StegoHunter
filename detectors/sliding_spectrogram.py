import os
import cv2
import numpy as np
from detectors.ocr_engine import OCREngine
from decoders.text_decoders import TextDecoderHub

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
        global_peak_mag = 1e-5
        for ch_sig in self.channels.values():
            global_peak_mag = max(global_peak_mag, float(np.max(np.abs(ch_sig))))

        for ch_name, channel_signal in self.channels.items():
            for t_start, t_end, w_name in time_slices:
                start_sample = int(t_start * self.sample_rate)
                end_sample = int(t_end * self.sample_rate)
                signal_slice = channel_signal[start_sample:end_sample]

                if len(signal_slice) < 512:
                    continue

                for n_fft in n_fft_list:
                    # Fine-grain hop length for readable horizontal letter width
                    if w_name == "Full_Track" and (t_end - t_start) > 8.0:
                        hop_length = max(128, int(len(signal_slice) / 1400))
                    else:
                        hop_length = max(32, n_fft // 16)
                    magnitude = self._stft(signal_slice, n_fft=n_fft, hop_length=hop_length)

                    db = 20 * np.log10(magnitude + 1e-9)
                    max_db = np.max(db)
                    dyn_range = 55.0
                    track_ref_db = 20 * np.log10(global_peak_mag * (n_fft / 2.0) + 1e-9)
                    effective_max_db = max(max_db, track_ref_db - 15.0)
                    min_db = effective_max_db - dyn_range
                    db_norm = np.clip((db - min_db) / (dyn_range + 1e-9) * 255.0, 0, 255).astype(np.uint8)
                    freqs = np.fft.rfftfreq(n_fft, 1.0 / self.sample_rate)

                    for band_name, f_min, f_max in bands:
                        idx_min = np.searchsorted(freqs, f_min)
                        idx_max = np.searchsorted(freqs, f_max)
                        idx_max = max(idx_max, idx_min + 5)

                        band_slice = db_norm[idx_min:idx_max, :]

                        # Signal gating: skip near-silent background dither/noise floor
                        if np.max(band_slice) < 90 or (np.mean(band_slice) < 0.5 and np.max(band_slice) < 110):
                            continue
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
                        ocr_res = self.ocr.scan_image(img_contrast)
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

        return self.all_snapshots, self.detected_anomalies, top_highlights

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
