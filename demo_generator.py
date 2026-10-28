import io
import wave
import zipfile
import numpy as np
import cv2

def generate_multi_stego_demo(filename="sample_challenge.wav", message="FLAG{4UD10_5T3G0_15_FUN}", duration=4.0, sample_rate=44100):
    """
    Generates a realistic multi-vector CTF & forensic challenge audio file:
    1. High-frequency hidden spectrogram text (14 kHz - 20 kHz).
    2. Background ambient sound.
    3. Trailing slack payload (appended ZIP archive with a secret note).
    """
    print(f"\n[+] Synthesizing multi-layer stego challenge: '{filename}'")
    print(f"    - Vector 1: Spectrogram watermarking: '{message}'")
    print(f"    - Vector 2: Trailing EOF Slack Space (ZIP payload with Base64 secret)")

    # 1. Canvas for Spectrogram Text
    h = 80
    w = 400
    img = np.zeros((h, w), dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.05
    thickness = 2
    text_size = cv2.getTextSize(message, font, font_scale, thickness)[0]
    text_x = max(10, (w - text_size[0]) // 2)
    text_y = (h + text_size[1]) // 2
    cv2.putText(img, message, (text_x, text_y), font, font_scale, 255, thickness, cv2.LINE_AA)
    img = np.flipud(img)

    # 2. Audio Synthesis
    total_samples = int(duration * sample_rate)
    audio = np.zeros(total_samples, dtype=np.float32)

    # Ambient audio (two gentle sine tones)
    t_full = np.linspace(0, duration, total_samples, endpoint=False)
    audio += 0.04 * np.sin(2 * np.pi * 440 * t_full) + 0.02 * np.sin(2 * np.pi * 880 * t_full)

    # Embed spectrogram text
    f_min, f_max = 14000, 20000
    freq_bins = np.linspace(f_min, f_max, h)
    samples_per_col = total_samples // w

    for col in range(w):
        active_rows = np.where(img[:, col] > 40)[0]
        if len(active_rows) == 0:
            continue
        start_idx = col * samples_per_col
        end_idx = min(start_idx + samples_per_col, total_samples)
        t_slice = t_full[start_idx:end_idx]

        slice_audio = np.zeros(len(t_slice), dtype=np.float32)
        for row in active_rows:
            slice_audio += (img[row, col] / 255.0) * np.sin(2 * np.pi * freq_bins[row] * t_slice)

        win = np.hanning(len(slice_audio))
        audio[start_idx:end_idx] += slice_audio * win * 0.15

    # Normalize audio
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.85

    # Write PCM WAV
    audio_int16 = (audio * 32767).astype(np.int16)
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

    # 3. Create a safe trailing ZIP archive with an encoded secret
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("secret_flag.txt", "SECRET_KEY: RkxBR3tTTEFDS19TUEFDRV9VUExPQURfQ0FSVkVEfQ==")
    zip_bytes = zip_buffer.getvalue()

    # Append ZIP bytes to WAV file (Slack Space Injection)
    with open(filename, 'ab') as af:
        af.write(zip_bytes)

    print(f"[✓] Created '{filename}' with {len(zip_bytes)} bytes of slack space payload appended.\n")
    return filename

if __name__ == "__main__":
    generate_multi_stego_demo()
