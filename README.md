# 🛡️ AudioStegoHunter & Forensics Suite

[![test](https://github.com/NGlory7/StegoHunter/actions/workflows/test.yml/badge.svg)](https://github.com/NGlory7/StegoHunter/actions/workflows/test.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Security: Zip Bomb Protected](https://img.shields.io/badge/Security-Zip_Bomb_Protected-red.svg)](#-güvenlik-ve-zip-bomb-savunması)
[![CTF: Multi-Vector Solved](https://img.shields.io/badge/CTF-Multi--Vector_Solved-orange.svg)](#-ctf-benchmark--doğrulanmış-çözümler)

Gelişmiş ses steganografisi analizi, adli bilişim (audio forensics), kayan pencere (sliding-window) spektrum radarı, OpenCV tabanlı harf tespiti ve otomatik çoklu deşifre motoru.

Yalnızca basit CTF soruları için değil; **gerçek dünyadaki ses dosyalarında (WAV/PCM) gizlenmiş şifreleri, metinleri, gömülü dosyaları ve anomali izlerini** saniyeler içinde otomatik olarak ortaya çıkarmak üzere geliştirilmiştir.

---

## 🚀 Öne Çıkan Yetenekler

### 1. 🔍 OpenCV MSER Harf Tespiti & Kayan Pencere Radarı
* **Sürekli Mikro-Pencere Taraması:** Ses dosyasını tek parça sıkıştırmak yerine 3.5 saniyelik pencereler (%50 örtüşmeli) halinde tarar.
* **OpenCV MSER Harf & Kelime Pankartı Tespiti:** Görüntüdeki harfleri taban çizgisi uyumu (baseline alignment) ve MSER ile tespit eder.
* **Davul Darbesi ve Gürültü Filtresi:** Tüm frekansı dikey kaplayan davul/zil patlamalarını eler; gerçek metin içeren yatay bantları öne çıkarır.
* **Otomatik Büyüteç Pankartı (Auto-Crop Banner):** Bulunan flag kesitini yüksek kontrastlı ve insan gözüyle rahatça okunabilir 1600+ px genişliğinde bir pankart olarak raporun en tepesine (**#1 VIP Highlight**) yerleştirir.
* **Stereo Fark Kanalı (`Channel_Diff_LR`):** Sol ve sağ kanalları birbirinden çıkararak merkezdeki vokal ve müzikleri yok eder, yalnızca gizli stego katmanını izole eder.

### 2. 🔑 Otomatik Çok Modlu Sinyal Dekoderleri (Multimodal Decoders)
* **DTMF Tuş Takımı & BigInt Çözücü:** Telefon tuşlama seslerini (DTMF) FFT ile okur; büyük tamsayı (BigInt) $\rightarrow$ Hex $\rightarrow$ ASCII dönüşümüyle bayrağı otomatik bulur.
* **Mors Alfabesi Dekoderi:** Ses dalgasındaki bip seslerini enerji zarfı analiziyle çözer; nokta/çizgi uzunlukları 2-kümeli k-means ile ayrılır, 25 ms'den kısa tıklama/davul darbeleri elenir.
* **Binary FSK Dekoderi:** İki tonlu frekans kaydırmalı modülasyonu (modem / Bell 202 tarzı) 1 ms çözünürlükle izler, baud hızını ve mark/space frekanslarını otomatik tahmin edip bit akışını metne çevirir.
* **Ultrasonik Bant Profili:** 15 kHz üstündeki enerji oranını ölçer; spektrogram yazısı ve gizli taşıyıcılar için erken uyarı verir.
* **PAM (Pulse Amplitude Modulation) Dekoderi:** Ses örneklerindeki 16 seviyeli (hex) ve 2 seviyeli (binary) ayrık genlik basamaklarını okuyup bayrağı çözer.
* **Kapsamlı Metin Dekoderi:** Base64, Base32, Hex, Binary (MSB/LSB-first), ondalık ASCII, ROT13, Sezar brute-force (25 kaydırma), Atbash ve tersine çevrilmiş (reversed) bayrak desenlerini (`flag{...}`, `picoCTF{...}`, `riftCTF{...}`) yakalar.
* **DTMF Karışık Genişlikli ASCII:** Ayraçsız `70 76 65 71 123 ...` (2 ve 3 haneli) ondalık kod dizilerini de çözer.

### 3. 🔬 LSB (En Önemsiz Bit) & Shannon Entropi Analizi
* 16-bit PCM ses örneklerinin LSB (Bit 0, Bit 1, Bit 2) katmanlarını inceler.
* **Shannon Entropisi:** Entropi ~8.00 seviyesine çıktığında (şifreli/sıkıştırılmış veri gömülme ihtimali) kırmızı alarm verir.
* **2D Bit-Plane Görselleştirmesi:** LSB bitlerini 2 boyutlu siyah-beyaz piksel matrisi (`lsb_bitplane_bit0.png`) olarak görselleştirir.
* **LSB İçinde Gömülü Dosya:** Bit düzlemine doğrudan yazılmış ZIP/PNG/PDF/ELF gibi dosya imzalarını bulur.
* 8/16/24/32-bit PCM desteği.

### 4. 🛡️ Güvenlik ve Zip Bomb Savunması (Decompression Shield)
* Dosyaların sonuna (slack space) veya içine gömülmüş arşivleri incelerken **Zip Bomb, Decompression Bomb ve DoS saldırılarına** karşı koruma sağlar:
  * **Sıkıştırma Oranı Kontrolü:** Maksimum 25:1 oran limiti ile bellek patlamalarını önler.
  * **Toplam Boyut Kotası:** 50 MB üzeri açılacak verileri diske yazmadan engeller.
  * **Dizin Atlama (Path Traversal) Koruması:** `../../` tarzı zararlı dosya yollarını filtreler.
  * **Akışkan Çıkartma:** Dosyaları belleğe tek seferde yüklemeden parça parça işler.

### 5. 📂 Adli Dosya Ayıklama (Slack Space & File Carving)
* WAV başlığındaki beyan edilen boyut ile gerçek dosya boyutu arasındaki fazlalık byte'ları (Slack Space) tespit eder.
* ZIP, RAR, 7Z, PNG, JPEG, PDF, GIF, GZIP, BZIP2, XZ, ELF, Ogg, FLAC gibi gömülü dosyaları güvenle ayıklar (`carved_payloads/`).
* Slack alanının başında değil, **içinde herhangi bir offset'te** (ör. dolgu + ZIP) saklanan dosyaları da bulur.

### 6. 🏷️ RIFF Metadata & Başlık Adli Analizi
* LIST/INFO (başlık, sanatçı, yorum) ve diğer chunk'lardaki gizli bayrakları tüm metin dekoderlerinden geçirir.
* RIFF boyut tutarsızlığı, standart dışı chunk kimlikleri ve chunk içine gömülü dosya imzaları için alarm üretir.
* IEEE float (32/64-bit) ve `WAVE_FORMAT_EXTENSIBLE` WAV dosyalarını doğru okur.

---

## 🏆 CTF Benchmark & Doğrulanmış Çözümler

DTMF, LSB ve PAM çözümleri her push'ta `tests/test_regression.py` ile otomatik doğrulanır. Ayrıca `tests/synth.py` içindeki deterministik sinyal üreticileriyle her dedektör (Morse, DTMF, PAM, FSK, LSB, carving, metadata, zip-bomb) için 160+ birim ve uçtan uca test çalışır:

```powershell
python -m pytest -q
```

 Örnek dosyaların kaynakları ve telif notu: [samples/README.md](samples/README.md).

Suite içindeki tüm modüller gerçek dünya CTF meydan okumalarında başarıyla test edilmiş ve doğrulanmıştır:

| Challenge & CTF | Stego Vektörü | Çözülen Flag |
| :--- | :--- | :--- |
| **RiftCTF (chall3)** | Görsel Spektrogram (OpenCV MSER) | `riftCTF{Music_is_Fr33d0m}` (spektrogram görüntüsünden göz ile okunur) |
| **NahamCon CTF (Beep Boop)** | DTMF Tuş Takımı & BigInt / Hex | `flag{do_you_speak_the_beep_boop}` |
| **NahamCon CTF (Walkman)** | 16-bit PCM LSB Bit Düzlemi | `flag{do_that_bit_again}` |
| **PicoCTF (Surfing Waves)** | 16-Seviyeli PAM Genlik Modülasyonu | `picoCTF{mU21C_1s_1337_115155af}` |

---

## 📦 Kurulum

1. Repoyu klonlayın:
```bash
git clone https://github.com/KULLANICI_ADINIZ/AudioStegoHunter.git
cd AudioStegoHunter
```

2. Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```

---

## 💻 Kullanım

### 🔹 Temel Analiz ve Tarayıcıda Otomatik Rapor Açma
```powershell
python stego_hunter.py samples/riftctf_chall3.wav --open
```

### 🔹 Özel Çıktı Klasörü Belirleme
```powershell
python stego_hunter.py samples/flag.wav --out stego_output_beepboop --open
```

### 🔹 Kayan Pencere Parametrelerini Ayarlama
```powershell
python stego_hunter.py ses_dosyasi.wav --window 3.5 --overlap 0.5
```

### 🔹 Sentetik Çok Vektörlü Stego Dosyası Üretip Test Etme (`--demo`)
```powershell
python stego_hunter.py --demo --flag "FLAG{super_gizli_mesaj}" --open
```

---

## 📁 Proje Mimarisi

```text
AudioStegoHunter/
├── core/
│   ├── audio_loader.py       # RIFF/WAV başlık ayrıştırma, PCM ve slack space okuyucu
│   ├── crypto_hasher.py      # MD5, SHA-1, SHA-256 parmak izi hesaplayıcı
│   └── safety.py             # Zip bomb, bellek tüketimi ve dizin atlama savunması
├── detectors/
│   ├── file_carver.py        # Gömülü arşiv ve dosya kazıma (Magic bytes)
│   ├── lsb_analyzer.py       # Bit 0-2 entropi analizi ve 2D bit-plane üretimi
│   ├── metadata_scanner.py   # RIFF chunk metadata ve başlık tutarlılık analizi
│   ├── ocr_engine.py         # QR kod ve görsel sembol tarayıcı
│   ├── signal_decoders.py    # DTMF, Morse, PAM, FSK dekoderleri ve spektral bant profili
│   └── sliding_spectrogram.py # Kayan pencere radarı, OpenCV MSER harf tespiti ve büyüteç
├── decoders/
│   └── text_decoders.py      # Base64/32, Hex, Binary, ROT13, Sezar, Atbash ve Flag Regex motoru
├── ui/
│   ├── console_hud.py        # Terminal arayüzü ve Rich renkli tabloları
│   └── web_dashboard.py      # İnteraktif HTML5 adli bilişim raporu üreticisi
├── tests/                    # pytest: regresyon, sentetik sinyal ve uçtan uca testler
├── samples/                  # Doğrulanmış CTF test sesleri (flag.wav, chall3.wav, vb.)
├── demo_generator.py         # Çok katmanlı sentetik test ses üreticisi
├── stego_hunter.py           # Ana CLI çalıştırma noktası
├── requirements.txt          # Python bağımlılıkları
├── .gitignore                # Git yoksayma kuralları
├── LICENSE                   # MIT Lisansı
└── README.md                 # Proje dokümantasyonu
```

---

## 📊 Çıktılar ve Raporlar

Tarama tamamlandığında belirtilen çıktı dizininde (varsayılan: `stego_output/`):
* **`index.html`:** En belirgin harf kesitlerini (#1 VIP Kartı) içeren modern, karanlık modlu interaktif dashboard.
* **`forensic_report.json`:** Otomasyon veya CI/CD boru hatları için yapılandırılmış JSON çıktısı.
* **`snapshots/`:** Her pencere ve frekans bandı için üretilmiş yüksek kontrastlı spektrogram ve büyüteç pankartları.
* **`carved_payloads/`:** Dosyadan güvenle ayıklanan gömülü arşiv ve dokümanlar.

---

## 📜 Lisans

Bu proje [MIT Lisansı](LICENSE) kapsamında açık kaynak olarak sunulmuştur.
