# Örnek ses dosyaları

Bu klasördeki dosyalar `tests/test_regression.py` tarafından regresyon testi olarak kullanılır.

| Dosya | Kaynak | Yöntem | Beklenen sonuç | Testte |
|---|---|---|---|---|
| `flag.wav` | NahamCon CTF — "Beep Boop" | DTMF | `flag{do_you_speak_the_beep_boop}` | ✅ |
| `wazaaa.wav` | NahamCon CTF — "Walkman" | 16-bit PCM LSB | `flag{do_that_bit_again}` | ✅ |
| `picoctf_surfing_waves.wav` | picoCTF — "Surfing the Waves" | 16 seviyeli PAM | `picoCTF{mU21C_1s_1337_115155af}` | ✅ |
| `riftctf_chall3.wav` | RiftCTF — chall3 | Spektrogramda görsel metin | `riftCTF{Music_is_Fr33d0m}` (göz ile okunur) | — |
| `sample_challenge.wav` | `demo_generator.py` ile üretildi | Spektrogram + ekli ZIP | `FLAG{4UD10_5T3G0_15_FUN}` | yanlış alarm kontrolü |

**Telif:** CTF dosyalarının hakları ilgili yarışma organizatörlerine aittir. Yalnızca eğitim ve aracın
doğruluğunu test etmek amacıyla burada tutulmaktadır. Hak sahibi talep ederse kaldırılacaktır;
bu durumda ilgili testler dosya yoksa atlanacak şekilde güncellenmelidir.

`riftctf_chall3.wav` spektrogramdaki yazıyı görüntü olarak çıkarır (OpenCV MSER); metnin otomatik okunması
(OCR) henüz güvenilir olmadığı için bu dosya otomatik teste dahil değildir.
