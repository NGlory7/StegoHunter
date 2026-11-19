"""StegoHunterEngine uçtan uca: tüm aşamalar tek dosyada birden çok vektörü yakalayıp rapora yazmalı."""
import base64
import json
import zipfile
import io

import numpy as np
import pytest

import synth
from stego_hunter import StegoHunterEngine


@pytest.fixture(scope="module")
def multi_vector_report(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("engine")

    # Vector A: FSK-modulated flag; Vector B: flag in INFO comment; Vector C: ZIP in slack
    fsk = synth.fsk_signal(b"flag{engine_fsk}")
    pcm = (np.clip(fsk, -1, 1) * 32767).astype("<i2").tobytes()
    info = synth.info_list_chunk({"ICMT": base64.b64encode(b"flag{engine_meta}").decode()})
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("n.txt", "flag{engine_zip}")
    wav = tmp / "multi.wav"
    wav.write_bytes(synth.build_riff([info, (b"data", pcm)]) + buf.getvalue())

    out = tmp / "out"
    report = StegoHunterEngine(str(wav), output_dir=str(out), window_sec=1.0, overlap=0.5).run()
    with open(out / "forensic_report.json", encoding="utf-8") as f:
        return report, json.load(f)


def test_engine_writes_html_report(multi_vector_report):
    report_path, _ = multi_vector_report
    with open(report_path, encoding="utf-8") as f:
        assert "<html" in f.read()


@pytest.mark.parametrize("flag", ["flag{engine_fsk}", "flag{engine_meta}", "flag{engine_zip}"])
def test_engine_report_contains_every_vector(multi_vector_report, flag):
    _, data = multi_vector_report
    assert flag in [s["content"] for s in data["secrets_found"]]


def test_engine_report_has_new_sections(multi_vector_report):
    _, data = multi_vector_report
    assert data["metadata"]["chunks"][0]["id"] == "LIST"
    lo, hi = data["fsk"][0]["mark_space_hz"]
    assert abs(lo - 1200) < 100 and abs(hi - 2200) < 100
    assert "band_ratios" in data["spectral_bands"]


def test_engine_rejects_non_wav(tmp_path):
    bad = tmp_path / "x.wav"
    bad.write_bytes(b"not a wav at all" * 10)
    with pytest.raises(SystemExit):
        StegoHunterEngine(str(bad), output_dir=str(tmp_path / "o")).run()
