"""Fast checks of the maths, no downloads and no AI model needed:  pytest -q"""
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "navcam3d"), str(ROOT / "orbital")]
import pds  # noqa: E402
from align import choose_align, find_hazards, fit_align  # noqa: E402
from fetch_orbital import make_tiff  # noqa: E402


# ---------- depth scaling ----------
def test_plain_fit_recovers_inverse_depth_relation():
    z = np.linspace(2, 20, 500)
    r = 3.0 / z + 0.1                       # model output = s * (1/z) + t
    assert np.allclose(fit_align("plain", r, z)(r), z, rtol=1e-6)


def test_power_fit_wins_when_relation_is_a_power_law():
    z = np.linspace(2, 60, 2000)
    r = 5.0 * z ** -0.7
    method, f = choose_align(r, z)
    assert method == "power"
    assert np.allclose(f(r), z, rtol=1e-6)


# ---------- hazards ----------
def test_finds_a_40cm_rock_and_ignores_a_10cm_bump():
    cell = 0.15
    h = np.zeros((120, 120))
    h[40:44, 40:44] = 0.40                  # rock, 60 cm across
    h[80:84, 80:84] = 0.10                  # pebble
    found = find_hazards(h, np.ones_like(h, bool), cell)
    assert len(found) == 1
    r, c, height, area = found[0]
    assert 40 <= r <= 44 and 40 <= c <= 44 and height > 0.3 and area == pytest.approx(16 * cell * cell)


# ---------- camera model and frames ----------
class _Cam(pds.CAHV):
    def __init__(self):
        self.C = np.array([0.5, 0.2, -1.9])
        self.A = np.array([0.9, 0.1, 0.42]); self.A /= np.linalg.norm(self.A)
        self.H = np.array([200.0, 1300.0, 50.0])
        self.V = np.array([100.0, 10.0, 1300.0])


def test_ray_through_projected_pixel_hits_the_point_again():
    cam = _Cam()
    P = np.array([6.0, 1.5, 0.3])
    u, v, depth = cam.project(P)
    ray = cam.rays(int(u) + 2, int(v) + 2)[int(round(v)), int(round(u))]
    back = cam.C + ray * depth / (ray @ cam.A)
    assert np.linalg.norm(back - P) < 0.02  # within 2 cm after rounding to the nearest pixel


def test_quaternion_gives_a_proper_rotation():
    q = np.array([0.332161, -0.0754662, -0.0622262, 0.938137]); q /= np.linalg.norm(q)
    Rm = pds.quat_to_matrix(q)
    assert np.allclose(Rm @ Rm.T, np.eye(3), atol=1e-9) and np.linalg.det(Rm) == pytest.approx(1)


# ---------- NASA PDS3 file reading ----------
def test_reads_a_pds3_label_and_big_endian_image(tmp_path):
    data = np.arange(6, dtype=">i2").reshape(2, 3)
    label = ("PDS_VERSION_ID = PDS3\nRECORD_BYTES = 64\n^IMAGE = (\"T.IMG\",2)\n/* comment */\n"
             "OBJECT = IMAGE\n  LINES = 2\n  LINE_SAMPLES = 3\n  BANDS = 1\n  SAMPLE_TYPE = MSB_INTEGER\n"
             "  SAMPLE_BITS = 16\nEND_OBJECT = IMAGE\nEND\n")
    (tmp_path / "T.LBL").write_text(label)
    (tmp_path / "T.IMG").write_bytes(b"\0" * 64 + data.tobytes())
    arr, entries = pds.read_product(tmp_path / "T.LBL")
    assert np.array_equal(arr, data.astype(np.float32))
    assert pds.get(entries, "LINES", "IMAGE") == "2"


# ---------- range-read TIFF rebuilding ----------
def test_synthetic_tiff_round_trip_uncompressed_strips():
    rows = np.random.default_rng(0).random((4, 7)).astype("<f4")
    tif = make_tiff(7, 4, [r.tobytes() for r in rows], {258: (3, [32]), 259: (3, [1]), 262: (3, [1]),
                                                        277: (3, [1]), 278: (4, [1]), 284: (3, [1]), 339: (3, [3])})
    assert np.array_equal(np.array(Image.open(io.BytesIO(tif))), rows)
