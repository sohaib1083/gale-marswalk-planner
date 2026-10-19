"""Tiny reader for MSL Navcam PDS3 products (.LBL + .IMG) plus the camera and frame maths.

Only what the lessons need: the label's key = value pairs, the image array, the CAHV camera
model and the rover-to-site transform. No third-party PDS library required.
"""
import re
from pathlib import Path

import numpy as np

SAMPLE = Path(__file__).resolve().parent.parent / "sample"
STEM = "NLB_486270860{}LF0481570NCAM00323M1"  # sol 1000, left Navcam, linearized, full frame


# ---------- label parsing ----------

def read_label(lbl_path):
    """Return the label as a list of (group_path, key, value_text) plus a flat lookup helper."""
    text = Path(lbl_path).read_text(errors="ignore")
    text = re.sub(r"/\*.*?\*/", "", text)  # drop comments
    entries, stack, key, buf = [], [], None, ""
    for line in text.splitlines():
        if key is not None:  # continuation of a multi-line value
            buf += " " + line.strip()
            if buf.count("(") <= buf.count(")") and buf.count('"') % 2 == 0:
                entries.append((tuple(stack), key, buf.strip()))
                key = None
            continue
        m = re.match(r"\s*([\^A-Z0-9_:]+)\s*=\s*(.*)$", line)
        if not m:
            continue
        k, v = m.group(1), m.group(2).strip()
        if k in ("GROUP", "OBJECT"):
            stack.append(v)
        elif k in ("END_GROUP", "END_OBJECT"):
            stack.pop()
        elif v.count("(") > v.count(")") or v.count('"') % 2:
            key, buf = k, v
        else:
            entries.append((tuple(stack), k, v))
    return entries


def get(entries, key, group=None):
    """First value for key (optionally inside a named group/object)."""
    for path, k, v in entries:
        if k == key and (group is None or group in path):
            return v
    raise KeyError(f"{key} (group={group})")


def num_tuple(v):
    return tuple(float(x) for x in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", v.split("<")[0]))


# ---------- image reading ----------

def read_image(kind):
    """kind = 'RAS' (picture), 'XYZ' (3 bands, metres, site frame) or 'RNG' (metres)."""
    return read_product(SAMPLE / (STEM.format(kind) + ".LBL"))


def read_product(lbl):
    """Read any MSL Navcam .LBL + .IMG pair. Returns (array, label entries)."""
    lbl = Path(lbl)
    img = lbl.with_suffix(".IMG")
    e = read_label(lbl)
    record_bytes = int(get(e, "RECORD_BYTES"))
    start_record = int(num_tuple(get(e, "^IMAGE").split(",")[-1])[0])
    lines = int(get(e, "LINES", "IMAGE"))
    samples = int(get(e, "LINE_SAMPLES", "IMAGE"))
    bands = int(get(e, "BANDS", "IMAGE"))
    stype, bits = get(e, "SAMPLE_TYPE", "IMAGE"), int(get(e, "SAMPLE_BITS", "IMAGE"))
    dtype = {("MSB_INTEGER", 16): ">i2", ("IEEE_REAL", 32): ">f4"}[(stype, bits)]
    offset = (start_record - 1) * record_bytes
    count = lines * samples * bands
    data = np.fromfile(img, dtype=dtype, count=count, offset=offset).astype(np.float32)
    data = data.reshape(bands, lines, samples)  # BAND_SEQUENTIAL
    return (data[0] if bands == 1 else data), e


# ---------- camera model (CAHV) ----------

class CAHV:
    """JPL's linear camera model. C = camera centre (m), A = pointing axis,
    H and V = horizontal / vertical vectors that already include focal length and image centre.

    project:  sample = (P-C)·H / (P-C)·A ,  line = (P-C)·V / (P-C)·A
    depth  :  (P-C)·A   (distance along the pointing axis, in metres)
    """

    def __init__(self, entries):
        g = "GEOMETRIC_CAMERA_MODEL_PARMS"
        self.C, self.A, self.H, self.V = (
            np.array(num_tuple(get(entries, f"MODEL_COMPONENT_{i}", g))) for i in range(1, 5)
        )
        self.frame = get(entries, "REFERENCE_COORD_SYSTEM_NAME", g).strip('"')

    def project(self, P):
        d = P - self.C
        z = d @ self.A
        return d @ self.H / z, d @ self.V / z, z

    def rays(self, samples, lines):
        """Unit ray direction for every pixel (x = sample, y = line)."""
        x, y = np.meshgrid(np.arange(samples, dtype=np.float64), np.arange(lines, dtype=np.float64))
        hx = self.H[None, None, :] - x[..., None] * self.A
        vy = self.V[None, None, :] - y[..., None] * self.A
        r = np.cross(vy, hx)
        r /= np.linalg.norm(r, axis=-1, keepdims=True)
        r *= np.sign(r @ self.A)[..., None]  # make every ray point forward
        return r


# ---------- coordinate frames ----------

def quat_to_matrix(q):
    """Rotation matrix from a unit quaternion given scalar-first (s, x, y, z)."""
    s, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - s * z), 2 * (x * z + s * y)],
        [2 * (x * y + s * z), 1 - 2 * (x * x + z * z), 2 * (y * z - s * x)],
        [2 * (x * z - s * y), 2 * (y * z + s * x), 1 - 2 * (x * x + y * y)],
    ])


def rover_to_site(entries):
    """Rotation R and offset t such that  p_site = R @ p_rover + t."""
    g = "ROVER_COORD_SYSTEM_PARMS"
    t = np.array(num_tuple(get(entries, "ORIGIN_OFFSET_VECTOR", g)))
    q = num_tuple(get(entries, "ORIGIN_ROTATION_QUATERNION", g))
    return quat_to_matrix(q), t
