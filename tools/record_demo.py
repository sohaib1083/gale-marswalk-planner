"""Record a ~30 second fly-through of the running app (Space Apps asks for a 30-second video or 7 slides).

    python3 tools/record_demo.py            # -> docs/media/demo_30s.mp4
    python3 tools/record_demo.py --fps 12   # faster to render on a laptop without a free GPU

Drives the real app in headless Chromium: Mars globe -> fly to Gale -> crater in 3D -> planning area with a planned
walk -> 3D terrain with the Navcam close-ups -> all stops together. Frames are stitched with ffmpeg.
"""
import argparse
import functools
import http.server
import math
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "media"
W, H = 1280, 720
CSS = """
body { padding: 0 !important; } header, .below, .credit, .rail, .tabs { display: none !important; }
.app { display: block !important; border: 0 !important; border-radius: 0 !important; }
.view { position: fixed !important; inset: 0 !important; height: 100vh !important; }
.hint, .view-controls { display: none !important; }
#cap { position: fixed; z-index: 90; left: 32px; bottom: 30px; color: #eef2f5; font: 600 26px/1.3 'Archivo', Arial, sans-serif;
       text-shadow: 0 2px 10px rgba(0,0,0,.85); max-width: 1000px; }
#cap small { display: block; font: 500 16px/1.4 'Public Sans', Arial, sans-serif; opacity: .92; margin-top: 4px; }
#brand { position: fixed; z-index: 90; right: 28px; top: 22px; color: #eef2f5; font: 600 14px 'Martian Mono', monospace;
         letter-spacing: .06em; text-shadow: 0 1px 6px rgba(0,0,0,.8); }
"""


class Quiet(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".html": "text/html; charset=utf-8"}

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--gpu", default="vulkan", choices=["vulkan", "swiftshader"])
    a = ap.parse_args()
    frames = OUT / "frames"
    shutil.rmtree(frames, ignore_errors=True)
    frames.mkdir(parents=True)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(ROOT / "app")))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/index.html"
    args = (["--use-angle=vulkan", "--enable-features=Vulkan", "--ignore-gpu-blocklist"] if a.gpu == "vulkan"
            else ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
    n = 0
    with sync_playwright() as p:
        b = p.chromium.launch(args=args)
        pg = b.new_page(viewport={"width": W, "height": H})
        pg.goto(url)
        pg.wait_for_function("window.marswalk && window.marswalk.S.route && window.marswalk.S.globe && window.marswalk.S.globe.o", timeout=240000)
        pg.add_style_tag(content=CSS)
        pg.evaluate("""() => { for (const id of ['cap', 'brand']) { const d = document.createElement('div'); d.id = id; document.body.appendChild(d); }
                              document.getElementById('brand').textContent = 'GALE MARSWALK PLANNER · ASTRONOX'; }""")

        def caption(title, sub):
            pg.evaluate("([t, s]) => { document.getElementById('cap').innerHTML = t + '<small>' + s + '</small>'; }", [title, sub])

        def snap():
            nonlocal n
            pg.evaluate("() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")
            pg.screenshot(path=str(frames / f"{n:05d}.jpg"), type="jpeg", quality=90)
            n += 1

        def orbit(view_js, seconds, start_deg, sweep_deg, radius_js, height_js):
            steps = int(seconds * a.fps)
            for i in range(steps):
                ang = math.radians(start_deg + sweep_deg * i / max(steps - 1, 1))
                pg.evaluate(f"""(ang) => {{ const o = {view_js}; const t = o.ctl.target; const r = {radius_js}, h = {height_js};
                    o.cam.position.set(t.x + r * Math.cos(ang), t.y + h, t.z + r * Math.sin(ang)); o.ctl.update(); }}""", ang)
                snap()

        # 1. Mars (6 s)
        caption("Plan a safe walk on Mars", "Starting from the whole planet: NASA's Viking colour map, 9 landing sites")
        pg.evaluate("() => { window.marswalk.S.globe.o.ctl.autoRotate = false; }")
        steps = int(3.5 * a.fps)
        for i in range(steps):
            pg.evaluate("(k) => { const G = window.marswalk.S.globe, o = G.o; const lon = G.gale.lon - 70 + 70 * k, lat = G.gale.lat + 10 - 10 * k;"
                        " const la = lat * Math.PI / 180, ph = (lon + 180) * Math.PI / 180, r = 2.7 - 0.6 * k;"
                        " o.cam.position.set(-Math.cos(ph) * Math.cos(la) * r, Math.sin(la) * r, Math.sin(ph) * Math.cos(la) * r); o.cam.lookAt(0, 0, 0); }",
                        i / max(steps - 1, 1))
            snap()
        caption("Gale Crater: where Curiosity landed in 2012", "Flying in…")
        steps = int(2.5 * a.fps)
        for i in range(steps):
            k = i / max(steps - 1, 1)
            e = k * k * (3 - 2 * k)
            pg.evaluate("(e) => { const G = window.marswalk.S.globe, o = G.o; const la = G.gale.lat * Math.PI / 180, ph = (G.gale.lon + 180) * Math.PI / 180, r = 2.1 - 0.98 * e;"
                        " o.cam.position.set(-Math.cos(ph) * Math.cos(la) * r, Math.sin(la) * r, Math.sin(ph) * Math.cos(la) * r); o.cam.lookAt(0, 0, 0); }", e)
            snap()

        # 2. Gale Crater in 3D (6 s)
        pg.evaluate("() => document.querySelector('.tabs button[data-view=gale]').click()")
        pg.wait_for_function("window.marswalk.S.gale && window.marswalk.S.gale.o", timeout=120000)
        pg.wait_for_timeout(1500)
        caption("Gale Crater in 3D, heights from orbit", "Curiosity's whole 35 km route since landing in orange · our planning area in blue")
        orbit("window.marswalk.S.gale.o", 6, 120, 70, "17000", "10000")

        # 3. Planning area: map + safe route (6 s)
        pg.evaluate("() => document.querySelector('.tabs button[data-view=map]').click()")
        pg.wait_for_timeout(1200)
        caption("Tap a start and an end: the safest walk", "Avoids slopes over 25° and rocks over 50 cm · 2 km of real HiRISE terrain")
        for i in range(int(6 * a.fps)):
            k = i / max(int(6 * a.fps) - 1, 1)
            pg.evaluate("(k) => { const S = window.marswalk.S, c = S.data.closeups.find(x => x.sol === 1100) || S.data.closeups[0];"
                        " const r = document.getElementById('map').getBoundingClientRect(), s = 0.55 + 2.2 * k * k;"
                        " S.view.s = s; S.view.x = r.width / 2 - (c.x - 150 * k) * s; S.view.y = r.height / 2 - (c.y - 120 * k) * s;"
                        " document.getElementById('lay-haz').dispatchEvent(new Event('change')); }", k)
            snap()

        # 4. 3D terrain with close-ups (6 s)
        pg.evaluate("() => document.querySelector('.tabs button[data-view=terrain]').click()")
        pg.wait_for_timeout(3500)
        caption("The same walk in 3D", "Orbital terrain, with ground models built from Curiosity's own photos at each stop")
        orbit("window.marswalk.S.t3d_o", 6, 200, 60, "650", "420")

        # 5. Ground close-ups, all stops (6 s)
        pg.evaluate("() => document.querySelector('#closeup-list button[data-sol=all]') ? document.querySelector('#closeup-list button[data-sol=all]').click() : document.querySelector('.tabs button[data-view=close]').click()")
        pg.wait_for_timeout(6000)
        caption("Every stop in 3D, from Curiosity's photos", "NASA stereo + Depth Anything V2 · rocks and ledges marked for the astronaut")
        orbit("window.marswalk.S.close.o", 6, 230, 80, "120", "70")
        b.close()
    srv.shutdown()
    out = OUT / "demo_30s.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", str(frames / "%05d.jpg"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", "-movflags", "+faststart", str(out)], check=True)
    shutil.rmtree(frames)
    print(f"{n} frames -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
