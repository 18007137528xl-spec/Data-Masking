"""Rebuild the illustrated console guide from a live console.

    python scripts/console_guide.py

Starts `deidkit serve` on a synthetic raw EDC drop, walks all five steps in a
headless browser, photographs each screen with numbered markers, and inlines
the shots into docs/guide.template.html to produce src/deidkit/guide.html --
the page the console serves at /guide.

Why a script and not a folder of PNGs: a guide whose screenshots no longer
match the page is worse than no guide, because the reader trusts the picture
over the screen in front of them. Rerun this whenever console.html changes.

Everything in the drop is invented. The folders are created under literal
Windows-style names (D:\\quarantine\\STUDY-001) so the shots show what a
steward on the Windows server will actually type.

Needs playwright with a Chromium it can find.
"""

from __future__ import annotations

import base64
import csv
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "docs" / "guide.template.html"
TARGET = ROOT / "src" / "deidkit" / "guide.html"
W = 1200


def make_drop(folder: Path) -> None:
    """A raw LB extract built to put every console state on screen."""
    random.seed(11)
    notes = [
        "", "", "", "",
        "haemolysed sample, redrawn",
        "patient's daughter Mary Hartley phoned about the result",
        "", "repeat requested by Dr. Okafor", "",
    ]
    rows = []
    for s in range(1, 31):
        site = f"{(s % 3) + 1:03d}"
        age = random.randint(24, 79)
        for v in (1, 2, 3):
            rows.append({
                "SUBJECT": f"{site}-{s:04d}",
                "SITEID": site,
                "VISIT": f"V{v}",
                "LBDAT": f"{random.randint(1, 28):02d}/0{v + 2}/2025",
                "LBTEST": random.choice(["Sodium", "Potassium", "ALT"]),
                "LBORRES": str(random.randint(20, 150)),
                "AGE": str(age),
                "LBCOMM": random.choice(notes),
                "ENTEREDBY": random.choice(["jchen", "mrossi", "akim"]),
                "PKGRP": "",
            })
    for i, r in enumerate(rows):
        r["PKGRP"] = "A" if i % 2 else "B"
    folder.mkdir(parents=True)
    with open(folder / "LB.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


MARK_JS = """
([sel, n, where]) => {
  const el = document.querySelector(sel);
  if (!el) return false;
  const r = el.getBoundingClientRect();
  const at = {
    left:  [r.left - 20,  r.top + Math.min(r.height / 2, 20)],
    right: [r.right + 20, r.top + r.height / 2],
    rail:  [r.right - 20, r.top + r.height / 2],
  }[where];
  const b = document.createElement('div');
  b.className = '__mark';
  b.textContent = n;
  Object.assign(b.style, {
    position: 'absolute', zIndex: 9999,
    left: (window.scrollX + at[0] - 13) + 'px',
    top:  (window.scrollY + at[1] - 13) + 'px',
    width: '26px', height: '26px', borderRadius: '50%',
    background: '#6D28D9', color: '#fff', font: '700 14px/26px system-ui',
    textAlign: 'center', boxShadow: '0 0 0 3px #fff, 0 2px 6px rgba(0,0,0,.25)',
  });
  document.body.appendChild(b);
  return true;
}
"""


def mark(pg, sel, n, where="left"):
    assert pg.evaluate(MARK_JS, [sel, str(n), where]), f"marker {n}: nothing at {sel}"


def clear(pg):
    pg.evaluate("document.querySelectorAll('.__mark').forEach(e => e.remove())")


def shot(pg, name, y=0, height=760, full=False):
    pg.wait_for_timeout(250)
    path = f"{OUT}/{name}.png"
    if full:
        pg.screenshot(path=path, full_page=True)
    else:
        pg.screenshot(path=path, full_page=True,
                      clip={"x": 0, "y": y, "width": W, "height": height})
    clear(pg)
    print("shot", name)


def doc_y(pg, sel):
    return pg.evaluate(
        f"document.querySelector({sel!r}).getBoundingClientRect().top + window.scrollY"
    )


def row_sel(pg, column):
    i = pg.evaluate(
        "c => [...document.querySelectorAll('#rows tr')]"
        ".findIndex(t => t.querySelector('.col').textContent === c) + 1",
        column,
    )
    assert i, column
    return f"#rows tr:nth-child({i})"




OUT = ""  # set by shoot(); shot() writes here


def shoot(url: str, out: str) -> None:
    global OUT
    OUT = out
    URL = url
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": W, "height": 900})
        pg.goto(URL)
        pg.wait_for_timeout(500)

        # --- the red box: a one-character typo in the folder ----------------------
        pg.fill("#dir", r"D:\quarantine\STUDY-OO1")
        pg.click("#btn-profile")
        pg.wait_for_selector("#err .note.halt")
        mark(pg, "#err .note.halt", 1)
        shot(pg, "err-folder", height=330)

        # --- 01 ------------------------------------------------------------------
        pg.fill("#dir", r"D:\quarantine\STUDY-001")
        pg.check("#raw")
        pg.fill("#source", "STUDY-001")
        pg.evaluate("document.getElementById('err').innerHTML = ''")
        mark(pg, '.steps button[data-step="0"]', 1, "rail")
        mark(pg, "#dir", 2)
        mark(pg, "#raw", 3)
        mark(pg, "#source", 4)
        mark(pg, "#k", 5)
        mark(pg, "#offsetkey", 6)
        mark(pg, "#btn-profile", 7)
        shot(pg, "s1-read", height=600)

        # --- 02 ------------------------------------------------------------------
        pg.click("#btn-profile")
        pg.wait_for_selector("#rows tr")
        pg.wait_for_timeout(300)
        mark(pg, "#notes .note", 1)
        mark(pg, "#tally", 2)
        mark(pg, "#btn-bulk", 3)
        mark(pg, "#rows tr:first-child .chip", 4)
        mark(pg, "#rows tr:first-child .pick", 5)
        shot(pg, "s2-review", height=760)

        # decide two rows by hand: PKGRP -> change to drop; LBCOMM -> OK
        first = pg.locator("#rows tr").nth(0)
        first.locator('.pick button[data-v="CHANGE"]').click()
        pg.wait_for_timeout(200)
        first = pg.locator("#rows tr").nth(0)
        first.locator('select[data-f="decision_treatment"]').select_option("drop")
        first.locator('input[data-f="steward_note"]').fill(
            "randomisation stratum; not needed for this corpus")
        first.locator('input[data-f="steward_note"]').dispatch_event("change")
        pg.locator("#rows tr").nth(1).locator('.pick button[data-v="OK"]').click()
        pg.wait_for_timeout(300)
        mark(pg, '#rows tr:first-child .pick button[data-v="CHANGE"]', 1, "right")
        mark(pg, '#rows tr:first-child select', 2)
        mark(pg, '#rows tr:first-child input[data-f="decision_params"]', 3)
        mark(pg, '#rows tr:first-child input[data-f="steward_note"]', 4)
        mark(pg, '#rows tr:nth-child(2) .pick', 5)
        mark(pg, '#rows tr:nth-child(3) td', 6)
        y = doc_y(pg, ".bar") - 10
        shot(pg, "s2-change", y=y, height=560)

        # a row carrying the two new surrogate parameters
        sel = row_sel(pg, "SUBJECT")
        mark(pg, sel + " .params", 1)
        mark(pg, sel + " .why", 2)
        y = doc_y(pg, sel) - 10
        h = pg.evaluate(f"document.querySelector({sel!r}).getBoundingClientRect().height")
        shot(pg, "s2-surrogate", y=y, height=h + 20)

        # accept the rest in one go
        pg.evaluate("window.scrollTo(0,0)")
        pg.click("#btn-bulk")
        pg.wait_for_timeout(400)
        mark(pg, "#tally", 1)
        mark(pg, '.steps button[data-step="2"]', 2, "rail")
        shot(pg, "s2-done", height=360)

        # --- 03 ------------------------------------------------------------------
        pg.click('.steps button[data-step="2"]')
        pg.fill("#who", "xiaofeng.li@tigermedgrp.com")
        pg.fill("#cnote", "first drop, reviewed column by column")
        pg.fill("#cout", r"D:\contracts\STUDY-001.yaml")
        pg.click("#btn-approve")
        pg.wait_for_timeout(500)
        pg.click('.steps button[data-step="2"]')
        pg.wait_for_timeout(200)
        mark(pg, "#who", 1)
        mark(pg, "#cnote", 2)
        mark(pg, "#cout", 3)
        mark(pg, "#btn-approve", 4)
        mark(pg, "#approved .kv", 5)
        shot(pg, "s3-sign", height=700)

        # --- 04 ------------------------------------------------------------------
        pg.click('.steps button[data-step="3"]')
        pg.fill("#out", r"D:\tiers\STUDY-001-Q3")
        pg.fill("#vault", r"D:\vault\STUDY-001.db")
        pg.fill("#operator", "xiaofeng.li@tigermedgrp.com")
        mark(pg, "#out", 1)
        mark(pg, "#vault", 2)
        mark(pg, "#operator", 3)
        mark(pg, "#fmt", 4)
        mark(pg, "#btn-run", 5)
        shot(pg, "s4-run", height=640)

        # --- 05 ------------------------------------------------------------------
        pg.click("#btn-run")
        pg.wait_for_selector("#results pre.out", timeout=30000)
        pg.wait_for_timeout(400)
        # The server here is Linux, where D:\tiers\STUDY-001-Q3 is one literal
        # folder name, so the workbook path prints doubled. On Windows it reads
        # D:\tiers\STUDY-001-Q3/STUDY-001-Q3.xlsx::LB -- show that.
        pg.evaluate(r"""
          document.querySelectorAll('#results dd').forEach(d => {
            d.textContent = d.textContent.replace(
              'D:\\tiers\\STUDY-001-Q3/D:\\tiers\\STUDY-001-Q3.xlsx',
              'D:\\tiers\\STUDY-001-Q3/STUDY-001-Q3.xlsx');
          });
        """)
        mark(pg, "#results pre.out", 1)
        mark(pg, "#results .note", 2)
        mark(pg, "#results .card", 3)
        if pg.locator("#results .note.warn").count():
            mark(pg, "#results .note.warn", 4)
            mark(pg, "#btn-queue", 5)
        mark(pg, "#results .note.info", 6)
        shot(pg, "s5-results", full=True)

        if pg.locator("#btn-queue").count():
            pg.click("#btn-queue")
            pg.wait_for_selector("#queue table")
            pg.wait_for_timeout(200)
            y = doc_y(pg, "#queue") - 12
            h = pg.evaluate("document.querySelector('#queue').getBoundingClientRect().height")
            shot(pg, "s5-queue", y=y, height=min(h + 24, 420))

        # --- the discard guard: back to step 1, read again -----------------------
        pg.click('.steps button[data-step="0"]')
        pg.click("#btn-profile")
        pg.wait_for_selector("#err .note.warn")
        mark(pg, "#btn-keep", 1)
        mark(pg, "#btn-discard", 2, "right")
        shot(pg, "guard-discard", height=300)
        pg.click("#btn-keep")

        # --- refresh mid-way: the page comes back where it was -------------------
        pg.reload()
        pg.wait_for_timeout(900)
        cur = pg.evaluate(
            "document.querySelector('.steps button[aria-current=\"true\"]').textContent")
        print("after reload, current step:", cur.strip())
        b.close()


def build(shots: Path) -> None:
    html = TEMPLATE.read_text(encoding="utf-8")

    def inline(m: re.Match) -> str:
        data = (shots / f"{m.group(1)}.png").read_bytes()
        return "data:image/png;base64," + base64.b64encode(data).decode()

    html = re.sub(r"\{\{img:([\w-]+)\}\}", inline, html)
    left = re.findall(r"\{\{img:[^}]*\}\}", html)
    assert not left, f"unresolved screenshots: {left}"
    TARGET.write_text(html, encoding="utf-8")
    print(f"wrote {TARGET.relative_to(ROOT)} ({TARGET.stat().st_size // 1024} KB)")


def main() -> int:
    from deidkit.vault import Vault

    work = Path(tempfile.mkdtemp(prefix="deidkit-guide-"))
    shots = work / "_shots"
    make_drop(work / r"D:\quarantine\STUDY-001")
    port = free_port()
    env = dict(os.environ, DEIDKIT_VAULT_KEY=Vault.generate_key())
    server = subprocess.Popen(
        [sys.executable, "-m", "deidkit.cli", "serve", "--port", str(port)],
        cwd=work, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        url = f"http://127.0.0.1:{port}/"
        for _ in range(50):
            try:
                urllib.request.urlopen(url, timeout=1)
                break
            except OSError:
                time.sleep(0.2)
        shots.mkdir()
        shoot(url, str(shots))
        build(shots)
    finally:
        server.terminate()
        server.wait(timeout=10)
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
