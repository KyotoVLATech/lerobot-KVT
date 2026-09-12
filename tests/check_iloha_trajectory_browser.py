"""Browser integration checks against an isolated local trajectory server.

uv run --no-project --with pyarrow --with numpy --with playwright python tests/check_iloha_trajectory_browser.py
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from iloha_trajectory_replay import build_trajectory, parse_settings  # noqa: E402
from iloha_trajectory_web import load_episode  # noqa: E402


def check_settings_replay(document, datasets_root):
    """The robot side must rebuild the previewed frames from the settings alone."""
    settings = parse_settings(document)

    def source(name, episode):
        data = load_episode(datasets_root, name, episode)
        return np.asarray(data["actions"], dtype=np.float64), float(data["fps"])

    trajectory = build_trajectory(settings, source)
    assert len(trajectory["actions"]) == document["trajectory"]["frames"], (
        len(trajectory["actions"]), document["trajectory"]["frames"])
    assert trajectory["duration"] == document["trajectory"]["duration"]
    assert trajectory["boundaries"] == document["trajectory"]["boundaries"]
    return trajectory


def check(url, datasets_root):
    screenshots = Path(".cache/trajectory-web")
    screenshots.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1550, "height": 1060}, device_scale_factor=1)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("console", lambda message: errors.append(f"console.{message.type}: {message.text}")
                if message.type == "error" and "favicon" not in message.text else None)
        page.goto(url, wait_until="networkidle")
        page.wait_for_function("window.ilohaStudio && !document.getElementById('export').disabled", timeout=60000)
        assert page.locator(".clip").count() == 2
        assert page.locator("#timeline-canvas").bounding_box()["width"] > page.locator("#viewport").bounding_box()["width"] * .8
        initial = page.evaluate("({duration:ilohaStudio.result.duration,frames:ilohaStudio.clips.map(c=>c.source.actions.length)})")
        assert initial["frames"] == [4991, 1101], initial
        for key in ("actual", "actualPaths", "stats", "currents", "jointStats", "modelVersion"):
            assert page.evaluate(f"!({key!r} in ilohaStudio.result)"), key
        print("Initial:", json.dumps(initial))
        page.locator("#scrub").evaluate("(el,t) => { el.value=t; el.dispatchEvent(new Event('input',{bubbles:true})); }", initial["duration"] * .65)
        page.wait_for_timeout(120)
        assert page.evaluate("ilohaStudio.currentTime > 0")
        page.screenshot(path=str(screenshots / "initial.png"), full_page=True)
        advanced = page.locator("#scene").screenshot()
        page.locator("#scrub").focus()
        page.keyboard.press("Home")
        page.wait_for_timeout(120)
        assert page.evaluate("ilohaStudio.currentTime === 0")
        assert page.locator("#scene").screenshot() != advanced

        def edit_and_wait(action):
            page.evaluate("window.previousResult = ilohaStudio.result")
            action()
            page.wait_for_function("ilohaStudio.result !== window.previousResult && !document.getElementById('export').disabled", timeout=60000)

        edit_and_wait(lambda: page.locator("#normal-speed").click())
        baseline = page.evaluate("ilohaStudio.result.duration")
        edit_and_wait(lambda: page.locator('[data-clip="0"] [data-replay="base_speed"]').fill("2"))
        assert page.evaluate("ilohaStudio.clips[1].replay.base_speed === 1")
        assert page.evaluate("ilohaStudio.result.duration") < baseline
        edit_and_wait(lambda: page.locator('[data-clip="1"] [data-replay="max_speedup"]').fill("3"))
        edit_and_wait(lambda: page.locator('[data-clip="0"] input[type="number"][data-trim="end"]').fill("30"))
        edit_and_wait(lambda: page.locator('[data-clip="1"] input[type="number"][data-trim="start"]').fill("2"))
        for mode in ("linear", "smooth", "crossfade", "cut"):
            edit_and_wait(lambda mode=mode: page.locator("#transition").select_option(mode))
            assert page.evaluate("ilohaStudio.result.boundaries[0].mode") == mode
        edit_and_wait(lambda: page.locator("#transition").select_option("smooth"))
        page.locator("#boundary").click()
        page.locator("#play").click()
        page.wait_for_timeout(250)
        page.locator("#play").click()
        page.locator('[data-view="top"]').click()
        page.locator('[data-view="perspective"]').click()
        page.locator("#show-right").uncheck()
        page.locator("#show-right").check()

        page.locator("#export").click()
        assert "フレーム" in page.locator("#export-summary").inner_text()
        with page.expect_download() as download:
            page.locator("#export-settings").click()
        document = json.loads(Path(download.value.path()).read_text(encoding="utf-8"))
        assert download.value.suggested_filename == "trajectory_settings.json"
        assert document["schema_version"] == 3
        assert document["speed_applied"] is True
        assert document["clips"][0]["replay"]["base_speed"] == 2
        assert document["clips"][1]["start"] == 2
        assert document["edit"]["mode"] == "smooth"
        assert document["trajectory"]["frames"] == page.evaluate("ilohaStudio.trajectory.actions.length")
        for key in ("actuator", "simulation_model", "actuator_limits_applied", "actions"):
            assert key not in document, key
        # The settings file alone must reproduce the previewed motion off-browser.
        trajectory = check_settings_replay(document, datasets_root)
        preview = np.asarray(page.evaluate("ilohaStudio.trajectory.actions"), dtype=np.float64)
        assert np.max(np.abs(preview - trajectory["actions"])) == 0.0
        print(f"Settings replay matches the preview exactly: {len(trajectory['actions'])} frames, "
              f"{trajectory['duration']:.2f} s")
        page.locator("#close-export").click()

        page.screenshot(path=str(screenshots / "edited.png"), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path=str(screenshots / "mobile.png"), full_page=True)
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
        assert not errors, errors
        print("Browser checks passed: load, seek/rewind, per-clip speed, trim, four transitions, "
              "play, views, settings export replayed bit-for-bit in Python, mobile.")
        browser.close()


def main():
    # Validate against an isolated copy, never in user datasets.
    repo = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="iloha-browser-") as tmp:
        root = Path(tmp)
        for name in ("iloha-best", "iloha-common"):
            shutil.copytree(repo / "datasets" / name, root / name)
        server = subprocess.Popen([sys.executable, str(repo / "iloha_trajectory_web.py"), "--datasets-root", str(root), "--port", "0"],
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
                                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            url = server.stdout.readline().strip().split(" ")[-1]
            assert url.startswith("http://"), url
            check(url, root)
        finally:
            server.terminate()
            server.wait(timeout=10)


if __name__ == "__main__":
    main()
