"""Browser integration checks against an isolated local trajectory server."""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright


def check(url, export_root):
    screenshots = Path(".cache/trajectory-web")
    screenshots.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1550, "height": 1060}, device_scale_factor=1)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url, wait_until="networkidle")
        page.wait_for_function("window.ilohaStudio && !document.getElementById('export').disabled", timeout=60000)
        assert page.locator(".clip").count() == 2
        assert page.locator("#timeline-canvas").bounding_box()["width"] > page.locator("#viewport").bounding_box()["width"] * .8
        initial = page.evaluate("({duration:ilohaStudio.result.motionDuration,frames:ilohaStudio.clips.map(c=>c.source.actions.length),stats:ilohaStudio.result.stats})")
        assert initial["frames"] == [4991, 1101], initial
        assert page.evaluate("ilohaStudio.result.modelVersion === 3")
        assert page.evaluate("ilohaStudio.result.jointStats.slice(0,3).map(d=>d.model).join(',') === 'RobStride 03,RobStride 06,RobStride 00'")
        assert page.locator("#model-warning").count() == 0
        assert "静止保持にも出力が不足しています" not in page.locator("body").inner_text()
        assert "RobStrideの定格トルクを超える保持区間があります" not in page.locator("body").inner_text()
        assert page.locator("#joint-diagnostics tr").count() == 6
        assert page.evaluate("ilohaStudio.settings.limits.every(l=>l.acceleration === 0)")
        assert page.locator("#gravity").count() == 0
        assert "NaN" not in page.locator("#joint-diagnostics").inner_text()
        print("J3 diagnostics:", page.evaluate("ilohaStudio.result.jointStats[2]"))
        print("Initial:", json.dumps(initial))
        page.locator("#scrub").evaluate("(el,t) => { el.value=t; el.dispatchEvent(new Event('input',{bubbles:true})); }", initial["duration"] * .65)
        page.wait_for_timeout(120)
        assert page.evaluate("ilohaStudio.currentTime > 0")
        page.screenshot(path=str(screenshots / "initial.png"), full_page=True)
        advanced = page.locator("#scene").screenshot()
        advanced_pixels = page.locator("#scene").evaluate("canvas=>canvas.toDataURL()")
        assert not page.locator("#show-actual-trace").is_checked()
        trace_time = page.evaluate("ilohaStudio.currentTime")
        page.locator("#show-actual-trace").check()
        page.wait_for_timeout(120)
        assert page.locator("#scene").evaluate("canvas=>canvas.toDataURL()") != advanced_pixels
        assert page.evaluate("ilohaStudio.currentTime") == trace_time
        page.locator("#show-actual-trace").uncheck()
        page.wait_for_timeout(120)
        assert page.locator("#scene").evaluate("canvas=>canvas.toDataURL()") == advanced_pixels
        page.locator("#scrub").focus()
        page.keyboard.press("Home")
        page.wait_for_timeout(120)
        assert page.evaluate("ilohaStudio.currentTime === 0")
        assert page.locator("#scene").screenshot() != advanced

        def edit_and_wait(action):
            page.evaluate("window.previousResult = ilohaStudio.result")
            action()
            page.wait_for_function("ilohaStudio.result !== window.previousResult && !document.getElementById('export').disabled", timeout=60000)

        original_limits = page.evaluate("ilohaStudio.settings.limits.map(l=>[l.current,l.velocity,l.acceleration])")
        edit_and_wait(lambda: page.locator("#current-basis").select_option("rms"))
        assert page.evaluate("ilohaStudio.settings.limits[2].kt === 1.48")
        assert page.evaluate("ilohaStudio.settings.limits.map(l=>[l.current,l.velocity,l.acceleration])") == original_limits
        edit_and_wait(lambda: page.locator("#current-basis").select_option("peak"))
        assert page.evaluate("Math.abs(ilohaStudio.settings.limits[2].kt - 1.48/Math.SQRT2) < 1e-10")

        # Compare the actual input datasets, without using copied observation.state
        # as measured feedback. These are model errors relative to ideal commands.
        edit_and_wait(lambda: page.locator("#normal-speed").click())
        baseline = page.evaluate("ilohaStudio.result.stats")
        assert baseline["rmsError"] < .003, baseline
        assert baseline["maxError"] < .05, baseline
        assert page.evaluate("Math.max(ilohaStudio.result.jointStats[2].maxAngleError,ilohaStudio.result.jointStats[8].maxAngleError) < .15")
        comparisons = {1: baseline}
        for speed in (2, 4):
            def set_speed(speed=speed):
                page.evaluate("speed=>{for(const c of ilohaStudio.clips){c.replay.base_speed=speed;c.replay.max_speedup=1;}document.getElementById('compute').click();}", speed)
            edit_and_wait(set_speed)
            comparisons[speed] = page.evaluate("ilohaStudio.result.stats")
        assert comparisons[2]["rmsError"] > comparisons[1]["rmsError"] * 1.5
        assert comparisons[4]["rmsError"] > comparisons[2]["rmsError"] * 2
        print("1x/2x/4x model comparison:", json.dumps(comparisons))
        edit_and_wait(lambda: page.locator("#normal-speed").click())
        edit_and_wait(lambda: page.locator("#apply-pp-limits").click())
        assert page.evaluate("ilohaStudio.settings.limits[2].acceleration === Math.PI/2")
        assert page.evaluate("ilohaStudio.result.stats.rmsError") > baseline["rmsError"]
        edit_and_wait(lambda: page.locator("#reset-limits").click())
        assert page.evaluate("ilohaStudio.settings.limits.every(l=>l.acceleration === 0)")
        page.locator("#holding-currents").locator("..").locator("summary").click()
        edit_and_wait(lambda: page.locator('[data-reserve="2"]').fill("1"))
        assert page.evaluate("Math.abs(ilohaStudio.result.jointStats[2].availableTorque - 3*ilohaStudio.settings.limits[2].kt) < 1e-9")

        edit_and_wait(lambda: page.locator('[data-clip="0"] [data-replay="base_speed"]').fill("2"))
        assert page.evaluate("ilohaStudio.clips[1].replay.base_speed === 1")
        assert page.evaluate("ilohaStudio.result.motionDuration") < initial["duration"]
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
        edit_and_wait(lambda: page.locator("#relax-limits").click())
        page.locator("#arm-tab").select_option("1")
        edit_and_wait(lambda: page.locator('[data-joint="6"][data-limit="current"]').fill("2"))
        assert page.evaluate("ilohaStudio.settings.limits[0].current !== ilohaStudio.settings.limits[6].current")
        page.locator("#export").click()
        with page.expect_download() as download_info:
            page.locator("#export-ideal-json").click()
        data = json.loads(Path(download_info.value.path()).read_text(encoding="utf-8"))
        assert data["coordinates"] == "iloha"
        assert len(data["actions"][0]) == 14
        assert data["fps"] == 30
        page.locator("#export-name").fill("browser-merged")
        page.locator("#export-dataset").click()
        page.wait_for_function("document.getElementById('export-result').textContent.includes('trajectory_settings.json')", timeout=30000)
        assert (export_root / "browser-merged/data/chunk-000/file-000.parquet").is_file()
        saved = json.loads((export_root / "browser-merged/trajectory_settings.json").read_text(encoding="utf-8"))
        assert saved["speed_applied"] is False
        assert saved["actuator_limits_applied"] is False
        assert saved["simulation_model"]["version"] == 3
        assert saved["actuator"]["limits"][2]["holdingCurrent"] == 1
        assert saved["simulation_model"]["motor_specs"][1]["model"] == "RobStride 06"
        assert saved["frames"] == len(data["actions"])
        assert saved["clips"][0]["replay"]["base_speed"] == 2
        page.locator("#export-dataset").click()
        page.wait_for_function("document.getElementById('export-result').textContent.includes('既にあります')")
        page.locator("#close-export").click()
        page.screenshot(path=str(screenshots / "edited.png"), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path=str(screenshots / "mobile.png"), full_page=True)
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
        assert not errors, errors
        print("Browser checks passed: load, seek/rewind, per-clip speed, trim, four transitions, play, views, limits, ideal dataset + settings export, overwrite protection, mobile.")
        browser.close()


def main():
    # Validate real browser writes in an isolated copy, never in user datasets.
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
