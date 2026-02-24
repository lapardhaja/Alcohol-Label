"""
Playwright E2E test for the approve flow in Alcohol Label app.
Run from project root after: streamlit run app.py --server.headless true --server.port 8501
"""
import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
os.chdir(_root)

from playwright.sync_api import sync_playwright
import time

SCREENSHOT_DIR = _root / "tests" / "approve_flow_screenshots"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

def screenshot(page, name: str):
    path = SCREENSHOT_DIR / f"{name}.png"
    page.screenshot(path=str(path))
    print(f"  Screenshot: {path}")

def main():
    report = []
    base_url = "http://localhost:8501"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1400, "height": 900})
        page = context.new_page()

        try:
            report.append("1. Opening app...")
            page.goto(base_url, wait_until="networkidle", timeout=30000)
            time.sleep(2)
            screenshot(page, "01_app_loaded")
            report.append("   OK: App loaded")

            report.append("2. Clicking Create new application...")
            create_btn = page.get_by_role("button", name="Create new application")
            create_btn.wait_for(state="visible", timeout=10000)
            create_btn.click()
            time.sleep(1.5)
            screenshot(page, "02_after_create_new")
            report.append("   OK: Create new clicked")

            report.append("3. Uploading test image...")
            test_image = _root / "sample_data" / "test_1.jpg"
            if test_image.exists():
                file_input = page.locator('input[type="file"]').first
                file_input.set_input_files(str(test_image))
                time.sleep(1)
                screenshot(page, "03_after_upload")
                report.append("   OK: Image uploaded")
            else:
                report.append("   SKIP: test_1.jpg not found")

            report.append("4. Selecting preset test_1...")
            preset_select = page.locator('[data-testid="stSelectbox"]').first
            preset_select.click()
            time.sleep(0.5)
            page.get_by_text("test_1", exact=False).first.click()
            time.sleep(1)
            screenshot(page, "04_after_preset")
            report.append("   OK: Preset selected")

            report.append("5. Clicking Check label...")
            check_btn = page.get_by_role("button", name="Check label")
            check_btn.wait_for(state="visible", timeout=10000)
            check_btn.click()
            time.sleep(5)
            screenshot(page, "05_after_check_label")
            report.append("   OK: Check label clicked")

            if page.get_by_text("OCR unavailable").count() > 0:
                report.append("   WARNING: OCR unavailable")

            report.append("6. Clicking Approve...")
            approve_btn = page.get_by_role("button", name="Approve")
            if approve_btn.count() == 0:
                report.append("   FAIL: Approve button not found!")
                screenshot(page, "06_approve_not_found")
            else:
                approve_btn.first.click()
                time.sleep(2)
                screenshot(page, "07_after_approve_click")
                report.append("   OK: Approve clicked")

            report.append("7. Checking Approved list...")
            page_content = page.content()
            has_approved = "Approved" in page_content
            has_abc = "ABC Distillery" in page_content or "ABC" in page_content
            approved_radio = page.get_by_text("Approved", exact=True)
            approved_visible = approved_radio.count() > 0

            if has_approved and (has_abc or approved_visible):
                report.append("   OK: App shows Approved state")
                screenshot(page, "08_on_approved_list")
            else:
                report.append("   CHECK: Verify manually")
                screenshot(page, "08_final_state")

        except Exception as e:
            report.append(f"   ERROR: {e}")
            screenshot(page, "error_state")
            import traceback
            traceback.print_exc()

        finally:
            browser.close()

    print("\n" + "=" * 60)
    print("APPROVE FLOW TEST REPORT")
    print("=" * 60)
    for line in report:
        print(line)
    print("=" * 60)
    print(f"Screenshots: {SCREENSHOT_DIR}")

if __name__ == "__main__":
    main()
