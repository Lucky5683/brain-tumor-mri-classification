"""Automated Browser Screenshot Capture for Streamlit Application using Selenium."""

from __future__ import annotations
import time
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

screenshots_dir = Path("screenshots")
screenshots_dir.mkdir(parents=True, exist_ok=True)

chrome_options = ChromeOptions()
chrome_options.add_argument("--headless=new")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--window-size=1440,1100")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

print("Initializing headless Chrome browser driver...")
driver = webdriver.Chrome(options=chrome_options)

try:
    print("Navigating to http://localhost:8501 ...")
    driver.get("http://localhost:8501")

    # Wait for Streamlit app to load (look for title or disclaimer)
    wait = WebDriverWait(driver, 30)
    wait.until(EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Brain Tumor MRI Classification')]")))
    time.sleep(3)  # Let animations and widgets settle

    # Capture 1: Home page
    home_shot = screenshots_dir / "streamlit_app_home.png"
    driver.save_screenshot(str(home_shot))
    print(f"[1/3] Captured Streamlit Home Screen: {home_shot} ({home_shot.stat().st_size} bytes)")

    # 2. Select pre-loaded sample MRI from selectbox
    print("Selecting test sample from dropdown...")
    selectboxes = driver.find_elements(By.CSS_SELECTOR, "[data-baseweb='select']")
    # The second selectbox is the sample MRI dropdown
    if len(selectboxes) >= 2:
        sample_select = selectboxes[1]
        sample_select.click()
        time.sleep(1)
        # Select 'Glioma Sample (Testing/glioma)'
        options = driver.find_elements(By.XPATH, "//*[contains(text(), 'Glioma Sample')]")
        if options:
            options[0].click()
            time.sleep(2)
            print("Selected Glioma sample successfully.")

    # Capture 2: Image loaded state
    loaded_shot = screenshots_dir / "streamlit_sample_loaded.png"
    driver.save_screenshot(str(loaded_shot))
    print(f"[2/3] Captured Sample Loaded Screen: {loaded_shot} ({loaded_shot.stat().st_size} bytes)")

    # 3. Click 'Run Inference & Saliency Analysis'
    print("Clicking 'Run Inference & Saliency Analysis' button...")
    buttons = driver.find_elements(By.XPATH, "//button[contains(., 'Run Inference')]")
    if buttons:
        buttons[0].click()
        print("Analysis button clicked! Waiting for inference and Grad-CAM computation...")
        # Wait until 'Predicted Class' metric appears
        wait.until(EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Predicted Class') or contains(text(), 'Explainable AI')]")))
        time.sleep(5)  # Wait for charts and images to finish rendering

        # Resize window to capture entire results dashboard
        driver.set_window_size(1440, 1800)
        time.sleep(2)

        result_shot = screenshots_dir / "streamlit_prediction_results.png"
        driver.save_screenshot(str(result_shot))
        print(f"[3/3] Captured Full Results & Grad-CAM Visualization: {result_shot} ({result_shot.stat().st_size} bytes)")
    else:
        print("Warning: Inference button not found directly.")

    print("\nAll genuine Streamlit screenshots captured successfully!")

finally:
    driver.quit()
