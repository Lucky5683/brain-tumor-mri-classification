import os
import time
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

opts = Options()
opts.add_argument("--headless=new")
opts.add_argument("--window-size=1440,1400")
opts.add_argument("--disable-gpu")
opts.add_argument("--no-sandbox")

driver = webdriver.Chrome(options=opts)
try:
    print("Navigating to http://localhost:8501 ...")
    driver.get("http://localhost:8501")
    wait = WebDriverWait(driver, 30)
    wait.until(EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Brain Tumor MRI Classification')]")))
    time.sleep(3)

    # Save home screen
    driver.save_screenshot("screenshots/streamlit_app_home.png")
    print("Saved screenshots/streamlit_app_home.png")

    # Upload test MRI scan using input[type='file']
    test_img_path = str(Path("dataset/Testing/glioma/Te-gl_10.jpg").resolve())
    print(f"Uploading MRI image file: {test_img_path}")
    file_input = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file']")))
    file_input.send_keys(test_img_path)
    time.sleep(4)

    # Save loaded state screen
    driver.save_screenshot("screenshots/streamlit_sample_loaded.png")
    print("Saved screenshots/streamlit_sample_loaded.png")

    # Find Run Inference button
    print("Waiting for 'Run Inference & Saliency Analysis' button...")
    button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Run Inference')]")))
    print("Clicking inference button...")
    driver.execute_script("arguments[0].click();", button)

    print("Analysis triggered! Waiting for inference and Grad-CAM computation...")
    # Wait until 'Predicted Class' or 'Explainable AI' appears
    wait.until(EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Predicted Class')]")))
    time.sleep(8)  # Give time for Grad-CAM images and charts to finish rendering

    # Enlarge window to capture full dashboard
    driver.set_window_size(1440, 2400)
    time.sleep(2)
    driver.save_screenshot("screenshots/streamlit_prediction_results.png")
    print("Saved screenshots/streamlit_prediction_results.png")

    print("\nAll Streamlit live interaction screenshots captured successfully!")

finally:
    driver.quit()
