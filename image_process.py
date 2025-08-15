import cv2
import numpy as np
import streamlit as st
import io
import zipfile

def display_image(img, caption="Image"):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption=caption, channels="RGB")

def denoise(img):
    return cv2.medianBlur(img, 5)

def brighten(img):
    return cv2.addWeighted(img, 1.5, np.zeros(img.shape, img.dtype), 0, 0)

def size_normalization(img, target_size=(400, 300)):
    return cv2.resize(img, target_size)

def illumination_normalization(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

def outline(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outlined = img.copy()
    cv2.drawContours(outlined, contours, -1, (0, 255, 0), 2)
    return outlined

st.title("Custom Image Processing App with ZIP Download")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    st.subheader("Original Image")
    display_image(image, "Uploaded Image")

    steps = {}

    st.write("### Select processing steps:")

    if st.button("Denoise Image"):
        denoised = denoise(image)
        display_image(denoised, "Denoised Image")
        steps["denoised.jpg"] = denoised

    if st.button("Brighten Image"):
        brightened = brighten(image)
        display_image(brightened, "Brightened Image")
        steps["brightened.jpg"] = brightened

    if st.button("Normalize Size"):
        normalized = size_normalization(image)
        display_image(normalized, "Size Normalized Image")
        steps["normalized.jpg"] = normalized

    if st.button("Illumination Normalization"):
        illum_norm = illumination_normalization(image)
        display_image(illum_norm, "Illumination Normalized Image")
        steps["illum_norm.jpg"] = illum_norm

    if st.button("Outline Image"):
        outlined = outline(image)
        display_image(outlined, "Outlined Image")
        steps["outlined.jpg"] = outlined

    if steps:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            for filename, img in steps.items():
                success, buffer = cv2.imencode(".jpg", img)
                if success:
                    zip_file.writestr(filename, buffer.tobytes())
        zip_buffer.seek(0)

        st.download_button(
            label="Download Selected Images (ZIP)",
            data=zip_buffer,
            file_name="selected_images.zip",
            mime="application/zip"
        )
