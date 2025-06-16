# app.py
import streamlit as st
from PIL import Image
from pyzbar.pyzbar import decode
import face_recognition
import numpy as np

st.set_page_config(layout="centered", page_title="Smart QR + Face Verification")

st.title("🔍 Day 7 – Smart QR + Face Verification System")
st.markdown("---")

st.subheader("1. Upload QR Code Image")
qr_image_file = st.file_uploader("Choose a QR Code image...", type=['jpg', 'png', 'jpeg'])

qr_content = None
if qr_image_file is not None:
    try:
        qr_img = Image.open(qr_image_file)
        st.image(qr_img, caption='Uploaded QR Code', use_column_width=True)
        st.write("")
        st.subheader("📎 QR Code Data:")
        qr_data = decode(qr_img)
        if qr_data:
            for obj in qr_data:
                qr_content = obj.data.decode('utf-8')
                st.success(f"**QR Content:** `{qr_content}`")
                st.info("This content could be a User ID, a hash, or any identifier.")
        else:
            st.warning("No QR Code Detected in the uploaded image.")
    except Exception as e:
        st.error(f"Error processing QR code image: {e}")

st.markdown("---")

st.subheader("2. Upload Face Images for Comparison")
st.write("Upload two face images. Typically, one would be from a database (associated with the QR code) and the other a live capture.")

col1, col2 = st.columns(2)

with col1:
    face_image1_file = st.file_uploader("Upload Face Image (e.g., from QR/Database)", type=['jpg', 'png', 'jpeg'], key="face1")
    face_img1_pil = None
    if face_image1_file is not None:
        face_img1_pil = Image.open(face_image1_file)
        st.image(face_img1_pil, caption='Face from QR/Database', use_column_width=True)

with col2:
    face_image2_file = st.file_uploader("Upload Live Face Image to Compare", type=['jpg', 'png', 'jpeg'], key="face2")
    face_img2_pil = None
    if face_image2_file is not None:
        face_img2_pil = Image.open(face_image2_file)
        st.image(face_img2_pil, caption='Live Face to Compare', use_column_width=True)

st.markdown("---")

st.subheader("3. Validate Identity")

if st.button("Compare Faces and Validate Identity"):
    if face_img1_pil and face_img2_pil:
        st.subheader("🧠 Face Match Result")
        try:
            # Convert PIL Image to numpy array for face_recognition
            image1_np = np.array(face_img1_pil)
            image2_np = np.array(face_img2_pil)

            enc1 = face_recognition.face_encodings(image1_np)
            enc2 = face_recognition.face_encodings(image2_np)

            if enc1 and enc2:
                # Compare faces. tolerance can be adjusted (default is 0.6)
                # Lower tolerance means stricter match
                result = face_recognition.compare_faces([enc1[0]], enc2[0], tolerance=0.6)
                face_distances = face_recognition.face_distance([enc1[0]], enc2[0])

                if result[0]:
                    st.success(f"✅ Faces Match! Similarity (lower is better): {face_distances[0]:.2f}")
                    if qr_content:
                        st.subheader("Overall Identity Validation:")
                        st.success(f"**Identity Confirmed!**\n\nQR Content: `{qr_content}`\nFaces Match.")
                    else:
                        st.warning("Faces match, but QR content was not detected or processed.")
                else:
                    st.error(f"❌ Faces Do Not Match. Similarity (lower is better): {face_distances[0]:.2f}")
                    if qr_content:
                        st.error(f"Identity Not Confirmed: QR content (`{qr_content}`) does not match the live face.")
                    else:
                        st.error("Identity Not Confirmed: Faces do not match, and QR content was not detected.")
            else:
                st.error("Face not detected in one or both of the uploaded images. Please ensure clear face images.")
        except Exception as e:
            st.error(f"Error during face comparison: {e}. Please ensure images contain clear faces.")
    else:
        st.warning("Please upload both face images to compare.")

st.markdown("---")
st.info("This system can be integrated with databases for more robust identity management.")