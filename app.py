import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import random

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="VisionAI | Deep Retinal Suite",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# PREMIUM DEEP THEME CSS
# =========================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&display=swap');

    .stApp {
        background: radial-gradient(circle at 10% 20%, #0a192f 0%, #000000 100%);
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .header-section {
        padding: 3rem 2rem;
        text-align: center;
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border-radius: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 2.5rem;
    }

    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(10px);
        border-radius: 25px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        color: #f8fafc;
        margin-bottom: 20px;
    }

    .img-label {
        color: #38bdf8;
        font-weight: 700;
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 15px;
        display: block;
        text-align: center;
        border-bottom: 1px solid rgba(56, 189, 248, 0.2);
        padding-bottom: 8px;
    }

    .prescription-container {
        background: rgba(255, 255, 255, 0.05); 
        backdrop-filter: blur(15px);
        color: #f8fafc !important;
        padding: 40px;
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.5);
        margin-top: 2rem;
    }

    .explanation-card {
        background: rgba(14, 165, 233, 0.1); 
        border-left: 5px solid #38bdf8;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 25px;
    }

    .explanation-title {
        color: #38bdf8;
        font-weight: 800;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .rx-symbol {
        font-size: 3rem;
        background: linear-gradient(90deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }

    .status-normal { color: #10b981; font-weight: bold; border: 1px solid #10b981; padding: 5px 15px; border-radius: 50px; }
    .status-alert { color: #ef4444; font-weight: bold; border: 1px solid #ef4444; padding: 5px 15px; border-radius: 50px; }

    div[data-testid="stFileUploader"] {
        border: 2px dashed rgba(56, 189, 248, 0.4);
        padding: 20px;
        border-radius: 20px;
        background: rgba(255,255,255,0.02);
    }
</style>
""", unsafe_allow_html=True)

# =========================
# CORE LOGIC
# =========================
@st.cache_resource
def load_model():
    try:
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras import layers, models
        base_model = MobileNetV2(input_shape=(224,224,3), include_top=False, weights=None)
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(4, activation='softmax')
        ])
        model.build((None, 224, 224, 3))
        model.load_weights("final.weights.h5")
        return model
    except:
        return None

model = load_model()
class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

def get_gradcam(model, img_array):
    base_model = model.layers[0]
    last_conv = next(l for l in base_model.layers[::-1] if "conv" in l.name)
    grad_model = tf.keras.models.Model(inputs=base_model.input, outputs=[last_conv.output, base_model.output])
    with tf.GradientTape() as tape:
        conv_output, base_output = grad_model(img_array)
        x = base_output
        for layer in model.layers[1:]: x = layer(x)
        preds = x
        loss = preds[:, tf.argmax(preds[0])]
    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    heatmap = conv_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.maximum(tf.squeeze(heatmap), 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def generate_report(disease):
    explanations = {
        'CNV': "Choroidal Neovascularization: Pathological blood vessel growth beneath the retina, causing severe fluid leakage.",
        'DME': "Diabetic Macular Edema: Fluid accumulation in the macula affecting central vision due to diabetic complications.",
        'DRUSEN': "Retinal Drusen: Yellow lipid deposits under the retina, indicating potential age-related macular degeneration.",
        'NORMAL': "Healthy Retinal Morphology: No definitive signs of edema, hemorrhage, or lipid deposits detected."
    }
    risk = {'CNV': "🔴 CRITICAL", 'DME': "🔴 HIGH", 'DRUSEN': "🟡 MODERATE", 'NORMAL': "🟢 STABLE"}
    treatment = {
        'CNV': ["Anti-VEGF Injections", "Urgent Specialist Referral", "Weekly Amsler Grid Monitoring", "Monthly OCT Scans"],
        'DME': ["Strict Glycemic Control", "Intravitreal Injections", "Blood Pressure Management", "Endocrinology Review"],
        'DRUSEN': ["AREDS2 Vitamins", "UV Protection Sunglasses", "Smoking Cessation", "6-Month Follow-up"],
        'NORMAL': ["Annual Routine Exam", "20-20-20 Screen Rule", "Omega-3 Rich Diet", "Protective Eyewear"]
    }
    return explanations[disease], risk[disease], treatment[disease]

# =========================
# UI HEADER
# =========================
st.markdown("""
<div class="header-section">
    <h1 class="main-title">VISION AI PRO</h1>
    <p style="color: #94a3b8; font-size: 1.1rem; letter-spacing: 1px;">PRECISION RETINAL DIAGNOSTIC ENGINE</p>
</div>
""", unsafe_allow_html=True)

if model is None: 
    st.error("Weights file 'final.weights.h5' missing.")

col_u1, col_u2, col_u3 = st.columns([1, 2, 1])
with col_u2:
    uploaded_file = st.file_uploader("Upload Retinal OCT Scan", type=["jpg","png","jpeg"])

# =========================
# ANALYSIS DASHBOARD
# =========================
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image.resize((224,224)))/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    idx = np.argmax(prediction)
    disease = class_names[idx]
    confidence = float(np.max(prediction))
    expl, risk_lvl, treats = generate_report(disease)

    st.markdown("---")
    col_vis, col_rep = st.columns([1.2, 1.8], gap="large")

    with col_vis:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<span class="img-label">📸 Original OCT Input</span>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        
        st.markdown('<br><span class="img-label">🔥 Pathology Heatmap</span>', unsafe_allow_html=True)
        heatmap = get_gradcam(model, img_array)
        heatmap = cv2.resize(heatmap, (224,224))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(np.uint8(img_array[0] * 255), 0.6, heatmap, 0.4, 0)
        st.image(overlay, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_rep:
        status_class = "status-normal" if disease == "NORMAL" else "status-alert"
        st.markdown(f"""
        <div class="glass-card">
            <h5 style="color:#94a3b8; margin:0;">AI ANALYSIS STATUS</h5>
            <h1 style="color:white; margin:10px 0;">{disease} <span class="{status_class}">{risk_lvl}</span></h1>
            <p style="color:#38bdf8; font-weight:700;">Confidence: {round(confidence*100,2)}%</p>
        </div>
        """, unsafe_allow_html=True)
        st.progress(confidence)

        st.markdown("<h4 style='color:white; margin-top:20px;'>Neural Distribution</h4>", unsafe_allow_html=True)
        dist_cols = st.columns(4)
        for i, val in enumerate(prediction[0]):
            with dist_cols[i]:
                st.markdown(f"""
                <div style="text-align:center; padding:10px; background:rgba(255,255,255,0.05); border-radius:15px; border:1px solid rgba(255,255,255,0.1);">
                    <small style="color:#94a3b8;">{class_names[i]}</small><br>
                    <strong style="color:white;">{round(val*100,1)}%</strong>
                </div>
                """, unsafe_allow_html=True)

        treat_list_html = "".join([f"<li style='color:#cbd5e1; margin-bottom:8px;'>{t}</li>" for t in treats])
        
        st.markdown(f"""
        <div class="prescription-container">
            <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid rgba(255,255,255,0.1); padding-bottom:15px; margin-bottom:20px;">
                <div>
                    <h3 style="margin:0; color:#ffffff; letter-spacing:1px;">Clinical Evaluation Report</h3>
                    <small style="color:#94a3b8;">UID: OCT-{random.randint(1000,9999)} | Neural Engine v4.1</small>
                </div>
                <div class="rx-symbol">℞</div>
            </div>
            <div class="explanation-card">
                <div class="explanation-title">🔬 Neural Interpretation</div>
                <p style="margin:0; color:#e2e8f0; font-size: 1.1rem; font-style: italic; line-height:1.6;">"{expl}"</p>
            </div>
            <p style="color:#38bdf8; font-weight:700; text-transform:uppercase; font-size:0.85rem; letter-spacing:1px;">Management & Recommendations</p>
            <ul style="padding-left: 20px;">{treat_list_html}</ul>
            <div style="margin-top:30px; background:rgba(56, 189, 248, 0.1); padding:20px; border-radius:12px; border: 1px solid rgba(56, 189, 248, 0.2);">
                <strong style="color:#38bdf8;">Follow-up Protocol:</strong> <span style="color:#f8fafc;">Clinical correlation required. Monitoring advised as per clinical protocols.</span>
            </div>
            <p style="font-size:0.75rem; color:#64748b; margin-top:30px; text-align:center; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 15px;">
                * AI decision support tool. Verified ophthalmologist validation is mandatory.
            </p>
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""<div style="text-align:center; padding:100px; color:#ffffff; opacity:0.5;"><h2>Awaiting Input...</h2><p>Upload a retinal scan to initiate the diagnostic sequence.</p></div>""", unsafe_allow_html=True)
