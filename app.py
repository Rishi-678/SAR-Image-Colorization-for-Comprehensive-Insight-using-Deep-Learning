import streamlit as st

st.set_page_config(layout="wide")

# Apply CSS styling
st.markdown(""" 
<style> 
    .stApp { 
        background-image: url("https://upload.wikimedia.org/wikipedia/commons/b/b4/Artist%27s_concept_of_NISAR_over_Earth.jpg"); 
        background-size: cover; 
        background-position: center; 
        background-repeat: no-repeat; 
        background-attachment: fixed; 
        font-family: 'Poppins', sans-serif; 
    }

    .header {
        font-size: 32px;
        color: #FFFFFF;
        font-weight: bold;
        text-shadow: 2px 2px 10px rgba(0, 0, 0, 0.3);
        margin-left: 10px;
    }

    .logo img {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        border: 2px solid white;
    }

    .header-container {
        position: fixed;
        top: 55px;
        left: 35px;
        display: flex;
        align-items: center;
        z-index: 1000;
    }

    .description-box {
        position: fixed;
        bottom: 40px;
        left: 40px;
        width: 395px;
        background-color: rgba(255, 255, 255, 0.85);
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
        font-size: 15px;
        color: #333;
    }
</style>

<!-- Top-left Logo and Title -->
<div class="header-container">
    <div class="logo">
        <img src="https://cdn.iconscout.com/icon/free/png-512/free-earth-core-icon-download-in-svg-png-gif-file-formats--planet-space-pack-science-technology-icons-4940851.png?f=webp&w=512" alt="Logo">
    </div>
    <div class="header">SAR - Image Colorization</div>
</div>

<!-- Bottom-left Description -->
<div class="description-box">
    <p><strong>About the Project:</strong><br>
    This project uses state-of-the-art deep learning techniques to colorize grayscale Synthetic Aperture Radar (SAR) images. The goal is to enhance visual interpretation and make data more accessible for human analysis. The model is trained on satellite imagery and integrates CNN and GAN-based methods to generate realistic color approximations. It helps researchers, scientists, and analysts to interpret SAR data intuitively and quickly.</p>
</div>
""", unsafe_allow_html=True)

# Spacer to push the button downward
st.markdown("<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)

# Bottom-right "Next" button using layout
col1, col2, col3 = st.columns([8, 2, 2])
with col3:
    if st.button("Next", key="next_button"):
      st.switch_page("pages/UI-3.py")

