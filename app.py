import streamlit as st
import joblib
import numpy as np
import pandas as pd
from scipy.stats import boxcox

# ---------- Page setup ----------
st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="centered")

# ---------- Load saved artifacts ----------
model = joblib.load('laptop_price_model.pkl')
model_columns = joblib.load('model_columns.pkl')
ppi_lambda = joblib.load('ppi_boxcox_lambda.pkl')

st.title("💻 Laptop Price Predictor")
st.caption("Fill in the specs below and get an estimated market price.")
st.divider()

# ---------- Category lists (from training data) ----------
company_list = ['Apple', 'HP', 'Acer', 'Asus', 'Dell', 'Lenovo', 'Chuwi',
                 'MSI', 'Microsoft', 'Toshiba', 'Huawei', 'Xiaomi', 'Vero',
                 'Razer', 'Mediacom', 'Samsung', 'Google', 'Fujitsu', 'LG']

typename_list = ['Ultrabook', 'Notebook', 'Gaming', '2 in 1 Convertible',
                  'Workstation', 'Netbook']

opsys_list = ['macOS', 'No OS', 'Windows 10', 'Mac OS X', 'Linux',
              'Windows 10 S', 'Chrome OS', 'Windows 7', 'Android']

cpu_brand_list = ['Intel', 'AMD', 'Samsung']
gpu_brand_list = ['Intel', 'AMD', 'Nvidia', 'ARM']
storage_type_list = ['SSD', 'HDD', 'Flash Storage', 'Hybrid']

resolution_options = {
    "1366 x 768 (HD)": (1366, 768),
    "1600 x 900": (1600, 900),
    "1920 x 1080 (Full HD)": (1920, 1080),
    "1920 x 1200": (1920, 1200),
    "2256 x 1504": (2256, 1504),
    "2304 x 1440 (Retina)": (2304, 1440),
    "2560 x 1440 (Quad HD)": (2560, 1440),
    "2560 x 1600 (Retina)": (2560, 1600),
    "2736 x 1824 (Retina)": (2736, 1824),
    "2880 x 1800 (Retina)": (2880, 1800),
    "3200 x 1800 (Quad HD+)": (3200, 1800),
    "3840 x 2160 (4K Ultra HD)": (3840, 2160),
}

# ---------- User Inputs ----------
st.subheader("🏷️ Brand & Type")
c1, c2, c3 = st.columns(3)
with c1:
    company = st.selectbox("Company", company_list)
with c2:
    typename = st.selectbox("Type", typename_list)
with c3:
    opsys = st.selectbox("Operating System", opsys_list)

st.subheader("⚙️ Performance")
c1, c2, c3 = st.columns(3)
with c1:
    ram = st.select_slider("RAM (GB)", options=[1, 2, 4, 6, 8, 12, 16, 24, 32, 64], value=8)
with c2:
    cpu_brand = st.selectbox("CPU Brand", cpu_brand_list)
with c3:
    speed_ghz = st.slider("CPU Speed (GHz)", min_value=0.9, max_value=3.6, value=2.5, step=0.1)

c1, c2 = st.columns(2)
with c1:
    gpu_brand = st.selectbox("GPU Brand", gpu_brand_list)
with c2:
    storage_type = st.selectbox("Primary Storage Type", storage_type_list)

storage_size = st.select_slider("Total Storage (GB)", options=[8, 16, 32, 64, 128, 180, 240, 256, 500, 512, 1000, 2000], value=256)

st.subheader("🖥️ Display")
c1, c2 = st.columns(2)
with c1:
    inches = st.slider("Screen Size (Inches)", min_value=10.0, max_value=18.4, value=15.6, step=0.1)
with c2:
    resolution_label = st.selectbox("Resolution", list(resolution_options.keys()), index=2)
    width, height = resolution_options[resolution_label]

c1, c2 = st.columns(2)
with c1:
    has_touchscreen = st.checkbox("Touchscreen")
with c2:
    has_ips = st.checkbox("IPS Panel")

st.subheader("⚖️ Build")
weight = st.slider("Weight (kg)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)

st.divider()

# ---------- Predict Button ----------
if st.button("🔮 Predict Price", use_container_width=True, type="primary"):

    # --- Numeric transformations (must match training exactly) ---
    ram_transformed = np.log(ram)
    weight_transformed = np.log(weight)
    speed_ghz_squared = speed_ghz ** 2
    total_storage_sqrt = np.sqrt(storage_size)
    has_ssd = 1 if storage_type == 'SSD' else 0

    ppi_raw = (((width ** 2) + (height ** 2)) ** 0.5) / inches
    # Apply the SAME boxcox lambda learned during training
    ppi_boxcox = boxcox([ppi_raw], lmbda=ppi_lambda)[0]

    # --- Build a single-row dataframe with raw feature values ---
    input_dict = {
        'Inches': inches,
        'Weight_transformed': weight_transformed,
        'Ram_transformed': ram_transformed,
        'Speed_GHz_squared': speed_ghz_squared,
        'Total_storage_sqrt': total_storage_sqrt,
        'Has_SSD': has_ssd,
        'PPI_boxcox': ppi_boxcox,
        'Has_Touchscreen': int(has_touchscreen),
        'Has_IPS': int(has_ips),
        f'Company_{company}': 1,
        f'TypeName_{typename}': 1,
        f'OpSys_{opsys}': 1,
        f'Cpu_Brand_{cpu_brand}': 1,
        f'Gpu_Brand_{gpu_brand}': 1,
        f'Storage1_type_{storage_type}': 1,
    }

    # Start with all model columns set to 0 (float dtype so decimals fit fine)
    input_df = pd.DataFrame(0, index=[0], columns=model_columns, dtype=float)
    for col, val in input_dict.items():
        if col in input_df.columns:
            input_df.at[0, col] = val
        # else: it's the dropped "first category" from one-hot encoding — correctly stays 0

    input_df = input_df[model_columns]  # ensure exact column order

    # --- Predict (model outputs log scale) ---
    predicted_log_price = model.predict(input_df)[0]
    predicted_price = np.exp(predicted_log_price)

    st.metric(label="Estimated Price", value=f"₹{predicted_price:,.0f}")
    st.caption("This is an estimate based on a Random Forest model trained on historical listings — actual market prices may vary.")
