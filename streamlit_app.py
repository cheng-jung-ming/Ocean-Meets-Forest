import streamlit as st
import joblib
import pandas as pd
import requests
import io
import shap
import matplotlib.pyplot as plt
import platform

# 跨平台字型設定
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Microsoft JhengHei'
elif platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'Noto Sans CJK TC'  # Linux

plt.rcParams['axes.unicode_minus'] = False  # 負號使用 ASCII 減號，避免亂碼

@st.cache_resource
def load_explainer():
    url = "https://huggingface.co/jung-ming/Ocean-Meets-Forest/resolve/main/explainer.pkl"
    r = requests.get(url)
    return joblib.load(io.BytesIO(r.content))

explainer = load_explainer()

# 載入模型與 LabelEncoder
bundle = joblib.load("rf_model_with_encoder.pkl")
model = bundle["model"]
le = bundle["label_encoder"]

# 建立映射
ship_type_to_code = dict(zip(le.classes_, le.transform(le.classes_)))

# Streamlit UI
st.title("🚢 台中港艘次預測系統")
st.markdown("請輸入以下資訊，模型將預測該月艘次數")

port_count = st.selectbox("航線組合數", list(range(1, 100)))
year = st.selectbox("年", [2020, 2021, 2022, 2023, 2024, 2025])
month = st.selectbox("月", list(range(1, 13)))
ship_type = st.selectbox("船舶種類", list(ship_type_to_code.keys()))

if st.button("🔮 開始預測"):
    ship_type_encoded = ship_type_to_code[ship_type]
    input_df = pd.DataFrame({
        "航線組合數": [port_count],
        "年": [year],
        "月": [month],
        "船舶種類_編碼": [ship_type_encoded]
    })
    pred = model.predict(input_df)[0]
    st.success(f"預測結果：🚢 約為 {pred:.2f} 艘次")

    st.subheader("🧠 模型決策解釋圖（SHAP Waterfall）")

    shap_values = explainer(input_df)
    fig, ax = plt.subplots(figsize=(8, 4))
    shap.plots.waterfall(shap_values[0], show=False)

    # 修正負號顯示：把 unicode 負號 \u2212 替換成 ASCII 減號 -
    for text in ax.texts:
        if text.get_text().startswith('\u2212'):
            new_text = text.get_text().replace('\u2212', '-')
            text.set_text(new_text)

    st.pyplot(fig)
    plt.close(fig)
