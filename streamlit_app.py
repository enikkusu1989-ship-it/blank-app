import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from streamlit_drawable_canvas import st_canvas

st.set_page_config(
    page_title="手書き数字AI",
    page_icon="🔢",
    layout="centered"
)

st.title("🔢 手書き数字AI")
st.write("下のキャンバスに 0〜9 の数字を書いてください。")
st.write("AIがあなたの書いた数字を判定します。")


# =========================
# AIモデル
# =========================

@st.cache_resource
def train_model():
    # 手書き数字データセットを読み込む
    digits = load_digits()

    X = digits.data
    y = digits.target

    # SVMで学習
    model = SVC(
        kernel="rbf",
        probability=True
    )

    model.fit(X, y)

    return model


model = train_model()


# =========================
# 手書きキャンバス
# =========================

canvas = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",
    stroke_width=18,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)


# =========================
# 判定ボタン
# =========================

if st.button("🤖 数字を判定する", use_container_width=True):

    if canvas.image_data is None:
        st.warning("まず数字を書いてください！")
        st.stop()

    # RGBA → グレースケール
    image = Image.fromarray(
        canvas.image_data.astype("uint8")
    ).convert("L")

    # 白い部分を探す
    array = np.array(image)

    ys, xs = np.where(array > 30)

    if len(xs) == 0:
        st.warning("数字が見つかりませんでした。")
        st.stop()

    # 数字部分だけ切り抜く
    left = xs.min()
    right = xs.max() + 1
    top = ys.min()
    bottom = ys.max() + 1

    cropped = image.crop((left, top, right, bottom))

    # 正方形のキャンバスにする
    size = max(cropped.size)

    square = Image.new("L", (size, size), 0)

    x = (size - cropped.width) // 2
    y = (size - cropped.height) // 2

    square.paste(cropped, (x, y))

    # 8×8に縮小
    resized = square.resize(
        (8, 8),
        Image.Resampling.LANCZOS
    )

    # numpy化
    data = np.array(resized).astype(float)

    # 0〜16程度に正規化
    data = data / 255.0 * 16.0

    # AIに入力
    prediction = model.predict(
        data.reshape(1, -1)
    )[0]

    probabilities = model.predict_proba(
        data.reshape(1, -1)
    )[0]

    confidence = probabilities[prediction] * 100

    # 結果表示
    st.success(
        f"🎯 AIの判定： **{prediction}**"
    )

    st.metric(
        "AIの確信度",
        f"{confidence:.1f}%"
    )

    # 認識した画像を表示
    st.subheader("AIが見た数字")

    st.image(
        resized,
        width=200,
        caption=f"認識結果：{prediction}"
    )

    # 各数字の確率
    st.subheader("数字ごとの確率")

    probability_data = {
        str(i): float(probabilities[i] * 100)
        for i in range(10)
    }

    st.bar_chart(probability_data)