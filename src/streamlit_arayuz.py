# Brain Tumor MRI Classification

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

# model yolu
model_path = "modeller/beyin_tumoru_cnn_modeli.keras"

# sınıf isimleri
class_names = ["glioma", "meningioma", "notumor", "pituitary"]

# modeli yükle
model = tf.keras.models.load_model(model_path)

# sayfa başlığı
st.title("beyin tümörü mri sınıflandırma sistemi")

st.write("mri görüntüsü yükleyin ve model tahmin yapsın")

# dosya yükleme alanı
uploaded_file = st.file_uploader(
    "bir mri görüntüsü seç",
    type=["jpg", "jpeg", "png"]
)

# dosya yüklendiyse
if uploaded_file is not None:

    # resmi aç ve rgb formata çevir
    img = Image.open(uploaded_file).convert("RGB")

    # yüklenen resmi göster
    st.image(img, caption="yüklenen mri görüntüsü", width=300)

    # resmi modelin istediği boyuta getir
    img = img.resize((224, 224))

    # resmi numpy dizisine çevir
    img_array = image.img_to_array(img)

    # piksel değerlerini 0-1 aralığına getir
    img_array = img_array / 255.0

    # modele uygun batch boyutu ekle
    img_array = np.expand_dims(img_array, axis=0)

    # model ile tahmin yap
    prediction = model.predict(img_array, verbose=0)

    # en yüksek olasılığa sahip sınıfı bul
    predicted_index = np.argmax(prediction)

    # tahmin edilen sınıf adı
    predicted_class = class_names[predicted_index]

    # güven oranı
    confidence = prediction[0][predicted_index] * 100

    # sonucu göster
    st.success(f"tahmin edilen sınıf: {predicted_class}")

    # güven oranını göster
    st.info(f"güven oranı: %{confidence:.2f}")

    # tüm sınıf olasılıklarını göster
    st.subheader("sınıf olasılıkları")

    for i, class_name in enumerate(class_names):
        st.write(f"{class_name}: %{prediction[0][i] * 100:.2f}")