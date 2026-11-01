# Brain Tumor MRI Classification

import numpy as np  # sayısal işlemler
import tensorflow as tf  # modeli yüklemek için
from tensorflow.keras.preprocessing import image  # resmi hazırlamak için

model_path = "modeller/beyin_tumoru_cnn_modeli.keras"  # eğitilen cnn modeli
image_path = "ornek_resim.jpg"  # tahmin yapılacak resim

class_names = ["glioma", "meningioma", "notumor", "pituitary"]  # sınıf isimleri

model = tf.keras.models.load_model(model_path)  # modeli yükler

img = image.load_img(image_path, target_size=(224, 224))  # resmi 224x224 yapar
img_array = image.img_to_array(img)  # resmi diziye çevirir
img_array = img_array / 255.0  # normalize eder
img_array = np.expand_dims(img_array, axis=0)  # batch boyutu ekler

prediction = model.predict(img_array)  # tahmin yapar
predicted_index = np.argmax(prediction)  # en yüksek olasılığı seçer
predicted_class = class_names[predicted_index]  # sınıf adını alır
confidence = prediction[0][predicted_index] * 100  # güven yüzdesi

print("tahmin edilen sınıf:", predicted_class)
print("güven oranı:", round(confidence, 2), "%")

print("\nsınıf olasılıkları:")
for i, class_name in enumerate(class_names):
    print(class_name, ":", round(prediction[0][i] * 100, 2), "%")