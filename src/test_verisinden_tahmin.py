# Brain Tumor MRI Classification

import os  # klasör işlemleri
import numpy as np  # sayısal işlemler
import tensorflow as tf  # modeli yüklemek için
from tensorflow.keras.preprocessing import image  # resmi hazırlamak için

model_path = "modeller/beyin_tumoru_cnn_modeli.keras"  # eğitilmiş model
test_path = "veri_seti/Testing"  # test klasörü

class_names = ["glioma", "meningioma", "notumor", "pituitary"]  # sınıflar

model = tf.keras.models.load_model(model_path)  # modeli yükler

print("\nseçilebilecek sınıflar:")
for i, class_name in enumerate(class_names):
    print(f"{i + 1}. {class_name}")

class_choice = int(input("\nbir sınıf seç 1-4: ")) - 1  # kullanıcıdan sınıf alır
selected_class = class_names[class_choice]  # seçilen sınıf adı

class_folder = os.path.join(test_path, selected_class)  # sınıf klasör yolu
image_files = os.listdir(class_folder)  # görselleri listeler

print(f"\n{selected_class} sınıfındaki ilk 20 görsel:")
for i, img_name in enumerate(image_files[:20]):
    print(f"{i + 1}. {img_name}")

image_choice = int(input("\nbir görsel seç 1-20: ")) - 1  # kullanıcıdan görsel alır
selected_image = image_files[image_choice]  # seçilen görsel adı

image_path = os.path.join(class_folder, selected_image)  # seçilen görsel yolu

img = image.load_img(image_path, target_size=(224, 224))  # resmi yükler
img_array = image.img_to_array(img)  # resmi diziye çevirir
img_array = img_array / 255.0  # normalize eder
img_array = np.expand_dims(img_array, axis=0)  # batch boyutu ekler

prediction = model.predict(img_array, verbose=0)  # tahmin yapar
predicted_index = np.argmax(prediction)  # en yüksek sonucu alır
predicted_class = class_names[predicted_index]  # sınıf ismini alır
confidence = prediction[0][predicted_index] * 100  # güven oranı

print("\n----------------------------")
print("gerçek sınıf:", selected_class)
print("seçilen görsel:", selected_image)
print("tahmin edilen sınıf:", predicted_class)
print("güven oranı:", round(confidence, 2), "%")

print("\nsınıf olasılıkları:")
for i, class_name in enumerate(class_names):
    print(class_name, ":", round(prediction[0][i] * 100, 2), "%")
print("----------------------------")