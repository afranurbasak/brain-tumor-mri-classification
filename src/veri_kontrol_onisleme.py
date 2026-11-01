import os  # klasör işlemleri
from PIL import Image  # görsel kontrolü

train_path = "veri_seti/Training"  # eğitim verisi yolu
test_path = "veri_seti/Testing"  # test verisi yolu

# kontrol edilecek yollar
paths = [train_path, test_path]

print("\nveri seti kontrolü başlatıldı\n")

total_images = 0  # toplam görsel sayısı
broken_images = 0  # bozuk görsel sayısı

for path in paths:

    print(f"\nkontrol edilen klasör: {path}")

    classes = os.listdir(path)  # sınıf klasörleri

    for class_name in classes:

        class_path = os.path.join(path, class_name)

        # klasördeki dosyalar
        image_files = os.listdir(class_path)

        print(f"\n{class_name} -> {len(image_files)} görsel")

        # boş klasör kontrolü
        if len(image_files) == 0:
            print("uyarı: klasör boş")

        # görselleri kontrol et
        for img_name in image_files:

            img_path = os.path.join(class_path, img_name)

            try:
                img = Image.open(img_path)  # resmi açmayı dener

                width, height = img.size  # boyut bilgisi alır

                total_images += 1  # toplam görsel artırılır

                # çok küçük görsel kontrolü
                if width < 50 or height < 50:
                    print(f"küçük görsel bulundu: {img_name}")

            except Exception as e:

                broken_images += 1  # bozuk görsel artırılır

                print(f"bozuk görsel: {img_name}")

print("\n----------------------------")
print("kontrol tamamlandı")
print("toplam görsel sayısı:", total_images)
print("bozuk görsel sayısı:", broken_images)
print("----------------------------")