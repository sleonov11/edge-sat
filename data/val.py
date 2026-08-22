import os, shutil, random

if not os.path.exists("val"):
    os.makedirs("val/0")
    os.makedirs("val/1")
    for label in ["0", "1"]:
        files = os.listdir(f"train/{label}")
        random.shuffle(files)
        val_files = files[:int(len(files)*0.2)] # берем 20% на проверку
        for f in val_files:
            shutil.move(f"train/{label}/{f}", f"val/{label}/{f}")
    print("Папка val создана! Теперь данные разделены физически.")