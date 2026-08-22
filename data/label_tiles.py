import cv2
import os
import numpy as np

# === НАСТРОЙКИ ===
SRC_DIR = "images"      # папка с вашими 8 большими снимками
DEST_1 = "train/1"   # сюда ляжет ВОДА (и тайлы с кораблями из HRSC)
DEST_0 = "train/0"   # сюда ляжет СУША/облака
TILE = 256
OVERLAP = 64                 # 64 пикселя перекрытия, чтобы корабль не разрезало пополам
STEP = TILE - OVERLAP        # 192

os.makedirs(DEST_1, exist_ok=True)
os.makedirs(DEST_0, exist_ok=True)

# Функция автоклассификации по цвету (грубая, чтобы отсеять очевидное)
def is_obvious_water(tile):
    # Средние значения каналов BGR
    b, g, r = tile[:,:,0].mean(), tile[:,:,1].mean(), tile[:,:,2].mean()
    # Вода обычно тёмная и синяя/зеленоватая, суша обычно яркая и красная/зеленая
    if b > g and b > r and (b + g + r) / 3 < 120:
        return True
    return False

def is_obvious_land(tile):
    b, g, r = tile[:,:,0].mean(), tile[:,:,1].mean(), tile[:,:,2].mean()
    # Суша обычно яркая, преобладает зеленый или красный
    if (r > 80 or g > 80) and (r + g + b) / 3 > 80:
        # если тайл слишком светлый (например, песок, поле) - это точно земля
        if b < 80: 
            return True
    return False

print("Начинаю обработку...")
count_auto = 0
count_manual = 0

cv2.namedWindow("Labeler", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Labeler", 512, 512)

for img_name in os.listdir(SRC_DIR):
    img_path = os.path.join(SRC_DIR, img_name)
    img = cv2.imread(img_path)
    if img is None: continue
    
    h, w, _ = img.shape
    
    # Идем по сетке с перекрытием
    for y in range(0, h - TILE + 1, STEP):
        for x in range(0, w - TILE + 1, STEP):
            tile = img[y:y+TILE, x:x+TILE]
            
            # 1. Пытаемся отсортировать автоматически
            if is_obvious_water(tile):
                cv2.imwrite(os.path.join(DEST_1, f"{img_name}_{x}_{y}.jpg"), tile)
                count_auto += 1
                continue
                
            if is_obvious_land(tile):
                cv2.imwrite(os.path.join(DEST_0, f"{img_name}_{x}_{y}.jpg"), tile)
                count_auto += 1
                continue

            # 2. Если спорный случай (тень, берег, облако над водой) -> показываем человеку
            count_manual += 1
            cv2.imshow("Labeler", tile)
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q'): # выход
                cv2.destroyAllWindows()
                print(f"Готово! Авто: {count_auto}, Ручная разметка: {count_manual}")
                exit()
            elif key == ord('y'): # вода
                cv2.imwrite(os.path.join(DEST_1, f"{img_name}_{x}_{y}.jpg"), tile)
            elif key == ord('n'): # суша
                cv2.imwrite(os.path.join(DEST_0, f"{img_name}_{x}_{y}.jpg"), tile)
            elif key == ord('s'): # пропустить
                continue
            else:
                # если нажата другая клавиша - считаем как "n" (земля), чтобы не застрять
                cv2.imwrite(os.path.join(DEST_0, f"{img_name}_{x}_{y}.jpg"), tile)

cv2.destroyAllWindows()
print(f"Готово! Автоматически разложено: {count_auto}, размечено вручную: {count_manual}.")