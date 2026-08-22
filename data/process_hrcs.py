import cv2, os, random

path_1 = "train/1"
path_0 = "train/0"

target = len(os.listdir(path_1)) # 2507
current = len(os.listdir(path_0)) # 1821

files_0 = os.listdir(path_0)

i = 0
while current < target:
    f = random.choice(files_0)
    img = cv2.imread(os.path.join(path_0, f))
    
    choice = random.randint(0, 3)
    if choice == 0: aug = cv2.flip(img, 0)
    elif choice == 1: aug = cv2.flip(img, 1)
    elif choice == 2: aug = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    else: aug = cv2.rotate(img, cv2.ROTATE_180)
    
    cv2.imwrite(os.path.join(path_0, f"aug_{i}_{f}"), aug)
    current += 1
    i += 1

print(f"Итог: Вода={len(os.listdir('train/1'))}, Суша={len(os.listdir('train/0'))}")