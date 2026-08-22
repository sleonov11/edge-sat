import os

path_1 = "train/1"
path_0 = "train/0"

count_1 = len(os.listdir(path_1))
count_0 = len(os.listdir(path_0))

print(f"Воды (класс 1): {count_1}")
print(f"Суши (класс 0): {count_0}")
print(f"Всего изображений: {count_1 + count_0}")