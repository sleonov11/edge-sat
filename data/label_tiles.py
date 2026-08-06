import cv2
import pandas as pd
import numpy as np
import os
import sys
import argparse

TILE = 128

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img

def get_tile(img, x, y, tile_size=TILE):
    return img[y:y+tile_size, x:x+tile_size]

def save_labels(labels, filename):
    """Сохраняет метки в CSV, перезаписывая дубликаты последней меткой."""
    if os.path.exists(filename):
        existing = pd.read_csv(filename)
        new_df = pd.DataFrame(labels)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.drop_duplicates(subset=['image', 'x', 'y'], keep='last', inplace=True)
    else:
        combined = pd.DataFrame(labels)
    combined.to_csv(filename, index=False)

def main():
    parser = argparse.ArgumentParser(description='Ручная разметка тайлов')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Путь к папке с исходными изображениями')
    parser.add_argument('--features', type=str, default='features.csv',
                        help='Путь к features.csv (по умолчанию features.csv)')
    parser.add_argument('--labels_out', type=str, default='labels.csv',
                        help='Файл для сохранения меток (по умолчанию labels.csv)')
    args = parser.parse_args()

    image_dir = args.image_dir
    features_csv = args.features
    labels_csv = args.labels_out

    if not os.path.exists(features_csv):
        print(f"Файл {features_csv} не найден. Сначала запустите explore_features.py.")
        return

    df = pd.read_csv(features_csv)

    if os.path.exists(labels_csv):
        labels_df = pd.read_csv(labels_csv)
        df = df.merge(labels_df[['image', 'x', 'y', 'label']], on=['image', 'x', 'y'], how='left')
    else:
        df['label'] = float('nan')

    # Выбираем только неразмеченные тайлы
    unlabeled = df[df['label'].isna()]
    if len(unlabeled) == 0:
        print("Все тайлы уже размечены.")
        return

    unlabeled = unlabeled.sample(frac=1, random_state=42)
    print(f"Всего тайлов: {len(df)}, неразмечено: {len(unlabeled)}")
    print("Управление: 'y' – полезный, 'n' – пустой, 's' – пропустить, 'q' – выход")

    cv2.namedWindow("Tile", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tile", 256, 256)

    labels = []
    for idx, row in unlabeled.iterrows():
        img_file = row['image']
        x = int(row['x'])
        y = int(row['y'])
        img_path = os.path.join(image_dir, img_file)
        if not os.path.exists(img_path):
            print(f"Изображение {img_path} не найдено, пропускаем")
            continue

        img = load_image(img_path)
        tile = get_tile(img, x, y)
        cv2.imshow("Tile", tile)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('y'):
            label = 1
        elif key == ord('n'):
            label = 0
        elif key == ord('s'):
            continue
        else:
            print("Неверная клавиша, используйте y/n/s/q")
            continue

        labels.append({'image': img_file, 'x': x, 'y': y, 'label': label})

        if len(labels) % 10 == 0:
            save_labels(labels, labels_csv)

    if labels:
        save_labels(labels, labels_csv)

    cv2.destroyAllWindows()
    print(f"Готово. Метки сохранены в {labels_csv}")

if __name__ == '__main__':
    main()