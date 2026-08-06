import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Загружаем
features = pd.read_csv('features.csv')
labels = pd.read_csv('labels.csv')
df = features.merge(labels, on=['image', 'x', 'y'])

X = df[[f'f{i}' for i in range(16)]].values
y = df['label'].values

# Разделяем: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Обучаем Decision Tree (max_depth=3 — простое, интерпретируемое дерево)
clf = DecisionTreeClassifier(max_depth=3, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)

# Оценка
y_pred = clf.predict(X_test)
print("=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=['empty', 'useful']))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# Печатаем дерево
print("\n=== Decision Tree ===")
feature_names = [f'f{i}' for i in range(16)]
print(export_text(clf, feature_names=feature_names, decimals=3))