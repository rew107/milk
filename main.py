# Импорт библиотек

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
from itertools import cycle
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from itertools import cycle
from sklearn.preprocessing import label_binarize

file_path = 'milknew.csv'
# Загрузка данных
# Предположим, что у вас есть файл данных в формате CSV с именем 'milk_quality_dataset.csv'
dataset = pd.read_csv(file_path)

# Предобработка данных
# Пример предобработки: заполнение пропущенных значений, кодирование категориальных переменных и т.д.
print("Имена столбцов:", dataset.columns)
# Определение независимых и зависимых переменных
X = dataset[['pH', 'Temprature', 'Taste', 'Odor', 'Fat ', 'Turbidity', 'Colour']]
y = dataset['Grade']

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабирование данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Создание и обучение модели случайного леса
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Предсказание на тестовом наборе
y_pred = model.predict(X_test_scaled)

# Оценка модели
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Вывод результатов
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)

# Вычисление матрицы ошибок
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Визуализация матрицы ошибок
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
classes = ['High', 'Low', 'Medium']
tick_marks = range(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Преобразование меток в бинарный формат для построения ROC-кривой
y_test_bin = label_binarize(y_test, classes=['High', 'Low', 'Medium'])
y_pred_bin = label_binarize(y_pred, classes=['High', 'Low', 'Medium'])

# Рассчитываем ROC-кривую и площадь под кривой (AUC) для каждого класса
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(len(classes)):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Визуализация ROC-кривой
plt.figure(figsize=(10, 6))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(len(classes)), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve (area = {roc_auc[i]:.2f}) for {classes[i]}')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
