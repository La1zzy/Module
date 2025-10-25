import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

file_path = "medical_data_200_patients.csv"
df = pd.read_csv(file_path)

df = df.drop(columns=['ID', 'Ім\'я'])

pressure_split = df['Тиск (систол/діастол)'].str.split('/', expand=True)
df['Тиск (систол)'] = pd.to_numeric(pressure_split[0])
df['Тиск (діастол)'] = pd.to_numeric(pressure_split[1])
df = df.drop(columns=['Тиск (систол/діастол)'])

target_column = 'Хронічні захворювання'
y = df[target_column]
X = df.drop(columns=[target_column])

le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_.tolist()

X_encoded = pd.get_dummies(X, columns=['Стать', 'Куріння', 'Фіз активність'], drop_first=False)
feature_names = X_encoded.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y_encoded,
    test_size=0.2,
    random_state=42
)

dt_classifier = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_classifier.fit(X_train, y_train)

y_pred = dt_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)

plt.figure(figsize=(25, 15))
plot_tree(dt_classifier,
          feature_names=feature_names,
          class_names=class_names,
          filled=True,
          rounded=True,
          proportion=False,
          fontsize=8)

plt.title("Дерево рішень для класифікації хронічних захворювань (N=200)")
plt.tight_layout()
plt.savefig("decision_tree_N200_final.png")
plt.close()

feature_importance = pd.Series(dt_classifier.feature_importances_, index=feature_names)

print("--- ЗВІТ ПО ДЕРЕВУ РІШЕНЬ (N=200) ---")
print(f"\nТочність моделі (Accuracy) на тестовому наборі: {accuracy:.4f}")
print("\nЗвіт про класифікацію (Classification Report):\n")
print(report)
print("\nТОП-5 ВАЖЛИВИХ ОЗНАК:\n")
print(feature_importance[feature_importance > 0].sort_values(ascending=False).nlargest(5).to_string())