# Загрузка библиотек
import pandas as pd
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import HashingVectorizer

# Загрузка датасета
print("Загрузка датасета:\n")
url = "heart.csv"
dataset = read_csv(url)

# Оценка строк и столбцов данных
print("Оценка кол-ва строк и столбцов данных:\n")
print(dataset.shape)

# Срез данных head
print("Срез данных head:\n")
print(dataset.head(20))

# Стастическая сводка методом describe
print("Стастическая сводка методом describe:\n")
print(dataset.describe())

# Распределение по атрибуту target
print("Распределение по атрибуту target(высокая вероятность сердечного приступа):\n")
print(dataset.groupby('target').size())

# Гистограмма распределения атрибутов датасета
print("Гистограмма распределения атрибутов датасета:\n")
dataset.hist()
pyplot.show()

# Разделение датасета на обучающую и контрольную выборки
print("Разделение датасета на обучающую и контрольную выборки:\n")
array = dataset.values

# Выбор первых 13-х столбцов 
X = array[:,:13]
print (X)
# Выбор 14-го столбца 
y = array[:,13]
print (y)

# Разделение X и y на обучающую и контрольную выборки 
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.33, random_state=1337)

# Загружаем алгоритмы модели
print("Загрузка алгоритмов модели:\n")
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# Оцениваем модель на каждой итерации (10 кросс-валидаций)
print("Оценка модели на каждой итерации:\n")
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Сравниванием алгоритмы
print("Сравнение алгоритмов:\n")
pyplot.boxplot(results, labels=names)
pyplot.title('Сравнение качества алгоритмов')
pyplot.show()

# Создаем прогноз на контрольной выборке
print("Создание прогноза на контрольной выборке:\n")
model = LogisticRegression(solver='liblinear', multi_class='ovr')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Оцениваем прогноз
print("Оценка прогноза:\n")
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
