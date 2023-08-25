import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

file_path = '2021-2022 Football Player Stats.csv'

df = pd.read_csv(file_path, delimiter=';', encoding='cp1252')

print('Unikalne pozycje w danych: ', df['Pos'].unique())
print('Liczba rekordów: ', len(df))
positions = ['GK', 'DF', 'MF', 'FW']
print('Wybrane pozycje: ', positions)
df = df[df['Pos'].isin(positions)]
print('Liczba rekordów wybraniu pozycji:', len(df))
print('Czy w dataset są NaN: ', df.isnull().values.any())
original_columns = df.columns.copy()
print("Liczba wszystkich cech:", len(df.columns))

X = df.select_dtypes(include='number')

deleted_columns = set(original_columns) - set(X.columns)

print("Nazwy usuniętych kolumn:", deleted_columns)

print("Liczba numerycznych cech:", len(X.columns))
print('')

y = df['Pos']

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

classifier = RandomForestClassifier()
classifier.fit(X_std, y)
importance_scores = classifier.feature_importances_

feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importance_scores})

# Sort the features by importance score in descending order
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

n_features = 80
top_features = feature_importances.head(n_features)['Feature'].tolist()
selected_X = X[top_features]

scaler = StandardScaler()
selected_X = scaler.fit_transform(selected_X)

X_train, X_test, y_train, y_test = train_test_split(selected_X, y, test_size=0.2)

classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

class_labels = y_test.unique()
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

report = classification_report(y_test, y_pred)
print(report)
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].text(0.1, 0.5, report, fontsize=10, fontfamily='monospace')
axs[0].axis('off')
axs[0].set_title('Classification Report')

cm = confusion_matrix(y_test, y_pred)
class_labels = sorted(y_test.unique())
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=class_labels, yticklabels=class_labels,
            ax=axs[1])
axs[1].set_title('Confusion Matrix')
axs[1].set_xlabel('Predicted')
axs[1].set_ylabel('Actual')

position_colors = {'DF': '#8B008B', 'FW': '#9AC0DE', 'GK': '#2E8B57', 'MF': '#FFFF00'}

axs[2].scatter(X_pca[:, 0], X_pca[:, 1], c=[position_colors[pos] for pos in y], label=y)
handles = []
for position in position_colors:
    handle = axs[2].scatter([], [], c=position_colors[position], label=position)
    handles.append(handle)
axs[2].legend(handles, position_colors.keys(), title='Positions', markerscale=1.2, loc='upper right')
axs[2].set_xlabel('PC1')
axs[2].set_ylabel('PC2')
axs[2].set_title('Football Players - PCA')

plt.tight_layout()
plt.show()

































import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


df = pd.read_csv('2021-2022 Football Player Stats.csv', delimiter=';', encoding='cp1252')
positions = ['GK', 'DF', 'MF', 'FW']
df = df[df['Pos'].isin(positions)]
X = df.select_dtypes(include='number')
y = df['Pos']

scaler = StandardScaler()
X = scaler.fit_transform(X)

pca = PCA(n_components=X.shape[1])

pca.fit(X)
eigenvalues = pca.explained_variance_
elbow_index=1.9
plt.figure(figsize=(10, 6))
n_components = np.arange(1, 21)
# n_components = np.arange(1, 139)
# plt.scatter(n_components, eigenvalues)
plt.scatter(n_components, eigenvalues[:20])
plt.xlabel('Liczba czynników')
plt.ylabel('Wartości własne')
plt.title('Wykres osypiska (Scree plot)')
xmin, xmax = plt.xlim()  # Get the x-axis limits
plt.axhline(y=elbow_index, color='red', linestyle='--', xmin=xmin, xmax=xmax)
plt.xticks(n_components)
plt.show()
































from time import perf_counter
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA

df = pd.read_csv('2021-2022 Football Player Stats.csv', delimiter=';', encoding='cp1252')
positions = ['GK', 'DF', 'MF', 'FW']
df = df[df['Pos'].isin(positions)]
X = df.select_dtypes(include='number')
y = df['Pos']

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

classifier = RandomForestClassifier()

classifier.fit(X_std, y)

times = []
best_ks=[]
best_scores=[[],[],[]]
best_k = 0
best_score = 0
importance_scores = classifier.feature_importances_

n_components = [13]

for i in n_components:
    print(i)


    n_components = i
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_std)
    classifier = RandomForestClassifier()
    classifier2 = KNeighborsClassifier()
    classifier3 = SVC()

    start_time = perf_counter()
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_std)
    scores = cross_val_score(classifier, X_pca, y, cv=5)
    best_scores[0].append(scores.mean())
    end_time = perf_counter()
    times.append(end_time-start_time)

    start_time = perf_counter()
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_std)
    scores2 = cross_val_score(classifier2, X_pca, y, cv=5)
    best_scores[1].append(scores2.mean())
    end_time = perf_counter()
    times.append(end_time-start_time)

    start_time = perf_counter()
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_std)
    scores3 = cross_val_score(classifier3, X_pca, y, cv=5)
    best_scores[2].append(scores3.mean())
    end_time = perf_counter()
    times.append(end_time-start_time)

    best_ks.append(i)



print(best_ks)
for i in range(3):
    print(best_scores[i], f"Elapsed time: {times[i]:.4f} seconds")






























import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('2021-2022 Football Player Stats.csv', delimiter=';', encoding='cp1252')
positions = ['GK', 'DF', 'MF', 'FW']
df = df[df['Pos'].isin(positions)]
X = df.select_dtypes(include='number')
y = df['Pos']

# Standardize the features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

forest = RandomForestClassifier().fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

forest_importances = pd.Series(importances)

fig, ax = plt.subplots()
colors = ['green' if i in [89, 100, 101, 97, 55] else 'blue' for i in range(len(forest_importances))]
forest_importances.plot.bar(yerr=std, color=colors, xticks=list(range(0, 138, 10)) + [138])
ax.set_title("Znaczenie cech (feature importances)")
ax.set_ylabel("Średni spadek szumu")
ax.set_xlabel("Indeks cechy")

fig.tight_layout()
plt.show()

importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
importance_df = importance_df.sort_values('Importance', ascending=False)
top_features = importance_df.head(5)['Feature']

fig, ax = plt.subplots()
sorted_colors = ['green' if feature in top_features.values else 'blue' for feature in importance_df['Feature']]
sorted_yerr = [std[importance_df['Feature'].tolist().index(feature)] for feature in importance_df['Feature']]
importance_df.plot.bar(x='Feature', y='Importance', xticks=list(range(0, 138, 10)) + [138], color=sorted_colors, yerr=sorted_yerr, ax=ax)
ax.set_title("Znaczenie cech (posortowane)")
ax.set_ylabel("Średni spadek szumu")
ax.set_xlabel("Numer cechy ")

fig.tight_layout()
plt.show()

print("5 najważniejszych cech:")
print(top_features)

















from time import perf_counter
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('2021-2022 Football Player Stats.csv', delimiter=';', encoding='cp1252')
positions = ['GK', 'DF', 'MF', 'FW']
df = df[df['Pos'].isin(positions)]
X = df.select_dtypes(include='number')
y = df['Pos']
start_time = perf_counter()

scaler = StandardScaler()
X_std = scaler.fit_transform(X)
classifier = RandomForestClassifier()
classifier.fit(X_std, y)



k_values = []
scores = []
importance_scores = classifier.feature_importances_

classifier = RandomForestClassifier()

feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importance_scores})

feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
for k in range(1, X.shape[1] + 1):
    print(k)


    top_features = feature_importances.head(k)['Feature'].tolist()

    selected_X = X[top_features]


    scaler = StandardScaler()
    selected_X = scaler.fit_transform(selected_X)

    scores.append(cross_val_score(classifier, selected_X, y, cv=5).mean())
    k_values.append(k)

best_index = scores.index(max(scores))
best_k = k_values[best_index]
print("Best k value:", best_k)
print("Best score:", max(scores))
end_time = perf_counter()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.4f} seconds")
plt.plot(k_values, scores)
plt.xlabel('Number of Features (k)')
plt.ylabel('Score')
plt.title('Performance of RandomForest with Features Importance')
plt.show()













import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler

file_path = '2021-2022 Football Player Stats.csv'

df = pd.read_csv(file_path, delimiter=';', encoding='cp1252')
positions = ['GK', 'DF', 'MF', 'FW']
df = df[df['Pos'].isin(positions)]
X = df.select_dtypes(include='number')
y = df['Pos']

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

svc = SVC()

k_values = []
scores = []

for k in range(1, X.shape[1] + 1):
    print(k)
    feature_selector = SelectKBest(score_func=chi2, k=k)
    X_selected = feature_selector.fit_transform(X, y)

    selected_indices = feature_selector.get_support(indices=True)

    selected_features = df.columns[selected_indices]
    selected_X = pd.DataFrame(X_selected, columns=selected_features)

    scaler = StandardScaler()
    selected_X = scaler.fit_transform(selected_X)

    scores.append(cross_val_score(svc, selected_X, y, cv=5).mean())
    k_values.append(k)


best_index = scores.index(max(scores))
best_k = k_values[best_index]
print("Best k value:", best_k)
print("Best score:", max(scores))

plt.plot(k_values, scores)
plt.xlabel('Number of Features (k)')
plt.ylabel('Score')
plt.title('Performance of SVC with SelectKBest')
plt.show()

















import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from time import perf_counter


df = pd.read_csv('2021-2022 Football Player Stats.csv', delimiter=';', encoding='cp1252')
positions = ['GK', 'DF', 'MF', 'FW']
df = df[df['Pos'].isin(positions)]
X = df.select_dtypes(include='number')
y = df['Pos']
start_time = perf_counter()

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

classifier = RandomForestClassifier()
classifier.fit(X_std, y)



k_values = []
scores = []
importance_scores = classifier.feature_importances_

classifier = KNeighborsClassifier()

feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importance_scores})

feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
for k in range(1, X.shape[1] + 1):
    print(k)


    top_features = feature_importances.head(k)['Feature'].tolist()

    selected_X = X[top_features]


    scaler = StandardScaler()
    selected_X = scaler.fit_transform(selected_X)

    scores.append(cross_val_score(classifier, selected_X, y, cv=5).mean())
    k_values.append(k)

best_index = scores.index(max(scores))
best_k = k_values[best_index]
print("Best k value:", best_k)
print("Best score:", max(scores))
end_time = perf_counter()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.4f} seconds")
# Plot the scores against k values
plt.plot(k_values, scores)
plt.xlabel('Number of Features (k)')
plt.ylabel('Score')
plt.title('Performance of KNN with Features Importance')
plt.show()