import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer

# Load the dataset
url = 'https://github.com/YBIFoundation/Dataset/raw/main/Cancer.csv'
df = pd.read_csv(url)

# Convert Y variable into 0 and 1
df['diagnosis'] = df['diagnosis'].replace({'M': 0, 'B': 1})

# Split the data into features (X) and target variable (y)
X = df.drop(['diagnosis'], axis=1)
y = df['diagnosis']

# Handle missing values using SimpleImputer (replace NaN with mean)
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Split the imputed data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Create and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Print results
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(cm)
