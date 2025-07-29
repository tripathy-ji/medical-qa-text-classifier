
import kagglehub
import os
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display # Import display for use in the script

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
     nltk.download('wordnet')

# Download the dataset
dataset_path = kagglehub.dataset_download(
    "thedevastator/comprehensive-medical-q-a-dataset"
)
print(f"Dataset downloaded to: {dataset_path}")

# Construct the full path to the train.csv file
train_csv_path = os.path.join(dataset_path, 'train.csv')

# Load the train.csv file into a pandas DataFrame
df = pd.read_csv(train_csv_path)

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Cleans the input text using NLTK."""
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and non-alphabetic tokens
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stop words and stem
    cleaned_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    # Join tokens back into a string
    return ' '.join(cleaned_tokens)

# Apply the cleaning function to the 'Question' and 'Answer' columns
df['cleaned_Question'] = df['Question'].apply(clean_text)
df['cleaned_Answer'] = df['Answer'].apply(clean_text)

# Display the first few rows with the new cleaned columns
print("Cleaned data preview:")
display(df[['Question', 'cleaned_Question', 'Answer', 'cleaned_Answer']].head())

# Select features (X) and labels (y)
X = df['cleaned_Question']
y = df['qtype']

# Instantiate TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the features
X_tfidf = tfidf_vectorizer.fit_transform(X)

print("Shape of TF-IDF matrix:", X_tfidf.shape)

# Find classes with only one sample
class_counts = y.value_counts()
single_sample_classes = class_counts[class_counts == 1].index

# Filter out samples belonging to these classes
filtered_indices = y[~y.isin(single_sample_classes)].index
X_filtered = X_tfidf[filtered_indices]
y_filtered = y[filtered_indices]

# Now split the filtered data
X_train, X_test, y_train, y_test = train_test_split(
    X_filtered,
    y_filtered,
    test_size=0.25,
    random_state=42,
    stratify=y_filtered
)

# Print the shapes of the resulting sets
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

# Instantiate the Logistic Regression model with L2 regularization
logistic_regression_model = LogisticRegression(random_state=42, penalty='l2', solver='liblinear')

# Fit the model to the training data
logistic_regression_model.fit(X_train, y_train)

print("Logistic Regression model trained successfully.")

# Predict the labels for the test set
y_pred = logistic_regression_model.predict(X_test)

# Calculate and print accuracy for the test set
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Calculate and print precision, recall, and F1-score for the test set
print("
Classification Report for Test Set:")
print(classification_report(y_test, y_pred))

# Predict the labels for the training set
y_train_pred = logistic_regression_model.predict(X_train)

# Calculate and print accuracy for the training set
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"
Training Accuracy: {train_accuracy:.4f}")

# Compare training and test accuracy to check for overfitting
print("
Overfitting Check:")
print(f"Training Accuracy ({train_accuracy:.4f}) vs Test Accuracy ({test_accuracy:.4f})")
if train_accuracy > test_accuracy:
    print("Training accuracy is higher than test accuracy, potentially indicating some overfitting.")
elif train_accuracy < test_accuracy:
    print("Test accuracy is higher than training accuracy, which is unusual and might need investigation.")
else:
    print("Training and test accuracies are similar, suggesting the model is generalizing well.")

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Get the unique class names from the test set
class_names = sorted(y_test.unique())

# Plot the confusion matrix using seaborn
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)

# Add labels and title
plt.title('Confusion Matrix for Logistic Regression Model')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Display the plot
plt.show()

# Interpret the confusion matrix
print("
Interpretation of the Confusion Matrix:")
print("The confusion matrix shows the counts of true positive, true negative, false positive, and false negative predictions for each class.")
print("The diagonal elements represent the number of instances where the predicted label matches the true label (correct predictions).")
print("Off-diagonal elements represent misclassifications.")
print("For example, the value in row i and column j indicates the number of instances of true class i that were predicted as class j.")
print("
Observations:")

