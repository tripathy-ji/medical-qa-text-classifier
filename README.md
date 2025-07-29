# Medical Q&A Classifier

## Project Description

This project aims to classify medical questions into different categories (e.g., symptoms, treatment, causes) based on their content. It utilizes the "comprehensive-medical-q-a-dataset" from Kaggle, which contains a collection of medical questions and their corresponding answers and categories. The project employs a natural language processing pipeline that includes text cleaning using NLTK, feature extraction using TF-IDF vectorization, and classification using a Logistic Regression model with L2 regularization to prevent overfitting.

## Project Structure

- `medical_qa_classifier.py`: The main Python script containing the code for data loading, cleaning, feature extraction, model training, evaluation, and visualization.
- `README.md`: This file, providing an overview of the project and instructions.
- `requirements.txt`: Lists the necessary Python libraries to run the project.

## Setup and Running the Project

To set up and run this project locally, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd medical-qa-classifier
    ```

2.  **Install dependencies:**

    Ensure you have Python and pip installed. Then, install the required libraries using the provided `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the main script:**

    Execute the Python script:

    ```bash
    python medical_qa_classifier.py
    ```

    This script will download the dataset (if not already present), preprocess the data, train the model, evaluate its performance, and display the evaluation metrics and a confusion matrix.

## Results

The Logistic Regression model achieved a test accuracy of approximately 98.00%. The classification report provides detailed precision, recall, and F1-scores for each category. A confusion matrix is plotted to visualize the model's performance across different classes, highlighting areas of strong performance and potential misclassifications (e.g., the 'considerations' category showed challenges in prediction).

## Future Improvements

- Investigate the performance issues with minority classes, such as the 'considerations' category, and explore techniques like oversampling, undersampling, or using different model architectures better suited for imbalanced datasets.
- Experiment with other text vectorization methods (e.g., Word Embeddings, Sentence Transformers) and more advanced machine learning models (e.g., Support Vector Machines, Naive Bayes, or deep learning models) to potentially improve classification accuracy and handle complex linguistic patterns.
- Implement cross-validation during model training for more robust evaluation and hyperparameter tuning.
