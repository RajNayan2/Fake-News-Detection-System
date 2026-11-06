
# Fake News Detection System

A Machine Learning-based Fake News Detection System that classifies news articles as **Real or Fake** using Natural Language Processing (NLP) and supervised learning algorithms.

## Features

- Classifies news articles as Real or Fake.
- Text preprocessing and cleaning.
- Tokenization and stopword removal.
- TF-IDF feature extraction.
- Multiple machine learning model implementation.
- Model performance evaluation and comparison.
- Achieved 96% test accuracy using SVM.

## Technologies Used

- Python
- Machine Learning
- Natural Language Processing (NLP)
- Scikit-learn
- TensorFlow
- Pandas
- NumPy
- TF-IDF Vectorization

## Machine Learning Algorithms

The following classification algorithms were implemented and evaluated:

1. Support Vector Machine (SVM)
2. Random Forest
3. Naive Bayes
4. Decision Tree

**Best Performing Model:** Support Vector Machine (SVM)

**Test Accuracy:** 96%

## Project Workflow

1. **Data Collection:** Collected real and fake news datasets.
2. **Data Preprocessing:** Cleaned text, removed unnecessary characters, and performed tokenization.
3. **Feature Extraction:** Applied TF-IDF vectorization to convert text into numerical features.
4. **Model Training:** Trained multiple machine learning classification algorithms.
5. **Model Evaluation:** Compared model performance using test data.
6. **Prediction:** Classified news articles as Real or Fake.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone <your-repository-url>
```

### 2. Navigate to the Project Directory

```bash
cd Fake-News-Detection-System
```

### 3. Install Dependencies

```bash
pip install pandas numpy scikit-learn tensorflow
```

### 4. Run the Project

Run the Python script or Jupyter Notebook used in your project.

```bash
python main.py
```

*Note: Adjust the execution command according to your actual project files.*

## Dataset

The project uses real and fake news datasets for training and testing the machine learning models.

Dataset source: Kaggle (True and Fake News Dataset).

## Results

The Support Vector Machine (SVM) model achieved **96% test accuracy**, outperforming the other evaluated models on the project's test dataset.

## Future Improvements

- Integrate real-time news verification APIs.
- Develop a web-based user interface.
- Improve model performance using advanced NLP techniques.
- Integrate external news sources for additional verification.
- Deploy the application as a web service.

## Author

**Raj Nayan**

B.Tech Computer Science and Engineering

---

If you find this project useful, consider giving the repository a star!
