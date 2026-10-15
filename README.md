# Fake Review Detection Using Machine Learning 

## Project Overview
Fake Review Detection is a machine learning project that identifies whether an online product review is Fake (CG – Computer Generated) or Real (OR – Original). The project uses review text, ratings, categories, and linguistic features to classify reviews.

## Objectives
- Detect fake and genuine online reviews.
- Compare traditional machine learning and deep learning approaches.
- Evaluate the importance of textual and linguistic features.
- Identify the most effective model using performance metrics.

## Models Used

### Module 1: TF-IDF + Linear SVM
This module converts review text into numerical features using TF-IDF and classifies reviews using a Linear Support Vector Machine (SVM).

Three experiments are performed:
- TF-IDF features only
- Linguistic features only
- TF-IDF and linguistic features combined

### Module 2: Hybrid BiLSTM + Attention + Feature Fusion
This module uses a Bidirectional Long Short-Term Memory (BiLSTM) network with an Attention mechanism. Text, linguistic, and category features are combined through Feature Fusion and passed through dense layers for classification.

## Dataset
The dataset used is `MLdatasetfinal.csv`.

- Total reviews: 1,909
- Fake reviews (CG): 955
- Real reviews (OR): 954
- Training data: 80%
- Testing data: 20%

### Features
- Review text
- Rating
- Category
- Text length
- Sentence count
- Average word length
- Vocabulary richness

## Technologies Used
- Python
- Jupyter Notebook
- Pandas and NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- Natural Language Processing (NLP)

## Evaluation Metrics
The models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score

## Results

| Model | Accuracy | F1-Score |
|---|---:|---:|
| TF-IDF + Linear SVM | 0.99476 | 0.99476 |
| Linguistic Features + Linear SVM | 0.87696 | 0.88279 |
| TF-IDF + Linguistic Features + Linear SVM | 0.98691 | 0.98708 |
| Hybrid BiLSTM + Attention + Feature Fusion | 0.99480 | 0.99480 |

The Hybrid BiLSTM model achieved an F1-score of 0.99480, while TF-IDF with Linear SVM achieved 0.99476. Both models performed very well on the test dataset.

## Project Structure
- `Model1_TFIDF_SVM.ipynb` – Traditional machine learning experiments
- `Model2_BiLSTM_Attention.ipynb` – Hybrid deep learning model
- `MLdatasetfinal.csv` – Dataset used for training and testing

*Note: Update the filenames above if your actual notebook filenames are different.*

## Conclusion
This project compares traditional machine learning and hybrid deep learning techniques for fake review detection. The results demonstrate that both TF-IDF with Linear SVM and Hybrid BiLSTM with Attention can effectively classify fake and real reviews.

## Future Enhancements
- Test the models on additional review datasets.
- Evaluate cross-category performance.
- Improve model interpretability.
- Develop a web application for real-time review classification.
