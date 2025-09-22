
# PegasusArc Human Activity Recognition Using Smartphones

## Overview
This project implements a machine learning system for human activity recognition using the UCI Human Activity Recognition Using Smartphones dataset. The system classifies activities (e.g., walking, sitting, standing) based on smartphone sensor data (accelerometer and gyroscope). It compares a baseline model using all available features with a model using a reduced feature set selected via K-Means clustering.

## Dataset
The dataset is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones). It includes:
- **Training Data** (`X_train.txt`): Sensor measurements with 561 features.
- **Training Labels** (`y_train.txt`): Activity labels (1-6) corresponding to activities like walking, sitting, standing, etc.

## Project Structure
The project is implemented in a Jupyter Notebook (`human_activity_recognition.ipynb`) with the following workflow:
1. **Data Loading**: Downloads and extracts the UCI HAR dataset, loading training data and labels into pandas DataFrames.
2. **Preprocessing**: 
   - Encodes activity labels using `LabelEncoder`.
   - Scales features using `StandardScaler`.
3. **Baseline Model**: Trains a Gaussian Naive Bayes classifier using all 561 features.
4. **Feature Reduction**: Applies K-Means clustering to select a subset of 50 features.
5. **Reduced Model**: Trains a Gaussian Naive Bayes classifier on the reduced feature set.
6. **Evaluation**: Compares accuracy and training time for both models.

## Requirements
To run this project, install the following Python packages:
- `requests`
- `beautifulsoup4`
- `pandas`
- `scikit-learn`
- `numpy`


## 📌 Usage

### Run the Notebook
1. Open **`human_activity_recognition.ipynb`** in Jupyter Notebook or Google Colab.  
2. Execute the cells sequentially to:
   - Download and preprocess the dataset.  
   - Train the baseline and reduced models.  
   - Display performance metrics.  

---

## 📊 Output

| Model                     | Accuracy  | Training Time | Features Used |
|----------------------------|----------|---------------|---------------|
| **Baseline (All Features)** | ~73.15%  | ~0.44 sec     | 561           |
| **Reduced (K-Means, 50 Features)** | ~81.31%  | ~0.016 sec    | 50            |

✅ The reduced model achieves **higher accuracy** and **significantly faster training time** compared to the baseline.

---

## 🚀 Future Improvements
- Experiment with alternative feature selection methods (e.g., PCA, feature importance from tree-based models).  
- Test other classifiers (e.g., Random Forest, SVM, or neural networks) for better performance.  
- Incorporate the test dataset (`X_test.txt`, `y_test.txt`) for a comprehensive evaluation.  
- Implement **cross-validation** to ensure robust model performance.  

---

## 📜 License
This project is licensed under the **MIT License**. See the [LICENSE](./LICENSE) file for details.  

---

## 🙏 Acknowledgments
- **UCI Machine Learning Repository** for providing the dataset.  
- **scikit-learn** for the machine learning tools used in this project. 
