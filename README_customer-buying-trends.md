# A Neural Network Approach to Analyse Customer Metrics and Predict Buying Trends

A machine learning project that cleans customer data, engineers features and trains a **Multilayer Perceptron (MLP) neural network** to predict whether a customer will buy (subscribe). Built with **Python** and **PySpark MLlib** in **Google Colab**.

Developed by **Fathimath Shibla N** as part of an internship with **Tata Consultancy Services (TCS)**, Department of Computer Science and Information Technology, Yenepoya (Deemed to be University), Bengaluru Campus.

**Internal guide:** Mr. Syed Umar Abbasi

---

## Problem Statement

Customer data is usually large, unstructured and inconsistent. Extracting insights from it by hand is slow, and traditional methods often fail to capture complex patterns in buying behaviour. This project builds an automated pipeline that analyzes customer data and predicts buying trends.

## Objectives

**General objective:** develop a predictive model for customer buying behaviour.

**Specific objectives:**
- Clean and preprocess the raw data
- Transform raw data into a machine-readable format
- Build and train a neural network model
- Evaluate and improve model performance

## Results

| Metric | Score |
|---|---|
| Accuracy | 97.93% |
| Precision | 97.93% |
| Recall | 97.93% |

The model outputs a predicted label and class probabilities for each customer (`Subscribed` / `Not Subscribed`).

## Tech Stack

| Area | Tools |
|---|---|
| Language | Python |
| ML framework | PySpark MLlib |
| Environment | Google Colab |
| Visualization | Graphviz (model architecture diagram) |

## How It Works

```
Customer Data -> Data Preprocessing -> Feature Engineering -> Neural Network Model -> Prediction Output
```

1. **Data collection**: load the customer dataset (CSV).
2. **Data preprocessing**: remove null values and clean column names (special characters).
3. **Feature engineering**: encode categorical data into numeric form and assemble a feature vector.
4. **Feature scaling**: scale the features before training.
5. **Model design**: build an MLP classifier.
6. **Training and testing**: train on one split and predict on the held-out test set.
7. **Performance evaluation**: compute accuracy, precision and recall.

## Model Architecture

| Layer | Size |
|---|---|
| Input | 150 features |
| Hidden layer 1 | 16 neurons |
| Hidden layer 2 | 8 neurons |
| Output | 2 classes (Subscribed / Not Subscribed) |

```python
from pyspark.ml.classification import MultilayerPerceptronClassifier

input_size = len(train_df.select("features").first()[0])
layers = [input_size, 16, 8, 2]

mlp = MultilayerPerceptronClassifier(
    featuresCol="features",
    labelCol="label",
    layers=layers,
    maxIter=120,
    seed=42,
    blockSize=128
)

model = mlp.fit(train_df)
predictions = model.transform(test_df)
predictions.select("label", "prediction", "probability").show(10, truncate=False)
```

## Getting Started

### Prerequisites

- A Google account to run the notebook in [Google Colab](https://colab.research.google.com), or a local Python 3.9+ environment
- Java 8 or later (required by Spark when running locally)

### Run in Google Colab

1. Open the notebook (`customer_buying_trends.ipynb`) in Colab.
2. Upload your dataset, or mount Google Drive and set the file path.
3. Run all cells in order: install PySpark, preprocess, train, evaluate.

### Run locally

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install pyspark pandas graphviz jupyter
jupyter notebook
```

## Testing

All 10 test cases passed.

| ID | Test | Expected result | Status |
|---|---|---|---|
| TC_01 | Upload dataset | Dataset loaded successfully | Pass |
| TC_02 | Handle missing values | Null values removed | Pass |
| TC_03 | Column cleaning | Clean column names | Pass |
| TC_04 | Feature encoding | Categorical data converted to numeric | Pass |
| TC_05 | Train model | Model trained | Pass |
| TC_06 | Prediction | Predictions generated | Pass |
| TC_07 | Accuracy check | Accuracy displayed | Pass |
| TC_08 | Model saving | Model saved | Pass |
| TC_09 | Model loading | Model loaded | Pass |
| TC_10 | Visualization | Model graph displayed | Pass |

## Real-Time Deployment (Design)

The pipeline can be extended to live predictions:

- Collect data from live sources (websites, apps)
- Stream it continuously with Spark Streaming
- Deploy the trained model as an API
- Process incoming data and return predictions instantly

Example: an e-commerce site recommending products instantly based on user activity.

## Limitations

- Results depend on the quality and size of the dataset
- Model performance may vary with different data

## Future Work

- Improve model accuracy
- Train on larger and more diverse datasets

## Project Status

**Complete (100%)**

| Week | Milestone |
|---|---|
| 1 | Data preprocessing and preparation |
| 2 | Model development and training |
| 3 | Evaluation and performance improvement |

## Acknowledgements

- Tata Consultancy Services (TCS) for the internship opportunity
- Mr. Syed Umar Abbasi, internal guide
- Yenepoya (Deemed to be University)

## License

Add a license of your choice (for example, MIT) before publishing.
