# 💳 Credit Card Customer Segmentation using Unsupervised Machine Learning

An end-to-end Machine Learning project that segments credit card customers into meaningful groups using **K-Means Clustering**. This project includes complete data preprocessing, exploratory data analysis (EDA), clustering evaluation, PCA visualization, an interactive Streamlit dashboard, and customer cluster prediction.

---

## 📌 Project Overview

Customer segmentation is one of the most important applications of Unsupervised Machine Learning in the banking and financial sector.

Instead of predicting a target value, this project discovers hidden customer groups based on their credit card usage patterns. These customer segments can help banks:

- Identify high-value customers
- Detect low-engagement users
- Create personalized marketing campaigns
- Improve customer retention
- Optimize financial services

This project provides an interactive dashboard where users can dynamically perform clustering, visualize clusters, evaluate clustering quality, and predict the cluster of new customers.

---

# 🎯 Objectives

- Perform complete Exploratory Data Analysis (EDA)
- Clean and preprocess the dataset
- Handle missing values
- Scale numerical features
- Apply multiple clustering algorithms
- Determine the optimal number of clusters
- Evaluate clustering performance
- Visualize customer segments using PCA
- Save trained models
- Predict cluster membership for new customers
- Deploy everything inside an interactive Streamlit application

---

# 📂 Dataset

Dataset Used:

**Credit Card Customer Dataset**

The dataset contains customer information such as:

- Balance
- Purchases
- Cash Advance
- Credit Limit
- Payments
- Purchase Frequency
- Installments
- Transactions
- Minimum Payments
- Tenure
- and other financial attributes.

---

# 🛠 Technologies Used

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit

---

# 🧠 Machine Learning Workflow

```
Load Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Missing Value Handling
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Feature Scaling
(StandardScaler)
      │
      ▼
K-Means Clustering
      │
      ▼
Cluster Evaluation
      │
      ▼
PCA Visualization
      │
      ▼
Save Model
      │
      ▼
Predict New Customer Cluster
```

---

# ⚙ Algorithms Used

### Clustering Algorithms

- K-Means Clustering
- Hierarchical Clustering
- DBSCAN

### Data Preprocessing

- StandardScaler

### Dimensionality Reduction

- Principal Component Analysis (PCA)

### Cluster Evaluation Metrics

- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Score
- Elbow Method

---

# 📊 Features

## Home Dashboard

- Project overview
- Dataset information
- Machine Learning workflow
- Dataset preview

---

## Dataset Analysis

- Dataset preview
- Descriptive statistics
- Missing value analysis
- Correlation heatmap
- Feature distribution
- Boxplots
- Outlier detection (IQR)

---

## Dynamic Clustering

- Select any numerical features
- Automatic Best K selection
- Manual K selection
- Feature scaling
- Elbow curve
- Silhouette analysis
- Cluster distribution
- PCA visualization
- Cluster centers
- Cluster summary
- Download clustered dataset
- Save trained model

---

## Cluster Visualization

- Cluster-wise dataset preview
- Cluster statistics
- PCA scatter plot
- Feature comparison
- Boxplots
- Histograms
- Cluster profiling

---

## Predict Customer Cluster

- Enter customer details
- Predict customer segment
- View prediction summary
- Interpret customer behaviour

---

# 📈 Evaluation Metrics

This project evaluates clustering quality using:

- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index
- Elbow Method

These metrics help determine the optimal number of clusters and validate clustering performance.

---

# 📁 Project Structure

```
Credit-Card-Customer-Segmentation/
│
├── app.py
├── Credit_Card_Segmentation.ipynb
├── dataset/
│   └── CreditCard_data.csv
├── models/
│   ├── scaler.pkl
│   └── kmeans_model.pkl
├── screenshots/
├── requirements.txt
├── README.md
└── LICENSE
```

---

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/Credit-Card-Customer-Segmentation.git
```

---

## Navigate into Project

```bash
cd Credit-Card-Customer-Segmentation
```

---

## Create Virtual Environment

Windows

```bash
python -m venv venv
```

Activate

```bash
venv\Scripts\activate
```

Mac/Linux

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Streamlit App

```bash
streamlit run app.py
```

---

# 📸 Application Screenshots

Add screenshots here after uploading them.

Example:

```
screenshots/
    home.png
    clustering.png
    prediction.png
```

---

# Future Improvements

- Gaussian Mixture Models
- OPTICS Clustering
- Interactive Plotly visualizations
- Cluster recommendation engine
- Model deployment on Streamlit Cloud
- Docker support

---

# 👨‍💻 Author

**Rahul Gottemukkula**

B.Tech Computer Science and Engineering (Data Analytics)

Passionate about:

- Data Science
- Machine Learning
- Artificial Intelligence
- Data Analytics

LinkedIn:
(Add your profile)

GitHub:
(Add your GitHub)

---

# ⭐ If you found this project useful

Please consider giving this repository a ⭐
