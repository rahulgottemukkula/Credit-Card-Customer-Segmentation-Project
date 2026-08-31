import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import joblib
import os
import math
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from sklearn.decomposition import PCA

# Project directories
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

DATA_PATH = PROJECT_DIR / "Dataset" / "CreditCard_data.csv"
MODEL_PATH = PROJECT_DIR / "Models" / "kmeans_model.pkl"
SCALER_PATH = PROJECT_DIR / "Models" / "scaler.pkl"
st.set_page_config(
    page_title="Credit Card Customer Segmentation",
    page_icon="💳",
    layout="wide"
)
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

df = load_data()
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go To",
    [
        "🏠 Home",
        "📊 Dataset Analysis",
        "⚙ Dynamic Clustering",
        "📈 Cluster Visualization",
        "🧠 Predict Cluster"
    ]
)
if page == "🏠 Home":
    st.title("💳 Credit Card Customer Segmentation Dashboard")
    st.markdown("---")
    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown("""
        ## 📖 Project Overview

        This project applies **Unsupervised Machine Learning**
        to segment credit card customers into meaningful groups.

        The discovered clusters help banks and financial
        institutions identify different customer behaviours
        and design targeted marketing strategies.

        ### Objectives

        - Discover hidden customer groups
        - Compare clustering algorithms
        - Evaluate cluster quality
        - Visualize customer segments
        - Predict cluster membership for new customers
        """)
    with col2:

        st.info(
            """
            **Project Type**

            Unsupervised Learning

            **Algorithm**

            K-Means Clustering

            **Dataset**

            Credit Card Customers
            """
        )

    st.markdown("---")
    st.subheader("📊 Dataset Information")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Rows", df.shape[0])
    with c2:
        st.metric("Columns", df.shape[1])
    with c3:
        st.metric("Missing Values", int(df.isnull().sum().sum()))
    st.markdown("---")
    st.subheader("🧠 Machine Learning Workflow")
    st.success("""
    ✔ Data Loading

    ✔ Data Cleaning

    ✔ Exploratory Data Analysis (EDA)

    ✔ Feature Scaling

    ✔ K-Means Clustering

    ✔ Hierarchical Clustering

    ✔ DBSCAN

    ✔ PCA Visualization

    ✔ Model Evaluation

    ✔ Cluster Prediction
    """)
    st.markdown("---")
    st.subheader("📌 Algorithms Used")
    col1, col2 = st.columns(2)
    with col1:
        st.write("✅ K-Means Clustering")
        st.write("✅ Hierarchical Clustering")
        st.write("✅ DBSCAN")
    with col2:
        st.write("✅ StandardScaler")
        st.write("✅ PCA")
        st.write("✅ Silhouette Analysis")
    st.markdown("---")
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

# ======================================================
# DATASET ANALYSIS
# ======================================================
elif page == "📊 Dataset Analysis":
    st.title("📊 Dataset Analysis")
    st.markdown("---")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Dataset Preview",
            "Statistics",
            "Missing Values",
            "Correlation",
            "Feature Analysis"
        ]
    )
    # --------------------------------------------------
    # DATASET PREVIEW
    # --------------------------------------------------
    with tab1:
        st.subheader("Dataset Preview")
        rows = st.slider(
            "Number of rows to display",
            min_value=5,
            max_value=50,
            value=10
        )
        st.dataframe(df.head(rows), use_container_width=True)
        st.markdown("---")
        st.subheader("Dataset Shape")
        c1, c2 = st.columns(2)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
    # --------------------------------------------------
    # DESCRIPTIVE STATISTICS
    # --------------------------------------------------
    with tab2:
        st.subheader("Descriptive Statistics")
        numeric_df = df.select_dtypes(include=["number"])
        st.dataframe(numeric_df.describe().T, use_container_width=True)
    # --------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------
    with tab3:
        st.subheader("Missing Value Analysis")
        missing = df.isnull().sum()
        missing_df = missing.reset_index()
        missing_df.columns = ["Column", "Missing Values"]
        st.dataframe(missing_df, use_container_width=True)
        fig, ax = plt.subplots(figsize=(10,5))
        ax.bar(
            missing_df["Column"],
            missing_df["Missing Values"]
        )
        plt.xticks(rotation=90)
        st.pyplot(fig)
    # --------------------------------------------------
    # CORRELATION
    # --------------------------------------------------
    with tab4:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(12,8))
        numeric_df = df.select_dtypes(include=["number"])
        sns.heatmap(
            numeric_df.corr(),
            cmap="coolwarm",
            annot=True,
            ax=ax
        )
        st.pyplot(fig)
    # --------------------------------------------------
    # FEATURE ANALYSIS
    # --------------------------------------------------
    with tab5:
        st.subheader("Feature Analysis")
        numeric_columns = df.select_dtypes(include="number").columns.tolist()
        feature = st.selectbox(
            "Select Feature",
            numeric_columns
        )
        st.markdown("### Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean", round(df[feature].mean(),2))
        c2.metric("Median", round(df[feature].median(),2))
        c3.metric("Minimum", round(df[feature].min(),2))
        c4.metric("Maximum", round(df[feature].max(),2))
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Distribution")
            fig, ax = plt.subplots(figsize=(6,4))
            sns.histplot(
                df[feature],
                kde=True,
                ax=ax
            )
            st.pyplot(fig)
        with col2:
            st.markdown("### Box Plot")
            fig, ax = plt.subplots(figsize=(6,4))
            sns.boxplot(
                y=df[feature],
                ax=ax
            )
            st.pyplot(fig)
        st.markdown("---")
        st.subheader("Outlier Detection (IQR Method)")
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[
            (df[feature] < lower) |
            (df[feature] > upper)
        ]
        st.metric("Number of Outliers", len(outliers))
        if len(outliers) > 0:
            st.dataframe(
                outliers[[feature]],
                use_container_width=True
            )


# ======================================================
# DYNAMIC CLUSTERING
# ======================================================
elif page == "⚙ Dynamic Clustering":
    st.title("⚙ Dynamic Clustering")
    st.markdown(
        """
        Select the features you want to use for clustering.
        The application will automatically scale the data,
        determine the best number of clusters,
        and evaluate clustering performance.
        """
    )
    st.markdown("---")
    # Numeric Features
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    selected_features = st.multiselect(
        "Select Features",
        numeric_columns,
        default=numeric_columns
    )
    if len(selected_features) < 2:
        st.warning("Please select at least two features.")
        st.stop()
    st.markdown("---")
    X = df[selected_features].copy()
    X = X.fillna(X.median())
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    st.success("✅ Selected Features Successfully Scaled")
    st.markdown("---")
    mode = st.radio(
        "Choose Clustering Mode",
        [
            "Automatic Best K",
            "Manual K"
        ]
    )
    # =====================================================
    # AUTOMATIC
    # =====================================================
    if mode == "Automatic Best K":
        scores = []
        k_values = range(2,11)
        for k in k_values:
            model = KMeans(
                n_clusters=k,
                random_state=42,
                n_init=10
            )
            labels = model.fit_predict(X_scaled)
            score = silhouette_score(
                X_scaled,
                labels
            )
            scores.append(score)
        best_k = k_values[np.argmax(scores)]
        kmeans = KMeans(
            n_clusters=best_k,
            random_state=42,
            n_init=10
        )
        clusters = kmeans.fit_predict(X_scaled)
    # =====================================================
    # MANUAL
    # =====================================================
    else:
        best_k = st.slider(
            "Select Number of Clusters",
            2,
            10,
            4
        )
        kmeans = KMeans(
            n_clusters=best_k,
            random_state=42,
            n_init=10
        )
        clusters = kmeans.fit_predict(X_scaled)
    st.markdown("---")
    st.subheader("📊 Clustering Results")
    silhouette = silhouette_score(
        X_scaled,
        clusters
    )
    dbi = davies_bouldin_score(
        X_scaled,
        clusters
    )
    ch = calinski_harabasz_score(
        X_scaled,
        clusters
    )
    c1,c2,c3,c4 = st.columns(4)
    c1.metric(
        "Best K",
        best_k
    )
    c2.metric(
        "Silhouette Score",
        round(silhouette,3)
    )
    c3.metric(
        "Davies-Bouldin",
        round(dbi,3)
    )
    c4.metric(
        "Calinski-Harabasz",
        round(ch,2)
    )
    st.markdown("---")
    clustered_df = df.copy()
    clustered_df["Cluster"] = clusters
    st.subheader("Clustered Dataset")
    st.dataframe(
        clustered_df.head(),
        use_container_width=True
    )
    # =====================================================
    # ELBOW METHOD
    # =====================================================
    st.markdown("---")
    st.subheader("📉 Elbow Method")
    inertia = []
    for k in range(2, 11):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )
        model.fit(X_scaled)
        inertia.append(model.inertia_)
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(
        range(2,11),
        inertia,
        marker="o",
        linewidth=2
    )
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Curve")
    st.pyplot(fig)
    # =====================================================
    # SILHOUETTE SCORE CURVE
    # =====================================================
    st.markdown("---")
    st.subheader("📈 Silhouette Score Curve")
    sil_scores = []
    for k in range(2,11):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )
        labels = model.fit_predict(X_scaled)
        sil_scores.append(
            silhouette_score(
                X_scaled,
                labels
            )
        )
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(
        range(2,11),
        sil_scores,
        marker="o",
        linewidth=2
    )
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Analysis")
    st.pyplot(fig)
    # =====================================================
    # CLUSTER DISTRIBUTION
    # =====================================================
    st.markdown("---")
    st.subheader("📊 Cluster Distribution")
    cluster_counts = clustered_df["Cluster"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(
        cluster_counts.index.astype(str),
        cluster_counts.values
    )
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Customers")
    ax.set_title("Customers per Cluster")
    st.pyplot(fig)
    # =====================================================
    # PCA VISUALIZATION
    # =====================================================
    st.markdown("---")
    st.subheader("🎯 PCA Cluster Visualization")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame()
    pca_df["PC1"] = X_pca[:,0]
    pca_df["PC2"] = X_pca[:,1]
    pca_df["Cluster"] = clusters
    fig, ax = plt.subplots(figsize=(9,6))
    scatter = ax.scatter(
        pca_df["PC1"],
        pca_df["PC2"],
        c=pca_df["Cluster"]
    )
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("Customer Segments (PCA)")
    plt.colorbar(scatter)
    st.pyplot(fig)
    # =====================================================
    # CLUSTER CENTERS
    # =====================================================
    st.markdown("---")
    st.subheader("📍 Cluster Centers")
    centers = pd.DataFrame(
        kmeans.cluster_centers_,
        columns=selected_features
    )
    centers.index = [
        f"Cluster {i}"
        for i in range(best_k)
    ]
    st.dataframe(
        centers,
        use_container_width=True
    )

    st.dataframe(centers,use_container_width=True)
    # =====================================================
    # CLUSTER SUMMARY
    # =====================================================
    st.markdown("---")
    st.subheader("📋 Cluster Summary")
    summary = clustered_df.groupby("Cluster").mean(numeric_only=True)
    st.dataframe(
        summary,
        use_container_width=True
    )
    # =====================================================
    # CLUSTER SIZES
    # =====================================================
    st.markdown("---")
    st.subheader("👥 Customers in Each Cluster")
    cluster_size = clustered_df["Cluster"].value_counts().sort_index()
    cluster_table = pd.DataFrame({
        "Cluster": cluster_size.index,
        "Customers": cluster_size.values
    })
    st.dataframe(
        cluster_table,
        use_container_width=True
    )
    # =====================================================
    # CLUSTER PROFILE
    # =====================================================
    st.markdown("---")
    st.subheader("🧠 Cluster Insights")
    for cluster in summary.index:
        st.markdown(f"### Cluster {cluster}")
        top_features = (
            summary.loc[cluster]
            .sort_values(ascending=False)
            .head(5)
        )
        st.write(
            "Top Characteristics:"
        )
        st.dataframe(
            top_features,
            use_container_width=True
        )
    # =====================================================
    # DOWNLOAD DATASET
    # =====================================================
    st.markdown("---")
    st.subheader("⬇ Download Clustered Dataset")
    csv = clustered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name="clustered_customers.csv",
        mime="text/csv"
    )
    # =====================================================
    # SAVE MODEL
    # =====================================================
    st.markdown("---")
    st.subheader("💾 Save Model")
    save = st.button("Save KMeans Model")
    if save:
        joblib.dump(scaler, MODEL_PATH.parent / "scaler.pkl")
        joblib.dump(kmeans, MODEL_PATH)
        st.success("Model saved successfully!")

# ======================================================
# CLUSTER VISUALIZATION
# ======================================================
elif page == "📈 Cluster Visualization":
    st.title("📈 Cluster Visualization Dashboard")
    st.markdown(
        """
        Explore the generated customer segments.
        """
    )
    # -------------------------------------
    # Check saved model
    # -------------------------------------
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        st.warning("Saved model files were not found in the Models folder.")
        st.stop()
    scaler = joblib.load(SCALER_PATH)
    kmeans = joblib.load(MODEL_PATH)# -------------------------------------
    # Prepare Data
    # -------------------------------------
    numeric_df = df.select_dtypes(include=np.number)
    X = numeric_df.fillna(numeric_df.median())
    X_scaled = scaler.transform(X)
    clusters = kmeans.predict(X_scaled)
    clustered_df = numeric_df.copy()
    clustered_df["Cluster"] = clusters
    # -------------------------------------
    # Cluster Selection
    # -------------------------------------
    cluster_option = st.selectbox(
        "Select Cluster",
        sorted(clustered_df["Cluster"].unique())
    )
    cluster_data = clustered_df[
        clustered_df["Cluster"] == cluster_option
    ]
    st.success(
        f"Customers in Cluster {cluster_option}: {len(cluster_data)}"
    )
    st.markdown("---")
    # -------------------------------------
    # Dataset Preview
    # -------------------------------------
    st.subheader("Cluster Preview")
    st.dataframe(
        cluster_data.head(20),
        use_container_width=True
    )
    # -------------------------------------
    st.subheader("Cluster Statistics")
    st.dataframe(
        cluster_data.describe().T,
        use_container_width=True
    )
    # -------------------------------------
    # PCA Plot
    # -------------------------------------
    st.markdown("---")
    st.subheader("PCA Visualization")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "Cluster": clusters
    })
    fig, ax = plt.subplots(figsize=(10,6))
    scatter = ax.scatter(
        pca_df["PC1"],
        pca_df["PC2"],
        c=pca_df["Cluster"]
    )
    ax.set_title("Customer Segments (PCA)")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    plt.colorbar(scatter)
    st.pyplot(fig)
    # -------------------------------------
    # Feature Comparison
    # -------------------------------------
    st.markdown("---")
    st.subheader("Feature Comparison")
    feature = st.selectbox(
        "Choose Feature",
        numeric_df.columns
    )
    fig, ax = plt.subplots(figsize=(9,5))
    clustered_df.groupby("Cluster")[feature].mean().plot(
        kind="bar",
        ax=ax
    )
    ax.set_ylabel(feature)
    st.pyplot(fig)
    # -------------------------------------
    # Box Plot
    # -------------------------------------
    st.markdown("---")
    st.subheader("Feature Distribution")
    fig, ax = plt.subplots(figsize=(9,5))
    sns.boxplot(
        data=clustered_df,
        x="Cluster",
        y=feature,
        ax=ax
    )
    st.pyplot(fig)
    # -------------------------------------
    # Histogram
    # -------------------------------------
    st.markdown("---")
    st.subheader("Histogram")
    fig, ax = plt.subplots(figsize=(9,5))
    sns.histplot(
        data=clustered_df,
        x=feature,
        hue="Cluster",
        kde=True,
        ax=ax
    )
    st.pyplot(fig)
    # -------------------------------------
    # Cluster Means
    # -------------------------------------
    st.markdown("---")
    st.subheader("Cluster Profile")
    st.dataframe(
        clustered_df.groupby("Cluster").mean(),
        use_container_width=True
    )
# ======================================================
# PREDICT CLUSTER
# ======================================================
elif page == "🧠 Predict Cluster":
    st.title("🧠 Predict Customer Cluster")
    st.markdown("""
    Enter customer details below to predict the customer segment.
    """)
    st.markdown("---")
    # --------------------------------------------------
    # Check Model
    # --------------------------------------------------
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        st.error("Saved model files were not found in the Models folder.")
        st.stop()
    # --------------------------------------------------
    # Load Model
    # --------------------------------------------------
    scaler = joblib.load(SCALER_PATH)
    kmeans = joblib.load(MODEL_PATH)
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    input_data = {}
    st.subheader("Customer Details")
    col1, col2 = st.columns(2)
    for i, feature in enumerate(numeric_columns):
        default_value = float(df[feature].median())
        if i % 2 == 0:
            with col1:
                input_data[feature] = st.number_input(
                    feature,
                    value=default_value,
                    format="%.2f"
                )
        else:
            with col2:
                input_data[feature] = st.number_input(
                    feature,
                    value=default_value,
                    format="%.2f"
                )
    st.markdown("---")
    if st.button("Predict Cluster"):
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        prediction = kmeans.predict(input_scaled)[0]
        st.success(f"Predicted Customer Cluster : {prediction}")
        st.markdown("---")
        st.subheader("Prediction Summary")
        st.dataframe(input_df, use_container_width=True)
        st.metric(
            "Predicted Cluster",
            prediction
        )
        # -----------------------------
        # Simple Interpretation
        # -----------------------------
        st.subheader("Cluster Interpretation")
        if prediction == 0:
            st.info(
                "Cluster 0: Customers with relatively lower spending behaviour."
            )
        elif prediction == 1:
            st.info(
                "Cluster 1: Customers with moderate credit card usage."
            )
        elif prediction == 2:
            st.info(
                "Cluster 2: High-value customers with higher balances and purchases."
            )
        elif prediction == 3:
            st.info(
                "Cluster 3: Customers with different payment and credit utilisation patterns."
            )
        else:
            st.info(
                f"Cluster {prediction}: Customer segment identified by the trained model."
            )
