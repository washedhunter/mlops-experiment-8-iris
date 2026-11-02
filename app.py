"""Streamlit web app for the Iris species classifier (deployed on Streamlit Community Cloud).

Run locally:  streamlit run app.py
"""
import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris

from src.train import FEATURES, SPECIES, train

st.set_page_config(page_title="Iris Classifier - MLOps", page_icon="🌸", layout="centered")


@st.cache_resource
def get_model():
    return train()


model, metrics = get_model()

st.title("🌸 Iris Species Classifier")
st.caption("MLOps Experiment 8 - Git, GitHub Actions, Docker, Kubernetes, Prometheus, Grafana, Streamlit")

c1, c2 = st.columns(2)
c1.metric("Test accuracy", f"{metrics['accuracy']:.2%}")
c2.metric("F1 (macro)", f"{metrics['f1_macro']:.2f}")

st.subheader("Enter flower measurements (cm)")
col_a, col_b = st.columns(2)
sepal_length = col_a.slider("Sepal length", 4.0, 8.0, 5.8, 0.1)
sepal_width = col_b.slider("Sepal width", 2.0, 4.5, 3.0, 0.1)
petal_length = col_a.slider("Petal length", 1.0, 7.0, 4.3, 0.1)
petal_width = col_b.slider("Petal width", 0.1, 2.5, 1.3, 0.1)

row = [[sepal_length, sepal_width, petal_length, petal_width]]
idx = int(model.predict(row)[0])
proba = model.predict_proba(row)[0]

st.success(f"Predicted species: **{SPECIES[idx].title()}** ({proba[idx]:.1%} confidence)")
st.bar_chart(pd.DataFrame({"probability": proba}, index=[s.title() for s in SPECIES]))

with st.expander("Dataset preview"):
    iris = load_iris(as_frame=True)
    df = iris.frame.rename(columns=dict(zip(iris.feature_names, FEATURES)))
    df["species"] = df["target"].map(dict(enumerate(SPECIES)))
    st.dataframe(df.drop(columns="target").head(20), use_container_width=True)
