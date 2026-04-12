import streamlit as st
import numpy as np
import pandas as pd
from data_loader import get_reviews_and_labels
from tfidf_wrapper import (
    train_tfidf_model,
    predict_with_tfidf,
    evaluate_tfidf_cv,
    get_tfidf_top_features,
    get_tfidf_nonzero_features_count
)
from embedding_wrapper import (
    train_embedding_model,
    predict_with_embeddings,
    evaluate_embedding_cv,
    compute_cosine_similarity
)
from evaluation_wrapper import get_best_models_summary

st.set_page_config(page_title="Sentiment Analysis NLP Lab", layout="wide")

st.title("Sentiment Analysis NLP Lab")
st.markdown("""
Explore and experiment with different sentiment analysis approaches:
- **TF-IDF + Logistic Regression**: Traditional text features
- **Sentence Embeddings + Logistic Regression**: Modern pretrained semantic understanding

Tune regularization, adjust C values, test custom reviews, and compare models.
""")

reviews, labels = get_reviews_and_labels()

with st.sidebar:
    st.header("Configuration")
    
    model_type = st.radio(
        "Select Model Type",
        options=["TF-IDF + Logistic Regression", "Sentence Embeddings + Logistic Regression"],
        key="model_type"
    )
    
    if model_type == "TF-IDF + Logistic Regression":
        st.subheader("TF-IDF Settings")
        regularization = st.radio("Regularization", options=["L1", "L2"], key="reg_type")
        penalty_map = {"L1": "l1", "L2": "l2"}
        penalty = penalty_map[regularization]
        
        C_options = [0.001, 0.01, 0.1, 1, 2, 3, 5, 10, 100]
        C = st.select_slider("C Value (Inverse Regularization Strength)", options=C_options, value=1)
        
        st.markdown("### Advanced Options")
        with st.expander("Click to expand"):
            stop_words_enabled = st.checkbox("Remove Stop Words", value=True)
            sublinear_tf = st.checkbox("Use Sublinear TF Scaling", value=True)
            ngram_choice = st.radio("N-gram Range", options=["Unigrams + Bigrams (1,2)", "Unigrams (1,1)"])
            ngram_map = {"Unigrams (1,1)": (1, 1), "Unigrams + Bigrams (1,2)": (1, 2)}
            ngram_range = ngram_map[ngram_choice]
        
        stop_words = 'english' if stop_words_enabled else None
    
    else:
        st.subheader("Embedding Settings")
        st.info("Model: `all-MiniLM-L6-v2`\n\n384-dimensional semantic vectors")
        regularization = "L2"
        penalty = "l2"
        C_options = [0.001, 0.01, 0.1, 1, 2, 3, 5, 10, 100]
        C = st.select_slider("C Value (Inverse Regularization Strength)", options=C_options, value=1)

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["Predict", "Cross-Validation", "Comparison", "Similarity"])

with tab1:
    st.header("Custom Review Prediction")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        user_review = st.text_area("Enter a movie review:", placeholder="e.g., This movie was amazing!", height=100)
    
    with col2:
        st.write("")
        predict_button = st.button("Predict Sentiment", use_container_width=True)
    
    if predict_button and user_review.strip():
        try:
            if model_type == "TF-IDF + Logistic Regression":
                model, vectorizer, _ = train_tfidf_model(
                    reviews, labels, C, penalty=penalty,
                    ngram_range=ngram_range,
                    stop_words=stop_words,
                    sublinear_tf=sublinear_tf
                )
                prediction, confidence = predict_with_tfidf(model, vectorizer, user_review)
            else:
                model, embedder, _ = train_embedding_model(reviews, labels, C, penalty=penalty)
                prediction, confidence = predict_with_embeddings(model, embedder, user_review)
            
            st.success(f"Prediction: **{'Positive' if prediction == 1 else 'Negative'}**")
            st.metric("Confidence", f"{confidence:.1%}")
            
            if model_type == "TF-IDF + Logistic Regression":
                with st.expander("Model Interpretation"):
                    top_pos, top_neg = get_tfidf_top_features(model, vectorizer, top_n=5)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Top Positive Features")
                        for feat, coef in top_pos:
                            st.write(f"**{feat}**: {coef:.4f}")
                    with col2:
                        st.subheader("Top Negative Features")
                        for feat, coef in top_neg:
                            st.write(f"**{feat}**: {coef:.4f}")
            else:
                st.info("Embeddings use dense 384-dimensional semantic vectors. Individual dimensions are not human-interpretable like TF-IDF words.")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
    elif predict_button:
        st.warning("Please enter a review first.")

with tab2:
    st.header("5-Fold Cross-Validation Analysis")
    
    cv_col1, cv_col2 = st.columns([2, 1])
    
    with cv_col2:
        run_cv_button = st.button("Run Cross-Validation", use_container_width=True)
    
    if run_cv_button:
        try:
            with st.spinner("Running cross-validation..."):
                if model_type == "TF-IDF + Logistic Regression":
                    scores, mean_acc, std_dev = evaluate_tfidf_cv(
                        reviews, labels, C, penalty=penalty,
                        ngram_range=ngram_range,
                        stop_words=stop_words,
                        sublinear_tf=sublinear_tf,
                        cv=5
                    )
                    model, vectorizer, _ = train_tfidf_model(
                        reviews, labels, C, penalty=penalty,
                        ngram_range=ngram_range,
                        stop_words=stop_words,
                        sublinear_tf=sublinear_tf
                    )
                else:
                    scores, mean_acc, std_dev = evaluate_embedding_cv(
                        reviews, labels, C, penalty=penalty, cv=5
                    )
            
            st.success("Cross-validation complete!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Accuracy", f"{mean_acc:.4f}")
            with col2:
                st.metric("Std Dev", f"{std_dev:.4f}")
            with col3:
                st.metric("Folds", "5")
            
            st.subheader("Fold-by-Fold Results")
            fold_df = pd.DataFrame({
                'Fold': range(1, 6),
                'Accuracy': scores
            })
            st.dataframe(fold_df, use_container_width=True)
            
            if model_type == "TF-IDF + Logistic Regression":
                st.subheader("Feature Analysis (Trained on Full Dataset)")
                col1, col2 = st.columns(2)
                
                with col1:
                    nonzero_count = get_tfidf_nonzero_features_count(model, vectorizer)
                    st.metric("Non-zero Features", nonzero_count)
                
                with col2:
                    total_features = len(vectorizer.get_feature_names_out())
                    st.metric("Total Features", total_features)
                
                top_pos, top_neg = get_tfidf_top_features(model, vectorizer, top_n=5)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Top 5 Positive Features")
                    for feat, coef in top_pos:
                        st.write(f"**{feat}**: `{coef:.4f}`")
                with col2:
                    st.subheader("Top 5 Negative Features")
                    for feat, coef in top_neg:
                        st.write(f"**{feat}**: `{coef:.4f}`")
            else:
                st.info("Embedding dimensions (384D vectors) are not interpretable like TF-IDF words. The model learns semantic patterns automatically.")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")

with tab3:
    st.header("Model Comparison Summary")
    st.markdown("Baseline performance of three model configurations:")
    
    if st.button("Generate Comparison Table", use_container_width=True):
        try:
            with st.spinner("Computing baseline models..."):
                comparison_df = get_best_models_summary()
            st.dataframe(comparison_df, use_container_width=True)
            st.markdown("""
            **Notes:**
            - **TF-IDF + L1**: Sparse model; removes less important features (high regularization)
            - **TF-IDF + L2**: Dense model; keeps all features but reduces weights
            - **Embeddings + L2**: Semantic model; uses pretrained 384D vectors for better understanding
            """)
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.info("Click the button above to compute baseline model comparisons. This may take a moment.")

with tab4:
    st.header("Cosine Similarity Explorer")
    st.markdown("Measure semantic similarity between two texts using embeddings. Note: similarity ≠ sentiment.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        text1 = st.text_area("Text 1:", value="Amazing movie with great acting", height=80)
    
    with col2:
        text2 = st.text_area("Text 2:", value="Terrible acting and boring plot", height=80)
    
    if st.button("Compute Similarity", use_container_width=True):
        try:
            with st.spinner("Computing embeddings..."):
                similarity = compute_cosine_similarity(text1, text2)
            
            st.success(f"Cosine Similarity: **{similarity:.4f}**")
            
            if similarity > 0.7:
                interpretation = "Very similar semantically"
            elif similarity > 0.5:
                interpretation = "Moderately similar"
            elif similarity > 0.3:
                interpretation = "Somewhat related"
            else:
                interpretation = "Semantically distinct"
            st.markdown(f"**Interpretation**: {interpretation}")
            
            st.info("Cosine similarity measures how close the meaning of the texts is. High similarity means the sentences are semantically related, but it does not guarantee identical sentiment.")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    st.markdown("---")
    st.subheader("Example Comparisons from Training Data")
    
    examples = [
        ("Amazing movie with great acting", "Terrible acting and boring plot"),
        ("Loved it, fantastic experience", "Waste of time"),
        ("This was a great movie", "So bad I fell asleep"),
    ]
    
    for i, (ex1, ex2) in enumerate(examples):
        if st.button(f"Example {i+1}: Compute", key=f"ex{i}"):
            with st.spinner("Computing..."):
                sim = compute_cosine_similarity(ex1, ex2)
            st.write(f"**'{ex1}' vs '{ex2}'**")
            st.write(f"Similarity: **{sim:.4f}**")

st.markdown("---")
st.markdown("""
### Learn More
- **C Parameter**: Controls regularization strength. Lower C = stronger regularization. Higher C = weaker regularization.
- **L1 Regularization**: Sparse solution; can zero out features entirely.
- **L2 Regularization**: Dense solution; shrinks weights but keeps all features.
- **TF-IDF**: Counts word importance by frequency (traditional NLP).
- **Embeddings**: Uses pretrained neural models to capture semantic meaning.
- **Cosine Similarity**: Measures angle between vectors (0 = orthogonal, 1 = identical).
""")
