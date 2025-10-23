"""
Streamlit UI wrapper for the user's GradientBoostingClassifier pipeline + analysis.
This reproduces the original script behavior and visualizations, without changing the model.
Save as `app.py` and run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import streamlit as st
st.set_page_config(layout="wide", page_title="Model Explorer", page_icon="📊")

# --- Standard imports (same as original script) ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, log_loss
)
from sklearn.preprocessing import RobustScaler, OneHotEncoder, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.calibration import calibration_curve

import io, time, textwrap, traceback, os, warnings
warnings.filterwarnings("ignore")

# ---------------------- Helpers ----------------------
@st.cache_data
def load_dataframe(uploaded_file, default_path="test.csv"):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    elif os.path.exists(default_path):
        return pd.read_csv(default_path)
    else:
        return None

def run_full_pipeline(df, show_progress=True):
    # Wrap the user's original pipeline and analysis as closely as possible.
    # Returns a dict of results and figures to render in the UI.
    result = {}
    try:
        # --- Prelim ---
        if 'id' in df.columns:
            df = df.drop('id', axis=1)
        df = df.dropna(subset=['satisfaction'])

        y = df['satisfaction'].astype('category').cat.codes
        X = df.drop('satisfaction', axis=1)
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = [col for col in X.columns if col not in cat_cols]

        # --- Preprocessing ---
        num_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])
        cat_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        preprocessor = ColumnTransformer([
            ('num', num_pipe, num_cols),
            ('cat', cat_pipe, cat_cols)
        ])

        # --- Split & SMOTE ---
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.8, random_state=42, stratify=y
        )
        X_train_proc = preprocessor.fit_transform(X_train)
        X_test_proc = preprocessor.transform(X_test)
        X_train_proc, y_train_proc = SMOTE(random_state=42).fit_resample(X_train_proc, y_train)

        # --- Model (unchanged hyperparams) ---
        gb = GradientBoostingClassifier(
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=2,
            n_estimators=100,
            random_state=42
        )
        gb.fit(X_train_proc, y_train_proc)
        y_train_pred = gb.predict(X_train_proc)
        y_test_pred = gb.predict(X_test_proc)
        y_test_proba = gb.predict_proba(X_test_proc)

        # --- Statistics ---
        train_f1 = f1_score(y_train_proc, y_train_pred, average='macro')
        test_f1 = f1_score(y_test, y_test_pred, average='macro')
        train_acc = accuracy_score(y_train_proc, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        fit_diff = train_f1 - test_f1
        logloss = log_loss(y_test, y_test_proba)

        result['metrics'] = {
            'train_f1': float(train_f1),
            'test_f1': float(test_f1),
            'train_acc': float(train_acc),
            'test_acc': float(test_acc),
            'fit_diff': float(fit_diff),
            'logloss': float(logloss),
            'classification_report': classification_report(y_test, y_test_pred, output_dict=False)
        }

        # --- Feature names & classes (matching original) ---
        n_classes = len(np.unique(y))
        # get feature names (preserve same strategy as original script)
        if len(cat_cols) > 0:
            onehot = preprocessor.named_transformers_['cat']['onehot']
            onehot_names = list(onehot.get_feature_names_out(cat_cols))
        else:
            onehot_names = []
        feature_names = num_cols + onehot_names
        indices = np.argsort(gb.feature_importances_)[::-1]
        if n_classes > 2:
            y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))
        else:
            y_test_bin = None

        result['model'] = gb
        result['preprocessor'] = preprocessor
        result['X_test_proc'] = X_test_proc
        result['y_test'] = y_test
        result['y_test_proba'] = y_test_proba
        result['y_test_pred'] = y_test_pred
        result['feature_names'] = feature_names
        result['indices'] = indices
        result['n_classes'] = n_classes
        result['y_test_bin'] = y_test_bin

        # --- Create and store figures (matplotlib objects) ---
        figs = {}

        # 1. Confusion Matrix
        fig1, ax1 = plt.subplots(figsize=(6,5))
        sns.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title("Confusion Matrix")
        ax1.set_xlabel("Predicted")
        ax1.set_ylabel("True")
        figs['confusion'] = fig1

        # 2. ROC Curve
        fig2, ax2 = plt.subplots(figsize=(6,5))
        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_test, y_test_proba[:, 1])
            auc = roc_auc_score(y_test, y_test_proba[:, 1])
            ax2.plot(fpr, tpr, label=f'AUC={auc:.2f}')
            ax2.plot([0, 1], [0, 1], 'k--')
            ax2.set_title('ROC Curve')
            ax2.set_xlabel('False Positive Rate')
            ax2.set_ylabel('True Positive Rate')
            ax2.legend()
        else:
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_test_proba[:, i])
                ax2.plot(fpr, tpr, label=f'Class {i} (AUC={roc_auc_score(y_test_bin[:, i], y_test_proba[:, i]):.2f})')
            ax2.plot([0, 1], [0, 1], 'k--')
            ax2.set_title('ROC Curve (multi-class)')
            ax2.set_xlabel('False Positive Rate')
            ax2.set_ylabel('True Positive Rate')
            ax2.legend()
        figs['roc'] = fig2

        # 3. Precision-Recall Curve
        fig3, ax3 = plt.subplots(figsize=(6,5))
        if n_classes == 2:
            precision, recall, _ = precision_recall_curve(y_test, y_test_proba[:, 1])
            ax3.plot(recall, precision, label='PR Curve')
            ax3.set_title('Precision-Recall Curve')
            ax3.set_xlabel('Recall')
            ax3.set_ylabel('Precision')
            ax3.legend()
        else:
            for i in range(n_classes):
                precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_test_proba[:, i])
                ax3.plot(recall, precision, label=f'Class {i}')
            ax3.set_title('Precision-Recall Curve (multi-class)')
            ax3.set_xlabel('Recall')
            ax3.set_ylabel('Precision')
            ax3.legend()
        figs['pr'] = fig3

        # 4. Feature Importances
        fig4, ax4 = plt.subplots(figsize=(10,4))
        ax4.bar(range(len(indices)), gb.feature_importances_[indices], align="center")
        ax4.set_xticks(range(len(indices)))
        ax4.set_xticklabels(np.array(feature_names)[indices], rotation=90, fontsize=8)
        ax4.set_title("Feature Importances")
        fig4.tight_layout()
        figs['feat_imp'] = fig4

        # 5. Permutation Importance
        perm = permutation_importance(gb, X_test_proc, y_test, n_repeats=10, random_state=42)
        sorted_idx = perm.importances_mean.argsort()[::-1]
        fig5, ax5 = plt.subplots(figsize=(10,4))
        ax5.bar(range(len(sorted_idx)), perm.importances_mean[sorted_idx], align="center")
        ax5.set_xticks(range(len(sorted_idx)))
        ax5.set_xticklabels(np.array(feature_names)[sorted_idx], rotation=90, fontsize=8)
        ax5.set_title("Permutation Importances")
        fig5.tight_layout()
        figs['perm_imp'] = fig5

        # 6. Learning Curve
        train_sizes, train_scores, test_scores = learning_curve(
            gb, X_train_proc, y_train_proc, cv=5, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5), scoring='f1_macro'
        )
        fig6, ax6 = plt.subplots(figsize=(7,5))
        ax6.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label="Train F1")
        ax6.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label="CV F1")
        ax6.set_xlabel("Training examples")
        ax6.set_ylabel("F1 Score")
        ax6.set_title("Learning Curve")
        ax6.legend()
        figs['learning_curve'] = fig6

        # 7. Partial Dependence Plot (top 2 features) - wrapped in try
        try:
            fig7, axes7 = plt.subplots(1, 2, figsize=(14,5))
            # Use indices identified earlier
            PartialDependenceDisplay.from_estimator(
                gb, X_test_proc, [indices[0], indices[1]], feature_names=np.array(feature_names)[indices[:2]], ax=axes7
            )
            figs['pdp'] = fig7
        except Exception as e:
            # store the exception message to show in the UI instead of a figure
            figs['pdp_error'] = str(e)

        # 8. Prediction Probability Histogram
        fig8, ax8 = plt.subplots(figsize=(6,5))
        ax8.hist(np.max(y_test_proba, axis=1), bins=20)
        ax8.set_title("Histogram of Max Prediction Probabilities (Test)")
        ax8.set_xlabel("Max Probability")
        ax8.set_ylabel("Frequency")
        figs['prob_hist'] = fig8

        # 9. Actual vs Predicted Counts
        fig9, ax9 = plt.subplots(figsize=(6,4))
        sns.countplot(x=y_test, label='Actual', ax=ax9, alpha=0.6)
        sns.countplot(x=y_test_pred, label='Predicted', ax=ax9, alpha=0.4)
        ax9.set_title("Actual vs Predicted Class Counts")
        ax9.legend(['Actual', 'Predicted'])
        figs['counts'] = fig9

        # 10. Calibration Curve (same computation as original script)
        fig10, ax10 = plt.subplots(figsize=(6,5))
        prob_true, prob_pred = calibration_curve(y_test == y_test_pred, np.max(y_test_proba, axis=1), n_bins=10)
        ax10.plot(prob_pred, prob_true, marker='o')
        ax10.plot([0,1],[0,1],'k--')
        ax10.set_title("Calibration Curve")
        ax10.set_xlabel("Mean Predicted Probability")
        ax10.set_ylabel("Fraction of Positives")
        figs['calibration'] = fig10

        # 11. Cumulative Gains Chart (matching original)
        fig11, ax11 = plt.subplots(figsize=(6,5))
        fpr, tpr, thresholds = roc_curve((y_test == y_test_pred), np.max(y_test_proba, axis=1))
        ax11.plot(tpr, label='Cumulative Gains')
        ax11.plot([0,1],[0,1],'k--')
        ax11.set_title("Cumulative Gains Chart")
        ax11.set_xlabel("Proportion of Samples")
        ax11.set_ylabel("Proportion of Positives")
        ax11.legend()
        figs['gains'] = fig11

        result['figs'] = figs

        return result

    except Exception as e:
        # Return the traceback for display so the user can troubleshoot
        return {'error': traceback.format_exc()}


# ---------------------- Streamlit UI ----------------------
st.title("📊 Model Explorer — UI Wrapper (keeps your model & analysis intact)")
st.markdown(
    """
    This app wraps your original script into a friendly UI.  
    — Model, preprocessing, SMOTE, and plots are unchanged from your script.  
    — Upload a CSV with a `satisfaction` column or place `test.csv` in the working directory.
    """
)

with st.sidebar:
    st.header("Run options")
    uploaded_file = st.file_uploader("Upload CSV (optional)", type=["csv"])
    run_button = st.button("Run analysis", type="primary")
    st.markdown("---")
    st.write("Notes:")
    st.write("- The model hyperparameters and metrics are preserved exactly.")
    st.write("- If your original script errors, the stack trace will be shown below.")
    st.write("- Requires `satisfaction` column in dataset.")

# Show sample or preview
df_preview = load_dataframe(uploaded_file)
if df_preview is None:
    st.warning("No `test.csv` found and no file uploaded. Upload a CSV with a `satisfaction` column or place `test.csv` in the working directory.")
    st.stop()

st.subheader("Data preview")
st.dataframe(df_preview.head(200))

# Run when button pressed
if run_button:
    with st.spinner("Running pipeline and generating visualizations... (this may take a little while)"):
        result = run_full_pipeline(df_preview)
    if result is None:
        st.error("No result returned. Something went wrong.")
        st.stop()

    if 'error' in result:
        st.error("Pipeline raised an exception — displaying traceback:")
        st.code(result['error'])
        st.stop()

    # Metrics panel
    metrics = result['metrics']
    st.subheader("Model metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Train F1 (macro)", f"{metrics['train_f1']:.4f}", delta=None)
    c1.metric("Train Acc", f"{metrics['train_acc']:.4f}")
    c2.metric("Test F1 (macro)", f"{metrics['test_f1']:.4f}")
    c2.metric("Test Acc", f"{metrics['test_acc']:.4f}")
    c3.metric("Fit Difference (TrainF1 - TestF1)", f"{metrics['fit_diff']:.4f}")
    c3.metric("Test Log Loss", f"{metrics['logloss']:.4f}")

    st.markdown("**Classification report:**")
    st.text(metrics['classification_report'])

    # Layout plots in tabs
    figs = result['figs']
    tab1, tab2, tab3 = st.tabs(["Conf/ROC/PR", "Importances & Permutation", "Learning & PDP & Others"])

    with tab1:
        st.subheader("Confusion matrix")
        st.pyplot(figs['confusion'])
        st.subheader("ROC curve")
        st.pyplot(figs['roc'])
        st.subheader("Precision-Recall curve")
        st.pyplot(figs['pr'])

    with tab2:
        st.subheader("Feature importances (model)")
        st.pyplot(figs['feat_imp'])
        st.subheader("Permutation importances (test set)")
        st.pyplot(figs['perm_imp'])

    with tab3:
        st.subheader("Learning curve")
        st.pyplot(figs['learning_curve'])

        st.subheader("Partial dependence (top 2 features)")
        if 'pdp' in figs:
            st.pyplot(figs['pdp'])
        else:
            st.error("Partial dependence plot failed:")
            st.text(figs.get('pdp_error', 'Unknown error'))

        st.subheader("Probability histogram")
        st.pyplot(figs['prob_hist'])

        st.subheader("Actual vs Predicted counts")
        st.pyplot(figs['counts'])

        st.subheader("Calibration curve")
        st.pyplot(figs['calibration'])

        st.subheader("Cumulative gains chart")
        st.pyplot(figs['gains'])

    st.success("All statistics and visualizations complete for Model 1.")

    # Option to download model & preprocessor (pickle)
    import joblib
    buffer = io.BytesIO()
    artifact = {
        'model': result['model'],
        'preprocessor': result['preprocessor'],
        'feature_names': result['feature_names']
    }
    joblib.dump(artifact, buffer)
    buffer.seek(0)
    st.download_button("Download model + preprocessor (joblib)", buffer, file_name="model_artifact.joblib")

    st.info("Finished. If you want the same UI as a desktop app (CustomTkinter) or prefer a different layout, tell me and I'll produce that next.")
