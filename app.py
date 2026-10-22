import time
import random
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from joblib import Parallel, delayed

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.base import clone

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    precision_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Universal ML Feature Reduction Studio", page_icon="⚡", layout="wide")


# ==========================================
# 1. Auto-Data Preprocessor
# ==========================================

def auto_preprocess_dataset(df, target_col, user_id_cols=None):
    df_clean = df.copy()
    
    cols_to_drop = list(user_id_cols) if user_id_cols else []
    for col in df_clean.columns:
        if col == target_col or col in cols_to_drop:
            continue
        col_lower = str(col).lower()
        if any(keyword in col_lower for keyword in ['id', 'uuid', 'code', 'date', 'time', 'timestamp']):
            cols_to_drop.append(col)
        elif df_clean[col].nunique() == len(df_clean) and df_clean[col].dtype == 'object':
            cols_to_drop.append(col)

    df_clean = df_clean.drop(columns=cols_to_drop, errors='ignore')
    df_clean = df_clean.dropna(subset=[target_col])
    
    y_raw = df_clean[target_col]
    X_raw = df_clean.drop(columns=[target_col])
    
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y_raw), name=target_col)
    
    numeric_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_raw.select_dtypes(exclude=[np.number]).columns.tolist()
    
    if len(numeric_cols) > 0:
        num_imputer = SimpleImputer(strategy='median')
        X_num = pd.DataFrame(num_imputer.fit_transform(X_raw[numeric_cols]), columns=numeric_cols)
    else:
        X_num = pd.DataFrame()
        
    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X_cat_imp = pd.DataFrame(cat_imputer.fit_transform(X_raw[categorical_cols]), columns=categorical_cols)
        X_cat = pd.get_dummies(X_cat_imp, drop_first=True, dtype=float)
        X = pd.concat([X_num, X_cat], axis=1)
    else:
        X = X_num
        
    X.columns = [str(c).replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_') for c in X.columns]
    
    constant_cols = [c for c in X.columns if X[c].nunique() <= 1]
    X = X.drop(columns=constant_cols)
    
    return X, y, cols_to_drop


# ==========================================
# 2. Feature Reduction Algorithms
# ==========================================

class FastGeneticOptimizer:
    def __init__(self, fitness_func, n_agents=14, n_dim=30, max_iter=30, patience=8):
        self.fitness_func = fitness_func
        self.n_agents = n_agents
        self.n_dim = n_dim
        self.max_iter = max_iter
        self.patience = patience
        self.p = min(0.5, np.sqrt(self.n_dim) / self.n_dim)
        self.cutoff = 0.02
        self.X = np.random.choice([0, 1], size=(n_agents, n_dim), p=[1 - self.p, self.p])
        self.X_best = np.zeros(n_dim)
        self.best_score = -float('inf')

    def optimize(self):
        no_improve = 0
        for _ in range(self.max_iter):
            scores = Parallel(n_jobs=-1, prefer="threads")(
                delayed(self.fitness_func)(self.X[i]) for i in range(self.n_agents)
            )
            improved = False
            for i, score in enumerate(scores):
                if score > self.best_score + 1e-5:
                    self.best_score = score
                    self.X_best = self.X[i].copy()
                    improved = True
            if improved:
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

            new_X = [self.X_best.copy()]
            weights = [max(0.0, s) + 1e-6 for s in scores]
            while len(new_X) < self.n_agents:
                p1, p2 = random.choices(self.X, weights=weights, k=2)
                cut = random.randint(1, max(1, self.n_dim - 1))
                child = np.concatenate([p1[:cut], p2[cut:]])
                if random.random() < self.cutoff:
                    m_idx = random.randint(0, self.n_dim - 1)
                    child[m_idx] = 1 - child[m_idx]
                new_X.append(child)
            self.X = np.array(new_X)
        return self.X_best


class FastBoardOptimizer:
    def __init__(self, fitness_func, n_agents=14, n_dim=30, max_iter=30, patience=8):
        self.fitness_func = fitness_func
        self.n_agents = n_agents
        self.n_dim = n_dim
        self.max_iter = max_iter
        self.patience = patience
        self.p = min(0.5, np.sqrt(self.n_dim) / self.n_dim)
        self.cutoff = 3.0 / max(1, self.n_dim)
        self.X = np.random.choice([0, 1], size=(n_agents, n_dim), p=[1 - self.p, self.p])
        self.X_best = np.zeros(n_dim)
        self.best_score = -float('inf')

    def optimize(self):
        scores = Parallel(n_jobs=-1, prefer="threads")(
            delayed(self.fitness_func)(self.X[i]) for i in range(self.n_agents)
        )
        for i, score in enumerate(scores):
            if score > self.best_score:
                self.best_score = score
                self.X_best = self.X[i].copy()

        no_improve = 0
        for _ in range(self.max_iter):
            for i in range(self.n_agents):
                if np.array_equal(self.X[i], self.X_best):
                    continue
                p = np.random.rand()
                if p < 0.5:
                    mask = (self.X[i] != self.X_best) & (np.random.rand(self.n_dim) < 0.1)
                    self.X[i, mask] = 1 - self.X[i, mask]
                elif p < 0.75:
                    mask = np.random.rand(self.n_dim) < self.cutoff
                    self.X[i, mask] = 1 - self.X[i, mask]
                else:
                    self.X[i] = np.random.choice([0, 1], size=self.n_dim, p=[1 - self.p, self.p])

            scores = Parallel(n_jobs=-1, prefer="threads")(
                delayed(self.fitness_func)(self.X[i]) for i in range(self.n_agents)
            )
            improved = False
            for i, score in enumerate(scores):
                if score > self.best_score + 1e-5:
                    self.best_score = score
                    self.X_best = self.X[i].copy()
                    improved = True
            if improved:
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break
        return self.X_best


def _eval_col(model, metric_func, X_tr, y_tr, X_val, y_val, cols):
    clf = clone(model)
    clf.fit(X_tr[cols], y_tr)
    return metric_func(y_val, clf.predict(X_val[cols]))


def forward_selection_fast(model, metric_func, X_tr, y_tr, X_val, y_val, n_feat, max_candidates=60):
    remaining = list(X_tr.columns)
    random.seed(42)
    random.shuffle(remaining)
    selected = []
    max_f = min(n_feat, len(remaining))
    best_overall = -float('inf')

    for _ in range(max_f):
        candidates = remaining[:max_candidates] if len(remaining) > max_candidates else remaining
        scores = Parallel(n_jobs=-1, prefer="threads")(
            delayed(_eval_col)(model, metric_func, X_tr, y_tr, X_val, y_val, selected + [c])
            for c in candidates
        )
        best_idx = int(np.argmax(scores))
        best_overall = scores[best_idx]
        best_c = candidates[best_idx]
        selected.append(best_c)
        remaining.remove(best_c)
    return selected, best_overall


def backward_selection_fast(model, metric_func, X_tr, y_tr, X_val, y_val, baseline_score):
    selected = list(X_tr.columns)
    threshold = baseline_score * 0.97
    candidates = selected.copy()
    random.seed(42)
    random.shuffle(candidates)

    for col in candidates:
        if len(selected) <= 1:
            break
        trial = [c for c in selected if c != col]
        score = _eval_col(model, metric_func, X_tr, y_tr, X_val, y_val, trial)
        if score >= threshold:
            selected.remove(col)
        else:
            break
    return selected


# ==========================================
# 3. Models & Evaluation Metrics
# ==========================================

def get_default_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=42, class_weight='balanced'),
        "MLP (Neural Network)": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42, early_stopping=True),
        "SVM": SVC(random_state=42, class_weight='balanced', probability=True),
        "KNN": KNeighborsClassifier(n_jobs=-1),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        "Perceptron": Perceptron(random_state=42, n_jobs=-1, class_weight='balanced')
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric='logloss')
    if HAS_LGBM:
        models["LightGBM"] = LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1, class_weight='balanced', verbose=-1)
    return models


def compute_comprehensive_metrics(clf, X_te, y_true, y_pred, avg_m, classes):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=avg_m, zero_division=0)
    rec = recall_score(y_true, y_pred, average=avg_m, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=avg_m, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    roc_auc = np.nan
    pr_auc = np.nan
    
    try:
        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(X_te)
            if len(classes) == 2:
                roc_auc = roc_auc_score(y_true, y_score[:, 1])
                pr_auc = average_precision_score(y_true, y_score[:, 1])
            else:
                roc_auc = roc_auc_score(y_true, y_score, multi_class='ovr', average='macro')
                y_bin = label_binarize(y_true, classes=classes)
                pr_auc = average_precision_score(y_bin, y_score, average='macro')
        elif hasattr(clf, "decision_function"):
            y_score = clf.decision_function(X_te)
            if len(classes) == 2:
                roc_auc = roc_auc_score(y_true, y_score)
                pr_auc = average_precision_score(y_true, y_score)
    except Exception:
        pass

    return {
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1_Score": round(f1, 4),
        "ROC_AUC": round(float(roc_auc), 4) if not np.isnan(roc_auc) else np.nan,
        "PR_AUC": round(float(pr_auc), 4) if not np.isnan(pr_auc) else np.nan,
        "MCC": round(mcc, 4)
    }


# ==========================================
# 4. Main Application Interface
# ==========================================

st.title("⚡ Dynamic Feature Reduction & ML Benchmark Studio")
st.markdown("Upload any dataset file (`.csv` or `.xlsx`). The system will automatically clean the data, select features, and benchmark industry-standard models.")

uploaded_file = st.file_uploader("📂 Choose Dataset File:", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Cannot open file: {e}")
        st.stop()

    st.subheader("1. Dataset Configuration")
    c1, c2 = st.columns([1, 1])
    with c1:
        default_target_idx = 0 if 'Bankrupt?' in df_raw.columns else len(df_raw.columns) - 1
        target_col = st.selectbox("Select Target Column:", df_raw.columns, index=default_target_idx)
    with c2:
        other_cols = [c for c in df_raw.columns if c != target_col]
        user_drop_cols = st.multiselect("Select Columns to Drop (Optional):", other_cols)

    X, y, auto_dropped = auto_preprocess_dataset(df_raw, target_col, user_drop_cols)

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Total Samples", f"{X.shape[0]:,}")
    col_m2.metric("Available Features", f"{X.shape[1]:,}")
    col_m3.metric("Target Classes", f"{len(np.unique(y))}")
    
    class_dist = pd.Series(y).value_counts()
    min_max_ratio = (class_dist.min() / class_dist.max()) * 100
    col_m4.metric("Class Balance Ratio", f"{min_max_ratio:.1f}%",
                  delta="Balanced" if min_max_ratio >= 35 else "Imbalanced",
                  delta_color="normal" if min_max_ratio >= 35 else "inverse")

    if auto_dropped:
        st.caption(f"ℹ️ System automatically dropped {len(auto_dropped)} unnecessary columns: {', '.join(auto_dropped[:5])}{'...' if len(auto_dropped) > 5 else ''}")

    st.divider()

    st.subheader("2. Select Methods, Models & Optimization Metric")
    m_col1, m_col2, m_col3 = st.columns([2, 2, 1])
    with m_col1:
        methods_selected = st.multiselect(
            "Select Feature Reduction Methods:",
            ["Full Features", "Variance Threshold", "Correlation Matrix", "PCA", "Forward Selection", "Backward Selection", "Genetic Algorithm", "New Model (Binary Swarm)"],
            default=["Full Features", "Variance Threshold", "PCA", "Forward Selection", "New Model (Binary Swarm)"]
        )
    with m_col2:
        all_models = get_default_models()
        default_model_picks = ["Logistic Regression", "Random Forest", "Hist Gradient Boosting", "MLP (Neural Network)"]
        if HAS_XGB:
            default_model_picks.append("XGBoost")
        if HAS_LGBM:
            default_model_picks.append("LightGBM")
            
        models_selected = st.multiselect(
            "Select Machine Learning Models:",
            list(all_models.keys()),
            default=[m for m in default_model_picks if m in all_models]
        )
    with m_col3:
        opt_metric_choice = st.selectbox(
            "Wrapper Search Metric:",
            ["F1_Score", "MCC", "Accuracy", "Recall", "Precision"],
            index=0
        )

    default_k = max(2, min(15, int(X.shape[1] * 0.4)))
    target_k = st.slider("Target Number of Features (Target k) for Forward / PCA:", min_value=2, max_value=max(3, X.shape[1]), value=default_k)

    # ปุ่มคำนวณและเก็บผลลง session_state
    if st.button("🚀 Compute & Run Benchmark", type="primary", use_container_width=True):
        if not methods_selected or not models_selected:
            st.warning("Please select at least one method and one model.")
        else:
            prog_bar = st.progress(0, text="Preparing and splitting dataset...")
            
            unique_classes = np.unique(y)
            avg_m = 'macro' if len(unique_classes) > 2 else 'binary'
            metric_fns = {
                'Accuracy': lambda yt, yp: accuracy_score(yt, yp),
                'Recall': lambda yt, yp: recall_score(yt, yp, average=avg_m, zero_division=0),
                'F1_Score': lambda yt, yp: f1_score(yt, yp, average=avg_m, zero_division=0),
                'Precision': lambda yt, yp: precision_score(yt, yp, average=avg_m, zero_division=0),
                'MCC': lambda yt, yp: matthews_corrcoef(yt, yp)
            }
            opt_metric = metric_fns[opt_metric_choice]

            X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
                X_train_full, y_train_full, test_size=0.3, random_state=42, stratify=y_train_full
            )

            scaler_full = StandardScaler().set_output(transform="pandas")
            X_tr_scaled = scaler_full.fit_transform(X_train_full)
            X_te_scaled = scaler_full.transform(X_test_full)

            scaler_sub = StandardScaler().set_output(transform="pandas")
            X_tr_sub_s = scaler_sub.fit_transform(X_tr_raw)
            X_val_sub_s = scaler_sub.transform(X_val_raw)

            results_list = []
            selected_features_dict = {}
            total_ops = len(methods_selected)
            cur_op = 0

            def record_run(method_name, model_name, X_tr_use, X_te_use, sel_time, feat_list):
                clf = clone(all_models[model_name])
                t_start = time.time()
                clf.fit(X_tr_use, y_train_full)
                y_pred = clf.predict(X_te_use)
                train_time = time.time() - t_start

                eval_scores = compute_comprehensive_metrics(clf, X_te_use, y_test_full, y_pred, avg_m, unique_classes)

                row_data = {
                    "Method": method_name,
                    "Model": model_name,
                    "Num_Features": X_tr_use.shape[1],
                    "Selection_Time_s": round(sel_time, 4),
                    "Train_Time_s": round(train_time, 4),
                    **eval_scores
                }
                results_list.append(row_data)
                selected_features_dict[f"{method_name} - {model_name}"] = feat_list

            if "Full Features" in methods_selected:
                cur_op += 1
                prog_bar.progress(cur_op / total_ops, text="Processing: Full Features...")
                for m_name in models_selected:
                    record_run("Full Features", m_name, X_tr_scaled, X_te_scaled, 0.0, list(X.columns))

            t0 = time.time()
            mm_scaler = MinMaxScaler().set_output(transform="pandas")
            X_tr_mm = mm_scaler.fit_transform(X_train_full)
            X_te_mm = mm_scaler.transform(X_test_full)
            try:
                vt = VarianceThreshold(threshold=0.035).set_output(transform="pandas")
                X_tr_vt = vt.fit_transform(X_tr_mm)
                X_te_vt = vt.transform(X_te_mm)
                vt_cols = list(X_tr_vt.columns)
            except ValueError:
                top_idx = X_tr_mm.var().sort_values(ascending=False).index[:max(2, target_k)]
                vt_cols = list(top_idx)
                X_tr_vt, X_te_vt = X_tr_mm[vt_cols], X_te_mm[vt_cols]
            vt_time = time.time() - t0

            if "Variance Threshold" in methods_selected:
                cur_op += 1
                prog_bar.progress(cur_op / total_ops, text="Processing: Variance Threshold...")
                std_vt = StandardScaler().set_output(transform="pandas")
                X_tr_vt_s = std_vt.fit_transform(X_tr_vt)
                X_te_vt_s = std_vt.transform(X_te_vt)
                for m_name in models_selected:
                    record_run("Variance Threshold", m_name, X_tr_vt_s, X_te_vt_s, vt_time, vt_cols)

            if "Correlation Matrix" in methods_selected:
                cur_op += 1
                prog_bar.progress(cur_op / total_ops, text="Processing: Correlation Matrix...")
                t0 = time.time()
                corr_m = X_train_full.corr().abs()
                upper = corr_m.where(np.triu(np.ones(corr_m.shape), k=1).astype(bool))
                to_drop = [c for c in upper.columns if any(upper[c] > 0.65)]
                corr_cols = [c for c in X_train_full.columns if c not in to_drop]
                if not corr_cols:
                    corr_cols = [X_train_full.columns[0]]
                c_time = time.time() - t0
                for m_name in models_selected:
                    record_run("Correlation Matrix", m_name, X_tr_scaled[corr_cols], X_te_scaled[corr_cols], c_time, corr_cols)

            if "PCA" in methods_selected:
                cur_op += 1
                prog_bar.progress(cur_op / total_ops, text="Processing: PCA...")
                t0 = time.time()
                n_comp = min(X_tr_scaled.shape[0], X_tr_scaled.shape[1], target_k)
                pca = PCA(n_components=n_comp, random_state=42)
                pca_cols = [f"PC_{i+1}" for i in range(n_comp)]
                X_tr_pca = pd.DataFrame(pca.fit_transform(X_tr_scaled), columns=pca_cols)
                X_te_pca = pd.DataFrame(pca.transform(X_te_scaled), columns=pca_cols)
                pca_time = time.time() - t0
                for m_name in models_selected:
                    record_run("PCA", m_name, X_tr_pca, X_te_pca, pca_time, pca_cols)

            search_features = vt_cols[:60] if len(vt_cols) > 60 else vt_cols
            X_tr_search = X_tr_sub_s[search_features]
            X_val_search = X_val_sub_s[search_features]
            surrogate_model = GaussianNB()

            if "Forward Selection" in methods_selected:
                cur_op += 1
                prog_bar.progress(cur_op / total_ops, text="Processing: Forward Selection...")
                for m_name in models_selected:
                    t0 = time.time()
                    s_cols, _ = forward_selection_fast(surrogate_model, opt_metric, X_tr_search, y_tr, X_val_search, y_val, target_k)
                    f_time = time.time() - t0
                    record_run("Forward Selection", m_name, X_tr_scaled[s_cols], X_te_scaled[s_cols], f_time, s_cols)

            if "Backward Selection" in methods_selected:
                cur_op += 1
                prog_bar.progress(cur_op / total_ops, text="Processing: Backward Selection...")
                for m_name in models_selected:
                    t0 = time.time()
                    base_c = clone(surrogate_model).fit(X_tr_search, y_tr)
                    b_score = opt_metric(y_val, base_c.predict(X_val_search))
                    s_cols = backward_selection_fast(surrogate_model, opt_metric, X_tr_search, y_tr, X_val_search, y_val, b_score)
                    b_time = time.time() - t0
                    record_run("Backward Selection", m_name, X_tr_scaled[s_cols], X_te_scaled[s_cols], b_time, s_cols)

            for m_label, OptCls in [("Genetic Algorithm", FastGeneticOptimizer), ("New Model (Binary Swarm)", FastBoardOptimizer)]:
                if m_label in methods_selected:
                    cur_op += 1
                    prog_bar.progress(cur_op / total_ops, text=f"Processing: {m_label}...")
                    n_dim_s = len(search_features)
                    X_tr_np = X_tr_search.values
                    X_val_np = X_val_search.values
                    search_f_arr = np.array(search_features)

                    for m_name in models_selected:
                        def fit_fn(pos):
                            idx = np.where(pos == 1)[0]
                            if len(idx) == 0:
                                return 0.0
                            clf_opt = clone(surrogate_model)
                            clf_opt.fit(X_tr_np[:, idx], y_tr)
                            sc = opt_metric(y_val, clf_opt.predict(X_val_np[:, idx]))
                            penalty = 1.0 - (len(idx) / n_dim_s)
                            return (0.85 * sc) + (0.15 * penalty)

                        t0 = time.time()
                        opt = OptCls(fit_fn, n_agents=12, n_dim=n_dim_s, max_iter=25, patience=6)
                        best_mask = opt.optimize()
                        final_idx = np.where(best_mask == 1)[0]
                        if len(final_idx) == 0:
                            final_idx = np.array([0])
                        s_cols = list(search_f_arr[final_idx])
                        opt_time = time.time() - t0
                        record_run(m_label, m_name, X_tr_scaled[s_cols], X_te_scaled[s_cols], opt_time, s_cols)

            prog_bar.empty()
            
            # บันทึกผลลัพธ์ลง session_state ป้องกันการเด้งเริ่มใหม่
            st.session_state['benchmark_df'] = pd.DataFrame(results_list)
            st.session_state['selected_features'] = selected_features_dict
            st.session_state['available_features_count'] = X.shape[1]

    # ดึงผลลัพธ์จาก session_state มาแสดงผล (คลิกเปลี่ยนตัวชี้วัดแล้วจะไม่เริ่มคำนวณใหม่)
    if 'benchmark_df' in st.session_state:
        df_res = st.session_state['benchmark_df']
        selected_features_dict = st.session_state['selected_features']
        total_feats = st.session_state['available_features_count']

        st.success("🎉 Benchmark completed successfully!")

        t1, t2, t3, t4 = st.tabs([
            "🏆 Model Performance Comparison",
            "⚡ Speed vs Performance Trade-off",
            "🧬 Selected Features",
            "📋 Full Metrics Table & Export"
        ])
        
        with t1:
            available_plot_metrics = ["F1_Score", "ROC_AUC", "PR_AUC", "MCC", "Accuracy", "Recall", "Precision"]
            chosen_plot_metric = st.radio(
                "Select Evaluation Metric to Visualize:",
                available_plot_metrics,
                horizontal=True,
                key="chosen_metric_radio"
            )
            df_plot = df_res.dropna(subset=[chosen_plot_metric])
            fig_bar = px.bar(
                df_plot, x="Model", y=chosen_plot_metric, color="Method", barmode="group", text_auto=".3f",
                title=f"{chosen_plot_metric} Comparison by Model and Feature Reduction Method"
            )
            if not df_plot.empty:
                fig_bar.update_layout(yaxis_range=[max(0.0, df_plot[chosen_plot_metric].min() - 0.1), 1.05])
            st.plotly_chart(fig_bar, use_container_width=True)

        with t2:
            fig_scatter = px.scatter(
                df_res, x="Selection_Time_s", y="F1_Score", color="Method", symbol="Model", size="Num_Features",
                hover_data=["ROC_AUC", "PR_AUC", "MCC", "Accuracy", "Num_Features"],
                title="Trade-off: Selection Time (s) vs F1-Score (Point size = Num Features)"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with t3:
            chosen_key = st.selectbox("View selected features for each method:", list(selected_features_dict.keys()))
            f_list = selected_features_dict[chosen_key]
            st.info(f"Number of selected features: **{len(f_list)}** (out of {total_feats})")
            st.write(f_list)

        with t4:
            st.dataframe(df_res.sort_values(by="F1_Score", ascending=False), use_container_width=True)
            st.download_button(
                "📥 Download Full Benchmark Results (CSV)",
                data=df_res.to_csv(index=False).encode('utf-8'),
                file_name="industry_benchmark_results.csv",
                mime="text/csv"
            )
else:
    st.info("👆 Please upload a dataset file above to start processing.")