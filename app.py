"""
Superstore Sales, Profit & Loss-Risk Analytics Dashboard
=========================================================
Author  : Anand Kumar
Industry: Retail / E-commerce
Run     : streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay,
)
import sklearn as _sklearn
_OHE_SPARSE_KW = "sparse_output" if tuple(int(x) for x in _sklearn.__version__.split(".")[:2]) >= (1, 2) else "sparse"

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Superstore Analytics",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Plotting style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 110,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})
sns.set_style("whitegrid")

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & CLEANING  (cached)
# ─────────────────────────────────────────────────────────────────────────────
FILE_PATH = "Superstore.csv.csv"

@st.cache_data(show_spinner="Loading & cleaning dataset…")
def load_and_clean():
    try:
        df = pd.read_csv(FILE_PATH, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(FILE_PATH, encoding="latin-1")

    # Remove fully blank rows
    df = df[~df.isnull().all(axis=1)].reset_index(drop=True)
    # Remove exact duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # Parse mixed dates
    def parse_dates(series):
        parsed = pd.to_datetime(series, infer_datetime_format=True, errors="coerce")
        if parsed.isnull().any():
            mask = parsed.isnull()
            parsed[mask] = pd.to_datetime(series[mask], dayfirst=False, errors="coerce")
        return parsed

    df["Order Date"] = parse_dates(df["Order Date"])
    df["Ship Date"]  = parse_dates(df["Ship Date"])

    # Fix corrupt Category using Product ID prefix
    valid_cats = {"FUR": "Furniture", "OFF": "Office Supplies", "TEC": "Technology"}
    def fix_cat(row):
        if row["Category"] not in ["Furniture", "Office Supplies", "Technology"]:
            prefix = str(row["Product ID"])[:3]
            return valid_cats.get(prefix, row["Category"])
        return row["Category"]
    df["Category"] = df.apply(fix_cat, axis=1)

    # Clean encoding artifacts in Product Name
    def clean_enc(t):
        if not isinstance(t, str): return t
        t = t.replace("\ufffd", "")
        t = re.sub(r"[\x80-\x9f]", "", t)
        t = re.sub(r" {2,}", " ", t).strip()
        return t
    df["Product Name"] = df["Product Name"].apply(clean_enc)

    # Numeric types
    df["Sales"]    = pd.to_numeric(df["Sales"],    errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Discount"] = pd.to_numeric(df["Discount"], errors="coerce")
    df["Profit"]   = pd.to_numeric(df["Profit"],   errors="coerce")

    # Categorical types
    for c in ["Ship Mode","Segment","Country","Region","Category","Sub-Category"]:
        df[c] = df[c].astype("category")

    # Feature engineering
    df["Order_Year"]        = df["Order Date"].dt.year
    df["Order_Month"]       = df["Order Date"].dt.month
    df["Order_Quarter"]     = df["Order Date"].dt.quarter
    df["Order_DayOfWeek"]   = df["Order Date"].dt.dayofweek
    df["Shipping_Duration"] = (df["Ship Date"] - df["Order Date"]).dt.days.clip(lower=0)
    df["Loss_Flag"]         = (df["Profit"] < 0).astype(int)

    return df

df_full = load_and_clean()

# ─────────────────────────────────────────────────────────────────────────────
# ML MODEL  (cached)
# ─────────────────────────────────────────────────────────────────────────────
NUMERICAL_FEATURES   = ["Sales","Quantity","Discount","Shipping_Duration",
                         "Order_Year","Order_Month","Order_Quarter","Order_DayOfWeek"]
CATEGORICAL_FEATURES = ["Ship Mode","Segment","Region","Category","Sub-Category","State"]
TARGET               = "Loss_Flag"

@st.cache_resource(show_spinner="Training ML models…")
def train_models(df):
    ml_df = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [TARGET]].copy()
    # Cast nullable Int64 columns to float64 for sklearn compatibility
    for _c in NUMERICAL_FEATURES:
        ml_df[_c] = ml_df[_c].astype("float64")
    ml_df = ml_df.dropna()
    for c in CATEGORICAL_FEATURES:
        ml_df[c] = ml_df[c].astype(str)

    X = ml_df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = ml_df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    num_pipe = Pipeline([("scaler", StandardScaler())])
    cat_pipe = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore", **{_OHE_SPARSE_KW: False}))])
    pre = ColumnTransformer([("num", num_pipe, NUMERICAL_FEATURES),
                              ("cat", cat_pipe, CATEGORICAL_FEATURES)])

    lr = Pipeline([("preprocessor", pre), ("classifier",
           LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"))])
    rf = Pipeline([("preprocessor", pre), ("classifier",
           RandomForestClassifier(n_estimators=200, random_state=42,
                                   class_weight="balanced", n_jobs=-1))])
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    def metrics(pipe, Xt, yt):
        yp   = pipe.predict(Xt)
        yprob = pipe.predict_proba(Xt)[:, 1]
        return {
            "accuracy":  accuracy_score(yt, yp),
            "precision": precision_score(yt, yp, zero_division=0),
            "recall":    recall_score(yt, yp, zero_division=0),
            "roc_auc":   roc_auc_score(yt, yprob),
            "y_pred":    yp,
            "y_prob":    yprob,
            "cm":        confusion_matrix(yt, yp),
        }

    lr_m = metrics(lr, X_test, y_test)
    rf_m = metrics(rf, X_test, y_test)

    # Feature importance from RF
    ohe_names = (rf.named_steps["preprocessor"]
                   .named_transformers_["cat"]
                   .named_steps["ohe"]
                   .get_feature_names_out(CATEGORICAL_FEATURES).tolist())
    all_feats = NUMERICAL_FEATURES + ohe_names
    importances = rf.named_steps["classifier"].feature_importances_

    feat_df = pd.DataFrame({"Feature": all_feats, "Importance": importances})
    feat_df = feat_df.sort_values("Importance", ascending=False).head(20)

    return lr, rf, lr_m, rf_m, y_test, feat_df

lr_model, rf_model, lr_metrics, rf_metrics, y_test_global, feat_importance = train_models(df_full)
best_m = rf_metrics if rf_metrics["roc_auc"] >= lr_metrics["roc_auc"] else lr_metrics
best_label = "Random Forest" if rf_metrics["roc_auc"] >= lr_metrics["roc_auc"] else "Logistic Regression"

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("🔍 Filters")
st.sidebar.caption("Filters apply to Analytics sections (not ML training).")

all_years   = sorted(df_full["Order_Year"].dropna().unique().tolist())
all_regions = sorted(df_full["Region"].cat.categories.tolist())
all_cats    = sorted(df_full["Category"].cat.categories.tolist())
all_segs    = sorted(df_full["Segment"].cat.categories.tolist())
all_subcats = sorted(df_full["Sub-Category"].cat.categories.tolist())

sel_years   = st.sidebar.multiselect("Year",        all_years,   default=all_years)
sel_regions = st.sidebar.multiselect("Region",      all_regions, default=all_regions)
sel_cats    = st.sidebar.multiselect("Category",    all_cats,    default=all_cats)
sel_segs    = st.sidebar.multiselect("Segment",     all_segs,    default=all_segs)
sel_subcats = st.sidebar.multiselect("Sub-Category",all_subcats, default=all_subcats)

# Apply filters
df = df_full[
    (df_full["Order_Year"].isin(sel_years)) &
    (df_full["Region"].astype(str).isin(sel_regions)) &
    (df_full["Category"].astype(str).isin(sel_cats)) &
    (df_full["Segment"].astype(str).isin(sel_segs)) &
    (df_full["Sub-Category"].astype(str).isin(sel_subcats))
].copy()

if df.empty:
    st.warning("No data matches the selected filters. Please broaden your selection.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: fig to st
# ─────────────────────────────────────────────────────────────────────────────
def show_fig(fig, caption=""):
    st.pyplot(fig, use_container_width=True)
    if caption:
        st.caption(caption)
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE TITLE
# ─────────────────────────────────────────────────────────────────────────────
st.title("🏪 Superstore Sales, Profit & Loss-Risk Analytics")
st.markdown("**Author:** Anand Kumar | **Industry:** Retail / E-commerce | **Methodology:** 4-Tier Analytics Ladder")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — EXECUTIVE OVERVIEW / KPI CARDS
# ─────────────────────────────────────────────────────────────────────────────
st.header("1. Executive Overview")

total_sales   = df["Sales"].sum()
total_profit  = df["Profit"].sum()
total_qty     = df["Quantity"].sum()
n_orders      = df["Order ID"].nunique()
n_customers   = df["Customer ID"].nunique()
profit_margin = total_profit / total_sales * 100 if total_sales else 0
loss_pct      = df["Loss_Flag"].mean() * 100

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Sales",       f"${total_sales:,.0f}")
c2.metric("Total Profit",      f"${total_profit:,.0f}")
c3.metric("Profit Margin",     f"{profit_margin:.1f}%")
c4.metric("Total Orders",      f"{n_orders:,}")
c5.metric("Unique Customers",  f"{n_customers:,}")
c6.metric("Loss-making Lines", f"{loss_pct:.1f}%")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — SALES ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
st.header("2. Sales Analysis")

monthly = (
    df.assign(YM=df["Order Date"].dt.to_period("M"))
    .groupby("YM")[["Sales","Profit"]]
    .sum()
    .reset_index()
    .sort_values("YM")
)
monthly["YM_str"] = monthly["YM"].astype(str)

fig, ax1 = plt.subplots(figsize=(13, 4))
ax2 = ax1.twinx()
ax1.bar(range(len(monthly)), monthly["Sales"],  color="steelblue", alpha=0.6, label="Sales")
ax2.plot(range(len(monthly)), monthly["Profit"], color="tomato", linewidth=1.8,
         marker="o", markersize=3, label="Profit")
ax2.axhline(0, color="black", linewidth=0.7, linestyle="--")
tick_pos = list(range(0, len(monthly), max(1, len(monthly)//10)))
ax1.set_xticks(tick_pos)
ax1.set_xticklabels([monthly["YM_str"].iloc[i] for i in tick_pos], rotation=45, ha="right")
ax1.set_ylabel("Sales ($)", color="steelblue")
ax2.set_ylabel("Profit ($)", color="tomato")
ax1.set_title("Monthly Sales and Profit Trend")
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc="upper left")
plt.tight_layout()
show_fig(fig, "Sales bars (left axis) and Profit line (right axis) by month.")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — PROFIT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
st.header("3. Profit Analysis")
col_l, col_r = st.columns(2)

with col_l:
    # Profit by Year
    yr_profit = df.groupby("Order_Year")["Profit"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(yr_profit["Order_Year"].astype(str), yr_profit["Profit"], color="mediumseagreen")
    ax.set_title("Annual Profit"); ax.set_xlabel("Year"); ax.set_ylabel("Profit ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"${v/1e3:.0f}K"))
    plt.tight_layout()
    show_fig(fig)

with col_r:
    # Profit by Quarter (box)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    q_data = [df[df["Order_Quarter"]==q]["Profit"].values for q in [1,2,3,4]]
    ax.boxplot(q_data, labels=["Q1","Q2","Q3","Q4"], patch_artist=True,
               boxprops=dict(facecolor="lightblue"))
    ax.axhline(0, color="red", linestyle="--", linewidth=0.8)
    ax.set_title("Profit Distribution by Quarter")
    ax.set_ylabel("Profit ($)")
    plt.tight_layout()
    show_fig(fig)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — CATEGORY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
st.header("4. Category Analysis")
cat_perf = (
    df.groupby("Category")[["Sales","Profit"]]
    .sum().reset_index().sort_values("Sales", ascending=False)
)
cat_perf["Margin %"] = (cat_perf["Profit"] / cat_perf["Sales"] * 100).round(2)

col_l, col_r = st.columns(2)
with col_l:
    x = np.arange(len(cat_perf))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(x - 0.2, cat_perf["Sales"],  0.35, label="Sales",  color="steelblue")
    ax.bar(x + 0.2, cat_perf["Profit"], 0.35, label="Profit",
           color=["mediumseagreen" if p >= 0 else "tomato" for p in cat_perf["Profit"]])
    ax.set_xticks(x); ax.set_xticklabels(cat_perf["Category"].tolist())
    ax.set_title("Sales & Profit by Category"); ax.set_ylabel("Amount ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"${v/1e6:.1f}M" if abs(v)>=1e6 else f"${v/1e3:.0f}K"))
    ax.legend(); plt.tight_layout()
    show_fig(fig)

with col_r:
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = ["tomato" if m < 5 else "steelblue" for m in cat_perf["Margin %"]]
    ax.bar(cat_perf["Category"], cat_perf["Margin %"], color=colors)
    ax.set_title("Profit Margin % by Category"); ax.set_ylabel("Margin %")
    plt.tight_layout()
    show_fig(fig)

st.dataframe(cat_perf.set_index("Category").style.format({"Sales":"${:,.0f}","Profit":"${:,.0f}","Margin %":"{:.1f}%"}))
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — REGIONAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
st.header("5. Regional Analysis")
region_perf = (
    df.groupby("Region")[["Sales","Profit"]]
    .sum().reset_index().sort_values("Sales", ascending=False)
)
region_perf["Margin %"] = (region_perf["Profit"] / region_perf["Sales"] * 100).round(2)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, col, ylabel, color in zip(
    axes,
    ["Sales","Profit","Margin %"],
    ["Sales ($)","Profit ($)","Margin (%)"],
    ["steelblue","mediumseagreen","mediumpurple"]
):
    vals = region_perf[col]
    bar_colors = [color if v >= 0 else "tomato" for v in vals]
    ax.bar(region_perf["Region"], vals, color=bar_colors)
    ax.set_title(ylabel); ax.set_ylabel(ylabel)
    if col != "Margin %":
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"${v/1e6:.1f}M" if abs(v)>=1e6 else f"${v/1e3:.0f}K"))
    ax.axhline(0, color="black", linewidth=0.5)
fig.suptitle("Performance by Region", fontsize=12, fontweight="bold")
plt.tight_layout()
show_fig(fig)

st.dataframe(region_perf.set_index("Region").style.format({"Sales":"${:,.0f}","Profit":"${:,.0f}","Margin %":"{:.1f}%"}))
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — SUB-CATEGORY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
st.header("6. Sub-Category Analysis")
subcat_perf = (
    df.groupby("Sub-Category")[["Sales","Profit"]]
    .sum().reset_index().sort_values("Profit", ascending=True)
)

fig, ax = plt.subplots(figsize=(10, 6))
colors = ["tomato" if p < 0 else "steelblue" for p in subcat_perf["Profit"]]
ax.barh(subcat_perf["Sub-Category"], subcat_perf["Profit"], color=colors)
ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel("Total Profit ($)")
ax.set_title("Sub-Category Profitability (Red = Net Loss)")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"${v/1e3:.0f}K"))
plt.tight_layout()
show_fig(fig, "Sub-categories in red are net loss-making in the selected filter scope.")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — DISCOUNT vs PROFIT
# ─────────────────────────────────────────────────────────────────────────────
st.header("7. Discount vs Profit Analysis")
col_l, col_r = st.columns(2)

with col_l:
    fig, ax = plt.subplots(figsize=(5, 4))
    cats_u = df["Category"].astype(str).unique()
    palette = {"Furniture":"steelblue","Office Supplies":"mediumseagreen","Technology":"tomato"}
    for cat in cats_u:
        sub = df[df["Category"].astype(str) == cat]
        ax.scatter(sub["Discount"], sub["Profit"], alpha=0.2, s=8,
                   color=palette.get(cat,"grey"), label=cat)
    m, b = np.polyfit(df["Discount"], df["Profit"], 1)
    xl = np.linspace(df["Discount"].min(), df["Discount"].max(), 200)
    ax.plot(xl, m*xl+b, color="black", linewidth=1.5, label=f"Trend (slope={m:.0f})")
    ax.axhline(0, color="grey", linewidth=0.7, linestyle="--")
    ax.set_xlabel("Discount"); ax.set_ylabel("Profit ($)")
    ax.set_title("Discount vs Profit Scatter")
    ax.legend(markerscale=2); plt.tight_layout()
    show_fig(fig)

with col_r:
    disc_bins = pd.cut(df["Discount"],
                       bins=[-0.01,0,0.1,0.2,0.3,0.4,0.5,1.0],
                       labels=["0%","1–10%","11–20%","21–30%","31–40%","41–50%",">50%"])
    disc_loss = df.groupby(disc_bins)["Loss_Flag"].agg(["sum","count"])
    disc_loss.columns = ["Loss","Total"]
    disc_loss["Loss Rate %"] = (disc_loss["Loss"]/disc_loss["Total"]*100).round(1)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(disc_loss.index.astype(str), disc_loss["Loss Rate %"],
           color=["tomato" if r > 50 else "steelblue" for r in disc_loss["Loss Rate %"]])
    ax.set_xlabel("Discount Band"); ax.set_ylabel("Loss Rate (%)")
    ax.set_title("Loss Rate by Discount Band")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    show_fig(fig)

corr_dp = df["Discount"].corr(df["Profit"])
st.info(f"📊 Pearson correlation between Discount and Profit: **{corr_dp:.4f}** "
        f"(negative = higher discount associated with lower profit)")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — ML MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
st.header("8. ML Model Performance")
st.caption("⚙️ Models are trained on the full dataset (80/20 split). Filters do not affect training.")

model_comp = pd.DataFrame({
    "Model":     ["Logistic Regression", "Random Forest"],
    "Accuracy":  [f"{lr_metrics['accuracy']:.4f}",  f"{rf_metrics['accuracy']:.4f}"],
    "Precision": [f"{lr_metrics['precision']:.4f}", f"{rf_metrics['precision']:.4f}"],
    "Recall":    [f"{lr_metrics['recall']:.4f}",    f"{rf_metrics['recall']:.4f}"],
    "ROC-AUC":   [f"{lr_metrics['roc_auc']:.4f}",   f"{rf_metrics['roc_auc']:.4f}"],
})
st.dataframe(model_comp, use_container_width=True)

col_l, col_r = st.columns(2)

# ── Confusion Matrices ────────────────────────────────────────────────────────
with col_l:
    st.subheader("9. Confusion Matrices")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, mets, name in zip(axes,
                               [lr_metrics, rf_metrics],
                               ["Logistic Reg.", "Random Forest"]):
        disp = ConfusionMatrixDisplay(mets["cm"], display_labels=["No Loss","Loss"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name)
    plt.tight_layout()
    show_fig(fig)

# ── ROC Curve ────────────────────────────────────────────────────────────────
with col_r:
    st.subheader("10. ROC-AUC Curve")
    fig, ax = plt.subplots(figsize=(5, 4))
    for mets, name, color in [
        (lr_metrics, f"LR  (AUC={lr_metrics['roc_auc']:.3f})", "steelblue"),
        (rf_metrics, f"RF  (AUC={rf_metrics['roc_auc']:.3f})", "tomato"),
    ]:
        fpr, tpr, _ = roc_curve(y_test_global, mets["y_prob"])
        ax.plot(fpr, tpr, label=name, linewidth=2, color=color)
    ax.plot([0,1],[0,1], "k--", linewidth=1, label="Chance")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Loss-Risk Classification")
    ax.legend(loc="lower right"); plt.tight_layout()
    show_fig(fig)

st.divider()

# ── Feature Importance ────────────────────────────────────────────────────────
st.subheader("Feature Importance (Random Forest — Top 20)")
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(feat_importance["Feature"][::-1], feat_importance["Importance"][::-1], color="steelblue")
ax.set_xlabel("Importance (Mean Decrease Impurity)")
ax.set_title("Top 20 Feature Importances")
plt.tight_layout()
show_fig(fig)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — LOSS-RISK PREDICTION (on filtered data)
# ─────────────────────────────────────────────────────────────────────────────
st.header("11. Loss-Risk Prediction (Filtered Records)")

rf_input_df = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()
for _c in NUMERICAL_FEATURES:
    rf_input_df[_c] = rf_input_df[_c].astype("float64")
for c in CATEGORICAL_FEATURES:
    rf_input_df[c] = rf_input_df[c].astype(str)

rf_input_df = rf_input_df.dropna()
if len(rf_input_df) > 0:
    loss_probs = rf_model.predict_proba(rf_input_df)[:, 1]
    risk_df = df.loc[rf_input_df.index].copy()
    risk_df["Loss_Probability"] = loss_probs
    risk_df["Risk_Level"] = pd.cut(loss_probs,
                                    bins=[-0.01, 0.3, 0.5, 0.7, 1.01],
                                    labels=["Low","Medium","High","Very High"])

    col_a, col_b = st.columns(2)
    with col_a:
        rl_counts = risk_df["Risk_Level"].value_counts().reindex(["Low","Medium","High","Very High"])
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar(rl_counts.index, rl_counts.values,
               color=["mediumseagreen","gold","orangered","darkred"])
        ax.set_title("Order Lines by Risk Level"); ax.set_ylabel("Count")
        plt.tight_layout()
        show_fig(fig)

    with col_b:
        top_risk = (
            risk_df.sort_values("Loss_Probability", ascending=False)
            [["Order ID","Category","Sub-Category","Region","Discount",
              "Sales","Loss_Probability","Risk_Level"]]
            .head(20)
        )
        st.dataframe(top_risk.style.format({"Discount":"{:.0%}","Sales":"${:,.2f}","Loss_Probability":"{:.3f}"}),
                     use_container_width=True)
else:
    st.warning("No records available for scoring after filtering.")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 — PRESCRIPTIVE RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────
st.header("12. Prescriptive Recommendations")

disc_bands = pd.cut(df["Discount"], bins=[-0.01,0,0.1,0.2,0.3,0.4,0.5,1.0],
                    labels=["0%","1–10%","11–20%","21–30%","31–40%","41–50%",">50%"])
disc_loss_rec = df.groupby(disc_bands)["Loss_Flag"].agg(["sum","count"])
disc_loss_rec.columns = ["Losses","Total"]
disc_loss_rec["Loss Rate %"] = (disc_loss_rec["Losses"]/disc_loss_rec["Total"]*100).round(1)
high_disc_bands = disc_loss_rec[disc_loss_rec["Loss Rate %"] > 50].index.tolist()

net_loss_subcats = (
    df.groupby("Sub-Category")["Profit"].sum()
    .reset_index()
    .query("Profit < 0")["Sub-Category"].tolist()
)

region_loss_rate = (
    df.groupby("Region")["Loss_Flag"]
    .agg(["sum","count"])
    .assign(**{"Loss Rate %": lambda x: (x["sum"]/x["count"]*100).round(1)})
)
worst_region = region_loss_rate["Loss Rate %"].idxmax()

recs = [
    ("🔴 REC 1 — Discount Cap Policy",
     f"Discount bands with > 50% loss rate: **{', '.join(str(b) for b in high_disc_bands) or 'None in filtered scope'}**. "
     f"Implement a hard cap or require manager approval for discounts exceeding 30%."),
    ("🟠 REC 2 — Sub-Category Pricing Review",
     f"Net loss-making sub-categories: **{', '.join(net_loss_subcats) or 'None in filtered scope'}**. "
     f"Conduct cost-plus pricing reviews and evaluate supplier agreements."),
    (f"🟡 REC 3 — Regional Audit: {worst_region}",
     f"The **{worst_region}** region has the highest loss rate "
     f"({region_loss_rate.loc[worst_region, 'Loss Rate %']:.1f}%) among filtered regions. "
     f"Assign a regional manager to review discount approvals and product mix."),
    ("🟢 REC 4 — Real-Time Loss-Risk Scoring",
     f"Deploy the trained **{best_label}** (ROC-AUC = {best_m['roc_auc']:.3f}) as an order-entry risk scorer. "
     f"Flag orders with predicted loss probability > 0.50 for review before processing."),
    ("🔵 REC 5 — Furniture/Tables Strategy",
     "Tables is a net loss-making sub-category. Consider re-pricing, reducing maximum discounts "
     "on furniture, or bundling with higher-margin accessories to protect overall margin."),
]

for title, body in recs:
    with st.expander(title, expanded=True):
        st.markdown(body)

st.divider()
st.caption("Superstore Analytics Dashboard — Anand Kumar | All metrics computed from actual Superstore.csv.csv dataset.")
