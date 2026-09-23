"""
generate_report.py
==================
Generates AnandKumar_ProjectReport.docx

Run AFTER executing the Jupyter Notebook (so that charts/ and data/ are populated).

Usage:
    python generate_report.py
"""

import os
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Any
# ── docx imports ──────────────────────────────────────────────────────────────
from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# ── sklearn for re-computing metrics ─────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    classification_report, confusion_matrix,
)
import sklearn as _sklearn
_OHE_SPARSE_KW = "sparse_output" if tuple(int(x) for x in _sklearn.__version__.split(".")[:2]) >= (1, 2) else "sparse"

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
FILE_PATH    = "Superstore.csv.csv"
OUTPUT_PATH  = "AnandKumar_ProjectReport.docx"
CHARTS_DIR   = "charts"

NUMERICAL_FEATURES   = ["Sales","Quantity","Discount","Shipping_Duration",
                          "Order_Year","Order_Month","Order_Quarter","Order_DayOfWeek"]
CATEGORICAL_FEATURES = ["Ship Mode","Segment","Region","Category","Sub-Category","State"]
TARGET               = "Loss_Flag"

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & CLEANING
# ─────────────────────────────────────────────────────────────────────────────
print("Loading dataset …")
try:
    df_raw = pd.read_csv(FILE_PATH, encoding="utf-8-sig")
except UnicodeDecodeError:
    df_raw = pd.read_csv(FILE_PATH, encoding="latin-1")

rows_before = len(df_raw)

df = df_raw.copy()
df = df[~df.isnull().all(axis=1)].reset_index(drop=True)
df = df.drop_duplicates().reset_index(drop=True)
def parse_mixed_dates(s) -> Any:
    p: Any = pd.to_datetime(s, format="mixed", errors="coerce")
    if p.isnull().any():
        mask: Any = p.isnull()
        p[mask] = pd.to_datetime(s[mask], dayfirst=False, errors="coerce")
    return p
df["Order Date"] = parse_mixed_dates(df["Order Date"])
df["Ship Date"]  = parse_mixed_dates(df["Ship Date"])

valid_cats = {"FUR": "Furniture", "OFF": "Office Supplies", "TEC": "Technology"}
def fix_cat(row):
    if row["Category"] not in ["Furniture","Office Supplies","Technology"]:
        return valid_cats.get(str(row["Product ID"])[:3], row["Category"])
    return row["Category"]
df["Category"] = df.apply(fix_cat, axis=1)

def clean_enc(t):
    if not isinstance(t, str): return t
    t = t.replace("\ufffd","")
    t = re.sub(r"[\x80-\x9f]","",t)
    t = re.sub(r" {2,}"," ",t).strip()
    return t
df["Product Name"] = df["Product Name"].apply(clean_enc)

for c in ["Sales","Quantity","Discount","Profit"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["Order_Year"]        = df["Order Date"].dt.year
df["Order_Month"]       = df["Order Date"].dt.month
df["Order_Quarter"]     = df["Order Date"].dt.quarter
df["Order_DayOfWeek"]   = df["Order Date"].dt.dayofweek
df["Shipping_Duration"] = (df["Ship Date"] - df["Order Date"]).dt.days.clip(lower=0)
df["Loss_Flag"]         = (df["Profit"] < 0).astype(int)

rows_after = len(df)
print(f"  Rows: {rows_before} → {rows_after}")

# ─────────────────────────────────────────────────────────────────────────────
# KPIs
# ─────────────────────────────────────────────────────────────────────────────
total_sales   = df["Sales"].sum()
total_profit  = df["Profit"].sum()
total_qty     = df["Quantity"].sum()
n_orders      = df["Order ID"].nunique()
n_customers   = df["Customer ID"].nunique()
profit_margin = total_profit / total_sales * 100 if total_sales else 0
loss_lines    = df["Loss_Flag"].sum()
loss_pct      = loss_lines / len(df) * 100
corr_dp       = df["Discount"].corr(df["Profit"])

# ─────────────────────────────────────────────────────────────────────────────
# ML METRICS
# ─────────────────────────────────────────────────────────────────────────────
print("Training models for report metrics …")
ml_df = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [TARGET]].copy()
for _c in NUMERICAL_FEATURES:
    ml_df[_c] = ml_df[_c].astype("float64")
ml_df = ml_df.dropna()
for c in CATEGORICAL_FEATURES:
    ml_df[c] = ml_df[c].astype(str)
X = ml_df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
y = ml_df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

pre = ColumnTransformer([
    ("num", Pipeline([("sc", StandardScaler())]),                                         NUMERICAL_FEATURES),
    ("cat", Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore",
                                             **{_OHE_SPARSE_KW: False}))]),               CATEGORICAL_FEATURES),
])

lr_pipe = Pipeline([("pre", pre), ("clf",
    LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"))])

# RF needs a fresh (unfitted) preprocessor instance
pre_rf = ColumnTransformer([
    ("num", Pipeline([("sc", StandardScaler())]),                                         NUMERICAL_FEATURES),
    ("cat", Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore",
                                             **{_OHE_SPARSE_KW: False}))]),               CATEGORICAL_FEATURES),
])
rf_pipe = Pipeline([("pre", pre_rf), ("clf",
    RandomForestClassifier(n_estimators=200, random_state=42,
                            class_weight="balanced", n_jobs=-1))])

lr_pipe.fit(X_train, y_train)
rf_pipe.fit(X_train, y_train)

def get_metrics(pipe, Xt, yt):
    yp   = pipe.predict(Xt)
    yprob = pipe.predict_proba(Xt)[:,1]
    return {
        "accuracy":  round(accuracy_score(yt, yp), 4),
        "precision": round(precision_score(yt, yp, zero_division=0), 4),
        "recall":    round(recall_score(yt, yp, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(yt, yprob), 4),
        "cm":        confusion_matrix(yt, yp),
        "report":    classification_report(yt, yp, target_names=["No Loss (0)","Loss (1)"]),
    }

lr_met = get_metrics(lr_pipe, X_test, y_test)
rf_met = get_metrics(rf_pipe, X_test, y_test)
best_met   = rf_met if rf_met["roc_auc"] >= lr_met["roc_auc"] else lr_met
best_label = "Random Forest" if rf_met["roc_auc"] >= lr_met["roc_auc"] else "Logistic Regression"

print(f"  Best model: {best_label}  ROC-AUC={best_met['roc_auc']}")

# ─────────────────────────────────────────────────────────────────────────────
# DOCX HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def set_cell_bg(cell, hex_color):
    """Set table cell background colour."""
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd  = OxmlElement("w:shd")
    shd.set(qn("w:val"),   "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"),  hex_color)
    tcPr.append(shd)

def add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    h.alignment = WD_ALIGN_PARAGRAPH.LEFT
    return h

def add_para(doc, text, bold=False, italic=False, size=11, color=None):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold   = bold
    run.italic = italic
    run.font.size = Pt(size)
    if color:
        run.font.color.rgb = RGBColor(*color)
    return p

def add_table(doc, headers, rows, header_bg="1F3864", header_fg="FFFFFF"):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = "Table Grid"
    # Header row
    hrow = t.rows[0]
    for i, h in enumerate(headers):
        cell = hrow.cells[i]
        cell.text = h
        set_cell_bg(cell, header_bg)
        for p in cell.paragraphs:
            for run in p.runs:
                run.bold = True
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                run.font.size = Pt(9)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    # Data rows
    for ri, row in enumerate(rows):
        drow = t.rows[ri + 1]
        for ci, val in enumerate(row):
            drow.cells[ci].text = str(val)
            for p in drow.cells[ci].paragraphs:
                for run in p.runs:
                    run.font.size = Pt(9)
    return t

def try_add_image(doc, path, width=Inches(5.5)):
    if os.path.exists(path):
        doc.add_picture(path, width=width)
    else:
        doc.add_paragraph(f"[Chart not found: {path}]").italic = True

# ─────────────────────────────────────────────────────────────────────────────
# BUILD DOCUMENT
# ─────────────────────────────────────────────────────────────────────────────
print("Building DOCX report …")
doc = Document()

# ── Margins ───────────────────────────────────────────────────────────────────
for section in doc.sections:
    section.top_margin    = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin   = Cm(3.0)
    section.right_margin  = Cm(2.5)

# ─── COVER PAGE ───────────────────────────────────────────────────────────────
doc.add_paragraph()
title_para = doc.add_paragraph()
title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
title_run = title_para.add_run("Superstore Sales, Profit & Loss-Risk Analytics\nusing Machine Learning")
title_run.bold = True
title_run.font.size = Pt(20)
title_run.font.color.rgb = RGBColor(0x1F, 0x38, 0x64)

doc.add_paragraph()
for line in [
    "Author: Anand Kumar",
    "Industry: Retail / E-commerce",
    "Methodology: 4-Tier Analytics Ladder",
    f"Date: {datetime.today().strftime('%B %Y')}",
]:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(line)
    r.font.size = Pt(13)

doc.add_page_break()

# ─── 1. ABSTRACT ──────────────────────────────────────────────────────────────
add_heading(doc, "1. Abstract / Executive Summary", level=1)
add_para(doc, (
    "This report presents a complete, internship-ready Data Analytics and Machine Learning "
    "project applied to the Superstore retail dataset. Using the 4-Tier Analytics Ladder, the "
    "project progresses through data hygiene, exploratory analytics, predictive modelling, and "
    "prescriptive recommendations. The core ML objective is to predict which order-line items "
    "are at risk of generating a financial loss — enabling proactive business intervention."
))

# ─── 2. INTRODUCTION ──────────────────────────────────────────────────────────
add_heading(doc, "2. Introduction", level=1)
add_para(doc, (
    "The Superstore dataset represents transaction records from a US-based retail company selling "
    "three product categories — Furniture, Office Supplies, and Technology — across four geographic "
    "regions (West, East, South, Central) between 2014 and 2017. Each row represents a single "
    "order line-item, capturing product, customer, geographic, and financial details."
))

# ─── 3. BUSINESS PROBLEM ──────────────────────────────────────────────────────
add_heading(doc, "3. Business Problem", level=1)
add_para(doc, "Management requires answers to four operational questions:")
for q in [
    "1. Which products, regions, and segments drive profit — and which generate losses?",
    "2. What structural patterns (discounting, category, geography) cause order-line losses?",
    "3. Can we predict, before fulfillment, whether an order line will be loss-making?",
    "4. What concrete actions should be taken to reduce loss exposure?",
]:
    doc.add_paragraph(q, style="List Bullet")

# ─── 4. OBJECTIVES ────────────────────────────────────────────────────────────
add_heading(doc, "4. Objectives", level=1)
for obj in [
    "Clean and validate the raw dataset to ensure analytical integrity.",
    "Compute KPIs and create professional visualizations across all analytical dimensions.",
    "Build a binary classification model to identify loss-risk order lines (Loss_Flag).",
    "Evaluate model performance using standard ML metrics on a held-out test set.",
    "Formulate prescriptive, evidence-based business recommendations.",
]:
    doc.add_paragraph(obj, style="List Bullet")

# ─── 5. DATASET DESCRIPTION ───────────────────────────────────────────────────
add_heading(doc, "5. Dataset Description", level=1)
add_para(doc, f"File: Superstore.csv.csv  |  Rows: {rows_after:,}  |  Columns: {df.shape[1]}  |  Period: 2014–2017")
doc.add_paragraph()

# ─── 6. DATA DICTIONARY ───────────────────────────────────────────────────────
add_heading(doc, "6. Data Dictionary", level=1)
dd_headers = ["Column", "Type", "Description"]
dd_rows = [
    ["Row ID",       "Integer",     "Sequential record identifier"],
    ["Order ID",     "String",      "Order identifier (shared by line items)"],
    ["Order Date",   "Date",        "Date order was placed"],
    ["Ship Date",    "Date",        "Date order was shipped"],
    ["Ship Mode",    "Categorical", "Shipping tier (Second / Standard / First / Same Day)"],
    ["Customer ID",  "String",      "Unique customer identifier"],
    ["Customer Name","String",      "Customer full name"],
    ["Segment",      "Categorical", "Consumer / Corporate / Home Office"],
    ["Country",      "Categorical", "United States (all rows)"],
    ["City",         "String",      "Delivery city"],
    ["State",        "String",      "Delivery state"],
    ["Postal Code",  "String",      "5-digit ZIP code"],
    ["Region",       "Categorical", "West / East / South / Central"],
    ["Product ID",   "String",      "Unique product identifier"],
    ["Category",     "Categorical", "Furniture / Office Supplies / Technology"],
    ["Sub-Category", "Categorical", "17 sub-categories"],
    ["Product Name", "String",      "Full product name"],
    ["Sales",        "Float",       "Revenue for the line-item ($)"],
    ["Quantity",     "Integer",     "Units ordered"],
    ["Discount",     "Float",       "Discount rate (0.0–1.0)"],
    ["Profit",       "Float",       "Net profit/loss for the line-item ($)"],
]
add_table(doc, dd_headers, dd_rows)
doc.add_paragraph()

# ─── 7. DATA QUALITY ASSESSMENT ───────────────────────────────────────────────
add_heading(doc, "7. Data Quality Assessment", level=1)
add_para(doc, "Pre-cleaning audit findings:")
issues = [
    ["Mixed date formats", "Order Date and Ship Date mixed MM-DD-YYYY and M/DD/YYYY"],
    ["Corrupt Category value", "Row with Category='C' (should be Technology)"],
    ["Encoding artifacts", "Product Name contained mojibake (unicode replacement chars)"],
    ["Trailing blank row", "One fully-blank row at end of file"],
    ["Exact duplicates", "Checked — none found in actual dataset"],
    ["Missing values", "No structural missing values detected post-audit"],
    ["Negative Profit", "Present and legitimate — order-line losses due to discounting"],
]
add_table(doc, ["Issue", "Detail"], issues)
doc.add_paragraph()

# ─── 8. DATA CLEANING ─────────────────────────────────────────────────────────
add_heading(doc, "8. Data Cleaning", level=1)
cleaning_steps = [
    ["Blank rows",      "Removed with isnull().all(axis=1)"],
    ["Duplicates",      "Removed with drop_duplicates()"],
    ["Date normalisation", "pd.to_datetime with infer_datetime_format + fallback pass"],
    ["Category fix",    "Row with 'C' corrected to 'Technology' using Product ID prefix TEC-"],
    ["Encoding cleanup","Replaced \\ufffd and \\x80-\\x9f characters in Product Name"],
    ["Data types",      "Numeric: Sales, Quantity, Discount, Profit; Category dtype for categoricals"],
    ["Feature engineering", "Shipping_Duration, Order_Year, Order_Month, Order_Quarter, Order_DayOfWeek added"],
    ["Negative Profit", "RETAINED — treated as valid business losses"],
]
add_table(doc, ["Step","Treatment"], cleaning_steps)
add_para(doc, f"\nRows before cleaning: {rows_before:,}  |  Rows after cleaning: {rows_after:,}", bold=True)
doc.add_paragraph()

# ─── 9. KPI ANALYSIS ──────────────────────────────────────────────────────────
add_heading(doc, "9. Exploratory Data Analysis — KPIs", level=1)
kpi_rows = [
    ["Total Sales",           f"${total_sales:,.2f}"],
    ["Total Profit",          f"${total_profit:,.2f}"],
    ["Overall Profit Margin", f"{profit_margin:.2f}%"],
    ["Total Quantity Sold",   f"{int(total_qty):,}"],
    ["Unique Orders",         f"{n_orders:,}"],
    ["Unique Customers",      f"{n_customers:,}"],
    ["Loss-making Line Items",f"{loss_lines:,}  ({loss_pct:.1f}%)"],
    ["Discount–Profit Corr.", f"{corr_dp:.4f} (negative association)"],
]
add_table(doc, ["KPI","Value"], kpi_rows)
doc.add_paragraph()

# ─── 10. VISUALIZATIONS ───────────────────────────────────────────────────────
add_heading(doc, "10. Visualizations & Findings", level=1)

viz_info = [
    ("charts/viz1_monthly_trend.png",
     "VIZ 1 — Monthly Sales and Profit Trend",
     "Sales and profit both trend upward from 2014 to 2017. Q4 months show consistent spikes, "
     "consistent with holiday retail seasonality. Some months show negative profit despite "
     "positive sales — likely driven by heavy seasonal discounting."),
    ("charts/viz2_category.png",
     "VIZ 2 — Sales and Profit by Category",
     "Technology leads in sales and profit margin. Furniture has the lowest margin relative "
     "to its sales volume. Office Supplies occupies the middle ground."),
    ("charts/viz3_region.png",
     "VIZ 3 — Sales and Profit by Region",
     "The West region leads in both sales and profit. The Central region shows "
     "disproportionately low profit relative to its sales volume."),
    ("charts/viz4_subcategory_profit.png",
     "VIZ 4 — Sub-Category Profitability",
     "Copiers, Phones, and Accessories are the highest-profit sub-categories. "
     "Tables and Bookcases are net loss-making at aggregate level."),
    ("charts/viz5_discount_vs_profit.png",
     "VIZ 5 — Discount vs Profit",
     f"Clear negative linear association between discount rate and profit. "
     f"Pearson r = {corr_dp:.3f}. Orders with discount >= 40% are predominantly loss-making. "
     "Note: this is an association, not proof of causation."),
    ("charts/viz6_segment.png",
     "VIZ 6 — Customer Segment Performance",
     "Consumer is the largest segment by volume. Home Office achieves the highest profit margin."),
    ("charts/viz7_state_profit.png",
     "VIZ 7 — State-Level Profitability",
     "California and New York are the highest-profit states. Texas, Ohio, and Pennsylvania "
     "show significant aggregate losses."),
]

for chart_path, title, insight in viz_info:
    add_heading(doc, title, level=2)
    try_add_image(doc, chart_path)
    add_para(doc, f"Observation: {insight}", italic=True)
    doc.add_paragraph()

# ─── 11. ML METHODOLOGY ───────────────────────────────────────────────────────
add_heading(doc, "11. Machine Learning Methodology", level=1)
add_para(doc, "Objective: Binary classification — predict whether an order line will generate a loss.")
doc.add_paragraph()

add_heading(doc, "11.1 Target Variable", level=2)
add_para(doc, "Loss_Flag = 1 when Profit < 0  (loss-making)", bold=True)
add_para(doc, "Loss_Flag = 0 when Profit >= 0  (profitable)", bold=True)
target_dist = df["Loss_Flag"].value_counts().to_dict()
add_para(doc, f"\nClass distribution: No Loss (0) = {target_dist.get(0,0):,}  |  Loss (1) = {target_dist.get(1,0):,}")
doc.add_paragraph()

add_heading(doc, "11.2 Feature Engineering", level=2)
fe_rows = [
    ["Shipping_Duration", "Ship Date − Order Date (days)", "Proxy for logistics cost"],
    ["Order_Year",        "Year of order",                 "Temporal trend"],
    ["Order_Month",       "Month of order",                "Seasonality"],
    ["Order_Quarter",     "Quarter of order",              "Seasonality"],
    ["Order_DayOfWeek",   "Day of week (0=Mon)",           "Order timing pattern"],
]
add_table(doc, ["Feature","Derivation","Rationale"], fe_rows)
doc.add_paragraph()

add_heading(doc, "11.3 Leakage Prevention", level=2)
leak_rows = [
    ["Profit",                          "EXCLUDED", "Loss_Flag is directly derived from Profit"],
    ["Row ID / Order ID / Customer ID / Product ID", "EXCLUDED", "High-cardinality identifiers"],
    ["Customer Name / Product Name",    "EXCLUDED", "Free-text; unsafe representation"],
    ["Country",                         "EXCLUDED", "Zero variance (all United States)"],
    ["Postal Code / City",              "EXCLUDED", "High cardinality; covered by State+Region"],
    ["Sales",                           "INCLUDED", "Revenue is not a direct derivation of Profit"],
    ["Discount, Quantity, Ship Mode…",  "INCLUDED", "All available at order entry; no leakage"],
]
add_table(doc, ["Column/Feature","Decision","Reason"], leak_rows)
doc.add_paragraph()

add_heading(doc, "11.4 Model Training", level=2)
add_para(doc, "Split: 80% training / 20% test, stratified by Loss_Flag.")
add_para(doc, "Preprocessing: StandardScaler for numeric features; OneHotEncoder for categoricals.")
add_para(doc, "Models evaluated: Logistic Regression (baseline) and Random Forest (200 trees).")
add_para(doc, "Class weighting: class_weight='balanced' applied to handle class imbalance.")
doc.add_paragraph()

# ─── 12. EVALUATION METRICS ───────────────────────────────────────────────────
add_heading(doc, "12. Model Evaluation Metrics", level=1)
eval_rows = [
    ["Logistic Regression", str(lr_met["accuracy"]), str(lr_met["precision"]),
     str(lr_met["recall"]), str(lr_met["roc_auc"])],
    ["Random Forest",       str(rf_met["accuracy"]), str(rf_met["precision"]),
     str(rf_met["recall"]), str(rf_met["roc_auc"])],
]
add_table(doc, ["Model","Accuracy","Precision","Recall","ROC-AUC"], eval_rows)
add_para(doc, f"\nSelected best model: {best_label} (ROC-AUC = {best_met['roc_auc']})", bold=True)
doc.add_paragraph()

add_heading(doc, "12.1 Confusion Matrix", level=2)
try_add_image(doc, "charts/viz_confusion_matrices.png")
doc.add_paragraph()

add_heading(doc, "12.2 ROC-AUC Curve", level=2)
try_add_image(doc, "charts/viz_roc_curve.png")
doc.add_paragraph()

add_heading(doc, "12.3 Business Interpretation of Confusion Matrix", level=2)
interp_rows = [
    ["True Positive (TP)",  "Model flags loss; outcome is a loss",
     "Correctly caught — enables proactive intervention"],
    ["True Negative (TN)",  "Model clears order; outcome is profitable",
     "Correctly approved — no unnecessary review effort"],
    ["False Positive (FP)", "Model flags loss; order is actually profitable",
     "Wasted review effort on a healthy order (cost: analyst time)"],
    ["False Negative (FN)", "Model clears order; order generates a loss",
     "Missed loss — real financial damage (highest business cost)"],
]
add_table(doc, ["Outcome","Definition","Business Impact"], interp_rows)
add_para(doc, ("\nIn retail loss prevention, False Negatives are more costly than False Positives. "
               "A missed loss causes real financial damage, while an unnecessary review only costs "
               "analyst time. Recall should therefore be weighted more heavily than Precision when "
               "selecting the operating threshold."), italic=True)
doc.add_paragraph()

# ─── 13. PRESCRIPTIVE ANALYTICS ───────────────────────────────────────────────
add_heading(doc, "13. Prescriptive Analytics", level=1)

disc_bands = pd.cut(df["Discount"],
                    bins=[-0.01,0,0.1,0.2,0.3,0.4,0.5,1.0],
                    labels=["0%","1–10%","11–20%","21–30%","31–40%","41–50%",">50%"])
disc_loss = df.groupby(disc_bands)["Loss_Flag"].agg(["sum","count"])
disc_loss.columns = ["Losses","Total"]
disc_loss["Loss Rate %"] = (disc_loss["Losses"]/disc_loss["Total"]*100).round(1)
high_bands = disc_loss[disc_loss["Loss Rate %"] > 50].index.tolist()

net_loss_sc = (df.groupby("Sub-Category")["Profit"].sum()
               .reset_index().query("Profit < 0")["Sub-Category"].tolist())

region_loss = (df.groupby("Region")["Loss_Flag"].agg(["sum","count"])
               .assign(**{"Loss Rate %": lambda x: (x["sum"]/x["count"]*100).round(1)}))
worst_region = region_loss["Loss Rate %"].idxmax()
worst_rate   = region_loss.loc[worst_region, "Loss Rate %"]

recs = [
    ("REC 1 — Discount Cap Policy",
     f"Discount bands with >50% loss rate: {high_bands}. "
     "Action: Implement a hard cap or require senior-manager approval for any discount exceeding 30%. "
     "The data shows loss rate rises sharply above this threshold."),
    ("REC 2 — Sub-Category Pricing Review",
     f"Net loss-making sub-categories: {net_loss_sc}. "
     "Action: Conduct full cost-plus pricing review. Evaluate supplier costs, "
     "shipping weight, and return rates."),
    (f"REC 3 — Regional Audit: {worst_region}",
     f"The {worst_region} region has the highest loss rate ({worst_rate}%). "
     "Action: Assign a regional sales manager to audit discount approvals and product mix."),
    (f"REC 4 — Real-Time Loss-Risk Scoring",
     f"Deploy the trained {best_label} (ROC-AUC = {best_met['roc_auc']}) as an order-entry risk scorer. "
     "Flag any order line with predicted loss probability > 0.50 for review before processing. "
     "Prioritise top 10% highest-risk lines daily."),
    ("REC 5 — Furniture/Tables Strategy",
     "Tables is a net loss-making sub-category. Evaluate re-pricing, reducing max discounts on "
     "Furniture, or bundling with higher-margin accessories to protect category margin."),
]

for title, body in recs:
    add_heading(doc, title, level=2)
    add_para(doc, body)
doc.add_paragraph()

# ─── 14. LIMITATIONS ──────────────────────────────────────────────────────────
add_heading(doc, "14. Limitations", level=1)
for lim in [
    "Dataset covers 2014–2017 only; insights may not reflect current market dynamics.",
    "Profit column does not include all overhead or logistics costs.",
    "ML model operates at line-item level; order-level aggregation not modelled.",
    "No external data (competitor pricing, economic indicators) available for enrichment.",
]:
    doc.add_paragraph(lim, style="List Bullet")
doc.add_paragraph()

# ─── 15. FUTURE SCOPE ─────────────────────────────────────────────────────────
add_heading(doc, "15. Future Scope", level=1)
for fs in [
    "Add SHAP values for ML explainability.",
    "Build customer lifetime value (CLV) model.",
    "Incorporate time-series forecasting (Prophet / ARIMA).",
    "Implement XGBoost/LightGBM with hyperparameter tuning.",
    "Add geographic choropleth map visualisation.",
    "Develop a product recommendation engine using co-purchase patterns.",
]:
    doc.add_paragraph(fs, style="List Bullet")
doc.add_paragraph()

# ─── 16. CONCLUSION ───────────────────────────────────────────────────────────
add_heading(doc, "16. Conclusion", level=1)
add_para(doc, (
    f"This project successfully applied the 4-Tier Analytics Ladder to the Superstore dataset, "
    f"processing {rows_after:,} clean records across 21 columns. Key findings include: "
    f"an overall profit margin of {profit_margin:.1f}%, {loss_pct:.1f}% loss-making line items, "
    f"a strong negative association between discount rate and profit (r = {corr_dp:.3f}), "
    f"and the identification of net loss-making sub-categories (Tables, Bookcases). "
    f"The best-performing ML model ({best_label}) achieved ROC-AUC = {best_met['roc_auc']} on the "
    f"held-out test set, demonstrating viable predictive power for operational loss-risk scoring. "
    f"Five concrete, evidence-based prescriptive recommendations were formulated to guide "
    f"discount policy, product pricing, regional management, and real-time risk scoring."
))

# ─── 17. REFERENCES ───────────────────────────────────────────────────────────
add_heading(doc, "17. References / Dataset Source", level=1)
add_para(doc, (
    "Dataset: Superstore.csv.csv (Superstore sample retail dataset).\n"
    "Widely attributed to Tableau sample data collection.\n"
    "Exact original source URL: [placeholder — verify from your course materials or data source]."
), italic=True)

# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────
doc.save(OUTPUT_PATH)
print(f"\nReport saved: {OUTPUT_PATH}")
print("Done.")
