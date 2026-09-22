# Superstore Sales, Profit & Loss-Risk Analytics using Machine Learning

**Author:** Anand Kumar  
**Industry:** Retail / E-commerce  
**Methodology:** 4-Tier Analytics Ladder  


---

## Project Overview

This project applies a complete Data Analytics and Machine Learning pipeline to the Superstore retail dataset. Using the 4-Tier Analytics Ladder, it progresses from raw data hygiene through descriptive analytics, predictive machine learning, and prescriptive business recommendations — making it suitable for an academic internship or portfolio submission.

---

## Business Problem

A US-based retail superstore sells Furniture, Office Supplies, and Technology products across four geographic regions. Management needs to:

1. Understand which products, regions, and customer segments drive sales and profit.
2. Identify structural patterns that cause order-line losses.
3. Build a predictive model to automatically flag loss-risk orders at the point of entry.
4. Translate findings into concrete, actionable business recommendations to protect margins.

---

## Dataset Description

| Property        | Value                             |
|-----------------|-----------------------------------|
| File            | `Superstore.csv.csv`              |
| Rows            | ~9,994 order line-items           |
| Columns         | 21                                |
| Time Span       | 2014 – 2017                       |
| Geography       | United States (4 regions)         |
| Granularity     | One row per order line-item       |

### Column Reference

| Column        | Type        | Description                                |
|---------------|-------------|--------------------------------------------|
| Row ID        | Integer     | Sequential record identifier               |
| Order ID      | String      | Unique order identifier (shared by lines)  |
| Order Date    | Date        | Date order was placed                      |
| Ship Date     | Date        | Date order was shipped                     |
| Ship Mode     | Categorical | Shipping tier selected                     |
| Customer ID   | String      | Unique customer identifier                 |
| Customer Name | String      | Customer full name                         |
| Segment       | Categorical | Consumer / Corporate / Home Office         |
| Country       | Categorical | United States (all rows)                   |
| City          | String      | City of delivery                           |
| State         | String      | State of delivery                          |
| Postal Code   | String      | 5-digit ZIP code                           |
| Region        | Categorical | West / East / South / Central              |
| Product ID    | String      | Unique product identifier                  |
| Category      | Categorical | Furniture / Office Supplies / Technology   |
| Sub-Category  | Categorical | 17 sub-categories                          |
| Product Name  | String      | Full product name                          |
| Sales         | Float       | Revenue for the line-item                  |
| Quantity      | Integer     | Units ordered                              |
| Discount      | Float       | Discount rate (0.0 – 1.0)                  |
| Profit        | Float       | Net profit/loss for the line-item          |

---

## Dataset Source

> **Note:** The dataset file (`Superstore.csv.csv`) is included in the project workspace root.  
> The Superstore dataset is widely attributed to the Tableau sample dataset collection.  
> Original URL: _[placeholder — verify the exact source from your local copy or course materials]_

---

## Technologies Used

| Library        | Purpose                                  |
|----------------|------------------------------------------|
| pandas         | Data loading, cleaning, transformation   |
| numpy          | Numerical operations                     |
| scikit-learn   | ML pipelines, models, evaluation         |
| matplotlib     | Static visualizations                    |
| seaborn        | Statistical visualizations               |
| streamlit      | Interactive dashboard                    |
| python-docx    | Programmatic DOCX report generation      |
| openpyxl       | Excel file support (optional)            |

---

## Project Methodology — 4-Tier Analytics Ladder

### Tier 1 — Data Hygiene and Descriptive Analytics
- Load raw dataset with encoding fallback
- Remove blank trailing rows and exact duplicates
- Normalize mixed date formats (`MM-DD-YYYY` and `M/DD/YYYY`) to standard datetime
- Fix corrupt `Category` value (`"C"`) using Product ID prefix inference
- Clean mojibake encoding artifacts in `Product Name`
- Enforce correct data types for all columns
- Document every cleaning step

### Tier 2 — Exploratory and Diagnostic Analytics
- Compute core KPIs: Total Sales, Profit, Quantity, Orders, Customers, Margin, Loss %
- 7+ professional visualizations covering:
  - Monthly Sales & Profit Trend
  - Category Performance
  - Regional Performance
  - Sub-Category Profitability
  - Discount vs Profit Association
  - Customer Segment Performance
  - State-Level Profitability
- Empirical observations with explicit distinction between:
  - Descriptive facts
  - Associations / correlations
  - Business hypotheses

### Tier 3 — Predictive Machine Learning
- **Target:** `Loss_Flag` (1 = Profit < 0, 0 = Profit ≥ 0)
- **Leakage Prevention:** Profit excluded from features; full leakage audit documented
- **Feature Engineering:** Shipping Duration, Order Year/Month/Quarter/DayOfWeek
- **Models:** Logistic Regression (baseline) and Random Forest
- **Split:** 80% train / 20% test, stratified by target
- **Pipeline:** sklearn ColumnTransformer with StandardScaler + OneHotEncoder

### Tier 4 — Prescriptive Analytics
- Risk-bucket analysis (top 10% loss-risk records)
- 5 specific, data-backed business recommendations
- Operationally actionable rules tied to observed patterns

---

## Data Cleaning Approach

| Issue | Treatment |
|-------|-----------|
| Mixed date formats | `pd.to_datetime` with `infer_datetime_format=True` + fallback pass |
| Corrupt Category "C" | Corrected to "Technology" using Product ID prefix `TEC-` |
| Encoding artifacts in Product Name | Removed `\ufffd` and `\x80-\x9f` code points |
| Trailing blank rows | Detected with `isnull().all(axis=1)` and removed |
| Exact duplicate rows | Removed with `drop_duplicates()` |
| Negative Profit | Retained as legitimate business losses |
| Missing values | Verified; no structural missingness found |

---

## EDA Summary

| KPI | Value |
|-----|-------|
| Total Sales | Calculated from actual data |
| Total Profit | Calculated from actual data |
| Profit Margin | Calculated from actual data |
| Loss-making Lines | Calculated from actual data |
| Pearson r (Discount vs Profit) | Calculated from actual data |

_All KPI values appear in executed notebook output — not hardcoded here._

**Key Findings:**
- Q4 months show consistent sales spikes (seasonal pattern)
- Technology has the highest profit margin; Furniture the lowest
- West region is most profitable; Central shows disproportionate losses
- Tables and Bookcases are net loss-making sub-categories
- Discount ≥ 40% correlates strongly with loss-making orders

---

## Machine Learning Approach

### Target Variable

```
Loss_Flag = 1  when  Profit < 0   (loss-making order line)
Loss_Flag = 0  when  Profit ≥ 0   (profitable order line)
```

### Leakage Prevention

| Column | Decision | Reason |
|--------|----------|--------|
| Profit | **EXCLUDED** | Loss_Flag is directly derived from this |
| Loss_Flag | **TARGET** | Not a feature |
| Row ID, Order ID, Customer ID, Product ID | **EXCLUDED** | High-cardinality identifiers |
| Customer Name, Product Name | **EXCLUDED** | Free-text, no safe representation |
| Country | **EXCLUDED** | Zero variance (all United States) |
| Postal Code, City | **EXCLUDED** | High cardinality; covered by State + Region |
| Sales | **INCLUDED** | Revenue is not a direct derivation of Profit (cost structure unknown) |
| Discount, Quantity, Ship Mode, Segment, Region, Category, Sub-Category, State | **INCLUDED** | All available at order entry |
| Shipping Duration, Order temporal features | **INCLUDED** | Derived from available date fields |

### Model Evaluation Metrics

All metrics computed from actual test-set execution (20% held-out):

- **Accuracy** — Overall fraction of correct predictions
- **Precision** — Of all predicted losses, fraction that are actual losses
- **Recall** — Of all actual losses, fraction the model catches
- **ROC-AUC** — Discrimination ability across all thresholds

_Actual metric values appear in the executed notebook output._

---

## Business Insights

1. **Discount control** is the single highest-leverage lever — loss rate rises sharply above 30% discount
2. **Tables sub-category** is structurally loss-making at aggregate level
3. **Central region** underperforms relative to its sales volume
4. **Technology** is the most profitable category by margin
5. **Home Office segment** achieves the highest profit margin per revenue

---

## Prescriptive Recommendations

| # | Recommendation | Evidence |
|---|---------------|----------|
| 1 | Cap discounts at 30%; require manager approval above this | Loss rate > 50% for discounts > 40% |
| 2 | Pricing review for Tables and Bookcases | Net negative aggregate profit |
| 3 | Regional discount audit for Central region | Disproportionate loss rate |
| 4 | Deploy ML model as real-time loss-risk scorer | Random Forest ROC-AUC from actual test set |
| 5 | Furniture bundling strategy | Lowest category margin |

---

## Project Folder Structure

```
superstore data analytics/
│
├── Superstore.csv.csv              ← Original dataset (do not modify)
├── AnandKumar_SuperstoreAnalytics.ipynb  ← Main Jupyter Notebook
├── app.py                          ← Streamlit dashboard
├── requirements.txt                ← Python dependencies
├── README.md                       ← This file
├── generate_report.py              ← DOCX report generator script
├── AnandKumar_ProjectReport.docx   ← Generated project report
│
├── data/
│   └── superstore_cleaned.csv      ← Cleaned dataset (auto-generated by notebook)
│
├── charts/                         ← Saved chart images (auto-generated by notebook)
│   ├── viz1_monthly_trend.png
│   ├── viz2_category.png
│   ├── viz3_region.png
│   ├── viz4_subcategory_profit.png
│   ├── viz5_discount_vs_profit.png
│   ├── viz6_segment.png
│   ├── viz7_state_profit.png
│   ├── viz_target_dist.png
│   ├── viz_confusion_matrices.png
│   ├── viz_roc_curve.png
│   └── viz_feature_importance.png
│
└── screenshots/                    ← Dashboard screenshots (manual)
```

---

## Installation Instructions

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## How to Run the Jupyter Notebook

```bash
jupyter notebook AnandKumar_SuperstoreAnalytics.ipynb
```

Or, to run all cells non-interactively:

```bash
jupyter nbconvert --to notebook --execute AnandKumar_SuperstoreAnalytics.ipynb --output AnandKumar_SuperstoreAnalytics_executed.ipynb
```

The notebook must be run from the project root directory (where `Superstore.csv.csv` is located).

---

## How to Run the Streamlit Application

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`.

Ensure you are in the project root directory when running this command.

---

## How to Generate the DOCX Report

```bash
python generate_report.py
```

This will create `AnandKumar_ProjectReport.docx` in the project root.

> **Note:** The report generator reads the cleaned dataset and notebook outputs. Run the notebook first to generate `data/superstore_cleaned.csv` and chart images.

---

## GitHub Usage Instructions

```bash
# Initialize repository
git init
git add .
git commit -m "Initial commit: Superstore Analytics Project"

# Push to GitHub
git remote add origin https://github.com/<your-username>/superstore-analytics.git
git branch -M main
git push -u origin main
```

**Important:** Add a `.gitignore` to exclude virtual environments:

```
venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
```

---

## Limitations

1. The dataset covers 2014–2017 only; insights may not reflect current market dynamics.
2. The Profit column is the net reported margin but does not include overhead, warehouse, or logistics costs beyond what is captured in the raw data.
3. The ML model is trained on order line-item features; order-level or customer-level patterns are not fully exploited.
4. The model predicts loss risk at line-item level — aggregated order-level loss prediction would require additional work.
5. No external data (e.g., competitor pricing, economic indicators) was available for enrichment.

---

## Future Improvements

1. Add SHAP values for ML explainability to make the model interpretable to business users
2. Build a customer lifetime value (CLV) model using Customer ID aggregation
3. Incorporate time-series forecasting (Prophet / ARIMA) for sales planning
4. Develop a product recommendation engine using co-purchase patterns
5. Add geographic visualisation (choropleth map by state)
6. Implement hyperparameter tuning with cross-validated grid search
7. Explore XGBoost / LightGBM for potentially higher ROC-AUC

---

*This project was created as an internship-ready, reproducible analytics portfolio piece using only the actual Superstore dataset.*
