# 🏡 Missing Value Imputation in Real Estate Pricing Data  
### Using Linear Regression

**Objective:**  
This project demonstrates how to handle missing target values using a **model-based imputation** technique rather than removing rows or filling missing values with averages.

**Dataset Used:** Real_Estate_with_Missing.csv  
**Guided by:** Aman Kharwal -Reference for conceptual understanding  
**Developed by:** *Pratik More*

---

## 📌 Project Overview

Many real-world datasets contain missing values. In this case study, the *House Price per Unit Area* column contained missing entries (some incorrectly recorded as `0`).  
Instead of dropping valuable data or using naive imputation, we use **Linear Regression to predict and fill the missing values.**

This method **preserves dataset integrity** and leads to more realistic downstream analysis.

---

## 🧠 Key Concepts Demonstrated

| Concept | Why It Matters |
|--------|---------------|
| Detecting & marking missing values | Avoids misleading results |
| Separating complete vs. incomplete rows | Ensures proper training strategy |
| Train/Test Model Workflow | Ensures fair model evaluation |
| Model-based imputation using `.predict()` | More realistic than mean/median fill |
| Safe update using `.loc[]` | Prevents overwriting correct data |
| Exporting cleaned dataset | Enables reproducibility |

---

## 🔄 Workflow Summary

The steps below outline the full workflow from raw data to cleaned dataset:

1. **Load the dataset** and inspect structure and missingness.
2. **Convert invalid pricing values (`0`) to `NaN`**, acknowledging they represent missing data.
3. **Separate the dataset** into:
   - `df_complete` → rows where price is present  
   - `df_missing` → rows where price is missing
4. **Train a Linear Regression model** using known price rows (`df_complete`).
5. **Predict missing price values** for `df_missing` based on learned relationships.
6. **Impute the predictions back into the original dataset** using `.loc[]` to avoid overwriting correct values.
7. **Export the final cleaned dataset** for analysis and modeling.

This workflow **retains valuable data** and avoids the bias introduced by naive imputation techniques.

---

## 🔧 Key Code Step (Imputation Logic)

```python
# Replace invalid values with NaN
df["House price of unit area"] = df["House price of unit area"].replace(0, np.nan)

# Split data based on availability of target values
df_complete = df[df[target].notna()]
df_missing = df[df[target].isna()]

# Train model on complete cases
model = LinearRegression()
model.fit(df_complete[features], df_complete[target])

# Impute missing values
df.loc[df[target].isna(), target] = model.predict(df_missing[features])
```
---

## 📝 Note on Interpretation

This CaseStudy was completed as a **conceptual learning case study** to demonstrate how **Linear Regression can be used to impute missing values** in a dataset.

Linear Regression is a **simple, interpretable model**, and it was chosen here to clearly illustrate the **imputation logic** — not to produce perfect real-world house price predictions.

As real estate pricing is influenced by **non-linear and geographic factors**, Linear Regression may sometimes produce **negative or unrealistic values**. In practical, production-level applications, models such as:

- **Random Forest Regression**
- **Gradient Boosting (XGBoost / LightGBM)**
- or **KNN-based imputation**

would provide more realistic results.

The objective here was to **learn and demonstrate the methodology**, not to deploy a pricing model.

> **Key takeaway:**  
> Model-based imputation helps retain valuable data and improves dataset completeness — but **model choice should always reflect real-world context.**

---

