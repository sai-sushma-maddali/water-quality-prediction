
# **Water Quality Prediction Using Machine Learning**

## **Project Overview**

This project focuses on developing a **data-driven framework** for predicting **Water Quality Index (WQI)** using field data from the **California Department of Water Resources (DWR)**.
The study aims to support sustainable water management by leveraging **machine learning (ML)** and **exploratory data analysis (EDA)** to uncover patterns in physicochemical parameters such as **pH**, **Dissolved Oxygen**, **Turbidity**, **Conductivity**, and **Temperature**.

The project is part of the **DATA 245 – Machine Learning** course at *San José State University*.

---

## **Objectives**

* Perform **EDA** to understand the structure, distribution, and quality of California’s water quality dataset.
* Standardize and clean multi-parameter field data for consistency and accuracy.
* Identify core water quality indicators and eliminate sparse or redundant features.
* Build the foundation for **Water Quality Index (WQI)** computation and predictive modeling using ML.

---
## **Live Demo**

https://water-quality-prediction-ca-dwr.streamlit.app/

---
## **Demo Videos**
The demo videos for this project are packaged as downloadable artifacts and attached to each GitHub Release. These videos provide a walkthrough of the UI, model predictions, and Augmented Reality demo.
You can find them under:

**GitHub → Repository → Releases → Assets**


---

## **Exploratory Data Analysis (EDA)**

### **1. Data Understanding**

* Data sourced from **California DWR Field Measurements (1913–2025)**.
* Explored key metadata including:
  `station_id`, `station_type`, `county_name`, `parameter`, and `fdr_result`.
* Visualized sample distribution across **time** (Figure 1) and **counties** (Figure 2) to identify data-rich regions suitable for analysis.

### **2. Data Cleaning and Standardization**

* Standardized units to ensure uniformity across stations:

  * °C → Water Temperature
  * µS/cm → Conductivity
  * mg/L → Dissolved Oxygen
  * NTU → Turbidity
* Built a **unit mapping dictionary** to resolve naming inconsistencies.
* Detected and replaced 296 outlier readings (e.g., pH > 14, DO > 20 mg/L, negative temperatures).

### **3. Missing Data and Sparse Features**

* Analyzed missingness for all parameters.
* Retained five high-coverage parameters (70–98%):
  `pH_pH units`, `DissolvedOxygen_mg/L`, `Turbidity_NTU`, `SpecificConductance_µS/cm`, `WaterTemperature_°C`.
* Dropped sparse parameters (>80% missing).
* Imputed missing coordinates using **county-level averages**.

### **4. Data Transformation**

* Converted data from **long to wide format** to have one row per sampling event.
* Each parameter became a separate column, preserving station metadata (county, station type, latitude, longitude).
* Conducted **correlation analysis** and visualized pairwise relationships among core parameters.

### **5. Key Insights**

* Data from **2000–2025** showed the highest completeness and consistency.
* Strong correlations observed between:

  * Dissolved Oxygen and Temperature (negative correlation)
  * Turbidity and Conductivity (moderate positive correlation)
* Missing data patterns confirmed uneven sampling between counties — surface water data dominated the dataset.
* Dataset now clean, standardized, and ready for WQI computation and ML modeling.

---

## **Next Steps**

* Compute **Water Quality Index (WQI)** using weighted arithmetic mean.
* Perform **feature engineering** and **feature selection**.
* Handle **class imbalance** using SMOTE.
* Train ML models (Random Forest, XGBoost, SVM) for WQI class prediction.
* Evaluate performance using **F1-score** and feature importance analysis.

---

## **Repository Structure**

```
water-quality-prediction/
│
├── documentation/
│   ├── Intermediate-project-status.pdf
│   ├── Predictive Modeling for Water Quality Assessment - project proposal.pdf
│   ├── Predictive_Modeling_for_Water_Quality_Assessment.pdf
│   ├── Water_Quality_Prediction-final-report.pdf
│   ├── Water-quality-prediction-slides.pdf
├── notebooks/                     # Jupyter notebooks for EDA and modeling
│   ├── Water Quality Testing.ipynb
│   ├── water_quality_analysis_ed_v2.ipynb
│   ├── water_quality_analysis_eda.ipynb
│   ├── water_quality_analysis_model_training.ipynb
├── src/
│   ├── app.py
│   ├── label_encoder.pkl
│   ├── processed_dataset_WQ.pkl
│   ├── requirements.txt
│   ├── water_quality_testing_xgb.py
│   ├── wqi_xgb_pipeline.pkl
├── README.md                      # Project overview

```

---

## **Technologies Used**

* **Python 3.10+**
* **Jupyter Notebook**
* **pandas**, **NumPy**, **matplotlib**, **seaborn**
* **scikit-learn**
* **XGBoost**, **imbalanced-learn**

---

## **Data Source**

California Department of Water Resources (DWR) –

https://data.ca.gov/dataset/water-quality-data

---


## **References**

* Kumar, R., & Singh, A. (2024). *Water quality prediction with machine learning algorithms.*
  *EPRA International Journal of Multidisciplinary Research (IJMR), 10(4), 45–53.*
  [https://doi.org/10.36713/epra16318](https://doi.org/10.36713/epra16318)
* Zhu, M. et al. (2022). *A review of the application of machine learning in water quality evaluation.*
  *Eco-Environment & Health, 1(2), 107–116.*
  [https://doi.org/10.1016/j.eehl.2022.06.001](https://doi.org/10.1016/j.eehl.2022.06.001)

*(Additional references included in project report)*

---
