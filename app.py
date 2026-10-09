import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(
    page_title="Healthcare Automated Insight Generation",
    page_icon="📊",
    layout="wide"
)

# --- State Initialization ---
defaults = {
    "trend_threshold": 10.0, 
    "iqr_multiplier": 1.5, 
    "corr_threshold": 0.70, 
    "dataset_path": "data/healthcare_performance.csv"
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def update_trend():
    st.session_state.trend_threshold = st.session_state._trend
def update_iqr():
    st.session_state.iqr_multiplier = st.session_state._iqr
def update_corr():
    st.session_state.corr_threshold = st.session_state._corr
def update_path():
    st.session_state.dataset_path = st.session_state._path

# --- Core Logic Functions ---
@st.cache_data
def load_data(filepath):
    if not os.path.exists(filepath):
        return None, f"File not found: {filepath}"
    try:
        df = pd.read_csv(filepath)
        return df, "Success"
    except Exception as e:
        return None, f"Error reading CSV: {e}"

def validate_data(df):
    required_cols = ["month", "district", "anc_coverage", "institutional_delivery", "immunization", "high_risk_cases"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return False, f"Missing required columns: {missing}"
    
    numeric_cols = ["anc_coverage", "institutional_delivery", "immunization", "high_risk_cases"]
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            return False, f"Column '{col}' must be numeric."
            
    try:
        pd.to_datetime(df["month"], format="%Y-%m")
    except Exception:
        return False, "Column 'month' must be in YYYY-MM format."
        
    return True, "Data is valid."

def get_processed_data(df):
    data = df.copy()
    data["month"] = pd.to_datetime(data["month"], format="%Y-%m")
    return data

def detect_trends(df, threshold):
    indicators = ["anc_coverage", "institutional_delivery", "immunization", "high_risk_cases"]
    data = df.sort_values(["district", "month"]).copy()
    
    results = []
    for indicator in indicators:
        data["prev"] = data.groupby("district")[indicator].shift(1)
        data["curr"] = data[indicator]
        
        valid = data["prev"].notna() & (data["prev"] != 0)
        temp = data[valid].copy()
        temp["change_pct"] = ((temp["curr"] - temp["prev"]) / temp["prev"]) * 100
        
        flagged = temp[temp["change_pct"].abs() >= threshold].copy()
        
        for _, row in flagged.iterrows():
            results.append({
                "district": row["district"],
                "month": row["month"].strftime("%Y-%m"),
                "indicator": indicator,
                "previous_value": row["prev"],
                "current_value": row["curr"],
                "change_pct": row["change_pct"],
                "direction": "Increase" if row["change_pct"] > 0 else "Decrease"
            })
            
    return pd.DataFrame(results, columns=["district", "month", "indicator", "previous_value", "current_value", "change_pct", "direction"])

def detect_outliers(df, multiplier):
    indicators = ["anc_coverage", "institutional_delivery", "immunization", "high_risk_cases"]
    results = []
    
    for indicator in indicators:
        values = df[indicator].dropna()
        if len(values) < 4:
            continue
            
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        mask = (df[indicator] < lower_bound) | (df[indicator] > upper_bound)
        flagged = df[mask]
        
        for _, row in flagged.iterrows():
            results.append({
                "district": row["district"],
                "month": row["month"].strftime("%Y-%m"),
                "indicator": indicator,
                "value": row[indicator],
                "lower_bound": round(lower_bound, 2),
                "upper_bound": round(upper_bound, 2),
                "outlier_type": "Low" if row[indicator] < lower_bound else "High"
            })
            
    return pd.DataFrame(results, columns=["district", "month", "indicator", "value", "lower_bound", "upper_bound", "outlier_type"])

def calculate_correlations(df, threshold):
    indicators = ["anc_coverage", "institutional_delivery", "immunization", "high_risk_cases"]
    corr_matrix = df[indicators].corr(method='pearson')
    
    results = []
    for i in range(len(indicators)):
        for j in range(i + 1, len(indicators)):
            ind1 = indicators[i]
            ind2 = indicators[j]
            coef = corr_matrix.loc[ind1, ind2]
            
            if abs(coef) >= threshold:
                results.append({
                    "indicator_1": ind1,
                    "indicator_2": ind2,
                    "correlation": round(coef, 4),
                    "direction": "Positive" if coef > 0 else "Negative"
                })
    return corr_matrix, pd.DataFrame(results, columns=["indicator_1", "indicator_2", "correlation", "direction"])

def generate_insights(trends_df, outliers_df, corr_df):
    insights = []
    idx = 1
    
    if not trends_df.empty:
        for _, row in trends_df.iterrows():
            sev = "High" if abs(row["change_pct"]) > 25 else "Medium" if abs(row["change_pct"]) > 15 else "Low"
            insights.append({
                "insight_id": f"INS-{idx:04d}",
                "type": "Trend",
                "indicator": row["indicator"],
                "entity": row["district"],
                "period": row["month"],
                "metric": f"{row['change_pct']:.1f}%",
                "change": row["direction"],
                "severity": sev,
                "explanation": f"Significant {row['direction'].lower()} in {row['indicator']} for {row['district']} ({row['previous_value']} -> {row['current_value']})."
            })
            idx += 1
            
    if not outliers_df.empty:
        for _, row in outliers_df.iterrows():
            sev = "High"
            insights.append({
                "insight_id": f"INS-{idx:04d}",
                "type": "Outlier",
                "indicator": row["indicator"],
                "entity": row["district"],
                "period": row["month"],
                "metric": str(row["value"]),
                "change": row["outlier_type"],
                "severity": sev,
                "explanation": f"Statistical anomaly: {row['indicator']} value of {row['value']} is unusually {row['outlier_type'].lower()} (bounds: {row['lower_bound']} to {row['upper_bound']})."
            })
            idx += 1
            
    if not corr_df.empty:
        for _, row in corr_df.iterrows():
            sev = "Medium" if abs(row["correlation"]) < 0.85 else "High"
            insights.append({
                "insight_id": f"INS-{idx:04d}",
                "type": "Correlation",
                "indicator": f"{row['indicator_1']} & {row['indicator_2']}",
                "entity": "All Districts",
                "period": "All Months",
                "metric": f"r={row['correlation']}",
                "change": row["direction"],
                "severity": sev,
                "explanation": f"Exploratory association found between {row['indicator_1']} and {row['indicator_2']}. Note: Correlation does not imply causation."
            })
            idx += 1
            
    return pd.DataFrame(insights, columns=["insight_id", "type", "indicator", "entity", "period", "metric", "change", "severity", "explanation"])

# --- Main App ---
st.title("Healthcare Automated Insight Generation")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Page", [
    "Overview",
    "Part A — Data Loading & Validation",
    "Part B — Trend Detection",
    "Part C — Outlier Detection",
    "Part D — Correlation Detection",
    "Part E — Automated Insights",
    "Part F — Visualizations"
])

# Global Data Processing Pipeline
raw_df, load_msg = load_data(st.session_state.dataset_path)
df_valid = False
processed_df = None

trends_df = pd.DataFrame()
outliers_df = pd.DataFrame()
corr_matrix = pd.DataFrame()
corr_df = pd.DataFrame()
insights_df = pd.DataFrame()

if raw_df is not None:
    is_valid, v_msg = validate_data(raw_df)
    if is_valid:
        df_valid = True
        processed_df = get_processed_data(raw_df)
        
        trends_df = detect_trends(processed_df, st.session_state.trend_threshold)
        outliers_df = detect_outliers(processed_df, st.session_state.iqr_multiplier)
        corr_matrix, corr_df = calculate_correlations(processed_df, st.session_state.corr_threshold)
        insights_df = generate_insights(trends_df, outliers_df, corr_df)
        
        # Save results to the results folder
        os.makedirs("results", exist_ok=True)
        trends_df.to_csv("results/trend_results.csv", index=False)
        outliers_df.to_csv("results/outlier_results.csv", index=False)
        corr_matrix.to_csv("results/corr_matrix.csv", index=True)
        corr_df.to_csv("results/corr_pairs.csv", index=False)
        insights_df.to_csv("results/insights.csv", index=False)

if page == "Overview":
    st.header("Overview")
    if not df_valid:
        st.warning("Please navigate to Part A and load a valid dataset.")
    else:
        st.write("Summary metrics from the current dataset and configuration:")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Districts", processed_df["district"].nunique())
        col2.metric("Total Records", len(processed_df))
        col3.metric("Total Generated Insights", len(insights_df))
        
        col4, col5, col6 = st.columns(3)
        col4.metric("Trends Detected", len(trends_df))
        col5.metric("Outliers Detected", len(outliers_df))
        col6.metric("Correlation Pairs", len(corr_df))
        
        st.subheader("Common Insights Preview")
        st.write("A combined view of the most significant statistical findings.")
        if not insights_df.empty:
            st.dataframe(insights_df.head(10), use_container_width=True, hide_index=True)
            st.info("Navigate to Part E for full insights and filtering options.")
        else:
            st.info("No insights generated with current thresholds.")

elif page == "Part A — Data Loading & Validation":
    st.header("Part A — Data Loading & Validation")
    
    st.text_input("Dataset File Path", value=st.session_state.dataset_path, key="_path", on_change=update_path)
        
    if raw_df is None:
        st.error(f"Error loading dataset: {load_msg}")
    else:
        st.success(f"Dataset loaded successfully from `{st.session_state.dataset_path}`.")
        
        if is_valid:
            st.success(f"Validation passed: {v_msg}")
        else:
            st.error(f"Validation failed: {v_msg}")
            
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Shape:** {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")
            st.write("**Data Types:**")
            st.dataframe(pd.DataFrame(raw_df.dtypes, columns=["Type"]).astype(str), use_container_width=True)
        with col2:
            st.write("**Missing Values per Column:**")
            st.dataframe(pd.DataFrame(raw_df.isna().sum(), columns=["Missing Count"]), use_container_width=True)
            
        st.subheader("Dataset Explorer")
        districts = ["All"] + sorted(raw_df["district"].dropna().unique().tolist()) if "district" in raw_df.columns else ["All"]
        months = ["All"] + sorted(raw_df["month"].dropna().unique().tolist()) if "month" in raw_df.columns else ["All"]
        
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            selected_dist = st.selectbox("Filter District", districts)
        with f_col2:
            selected_month = st.selectbox("Filter Month", months)
            
        preview_df = raw_df.copy()
        if "district" in preview_df.columns and selected_dist != "All":
            preview_df = preview_df[preview_df["district"] == selected_dist]
        if "month" in preview_df.columns and selected_month != "All":
            preview_df = preview_df[preview_df["month"] == selected_month]
            
        st.write("First 5 rows (matching filters):")
        st.dataframe(preview_df.head(5), use_container_width=True)

elif page == "Part B — Trend Detection":
    st.header("Part B — Trend Detection")
    if not df_valid:
        st.warning("Please load a valid dataset in Part A.")
    else:
        st.slider("Percentage change threshold", min_value=1.0, max_value=50.0, value=st.session_state.trend_threshold, step=1.0, key="_trend", on_change=update_trend)
        
        col1, col2 = st.columns(2)
        districts = ["All"] + sorted(processed_df["district"].unique().tolist())
        with col1:
            selected_district = st.selectbox("Select district", districts)
        with col2:
            indicators = ["All indicators", "anc_coverage", "institutional_delivery", "immunization", "high_risk_cases"]
            selected_indicator = st.selectbox("Select indicator", indicators)
            
        res = trends_df.copy()
        if not res.empty:
            if selected_district != "All":
                res = res[res["district"] == selected_district]
            if selected_indicator != "All indicators":
                res = res[res["indicator"] == selected_indicator]
                
        st.subheader("Detected Trends")
        st.metric("Significant trends", len(res))
        
        if res.empty:
            st.info("No trends meet the selected criteria.")
        else:
            res["change_pct"] = res["change_pct"].round(2)
            st.dataframe(res, use_container_width=True, hide_index=True)
            csv = res.to_csv(index=False).encode('utf-8')
            st.download_button("Download trend results", data=csv, file_name="trend_results.csv", mime="text/csv")

elif page == "Part C — Outlier Detection":
    st.header("Part C — Outlier Detection")
    if not df_valid:
        st.warning("Please load a valid dataset in Part A.")
    else:
        st.slider("IQR multiplier", min_value=1.0, max_value=3.0, value=st.session_state.iqr_multiplier, step=0.1, key="_iqr", on_change=update_iqr)
        
        col1, col2 = st.columns(2)
        districts = ["All"] + sorted(processed_df["district"].unique().tolist())
        with col1:
            selected_district = st.selectbox("Select district", districts)
        with col2:
            indicators = ["All indicators", "anc_coverage", "institutional_delivery", "immunization", "high_risk_cases"]
            selected_indicator = st.selectbox("Select indicator", indicators)
            
        res = outliers_df.copy()
        if not res.empty:
            if selected_district != "All":
                res = res[res["district"] == selected_district]
            if selected_indicator != "All indicators":
                res = res[res["indicator"] == selected_indicator]
                
        st.subheader("Detected Outliers")
        st.write("Note: An outlier is a statistical anomaly based on the IQR method, not necessarily a healthcare failure.")
        st.metric("Outliers detected", len(res))
        
        if res.empty:
            st.info("No outliers detected with the current settings.")
        else:
            st.dataframe(res, use_container_width=True, hide_index=True)
            csv = res.to_csv(index=False).encode('utf-8')
            st.download_button("Download outlier results", data=csv, file_name="outlier_results.csv", mime="text/csv")

elif page == "Part D — Correlation Detection":
    st.header("Part D — Correlation Detection")
    if not df_valid:
        st.warning("Please load a valid dataset in Part A.")
    else:
        st.slider("Absolute correlation threshold", min_value=0.10, max_value=1.00, value=st.session_state.corr_threshold, step=0.05, key="_corr", on_change=update_corr)
        
        st.subheader("Correlation Heatmap")
        st.write("Pearson correlation across numerical indicators.")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
        st.pyplot(fig)
        
        st.subheader("Flagged Indicator Pairs")
        st.write("Note: Correlation does not imply causation. The dataset is small and exploratory.")
        if corr_df.empty:
            st.info("No correlation pairs meet the selected threshold.")
        else:
            st.dataframe(corr_df, use_container_width=True, hide_index=True)
            
            col1, col2 = st.columns(2)
            with col1:
                csv1 = corr_matrix.to_csv(index=True).encode('utf-8')
                st.download_button("Download matrix", data=csv1, file_name="corr_matrix.csv", mime="text/csv")
            with col2:
                csv2 = corr_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download pairs", data=csv2, file_name="corr_pairs.csv", mime="text/csv")

elif page == "Part E — Automated Insights":
    st.header("Part E — Automated Insights")
    if not df_valid:
        st.warning("Please load a valid dataset in Part A.")
    else:
        st.write("Combined insights generated dynamically from Trends, Outliers, and Correlations.")
        st.write("**Severity Rules:**")
        st.write("- **Trend**: High if > 25% change, Medium if > 15%, else Low.")
        st.write("- **Outlier**: Always High severity as it marks a statistical anomaly.")
        st.write("- **Correlation**: High if |r| >= 0.85, else Medium.")
        
        if insights_df.empty:
            st.info("No insights generated with the current global thresholds.")
        else:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                districts = ["All"] + sorted(list(set(insights_df["entity"].unique())))
                selected_district = st.selectbox("Entity", districts)
            with col2:
                types = ["All"] + sorted(list(set(insights_df["type"].unique())))
                selected_type = st.selectbox("Type", types)
            with col3:
                severities = ["All"] + sorted(list(set(insights_df["severity"].unique())))
                selected_severity = st.selectbox("Severity", severities)
            with col4:
                inds = ["All"] + sorted(list(set(insights_df["indicator"].unique())))
                selected_indicator = st.selectbox("Indicator", inds)
                
            res = insights_df.copy()
            if selected_district != "All":
                res = res[res["entity"] == selected_district]
            if selected_type != "All":
                res = res[res["type"] == selected_type]
            if selected_severity != "All":
                res = res[res["severity"] == selected_severity]
            if selected_indicator != "All":
                res = res[res["indicator"] == selected_indicator]
                
            st.metric("Filtered Insights Count", len(res))
            if res.empty:
                st.info("No insights match the selected filters.")
            else:
                st.dataframe(res, use_container_width=True, hide_index=True)
                csv = res.to_csv(index=False).encode('utf-8')
                st.download_button("Download insights", data=csv, file_name="insights.csv", mime="text/csv")

elif page == "Part F — Visualizations":
    st.header("Part F — Visualizations")
    if not df_valid:
        st.warning("Please load a valid dataset in Part A.")
    else:
        st.write("Use the visualizations below to explore the data and generated insights. These charts reflect current data and do not imply future predictions.")
        
        st.subheader("1. Insights by Severity")
        if not insights_df.empty:
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            sns.countplot(data=insights_df, x="severity", order=["Low", "Medium", "High"], palette="Set2", ax=ax1)
            ax1.set_title("Number of Insights by Severity")
            ax1.set_ylabel("Count")
            ax1.set_xlabel("Severity Level")
            st.pyplot(fig1)
        else:
            st.info("No insights available to visualize.")
            
        st.subheader("2. Correlation Heatmap")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax2)
        ax2.set_title("Pearson Correlation Heatmap")
        st.pyplot(fig2)
        
        st.subheader("3. Indicator Over Time")
        indicators = ["anc_coverage", "institutional_delivery", "immunization", "high_risk_cases"]
        selected_ind = st.selectbox("Select Indicator", indicators)
        
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        sns.lineplot(data=processed_df, x="month", y=selected_ind, hue="district", marker="o", ax=ax3)
        ax3.set_title(f"{selected_ind} Over Time by District")
        ax3.set_ylabel("Value")
        ax3.set_xlabel("Month")
        plt.xticks(rotation=45)
        st.pyplot(fig3)
