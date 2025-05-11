import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit config
st.set_page_config(page_title="Air Quality Analysis", layout="wide")
sns.set(style='whitegrid')

data = pd.read_csv("preprocessed_air_data.csv")
model = joblib.load("lightweight_air_pollution_model.pkl")
expected_features = joblib.load("model_features.pkl")

page = st.sidebar.radio("Select Page", ["Data Overview", "Exploratory Data Analysis", "Model Prediction"])

# Data Overview Page
if page == "Data Overview":
    st.title("Data Overview")

    st.markdown("""
    The dataset was cleaned to prepare it for future analysis. Key preprocessing steps included:

    - Duplicate rows were removed to avoid skewing results.
    - Rows with missing values in any pollutant column were deleted, as these are critical for air quality analysis.
    - Missing wind direction values were forward-filled to maintain temporal consistency.
    - A new `datetime` column was created by combining the `year`, `month`, `day`, and `hour` fields.
    - Time-based features `month_name` and `day_name` were extracted to support trend analysis.
    - The original date/time columns were dropped after transformation.
    - A comparison table was produced to show missing values before and after cleaning.
    - Summary outputs were added to confirm all preprocessing steps were correctly applied.
    """)


    st.write("Hourly air quality and weather data from four Beijing monitoring stations (Urban, Suburban, Rural, Hotspot).")

    # Dataset Preview
    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    # Data Shape and Types
    st.subheader("Data Shape and Types")
    st.markdown(f"- **Total Rows:** {data.shape[0]:,}")
    st.markdown(f"- **Total Columns:** {data.shape[1]}")

    dtypes_df = pd.DataFrame(data.dtypes, columns=["Data Type"])
    st.dataframe(dtypes_df.T)

    # Preprocessing Summary
    st.subheader("Preprocessing Summary")

    # Load original merged dataset
    original_data = pd.read_csv("merged_air_data.csv")

    # Duplicates rows removed
    original_rows = original_data.shape[0]
    deduped = original_data.drop_duplicates()
    duplicates_removed = original_rows - deduped.shape[0]

    # Rows deleted due to missing pollutants
    pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    rows_deleted = deduped.shape[0] - deduped.dropna(subset=pollutants).shape[0]

    # Wind direction forward-filled
    wd_missing_before = original_data['wd'].isnull().sum()
    wd_filled = wd_missing_before - data['wd'].isnull().sum()

    # Weather interpolation
    weather_cols = ['TEMP', 'DEWP', 'PRES', 'RAIN', 'WSPM']
    weather_filled = {}
    for col in weather_cols:
        filled = original_data[col].isnull().sum() - data[col].isnull().sum()
        weather_filled[col] = filled

    # Comparison of missing values
    missing_comparison = pd.DataFrame({
        'Missing Before': original_data.isnull().sum(),
        'Missing After': data.isnull().sum()
    }).T
    missing_comparison = missing_comparison.loc[:, (missing_comparison.loc['Missing Before'] > 0)]

    # Display key summaries
    st.markdown(f"- **Duplicate rows removed:** {duplicates_removed:,}")
    st.markdown(f"- **Rows deleted due to missing pollutant values:** {rows_deleted:,}")
    st.markdown(f"- **Wind direction values forward-filled:** {wd_filled:,}")
    st.markdown(f"- **Total weather values interpolated:** {sum(weather_filled.values()):,}")

    st.subheader("Weather Interpolation Summary")
    weather_df = pd.DataFrame(weather_filled, index=["Values Interpolated"])
    st.dataframe(weather_df)

    st.subheader("Missing Values Before vs After")
    st.dataframe(missing_comparison.round(0).astype("Int64"))

    st.markdown(f"**Original dataset shape:** {original_data.shape}")
    st.markdown(f"**Cleaned dataset shape:** {data.shape}")



# Exploratory Data Analysis Page
elif page == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")

    st.markdown("""
    Exploratory Data Analysis (EDA) was conducted to uncover patterns and relationships in the cleaned dataset using a range of statistical and visual techniques:

    - Univariate histograms revealed that most pollutants had a large number of low concentration values, with a few extreme outliers. Rainfall was frequently recorded as zero, while other weather variables had more balanced distributions.
    - A bar chart comparing average PM2.5 and PM10 concentrations by weekday showed slightly elevated pollution on weekends, possibly due to increased residential activity.
    - Scatter plots displayed moderate negative correlations between temperature and both PM2.5 and CO, indicating that pollution levels tend to decrease as temperatures rise.
    - A correlation heatmap highlighted a strong positive relationship between PM2.5 and PM10, and a negative correlation between O3 and most other pollutants—likely due to its dependence on sunlight.
    - Line plots over time revealed seasonal variation, with CO concentrations rising during colder months, and broader fluctuations in PM2.5 and PM10.
    - An hourly trend analysis showed morning and evening peaks in PM2.5, PM10, and NO2, likely reflecting traffic patterns, while O3 peaked in the afternoon due to sunlight exposure.

    These results provided meaningful insights into pollution behaviour across time, temperature, and atmospheric conditions, helping inform the selection of features for the predictive model.
    """)

    # Univariate - Pollution
    st.subheader("Pollution Data Distributions")
    fig, axs = plt.subplots(2, 3, figsize=(15, 6))
    pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    for i, col in enumerate(pollutants):
        ax = axs[i // 3, i % 3]
        clipped = data[col][data[col] < data[col].quantile(0.99)]
        ax.hist(clipped.dropna(), bins=30)
        ax.set_title(col)
        ax.set_xlabel("Concentration")
        ax.set_ylabel("Count")
    plt.suptitle("Pollutant Distributions", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    st.pyplot(fig, use_container_width=False)

    # Univariate - Weather
    st.subheader("Weather Data Distributions")
    weather = ['TEMP', 'DEWP', 'PRES', 'RAIN', 'WSPM']
    fig, axs = plt.subplots(2, 3, figsize=(15, 5))

    for i, col in enumerate(weather):
        ax = axs[i // 3, i % 3]
        clipped = data[col][data[col] < data[col].quantile(0.99)]
        ax.hist(clipped.dropna(), bins=30)
        ax.set_title(col)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")

    fig.delaxes(axs[1, 2])
    plt.suptitle("Weather Distributions", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    st.pyplot(fig, use_container_width=False)


    # Bivariate - Bar Chart Weekday
    st.subheader("Average PM2.5 and PM10 by Weekday")
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    barchart_data = data.groupby('day_name')[['PM2.5', 'PM10']].mean().reindex(weekday_order)
    fig, ax = plt.subplots(figsize=(8, 5))
    barchart_data.plot(kind='bar', ax=ax)
    ax.set_title("Average PM2.5 and PM10 by Weekday")
    ax.set_ylabel("Concentration")
    ax.set_xlabel("Day of Week")
    ax.grid(axis='y')
    ax.legend(title="Pollutant", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)

    # Bivariate - Scatter plots vs TEMP
    st.subheader("Pollution vs Temperature")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.scatterplot(x='TEMP', y='PM2.5', data=data, alpha=0.3, ax=axes[0])
    axes[0].set_title('PM2.5 by TEMP')
    axes[0].grid(True)
    sns.scatterplot(x='TEMP', y='CO', data=data, alpha=0.3, color='green', ax=axes[1])
    axes[1].set_title('CO by TEMP')
    axes[1].grid(True)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)

    # Multivariate - Correlation Heatmap
    st.subheader("Correlation Heatmap of Pollutants and Weather Features")
    fig, ax = plt.subplots(figsize=(7, 5))
    corr = data[pollutants + weather].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', square=True, ax=ax, annot_kws={"size": 8})
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)

    # Multivariate - Line plots over time
    st.subheader("Pollutant Trends Over Time")
    line_plot = data[['datetime', 'PM2.5', 'PM10', 'CO']].dropna().sort_values('datetime')
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(line_plot['datetime'], line_plot['PM2.5'], color='tab:blue')
    axes[0].set_title('PM2.5 Levels Over Time')
    axes[0].set_ylabel('PM2.5')
    axes[0].grid(True)

    axes[1].plot(line_plot['datetime'], line_plot['PM10'], color='tab:orange')
    axes[1].set_title('PM10 Levels Over Time')
    axes[1].set_ylabel('PM10')
    axes[1].grid(True)

    axes[2].plot(line_plot['datetime'], line_plot['CO'], color='tab:green')
    axes[2].set_title('CO Levels Over Time')
    axes[2].set_ylabel('CO')
    axes[2].set_xlabel('Date')
    axes[2].grid(True)

    for ax in axes:
        ax.tick_params(labelbottom=True)
        ax.set_xlabel("Date")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)

    # Multivariate - Hourly trend analysis
    st.subheader("Hourly Pollution Concentration Trends")
    hourly_pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'O3']
    hourly_avg = data.groupby(data["datetime"].dt.hour)[hourly_pollutants].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in hourly_avg.columns:
        ax.plot(hourly_avg.index, hourly_avg[col], label=col, marker='o')
    ax.set_title("Average Pollution Concentration by Hour of Day")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Concentration")
    ax.set_xticks(range(0, 24))
    ax.grid(True)
    ax.legend(title="Pollutant", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)


# Prediction Page
elif page == "Model Prediction":
    st.title("Predict PM2.5 Levels")
    st.write("Enter pollutant and weather values to predict PM2.5 concentration using the trained model.")

    with st.form("predict_form"):
        st.subheader("Input Features")

        def numeric_input(col, label=None):
            label = label or col
            mean_val = float(data[col].mean())
            return st.number_input(label, value=mean_val, step=1.0)

        col1, col2 = st.columns(2)

        with col1:
            temp = numeric_input("TEMP", "Temperature (°C)")
            dewp = numeric_input("DEWP", "Dew Point (°C)")
            pres = numeric_input("PRES", "Pressure (hPa)")
            pm10 = numeric_input("PM10")

        with col2:
            rain = numeric_input("RAIN", "Rainfall (mm)")
            wspm = numeric_input("WSPM", "Wind Speed (m/s)")
            no2 = numeric_input("NO2")
            co = numeric_input("CO")

        col3, col4 = st.columns(2)

        with col3:
            day = st.selectbox("Day of Week", sorted(data["day_name"].unique()), key="day")
            month = st.selectbox("Month", sorted(data["month_name"].unique()), key="month")

        with col4:
            wd = st.selectbox("Wind Direction", sorted(data["wd"].dropna().unique()), key="wd")
            hour = st.slider("Hour of Day", 0, 23, 12)

        submitted = st.form_submit_button("Predict")

    if submitted:
        # Start with all features set to 0
        input_dict = {feature: 0 for feature in expected_features}

        # Fill numeric inputs
        input_dict['TEMP'] = temp
        input_dict['DEWP'] = dewp
        input_dict['PRES'] = pres
        input_dict['RAIN'] = rain
        input_dict['WSPM'] = wspm
        input_dict['PM10'] = pm10
        input_dict['NO2'] = no2
        input_dict['CO'] = co
        input_dict['hour'] = hour

        # One-hot encode categorical selections
        day_col = f'day_name_{day}'
        month_col = f'month_name_{month}'
        wd_col = f'wd_{wd}'

        if day_col in input_dict:
            input_dict[day_col] = 1
        if month_col in input_dict:
            input_dict[month_col] = 1
        if wd_col in input_dict:
            input_dict[wd_col] = 1

        # Add to dataframe
        input_encoded = pd.DataFrame([input_dict])

        # Run model
        prediction = model.predict(input_encoded)[0]
        st.success(f"Predicted PM2.5 concentration: {prediction:.2f} µg/m³")

    # Final model information
    st.markdown("### Model Evaluation Summary")
    st.markdown("""
    The final predictive model was built using the random forest regressor, trained on the prepared dataset using the best hyperparameters identified during the grid search.
    The model was able to achieve a strong performance on the test set with metrics of:

    - Mean absolute error: 4.09  
    - Root mean squared error: 6.57  
    - R2 score: 0.993

    The low MAE and RMSE showcase the model’s consistency at making accurate predictions and the strong R2 score of 0.993 proves that the model is able to explain almost all variation in PM2.5 concentrations.

    To ensure the final model was robust, a 5-fold cross-validation was performed across the full dataset:

    - R2 score: 0.925  
    - Standard deviation: 0.009

    These results display a highly accurate predictive model suitable for deployment.
    """)

    st.markdown("### Grid Search Optimisation Details")
    st.markdown("""
    To optimise the performance of the random forest model on the dataset, a grid search was performed using a parameter grid. The search explored tree depth, estimators, feature selection strategies, and leaf/split constraints while balancing performance and runtime.

    The grid search tested 108 combinations using 5-fold cross-validation (540 total runs). The best configuration was:

    - Number of estimators = 200  
    - Maximum tree depth = None  
    - Minimum samples to split = 2  
    - Minimum samples at leaf = 1  
    - Max features = 'sqrt'

    This configuration achieved a cross-validated R2 score of **0.925**, making it ideal for final deployment.
    """)


