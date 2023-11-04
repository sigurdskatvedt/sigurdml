import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from catboost import CatBoostRegressor, Pool
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 200)


def preprocess_data(targets, observed, estimated, test):
    """
    Preprocess the data by resampling, merging with targets, and dropping unnecessary columns.

    Parameters:
    - targets: Target dataframe with 'time' and target values.
    - observed: Dataframe with observed features.
    - estimated: Dataframe with estimated features.
    - test: Dataframe with test features.

    Returns:
    - Preprocessed dataframes ready for training and testing.
    """

    # Ensure the datetime columns are in datetime format
    targets["time"] = pd.to_datetime(targets["time"])
    observed["date_forecast"] = pd.to_datetime(observed["date_forecast"])
    estimated["date_forecast"] = pd.to_datetime(estimated["date_forecast"])
    test["date_forecast"] = pd.to_datetime(test["date_forecast"])

    # Ensure data is sorted by date_forecast
    targets = targets.sort_values(by="time")
    observed = observed.sort_values(by="date_forecast")
    estimated = estimated.sort_values(by="date_forecast")
    test = test.sort_values(by="date_forecast")

    # Identify boolean columns
    boolean_features = [
        col for col in observed.columns if observed[col].dropna().isin([0.0, 1.0]).all()
    ]

    # Forward fill NaNs for boolean columns
    for df in [observed, estimated, test]:
        df[boolean_features] = df[boolean_features].fillna(method="ffill")

    # Forward fill for time-series data (for non-boolean columns)
    for df in [observed, estimated, test]:
        df[df.columns.difference(boolean_features)] = df[
            df.columns.difference(boolean_features)
        ].fillna(method="ffill")

    """  
    # Forward fill for time-series data
    observed.fillna(method='ffill', inplace=True)
    estimated.fillna(method='ffill', inplace=True)
    test.fillna(method='ffill', inplace=True)

    # Fill NaNs in boolean features with 0
    boolean_features = [col for col in observed.columns if observed[col].dropna().isin([0.0, 1.0]).all()]
    observed[boolean_features] = observed[boolean_features].fillna(method='ffill')
    estimated[boolean_features] = estimated[boolean_features].fillna(method='ffill')
    test[boolean_features] = test[boolean_features].fillna(method='ffill') 
    """

    # Resample observed, estimated, and test data to 1 hour using mean() as aggregator
    # and drop rows where all columns are NaN
    observed_resampled = (
        observed.set_index("date_forecast")
        .resample("1H")
        .mean()
        .dropna(how="all")
        .reset_index()
    )
    estimated_resampled = (
        estimated.set_index("date_forecast")
        .resample("1H")
        .mean()
        .dropna(how="all")
        .reset_index()
    )
    test_resampled = (
        test.set_index("date_forecast")
        .resample("1H")
        .mean()
        .dropna(how="all")
        .reset_index()
    )

    # Round boolean columns after resampling
    for df in [observed_resampled, estimated_resampled, test_resampled]:
        df[boolean_features] = df[boolean_features].round(0)

    observed_resampled["estimated"] = 0
    estimated_resampled["estimated"] = 1
    test_resampled["estimated"] = 1

    # Merge the observed and estimated data
    weather_data = pd.concat([observed_resampled, estimated_resampled])

    # Merge with target values
    merged_data = pd.merge(
        targets, weather_data, how="inner", left_on="time", right_on="date_forecast"
    )

    # Time-Based Features (training data)
    merged_data["hour"] = merged_data["date_forecast"].dt.hour
    merged_data["sin_hour"] = np.sin(2 * np.pi * merged_data["hour"] / 24)
    merged_data["cos_hour"] = np.cos(2 * np.pi * merged_data["hour"] / 24)
    # merged_data['day_of_week'] = merged_data['date_forecast'].dt.dayofweek
    merged_data["month"] = merged_data["date_forecast"].dt.month
    merged_data["sin_month"] = np.sin(2 * np.pi * merged_data["month"] / 12)
    merged_data["cos_month"] = np.cos(2 * np.pi * merged_data["month"] / 12)

    # Time-Based Features (test data)
    test_resampled["hour"] = test_resampled["date_forecast"].dt.hour
    test_resampled["sin_hour"] = np.sin(2 * np.pi * test_resampled["hour"] / 24)
    test_resampled["cos_hour"] = np.cos(2 * np.pi * test_resampled["hour"] / 24)
    # test_resampled['day_of_week'] = test_resampled['date_forecast'].dt.dayofweek
    test_resampled["month"] = test_resampled["date_forecast"].dt.month
    test_resampled["sin_month"] = np.sin(2 * np.pi * test_resampled["month"] / 12)
    test_resampled["cos_month"] = np.cos(2 * np.pi * test_resampled["month"] / 12)

    # Drop non-feature columns
    merged_data = merged_data.drop(
        columns=["time", "date_forecast", "pv_measurement", "snow_density:kgm3"]
    )
    test_resampled = test_resampled.drop(columns=["date_forecast", "snow_density:kgm3"])

    # fixing ceiling_height NaN value
    merged_data["ceiling_height_agl:m"].fillna(0, inplace=True)
    test_resampled["ceiling_height_agl:m"].fillna(0, inplace=True)

    return merged_data, test_resampled


locations = ["A", "B", "C"]
location_mapping = {"A": 1, "B": 2, "C": 3}
all_predictions = []
all_predictions_rf = []


for loc in locations:
    # Load your data
    train = pd.read_parquet(f"data/{loc}/train_targets.parquet").fillna(0)
    X_train_estimated = pd.read_parquet(f"data/{loc}/X_train_estimated.parquet")
    X_train_observed = pd.read_parquet(f"data/{loc}/X_train_observed.parquet")
    X_test_estimated = pd.read_parquet(f"data/{loc}/X_test_estimated.parquet")

    # save as csv for analysis
    """ train.to_csv(f'{loc}_csv/train_targets.csv')
    X_train_estimated.to_csv(f'{loc}_csv/X_train_estimated.csv')
    X_train_observed.to_csv(f'{loc}_csv/X_train_observed.csv')
    X_test_estimated.to_csv(f'{loc}_csv/X_test_estimated.csv') """

    # Preprocess data
    X_train, X_test = preprocess_data(
        train, X_train_observed, X_train_estimated, X_test_estimated
    )

    X_train["location"] = location_mapping[loc]
    X_test["location"] = location_mapping[loc]

    X_train.to_csv("X_train.csv")
    X_test.to_csv("X_test.csv")

    y = train["pv_measurement"].values

    # Ensure X and y have the same length
    min_length = min(len(X_train), len(y))
    X_train, y_train = X_train.iloc[:min_length], y[:min_length]

    X_train_data, X_eval_data, y_train_data, y_eval_data = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Create catboost Pool objects
    train_pool = Pool(
        data=X_train_data, label=y_train_data, cat_features=["location", "estimated"]
    )
    eval_pool = Pool(
        data=X_eval_data, label=y_eval_data, cat_features=["location", "estimated"]
    )

    # Initialize and Train model
    # categorical_features = ['dew_or_rime:idx', 'elevation:m', 'is_day:idx', 'is_in_shadow:idx', 'snow_drift:idx', 'wind_speed_w_1000hPa:ms']

    model = CatBoostRegressor(depth=9, iterations=1000, loss_function="MAE")
    model.fit(train_pool, use_best_model=True, eval_set=eval_pool)

    # Make predictions using X_test_estimated data
    predictions = model.predict(X_test)

    # Store the predictions in all_predictions list
    all_predictions.append(predictions)

    """ 
    # Initialize and Train RandomForest model
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)

    # Make predictions using X_test data
    predictions_rf = model_rf.predict(X_test)
    
    # Store the RandomForest predictions in all_predictions_rf list
    all_predictions_rf.append(predictions_rf) 
    
    final_predictions_rf = np.concatenate(all_predictions_rf)
    
    average_predictions = (np.array(final_predictions) + np.array(final_predictions_rf)) / 2.0
    """

# Concatenate all predictions
final_predictions = np.concatenate(all_predictions)


# Save the final_predictions to CSV
df = pd.DataFrame(final_predictions, columns=["prediction"])
df["id"] = df.index
df = df[["id", "prediction"]]
df["prediction"] = df["prediction"].apply(lambda x: max(0, x))
df.to_csv("final_predictions.csv", index=False)
