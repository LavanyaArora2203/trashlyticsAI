import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import joblib
import warnings
warnings.filterwarnings("ignore")


def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])

    # Aggregate bin-level to zone-level
    zone_daily = (
        df.groupby(['zone', 'date'])
          .agg(
              avg_fill=('fill_percentage', 'mean'),
              max_fill=('fill_percentage', 'max'),
              bins_over_80=('fill_percentage', lambda x: (x > 80).sum())
          )
          .reset_index()
    )

    zone_daily = zone_daily.sort_values(['zone', 'date'])
    return zone_daily


def train_zone_models(zone_daily):

    zone_models = {}

    zones = zone_daily['zone'].unique()

    for zone in zones:
        zone_df = zone_daily[zone_daily['zone'] == zone]
        series = zone_df.set_index('date')['avg_fill']

        # Safe ARIMA configuration
        model = ARIMA(series, order=(2, 1, 2))
        model_fit = model.fit()

        zone_models[zone] = model_fit

    return zone_models


def save_model(models, filename="waste_forecast_model.pkl"):
    joblib.dump(models, filename)
    print(f"Model saved as {filename}")


if __name__ == "__main__":

    FILE_PATH = "data\\waste_fill.csv"

    print("Loading data...")
    zone_data = load_and_prepare_data(FILE_PATH)

    print("Training models...")
    models = train_zone_models(zone_data)

    print("Saving model...")
    save_model(models)

    print("Done!")
