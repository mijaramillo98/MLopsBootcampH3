import pickle
import pandas as pd
import numpy as np
import sys

def ride_duration_prediction(year, month):
        
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)
        
    taxi_type="yellow"

    categorical = ['PULocationID', 'DOLocationID']

    def read_data(filename):
        df = pd.read_parquet(filename)
        
        df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
        df['duration'] = df.duration.dt.total_seconds() / 60

        df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

        df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
        
        return df

    df = read_data(f'./yellow_tripdata_{year:04d}-{month:02d}.parquet')
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predictions'] = y_pred
    output_file = f'./homework.parquet'
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )
    print(np.mean(y_pred))

def run():
    year = int(sys.argv[2]) # 2022
    month = int(sys.argv[1]) # 2

    ride_duration_prediction(
        year=year,
        month=month,
    )


if __name__ == '__main__':
    run()

