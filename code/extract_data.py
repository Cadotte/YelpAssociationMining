import os
import pandas as pd

for filename in os.listdir(os.getcwd() + "/../data/json/"):

    # For each json file
    if "json" in filename:
        json_path = os.getcwd() + "/../data/json/" + filename
        print("json file: " + json_path)
        csv_path = json_path.replace("json", "csv")
        csv_path = csv_path.replace("yelp_academic_dataset_", "")
        print("csv file: " + csv_path)
        if not os.path.exists(csv_path):

            # Read data chunk by chunk
            data_reader = pd.read_json(os.getcwd() + "/../data/json/" + filename, lines=True, chunksize=10**4)
            for chunk in data_reader:
                # print(chunk)
                # Save to CSV
                chunk.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path))