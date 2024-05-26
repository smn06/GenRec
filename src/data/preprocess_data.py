import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(input_file, output_file):
    # Load the dataset
    data = pd.read_csv(input_file)

    # Handle missing values (if any)
    data.fillna(data.mean(), inplace=True)

    # Normalize numerical features
    scaler = StandardScaler()
    data[['rating']] = scaler.fit_transform(data[['rating']])

    # Convert timestamps to datetime (optional)
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')

    # Split the data into training and test sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Save the processed data
    train_data.to_csv('train_' + output_file, index=False)
    test_data.to_csv('test_' + output_file, index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess user data.')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file')
    args = parser.parse_args()
    
    preprocess_data(args.input, args.output)
