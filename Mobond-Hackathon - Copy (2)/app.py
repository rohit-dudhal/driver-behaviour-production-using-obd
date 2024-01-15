from flask import Flask, request, jsonify
import pandas as pd
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 
# Load the trained K-Means model

@app.route('/predict_cluster', methods=['POST'])
def predict_cluster():
    try:
        # Load input data from the request
        kmeans_model = joblib.load("C:/Users/Rohit Dudhal/Documents/Personal_Data/Mobond-Hackathon - Copy (2)/kmeans_model_13.joblib")
        
        config  = request.get_json()
        if type(config) == dict:
            df = pd.DataFrame(config, index=[0])
        else:
            df = config
        print(df)
        predicted_cluster = kmeans_model.predict(df)
        print(predicted_cluster)
        return jsonify({'predicted_cluster': int(predicted_cluster[0])})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)