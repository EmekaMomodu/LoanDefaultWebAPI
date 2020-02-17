# Dependencies
from flask import Flask, request, jsonify
from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import sys

# API definition
app = Flask(__name__)


@app.route('/prediction', methods=['POST'])
def predict():
    if modelz:
        try:
            json_ = request.json
            print(json_)
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)

            prediction = list(modelz.predict(query))

            return jsonify({'prediction': str(prediction)})

        except:

            return jsonify({'trace': traceback.format_exc})

    else:

        print('train the model first')
        return 'no model here to use'


if __name__ == '__main__':
    try:
        port = int(sys.argv[1])  # This is for a command line input
    except:
        port = 12346  # if you don't provide any port the port will be set to 1

    modelz = joblib.load('model.pkl')  # Load model.pkl
    print('Model columns loaded')

    app.run(port=port, debug=True)
