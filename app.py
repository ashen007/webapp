import numpy as np
import pandas as pd
import pickle
import joblib
import flask

with open(f'model/sk_linear_model.pkl', 'rb') as file:
    model = pickle.load(file)

app = flask.Flask(__name__, template_folder='template')


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return flask.render_template('index.html')

    if flask.request.method == 'POST':
        inputs = pd.DataFrame(
            columns=['LotFrontage', 'MasVnrArea', 'TotalBsmtSF', 'firstFlrSF', 'GrLivArea', 'FullBath',
                     'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'MasVnrType', 'OverallQual',
                     'ExterQual', 'BsmtQual', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'YearBuilt',
                     'YearRemodAdd', 'GarageYrBlt', 'Age'])
        inputs = inputs.append(flask.request.form.to_dict(), ignore_index=True)
        inputs = inputs.astype(float)
        inputs = np.power(inputs, 1 / 5)
        result = np.power(model.predict(inputs), 5)

        return flask.render_template('index.html', result=round(result[0],2), )


if __name__ == '__main__':
    app.run()
