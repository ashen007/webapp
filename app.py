import numpy as np
import pandas as pd
import pickle
import flask
import plotly.graph_objs as go
from dashboards import init_dashboard

with open(f'model/sk_linear_model.pkl', 'rb') as file:
    model = pickle.load(file)

app = flask.Flask(__name__, template_folder='template')
init_dashboard(app)

data = pd.read_pickle('data/cleaned.pkl')
data['Age'] = data['YrSold'] - data['YearBuilt']


# bubble plot
def create_bubble():
    i = 0
    colors = ['#3391A6', '#25594A', '#3FA663', '#2D7345', '#262626']
    hover_text = []
    bubble_size = []

    for index, row in data.iterrows():
        hover_text.append(('Neighborhood:{neighborhood}<br>' +
                           'Age:{age}<br>' +
                           'Price:{price}<br>' +
                           'Year Sold:{yrsold}<br>' +
                           'Area:{area}<br>').format(neighborhood=row['Neighborhood'],
                                                     age=row['Age'],
                                                     price=row['SalePrice'],
                                                     yrsold=row['YrSold'],
                                                     area=row['LotArea']))
        bubble_size.append(np.power(row['LotArea'], 1))

    data['text'] = hover_text
    data['size'] = bubble_size
    sizeref = 2 * max(data['size']) / 800

    groups = list(data['YrSold'].unique())
    group_data = {group: data[data['YrSold'] == group] for group in groups}

    fig = go.Figure()

    for group_name, group in group_data.items():
        fig.add_trace(go.Scatter(x=group['SalePrice'],
                                 y=group['Age'],
                                 name=str(group_name),
                                 text=group['text'],
                                 marker_size=group['size'],
                                 marker_color=colors[i],
                                 ))
        i += 1

    fig.update_traces(mode='markers',
                      marker=dict(sizemode='area',
                                  sizeref=sizeref,
                                  line_width=1))

    fig.update_layout(
        height=750,
        width=1000,
        title=dict(text='<b>How sale price changes with age,total area<b>',
                   font=dict(size=24),
                   yanchor='bottom',
                   xanchor='center'),
        xaxis=dict(title=dict(text='sale price',
                              font=dict(size=12)),
                   gridcolor='LightGray',
                   type='log',
                   gridwidth=1),
        yaxis=dict(title=dict(text='property age (years)',
                              font=dict(size=12)),
                   gridcolor='LightGray',
                   gridwidth=1),
        paper_bgcolor='white',
        plot_bgcolor='white',
    )

    fig.layout.template = None

    return fig.to_json()


fig_json = create_bubble()


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return flask.render_template('index.html', bubble_plot=fig_json)

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

        if result[0] < 0:
            result = 0
        else:
            result = round(result[0], 2)

        return flask.render_template('index.html', result=result, bubble_plot=fig_json)


if __name__ == '__main__':
    app.run()
