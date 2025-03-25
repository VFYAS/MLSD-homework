import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import numpy as np
import catboost
import os

app = dash.Dash(__name__, title="SMS Spam Classifier Demo (CatBoost)")

MODEL_PATH = 'sms_spam_catboost_model.cbm'
LABELS = ['Ham', 'Spam']

@app.callback(
    Output('loading-model', 'children'),
    Input('interval-component', 'n_intervals')
)
def load_model(n):
    try:
        global MODEL
        
        if not os.path.exists(MODEL_PATH):
            return ""
        
        MODEL = catboost.CatBoostClassifier()
        MODEL.load_model(fname=MODEL_PATH)
        
        return ""
    except Exception as e:
        return f"Error loading model: {str(e)}"

def predict_spam(message):
    probs = MODEL.predict_proba([[message]])[0]
    
    predicted_class = np.argmax(probs)
    return {
        'class': LABELS[predicted_class],
        'confidence': float(probs[predicted_class]),
        'is_spam': predicted_class == 1
    }

app.layout = html.Div([
    html.H1("SMS Spam Classifier (CatBoost)", style={'textAlign': 'center'}),
    
    html.Div([
        html.Div(id='loading-model'),
        dcc.Interval(id='interval-component', interval=1000, max_intervals=1)
    ]),
    
    html.Div([
        dcc.Textarea(
            id='sms-input',
            placeholder='Enter the SMS message to classify...',
            style={'width': '100%', 'height': 100, 'marginBottom': 20}
        ),
        
        html.Button('Classify', id='classify-button', n_clicks=0, 
                   style={'backgroundColor': '#4CAF50', 'color': 'white', 'border': 'none', 
                          'padding': '10px 20px', 'textAlign': 'center', 'fontSize': 16}),
        
        html.Div(id='classification-output', style={'marginTop': 20, 'fontSize': 18})
    ], style={'maxWidth': '800px', 'margin': 'auto', 'padding': 20, 'marginTop': 20, 'boxShadow': '0px 0px 10px rgba(0,0,0,0.1)'}),
    
    html.Div([
        html.H3("Example messages to try:"),
        html.Ul([
            html.Li("WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! Call this number immediately: +1234567890", style={'color': 'red'}),
            html.Li("Meeting rescheduled to 3 PM tomorrow. Please confirm attendance.", style={'color': 'green'}),
            html.Li("URGENT We are trying to contact you Last weekends draw shows u have won a £1000", style={'color': 'red'}),
            html.Li("Hi mom, could you pick me up after practice at 5?", style={'color': 'green'})
        ])
    ], style={'maxWidth': '800px', 'margin': 'auto', 'padding': 20, 'marginTop': 20, 'boxShadow': '0px 0px 10px rgba(0,0,0,0.1)'})
])

@app.callback(
    Output('classification-output', 'children'),
    Output('classification-output', 'style'),
    Input('classify-button', 'n_clicks'),
    State('sms-input', 'value'),
    prevent_initial_call=True
)
def update_output(n_clicks, message):
    if not message:
        return "Please enter a message to classify", {'color': 'orange'}
    
    try:
        result = predict_spam(message)
        
        style = {'padding': '15px', 'borderRadius': '5px', 'marginTop': '20px'}
        
        if result['is_spam']:
            style['backgroundColor'] = 'rgba(255, 0, 0, 0.1)'
            style['border'] = '1px solid red'
            style['color'] = 'red'
        else:
            style['backgroundColor'] = 'rgba(0, 255, 0, 0.1)'
            style['border'] = '1px solid green'
            style['color'] = 'green'
            
        return [
            html.Div([
                html.H4(f"Classification: {result['class']}", style={'marginBottom': '10px'}),
                html.P(f"Confidence: {result['confidence']:.2%}")
            ])
        ], style
    
    except Exception as e:
        return f"Error in classification: {str(e)}", {'color': 'red'}

if __name__ == '__main__':
    app.run(debug=True)
