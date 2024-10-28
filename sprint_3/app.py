from flask import Flask, request, render_template
import pickle

import pandas

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/input')
def home():
    return render_template('input_form.html')

@app.route('/result', methods=['POST'])
def result():
    # Collect and validate the input
    #try:
        Civil_Twilight = (request.form.get('Civil_Twilight'))
        Side = request.form.get('Side')
        Temperature_f = float(request.form.get('Temperature_f'))
        Visibility_mi = float(request.form.get('Visibility_mi'))
        Weather_Condition = float(request.form.get('Weather_Condition'))
        Wind_Direction = float(request.form.get('Wind_Direction'))
        Distance_mi = float(request.form.get('Distance_mi'))
        Humidity_percent = float(request.form.get('Humidity_percent'))
        Precipitation_in = float(request.form.get('Precipitation_in'))
        Pressure_in = float(request.form.get('Pressure_in'))
        Wind_Speed_mph = float(request.form.get('Wind_Speed_mph'))


        input_features = [[Civil_Twilight, Side, Temperature_f, Visibility_mi, Weather_Condition, Wind_Direction, Distance_mi, Humidity_percent, Precipitation_in, Pressure_in, Wind_Speed_mph]]


        # Load the model and make a prediction
        with open('rf.pkl', 'rb') as model_file:
            rf = pickle.load(model_file)
        input_features=pandas.DataFrame(input_features, columns=['Civil_Twilight', 'Side', 'Temperature_f', 'Visibility_mi', 'Weather_Condition', 'Wind_Direction', 'Distance_mi', 'Humidity_percent', 'Precipitation_in', 'Pressure_in', 'Wind_Speed_mph'])
        #print(input_features)
        prediction = rf.predict(input_features)
        result = prediction[0]

        return render_template('result.html', res=result)
    #except Exception as e:
        # Handle exceptions (e.g., missing file, invalid input)
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
