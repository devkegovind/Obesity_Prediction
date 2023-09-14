from flask import Flask, request, render_template, jsonify
from src.pipelines.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

@app.route('/')
def home_page():
    return render_template('index.html')




@app.route('/predict', methods = ['GET', 'POST'])

def predict_datapoint():
    if request.method == "GET":
        return render_template('form.html')
    
    else:
        data = CustomData(
            Gender = request.form.get('Gender'),
            Age = float(request.form.get('Age')),
            Height = float(request.form.get('Height')),
            Weight = float(request.form.get('Weight')),
            family_history_with_overweight = request.form.get('family_history_with_overweight'),
            FAVC = request.form.get('FAVC'),
            FCVC = float(request.form.get('FCVC')),
            NCP = float(request.form.get('NCP')),
            CAEC = request.form.get('CAEC'),
            SMOKE = request.form.get('SMOKE'),
            CH2O = float(request.form.get('CH2O')),
            SCC = request.form.get('SCC'),
            FAF = float(request.form.get('FAF')),
            TUE = float(request.form.get('TUE')),
            CALC = request.form.get('CALC'),
            MTRANS = request.form.get('MTRANS')
        )

        final_new_data = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_new_data)

        results = (pred[0])

        return render_template('results.html', final_result = results)
    

if __name__ == "__main__":
    app.run(host = '0.0.0.0', debug = True)