import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler


#web server passes all the requests it receives from clients to app obj for handling
app = Flask(__name__,static_url_path='/static')
model = pickle.load(open('model.pkl', 'rb'))    #$$ 
scaler = pickle.load(open('scaler.pkl', 'rb'))

#for each url requested by the client, its associated code is executed.
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    #feature scaling of new input
    final_features = scaler.transform(final_features)  
    prediction = model.predict(final_features)
    y_probabilities_test = model.predict_proba(final_features) #$$
    y_prob_success = y_probabilities_test[:, 1]
    #return render_template('index.html', prediction_text = y_prob_success)
    #print("final features",final_features)
    #print("prediction:",prediction)
    output = round(prediction[0], 2)
    y_prob = round(y_prob_success[0], 3)
    #return render_template('index.html', prediction_text = str(prediction) + ' ' + str(y_probabilities_test) +  ' ' + str(output) + ' ' + str(y_prob))
    #print(output)

    if output == 0:
        return render_template('index.html', accuracy = "91.63%", prediction_text = 'More likely to be a BENIGN tumor (probability = {:.2f})'.format(y_prob_success[0])) #'THE PATIENT IS MORE LIKELY TO HAVE A BENIGN CANCER WITH PROBABILITY VALUE  {}'.format(y_prob))
    else:
        return render_template('index.html', accuracy = "91.63%", prediction_text = 'More likely to be a MALIGNANT tumor (probability = {:.2f})'.format(y_prob_success[0])) #'THE PATIENT IS MORE LIKELY TO HAVE A MALIGNANT CANCER WITH PROBABILITY VALUE  {}'.format(y_prob))
        
'''@app.route('/predict_api',methods=['POST'])
def predict_api():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)'''

if __name__ == "__main__":
    app.run(debug=True)
