from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

multiple_reg=pickle.load(open('multiple_reg.sav','rb'))


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction=multiple_reg.predict(final)
    output='{:.2f}'.format(prediction[0][0], 2)
   
    if output>str(0):
        return render_template('index.html',pred='the estimated Price of house is Rs {}'.format(output),bhai="kuch karna hain iska ab?")
    else:
        return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)
