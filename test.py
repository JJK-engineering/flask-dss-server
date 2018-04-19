from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)

@app.route('/test', methods=['POST', 'GET'])
def test():

## first test, ok
#    return 'Hello, World!'

# test returning data to polymer, ok     
    registration='nodefinition'
    #registration = request.get_data()  #testing
    #registration = request.values['registration'])  #testing
    registration = request.values.get('registration')
    print("registration:",registration)
    print("json:",request.get_json(force=True))  #None

    #if request.form.get('registration') == 'success':  #testing
    #if request.form['registration'] == 'success':  #testing 
    if registration == "success":
        return jsonify({'abc': 'successfully registered', 'registration': registration})

    return jsonify({'abc': 'registration unsuccessful', 'registration': request.values['registration']})
