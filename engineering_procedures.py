from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)


@app.route('/coordinates', methods=['POST', 'GET'])
def pyproj_test():

    import pyproj as pp

    x1=[]
    y1=[]
    lnglat=[]
    
    easting = request.values['easting']
    northing = request.values['northing']
    epsg = request.values['epsg']
    easting = request.values['easting'].split(',')
    northing = request.values['northing'].split(',')
    print('epsg: ',epsg)
    print('easting: ',easting)
    print('northing: ',northing)
    
    inProj = pp.Proj(init='epsg:'+str(epsg))
    outProj = pp.Proj(init='epsg:4326')  #WGS84

    try:
        for i in range(len(easting)):  #need to test len of easting == len of northing
            x1.append(float(easting[i]))
            y1.append(float(northing[i]))
            lnglat.append(pp.transform(inProj, outProj, float(easting[i]), float(northing[i])))

        print('lnglat: ', lnglat)
        return jsonify({'lng': [item[0] for item in lnglat], 'lat': [item[1] for item in lnglat]})

    except:
        pass
        
    return jsonify({'entry': 'invalid'})



@app.route('/test', methods=['POST', 'GET'])
def test():

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
