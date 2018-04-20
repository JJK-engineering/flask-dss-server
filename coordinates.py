from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)

@app.route('/coordinates', methods=['POST', 'GET'])
def pyproj_test():

# test using pyproj on flask from polymer, ok
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
    print('epsg: ',type(epsg))
    print('easting: ',type(easting))
    print('northing: ',type(northing))    
    
    #inProj = pp.Proj(init='epsg:2056')  #Swiss CH1903+ / LV95
    inProj = pp.Proj(init='epsg:'+str(epsg))
    outProj = pp.Proj(init='epsg:4326')  #WGS84

#    try:
#        x1, y1 = float(easting), float(northing)
#        print("easting,northing",x1,y1)
#        #print("epsg",inProj)
#        print("json:",request.get_json(force=True))  #testing json input 
#        print("is-json:", request.is_json)    
#        lng, lat = pp.transform(inProj,outProj,x1,y1)
#        print('lng,lat: ',lng,lat)
#
#        return jsonify({'lng': lng, 'lat': lat})

    try:
        for i in range(len(easting)):  #need to test len of easting == len of northing
            print(i)
            print('easting[i]: ',easting[i])
            print('easting[i]: ',float(easting[i]))
            #x1[i], y1[i] = float(easting[i]), float(northing[i])
            print("here we are")
            x1.append(float(easting[i]))
            y1.append(float(northing[i]))
            print("here we are again")
            print("easting,northing",x1,y1)
            print("json:",request.get_json(force=True))  #testing json input 
            print("is-json:", request.is_json)    
            #lng[item],lat[item] = pp.transform(inProj,outProj,x1,y1)
            print(pp.transform(inProj, outProj, float(easting[i]), float(northing[i])))
            lnglat.append(pp.transform(inProj, outProj, float(easting[i]), float(northing[i])))
            #print('lng,lat: ',lng,lat)
            print('lnglat: ', lnglat)

        print("end of try")    
        print(lnglat[0][1])
        print([item[0] for item in lnglat]) 
        #return jsonify({'lng': (lnglat[0][0],lnglat[1][0]), 'lat': (lnglat[0][1], lnglat[1][1])})
        return jsonify({'lng': [item[0] for item in lnglat], 'lat': [item[1] for item in lnglat]})

    except:
        pass
        
    return jsonify({'entry': 'invalid'})

