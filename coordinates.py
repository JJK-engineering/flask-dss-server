from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)

@app.route('/coordinates', methods=['POST', 'GET'])
def pyproj_test():

# test using pyproj on flask from polymer, ok
    import pyproj as pp

    easting = request.values['easting']
    northing = request.values['northing']
    epsg = request.values['epsg']
    #easting = request.values['easting'].split(',')
    #northing = request.values['northing'].split(',')
    print('epsg: ',epsg)
    print('easting: ',easting)
    print('northing: ',northing)
    print('epsg: ',type(epsg))
    print('easting: ',type(easting))
    print('northing: ',type(northing))    
    
    #inProj = pp.Proj(init='epsg:2056')  #Swiss CH1903+ / LV95
    inProj = pp.Proj(init='epsg:'+str(epsg))
    outProj = pp.Proj(init='epsg:4326')  #WGS84

    try:
        x1, y1 = float(easting), float(northing)
        print("easting,northing",x1,y1)
        #print("epsg",inProj)
        print("json:",request.get_json(force=True))  #testing json input 
        print("is-json:", request.is_json)    
        lng, lat = pp.transform(inProj,outProj,x1,y1)
        print('lng,lat: ',lng,lat)

        return jsonify({'lng': lng, 'lat': lat})

#    try:
#        for i in range(len(easting)):  #need to test len of easting == len of northing
#            print(i)
#            print('easting[i]: ',easting[i])
#            #x1[item], y1[item] = float(easting[item]), float(northing[item])
#            x1.append(float(easting[i]))
#            y1.append(float(northing[i]))
#            print("easting,northing",x1,y1)
#            print("json:",request.get_json(force=True))  #testing json input 
#            print("is-json:", request.is_json)    
#            #lng[item],lat[item] = pp.transform(inProj,outProj,x1,y1)
#            print(pp.transform(inProj, outProj, float(easting[i]), float(northing[i])))
#            lnglat.append(pp.transform(inProj, outProj, float(easting[i]), float(northing[i])))
#            #print('lng,lat: ',lng,lat)
#            print('lnglat: ', lnglat)#
#
#        #return jsonify({'lng': lng, 'lat': lat})
#        return lnglat

    except:
        pass
        
    return jsonify({'entry': 'invalid'})

