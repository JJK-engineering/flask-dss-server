
# set up environ
import sys, os
os.environ["DISPLAY"] = ":1.0"
# set wd for this procedure and project 
#   ToDo: read in a key to set working dir (with root to be set from sys config)
os.chdir("/home/kaelin_joseph/DSS.HydraulicConfinement/")

# import required python libraires
import numpy as np
from numpy import *
import pandas as pd
import geopandas as gpd
import shapely as sp

# python setup for qgis processing
from qgis.core import QgsApplication
from PyQt4.QtGui import QApplication
sys.path.append('/usr/share/qgis/python/plugins')  #export PYTHONPATH not needed in start script
from processing.core.Processing import Processing
import processing

# set up plotly in 'offline' mode
import plotly.graph_objs as go
import plotly.offline as plotly
from plotly.graph_objs import *
import json
import plotly  #needed for plotly.utils.PlotlyJSONEncoder

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



@app.route('/hydraulic_confinement', methods=['POST', 'GET'])

# procedure for calculating hydraulic confinement along a pressure tunnel 
#   safety factor against hydraulic confinement calculated at stationed points along tunnel alignment
#   calculation determines the minimum dstance from the stationed point to the terrain surface 

def hydraulic_confinement():

    # set up qgis processing
    app = QApplication([], True)  #True -> window display enabled
    QgsApplication.setPrefixPath("/usr", True)
    QgsApplication.initQgis()
    Processing.initialize() 

    
    print('stating hydraulic_confinement()')

    # code below is from Jupyter
    ############################

    
    # define required input files
    DTM = "data/in/NamAngTopogRaster.tif"  #DEM with surface topography
    Alignment = "data/in/NamAngAlignmentHRT.r1.csv"  #tunnel alignment
    DTM_slope='data/in/NamAngTopogSlopeRaster.tif'  #DEM containing slope angle as attribute


    # In[8]:


    # define required input data
    # mapping
    crs = {'init': 'epsg:32648'}  #define crs for project
    grass_region = "726000,729500,1670500,1674000"  #map region E1,E2,N1,N2
    # resolution for analysis
    grass_station_dist = 50  #use for low resolution
    c = 0.5  #ring buffer radius = c*h  (h=overburden)
    res = 25.0  #use for low resolution
    # rock properties
    #  ToDo: mv to calling parameter (how to handle multiple rock layers?)
    density_rock = 28.0  #kN/m3
    # hydraulic properties
    #  ToDo: mv to calling parameter
    max_static_water_level = 637.20 #MASL 


    # In[9]:


    # define temporary data
    Alignment_shp ='tmp/NamAngAlignment.shp'  #alignment shp from Alignment
    Alignment_grass_csv = 'tmp/NamAngAlignmentGrass.csv'  #alignment csv fixed for grass
    Alignment_line_shp = "tmp/NamAngAlignmentLine.shp"  #intermediate data
    Alignment_stationed_shp = "tmp/NamAngAlignmentStationed.shp"  #alignment shp containing station points
    Alignment_dtm_csv = "tmp/NamAngAlignmentDTM.csv"  #alignment including terrain elevations at station points
    Buffer_shp = "tmp/NamAngBuffer.shp"  #buffer shp containing ring grid points at a particular station point
    Buffer_all_csv = "tmp/NamAngBufferAll.csv"  # all station point ring buffers written to csv
    Buffer_all_shp = "tmp/NamAngBufferFinal.shp"
    Buffer_dtm_csv = "tmp/NamAngBufferDTM.csv"
    Buffer_slope_csv = "tmp/NamAngBufferSlope.csv"


    # In[10]:


    alignment = {
        'Station':["0+000","0+163.81","2+063.77","2+603.58","3+030.73"],
        'Northing':[1673035.25,1673051.68,1672130.47,1671662.07,1671268.20],
        'Easting':[726651.46,726815.60,728479.47,728758.97,728581.38],
        'Elevation':[625.850,625.543,504.226,469.758,464.220]
    }


    # In[11]:


    alignment_df = pd.DataFrame(alignment)
    alignment_df = alignment_df[['Station','Northing','Easting','Elevation']]  #for checking
    ##alignment_df  #checking


    # In[12]:


    # create stationed alignment as Alignment_stationed_shp                 #ToDo JK: make into stationing function
    alignment_df_grass = alignment_df.loc[:,["Easting", "Northing"]]  #x first and y second
    alignment_df_grass.to_csv(Alignment_grass_csv, header=False, index=False)  #no header
    # points to line, write output to Alignment_line_shp
    print('starting grass7:v.in.lines')
    processing.runalg("grass7:v.in.lines",Alignment_grass_csv,",",False,
                      grass_region,0,Alignment_line_shp)  #no spaces between commas
    print('completed grass7:v.in.lines')    
    # line to station points, ouput segmented polyline to Alignment_stationed_shp
    processing.runalg("grass7:v.to.points",Alignment_line_shp,grass_station_dist,1,True,
                      grass_region,-1,0.0001,0,Alignment_stationed_shp)  #no spaces between commas


    # In[13]:


    # create alignment_stationed_df from Alignment_stationed_shp            #ToDo JK: make into shp -> df function
    alignment_stationed_df = gpd.read_file(Alignment_stationed_shp)
    # create columns for x_align, y_align, then delete columns cat_ and geometry
    alignment_stationed_df["x_align"] = alignment_stationed_df.geometry.x
    alignment_stationed_df["y_align"] = alignment_stationed_df.geometry.y
    alignment_stationed_df = alignment_stationed_df.drop(columns =["cat_", "geometry"])


    # In[14]:


    ##alignment_stationed_df  #checking


    # In[15]:


    # add required fields to alignment_stationed_df
    # add "id" 
    alignment_stationed_df["id_point"] = alignment_stationed_df.index
    # add "distance_stat" referencing length between point n and point n+1
    alignment_stationed_df["distance_stat"] = np.nan
    for n in range(0, len(alignment_stationed_df)-1):
        alignment_stationed_df.iloc[n, alignment_stationed_df.columns.get_loc("distance_stat")] = (
            ((alignment_stationed_df.iloc[n +1]["x_align"] - alignment_stationed_df.iloc[n]["x_align"])**2
            +(alignment_stationed_df.iloc[n +1]["y_align"] - alignment_stationed_df.iloc[n]["y_align"])**2 )**(0.5) )
    # add "distance_stat_sum"
    alignment_stationed_df["distance_stat_sum"] = np.nan
    for n in range(0, len(alignment_stationed_df) -1):
        distance = ( alignment_stationed_df.loc[(alignment_stationed_df.id_point.isin(range(0,n +1))), 
                            "distance_stat"] )
        distances = distance.tolist()
        alignment_stationed_df.iloc[n, alignment_stationed_df.columns.get_loc("distance_stat_sum")] = (
                                                                                            sum(distances) )    
    ##alignment_stationed_df  #checking


    # In[16]:


    # add required field "distance_intermed_align" to alignment_df
    alignment_df["distance_intermed_align"] = np.nan
    for n in range(0, len(alignment_df) -1):
        alignment_df.iloc[n, alignment_df.columns.get_loc("distance_intermed_align")] = (
            ((alignment_df.iloc[n +1]["Easting"]-alignment_df.iloc[n]["Easting"])**2 
                 +(alignment_df.iloc[n +1]["Northing"]-alignment_df.iloc[n]["Northing"])**2 )**(0.5) )
    ##alignment_df  #checking


    # In[17]:


    # join alignment_df to alignment_stationed_df
    alignment_stationed_df = pd.merge(left= alignment_stationed_df, right = alignment_df, 
                     left_on = ["x_align","y_align"], 
                     right_on = ["Easting","Northing"], how = "left")
    # clean up alignment_stationed_df
    try:
        alignment_stationed_df = (
            alignment_stationed_df.drop(columns =["Point", "Type", "Northing", "Easting"]) )
    except:
        pass
    ##alignment_stationed_df  #checking


    # In[18]:


    # get id_points for alignment points in alignment_stationed_df 
    #   select points where Elevation of point not NaN
    id_points_align =  (
        alignment_stationed_df.loc[(alignment_stationed_df.Elevation.isin(alignment_df["Elevation"])), "id_point"] )
    id_points_align= id_points_align.tolist()


    # In[19]:


    # prepare intermediated data in alignment_stationed_df required to interpolate alignment elevations
    # fill in "Elevation" and "distance_intermed_align" for points in alignment_stationed_df 
    #   where points of alignment_points_df != alignment_df
    # why is this needed??                                                                                ToDo: JK
    for n in range(0, len(id_points_align) -1): 
        alignment_stationed_df.loc[(alignment_stationed_df.id_point.isin(range(id_points_align[n] +1, 
                                    id_points_align[n +1]))), "Elevation"] = ( 
                                                                        alignment_df["Elevation"][n] )
    for n in range(0, len(id_points_align) -1): 
        alignment_stationed_df.loc[(alignment_stationed_df.id_point.isin(range(id_points_align[n] +1, 
                                    id_points_align[n +1]))), "distance_intermed_align"] = ( 
                                                            alignment_df["distance_intermed_align"][n] )
    # add "distance_intermed_stat" to alignment_stationed_df 
    alignment_stationed_df["distance_intermed_stat"] = np.nan
    for n in range(0, 1):
        alignment_stationed_df.loc[(alignment_stationed_df.id_point.isin(range(id_points_align[n], 
                                    id_points_align[n +1]))), "distance_intermed_stat"] =  ( 
                                                        alignment_stationed_df["distance_stat_sum"] )
    for n in range(1, len(id_points_align) -1):
          alignment_stationed_df.loc[(alignment_stationed_df.id_point.isin(range(id_points_align[n], 
                                      id_points_align[n +1]))), "distance_intermed_stat"] = ( 
                                                 alignment_stationed_df["distance_stat_sum"] - 
                                                 alignment_stationed_df["distance_stat_sum"][id_points_align[n] -1] )
    ##alignment_stationed_df  #checking


    # In[20]:


    # interpolate alignment elevation ("z_align") at all station points and write to alignment_stationed_df
    # add variable "z_align" to alignment_stationed_df
    alignment_stationed_df["z_align"] = np.nan
    for i in range(0, len(alignment_stationed_df)):
        # alignment points
        if i in id_points_align:
            alignment_stationed_df.iloc[i, alignment_stationed_df.columns.get_loc("z_align")] = ( 
                                                            alignment_stationed_df.iloc[i]["Elevation"] )
        # stationed points
        else:
            id_points_align_plus_point_n = id_points_align
            id_points_align_plus_point_n.append(i)
            id_points_align_plus_point_n.sort()
            m = id_points_align_plus_point_n.index(i) +1  #index of point n +1 (next alignment point)
            n = id_points_align_plus_point_n[m]  #id_point of next alignment point  
            o = id_points_align_plus_point_n.index(i) -1  #index of point n -1 (previous alignment point)
            p = id_points_align_plus_point_n[o]  #id_point of previous alignment point

            alignment_stationed_df.iloc[i, alignment_stationed_df.columns.get_loc("z_align")] = ( 
                                                        alignment_stationed_df.iloc[p]["Elevation"] 
                                                        +(alignment_stationed_df.iloc[n]["Elevation"] 
                                                        -alignment_stationed_df.iloc[p]["Elevation"]) 
                                                            /alignment_stationed_df.iloc[i]["distance_intermed_align"]
                                                            *alignment_stationed_df.iloc[i-1]["distance_intermed_stat"] )
            id_points_align_plus_point_n.remove(i)  #needed ??                                           #ToDo JK
    alignment_stationed_df = alignment_stationed_df.drop(columns = ["distance_intermed_align"])
    alignment_stationed_df = alignment_stationed_df.drop(columns = ["distance_intermed_stat"])
    alignment_stationed_df = alignment_stationed_df.drop(columns = ["Elevation"])
    ##alignment_stationed_df  #checking


    # In[21]:


    # add required field "z_dtm_align" to alignment_stationed_df 
    # list of shapely geometry points                                             #ToDo JK: make df -> shp function           
    alignment_stationed_geometry = ( 
        [sp.geometry.Point(row['x_align'], row['y_align']) for key, row in alignment_stationed_df.iterrows()] )
    # create alignment_stationed_geometry_df
    alignment_stationed_geometry_df = ( 
        gpd.GeoDataFrame(alignment_stationed_df, geometry=alignment_stationed_geometry, crs = crs) )
    # write df to Alignment_stationed_shp (overwrite file)
    alignment_stationed_geometry_df.to_file(Alignment_stationed_shp, driver='ESRI Shapefile') 
    # get DTM values for alignment_points                                     #ToDo JK: make into what.points function
    #   write to Alignment_dtm_csv
    processing.runalg("grass7:r.what.points",DTM,Alignment_stationed_shp, "NA",",", 500,
                      True,False,False,False,False,grass_region,-1,0.0001,Alignment_dtm_csv)
    # create alignment_dtm_df (dataframe) from Alignment_dtm_csv 
    alignment_dtm_df = pd.read_csv(Alignment_dtm_csv)
    # rename col=tmp... to "z_dtm_align"
    alignment_dtm_df_col_tmp = [col for col in alignment_dtm_df.columns if 'tmp' in col]
    if len(alignment_dtm_df_col_tmp) != 1:
        print "Extraction of DTM col=tmp did not work properly for alignment. Please check"
        exit()
    alignment_dtm_df = alignment_dtm_df.rename(
        columns= {alignment_dtm_df_col_tmp[0]: "z_dtm_align"})
    # write alignment_dtm_df["z_dtm_align"] to alignment_stationed_df["z_dtm_align"]
    alignment_stationed_df["z_dtm_align"] = alignment_dtm_df["z_dtm_align"]
    alignment_stationed_df = alignment_stationed_df.drop(columns = ["geometry"])
    ##alignment_stationed_df #checking


    # In[22]:


    # Add require field "h" to alignment_stationed_df = overburden depth above station point 
    alignment_stationed_df["h"] = alignment_stationed_df["z_dtm_align"] - alignment_stationed_df["z_align"] 
    ##alignment_stationed_df  #checking


    # In[23]:


    # define make_buffer to get buffer grid points at all station points along alignment
    def make_buffer(point, overburden, c, res):
        h = overburden
        if h < 0.0:
            print "Overburden is negative. Please check"
            exit()
        intvls_r = max(int(h*c / res), 1)  #number of intervals along the buffer radius, close enough
        res_r = h*c / intvls_r  #effective resolution along the radius
        buffer = np.array(point)  #initialize buffer, first item is exactly at station point
        # calculate local coordinates for grid along a ring and add to point coor
        for i in range(intvls_r):
            r = c*h - i*res_r
            perim = 2 * r * pi 
            intvls_c = max(int(perim/res), 1)  #number of intervals along a ring, close enough
            item = np.array([0.0, 0.0])  #initialize       
            for j in range(intvls_c):
                item[0] = (sin((2*pi) / intvls_c *(j+1)) *r) + point[0]
                item[1] = (cos((2*pi) / intvls_c *(j+1)) *r) + point[1]
                buffer = np.vstack((buffer, item))  #vstack works with arrays of diff nr of items, append does not        
        return buffer


    # In[24]:


    # create csv file with all buffer points     #ToDo KLK: make check plot of alignment, station points, buffer_all
    # point = alignment_stationed_xy
    # create alignment_stationed_xy from alignment_stationed_df with x,y of all station points
    alignment_stationed_xy = alignment_stationed_df.as_matrix(columns=['x_align','y_align'])
    # overburden = alignment_stationed_h
    alignment_stationed_h = alignment_stationed_df.as_matrix(columns=['h'])
    # initialize buffer_df, buffer_all_df, buffer_all
    buffer_all = {}
    buffer_df = pd.DataFrame(columns=["id_point", "x_align", "y_align", "z_align", "h" ,"x_buffer", "y_buffer"])
    buffer_all_df = pd.DataFrame(columns=["id_point", "x_align", "y_align", "z_align", "h","x_buffer", "y_buffer"])
    for n in range(0, len(alignment_stationed_df)): 
        buffer_point = make_buffer(point=alignment_stationed_xy[n], overburden=alignment_stationed_h[n], c=c, res=res)
        buffer_all[n] = buffer_df.copy(deep=False)  #copy of initialized buffer_df
        #print("n: ", n)
        #print(buffer_all)
        buffer_all[n]["id_point"] = [n] * len(buffer_point)  #list with len(buffer_point) number of n values) 
        buffer_all[n]["stat_sum"] = (  #make stat_sum correct for point n
            [alignment_stationed_df.iloc[n-1, alignment_stationed_df.columns.get_loc("distance_stat_sum")]] * len(buffer_point) ) 
        buffer_all[n]["dist_stat"] = (  #make stat_sum correct for point n
            [alignment_stationed_df.iloc[n-1, alignment_stationed_df.columns.get_loc("distance_stat")]] * len(buffer_point) ) 
        buffer_all[n]["x_align"] = ( 
            [alignment_stationed_df.iloc[n, alignment_stationed_df.columns.get_loc("x_align")]] * len(buffer_point) )      
        buffer_all[n]["y_align"] = ( 
            [alignment_stationed_df.iloc[n, alignment_stationed_df.columns.get_loc("y_align")]] * len(buffer_point) )      
        buffer_all[n]["z_align"] = ( 
            [alignment_stationed_df.iloc[n, alignment_stationed_df.columns.get_loc("z_align")]] * len(buffer_point) )           
        buffer_all[n]["h"] = ( 
            [alignment_stationed_df.iloc[n, alignment_stationed_df.columns.get_loc("h")]] * len(buffer_point) )           
        buffer_all[n]["x_buffer"] = buffer_point[0:,0]
        buffer_all[n]["y_buffer"] = buffer_point[0:,1]
        buffer_all_df = pd.concat([buffer_all_df, buffer_all[n]])
        #print(buffer_all_df)
    # add variable "id_buffer_point" to buffer_all_df
    buffer_all_df = buffer_all_df.reset_index(drop=True)
    buffer_all_df["id_buffer_point"] = buffer_all_df.index    
    # save buffer_all_df to csv
    buffer_all_df.to_csv(Buffer_all_csv, sep=",", na_rep="NaN")


    # In[25]:


    # add required field "z_dtm_buffer" and calcualted "dist" to buffer_all_df 
    # add required field "slope" to buffer_all_df 
    # buffer_all_df to Buffer_all_shp                                   #ToDo JK: reuse df -> shp function from above
    # list of shapely geometry points
    buffer_all_geometry = ( 
        [sp.geometry.Point(row['x_buffer'], row['y_buffer']) for key, row in buffer_all_df.iterrows()] )
    # create buffer_all_geometry_df
    buffer_all_geometry_df = gpd.GeoDataFrame(buffer_all_df, geometry=buffer_all_geometry, crs = crs)
    # write df to Buffer_all_shp
    buffer_all_geometry_df.to_file(Buffer_all_shp, driver='ESRI Shapefile') 
    #print(buffer_all_geometry_df.head())
    # get DTM values for Buffer_all_shp                             #ToDo JK: reuse what.points function from above
    #   write to Buffer_dtm_csv
    processing.runalg("grass7:r.what.points",DTM,Buffer_all_shp, "NA",",", 500,True,False,False,False,False,
                      grass_region,-1,0.0001,Buffer_dtm_csv)
    # create buffer_dtm_df (dataframe) from Buffer_dtm_csv
    buffer_dtm_df = pd.read_csv(Buffer_dtm_csv)
    # rename col=tmp... to "z_dtm_buffer"
    buffer_dtm_df_col_tmp = [col for col in buffer_dtm_df.columns if 'tmp' in col]
    if len(buffer_dtm_df_col_tmp) != 1:
        print "Extraction of DTM col=tmp did not work properly for buffer. Please check"
        exit()
    buffer_dtm_df = buffer_dtm_df.rename(
        columns= {buffer_dtm_df_col_tmp[0]: "z_dtm_buffer"})
    # write buffer_dtm_df["z_dtm"] to buffer_all_df["z_dtm"]
    buffer_all_df["z_dtm_buffer"] = buffer_dtm_df["z_dtm_buffer"]
    # get slope values for Buffer_all_shp
    #   write to Buffer_slope_csv
    processing.runalg("grass7:r.what.points",DTM_slope,Buffer_all_shp, "NA",",", 500,True,False,False,False,False,
                      grass_region,-1,0.0001,Buffer_slope_csv)
    # create buffer_dtm_df (dataframe) from Buffer_dtm_csv
    buffer_slope_df = pd.read_csv(Buffer_slope_csv)
    # rename col=tmp... to "z_dtm_buffer"
    buffer_slope_df_col_tmp = [col for col in buffer_slope_df.columns if 'tmp' in col]
    if len(buffer_slope_df_col_tmp) != 1:
        print "Extraction of Slope col=tmp did not work properly for buffer. Please check"
        exit()
    buffer_slope_df = buffer_slope_df.rename(
        columns= {buffer_slope_df_col_tmp[0]: "slope"})
    # write buffer_dtm_df["z_dtm"] to buffer_all_df["z_dtm"]
    buffer_all_df["slope"] = buffer_slope_df["slope"]
    # calculate "dist" between each buffer point and associated alignment point 
    buffer_all_df["dist"] = (((buffer_all_df["x_align"] - buffer_all_df["x_buffer"])**2 + 
                             (buffer_all_df["y_align"] - buffer_all_df["y_buffer"]) **2) +
                             (buffer_all_df["z_dtm_buffer"] - buffer_all_df["z_align"]) **2) **(0.5)
    # clean up
    buffer_all_df = buffer_all_df.drop(columns =["geometry"])


    # In[26]:


    ##buffer_all_df #checking


    # In[27]:


    # calculate minimum distance to terrain in each buffer ring
                                                    #ToDo KLK: make check plot of id_buffer_points with "min_dist"
                                                    #ToDo JK: add station field
    buffer_all_df["min_dist"] = np.nan
    for n in range(0, len(alignment_stationed_df)):
        buffer_all_df_sel = buffer_all_df.loc[(buffer_all_df["id_point"] == n),]
        dist_idxmin=buffer_all_df_sel['dist'].idxmin()
        buffer_all_df.loc[(buffer_all_df["id_buffer_point"] == dist_idxmin), "min_dist"] = "MIN"
    buffer_all_df.to_csv(Buffer_all_csv, header=True, index=False)  #no header
    ##buffer_all_df.loc[(buffer_all_df["min_dist"] == "MIN"),]  #checking


    # In[28]:


    # calculate hydraulic confinement safety factor at each station point
    #   required input data: reference maximum water pressure elevation (static or dynamic ??) 
    buffer_all_df_sel = buffer_all_df.loc[(buffer_all_df["min_dist"] == "MIN"),]
    dist = array(buffer_all_df_sel['dist'])
    slope = array(buffer_all_df_sel['slope'])
    z_align = array(buffer_all_df_sel['z_align'])
    stat_sum = array(buffer_all_df_sel['stat_sum'])
    stat_sum[0] = 0  #correction for station_stat_sum being for n-1 above (to be fixed above)
    def hydr_conf_sf(density_rock, max_static_level, z_align, min_dist, slope):
      density_water = 9.805
      static_head = max_static_level - z_align 
      sf = (min_dist * density_rock *  cos(slope*pi/180.)) / (static_head * density_water)
      return sf
    FS = hydr_conf_sf(28.0, max_static_water_level, z_align, dist, slope)
    buffer_all_df_sel["FS"] = np.nan
    for n in range(0, len(buffer_all_df_sel)):
        buffer_all_df_sel.iloc[n, buffer_all_df_sel.columns.get_loc("FS")] = FS[n]
    buffer_all_df_sel = buffer_all_df_sel.drop(columns =["x_align", "y_align", "min_dist", "id_buffer_point"])


    # In[29]:


    ##buffer_all_df_sel  #checking


    # In[30]:


    # calculate hydraulic confinement safety factor at each station point
    #   required input data: reference maximum water pressure elevation (static or dynamic ??) 
    buffer_all_df_sel = buffer_all_df.loc[(buffer_all_df["min_dist"] == "MIN"),]
    dist = array(buffer_all_df_sel['dist'])
    slope = array(buffer_all_df_sel['slope'])
    z_align = array(buffer_all_df_sel['z_align'])
    stat_sum = array(buffer_all_df_sel['stat_sum'])
    stat_sum[0] = 0  #correction for station_stat_sum being for n-1 above (to be fixed above)
    FS = (dist * 28.0 * cos(slope*pi/180.)) / ((637.20 - z_align) * 10)
    buffer_all_df_sel["FS"] = np.nan
    for n in range(0, len(buffer_all_df_sel)):
        buffer_all_df_sel.iloc[n, buffer_all_df_sel.columns.get_loc("FS")] = FS[n]
    buffer_all_df_sel = buffer_all_df_sel.drop(columns =["x_align", "y_align", "min_dist", "id_buffer_point"])


    # In[31]:


    ##buffer_all_df_sel  #checking


    # In[32]:


    # initialize variables
    print('starting plotly')
    
    x_data = []
    y_data = []
    traces = []
    annotations = []

    fs = buffer_all_df_sel['FS'].tolist(),
    c_fs = []
    for i in range(len(buffer_all_df_sel.index)):
        if fs[0][i] <= 1.4:
            c_fs.append('red')
        elif 1.4 < fs[0][i] < 1.8:
            c_fs.append('yellow')
        else:
            c_fs.append('green')
    slope = buffer_all_df_sel['slope'].tolist(),
    c_slope = []
    for i in range(len(buffer_all_df_sel.index)):
        if slope[0][i] > 30:
            c_slope.append('red')
        else:
            c_slope.append('green')        

    x_data = [buffer_all_df_sel['dist_stat'].tolist()]
    y_data = ['hydraulic confinement']

    for i in range(0, len(x_data[0])):
        for xd, yd in zip(x_data, y_data):
            traces.append(go.Bar(
                x=[xd[i]],
                y=[yd],
                width = 0.3,
                orientation='h',
                marker=dict(
                    color = c_fs[i],
                ),
                hoverinfo = 'none',
            ))

    layout = go.Layout(
        xaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
            domain=[0.08, 1]  #horizontal extent of bar
        ),
        yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
        ),
        barmode='stack',
        margin=dict(
            l=0,
            r=0,
            t=0,
            b=0
        ),
        showlegend=False,
        ####width=945,
        ####height=70,
    )

    for yd, xd in zip(y_data, x_data):
        # labeling the y-axis
        annotations.append(dict(xref='paper', yref='y',
                                x=0.09, y=yd, #x is position of text
                                xanchor='left',
                                text=str(yd),
                                font=dict(family='Arial', size=14,
                                          color='white'),
                                showarrow=False, align='left'))
    layout['annotations'] = annotations

    ##fig = go.Figure(data=traces, layout=layout)

    print('completed plotly')

    #print(fig)

    
    ############################
    # code above is from Jupyter
    #   in plotly comment out  ##print(fig)
    #   in plotly comment out  ##fig = go.Figure(data=traces, layout=layout)
    #   in plotly comment out  ##width=945,
    #   in plotly comment out  ##height=70,

    
    # prepare plotly data for returning to Polymer
    plot = [
        dict(
            data=[traces],
            layout=[layout]
        )
    ]

    # Convert the figures to JSON
    #  PlotlyJSONEncoder appropriately converts pandas, datetime, etc
    #  objects to their JSON equivalents
    plotJSON = json.dumps(plot, cls=plotly.utils.PlotlyJSONEncoder)
    #print(plotJSON)

    
    try:
        #return jsonify(fig)
        #return traces
        #return jsonify({'entry': 'valid'})  #ok
        return(plotJSON)
        
    except:
        pass
        
    return jsonify({'entry': 'invalid'})
