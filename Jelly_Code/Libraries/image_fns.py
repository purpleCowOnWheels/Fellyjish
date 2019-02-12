import sys, os, pdb, shutil, datetime as dt, pandas as pd, numpy as np, time, matplotlib.pyplot as plt
try:
	import cv2
except ImportError:
	os.system( 'pip install opencv-python' ) #do this since its a fairly obscure package with weird naming convention
	import cv2

from skimage    import io, color, measure, draw, img_as_bool
from scipy      import optimize
from time       import sleep
from math       import cos, sin, radians, floor, sqrt
from bisect     import bisect


def video_to_frames(input_loc, output_loc):    
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print ( "  >> Number of frames: ", video_length)
    count = 0
    print ("  >> Converting video...")
    # Start converting the video
    while cap.isOpened():
        if( count % 1000 == 0 ): print( "    ++ Completed frame: ", count, '...' ) 
        # Extract the frame
        ret, frame = cap.read()
        # Write the results back to output location.
        cv2.imwrite(output_loc + "/%#05d.jpg" % (count+1), frame)
        count = count + 1
        # If there are no more frames left
        if (count > (video_length-1)):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats

            print ( "  >> %d frames successfully extracted." % count)
            print ( "    ++ %d seconds..." % (time_end-time_start))
            break

def locateContours( img, threshold = 127 ):
    ## (1) Read
    # img = cv2.imread("img04.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ## (2) Threshold
    th, threshed = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)

    ## (3) Find the first contour that greate than 100, locate in centeral region
    ## Adjust the parameter when necessary
    cnts = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnts = sorted(cnts, key=cv2.contourArea)
    H,W = img.shape[:2]
    for cnt in cnts:
        x,y,w,h = cv2.boundingRect(cnt)
        if cv2.contourArea(cnt) > 100 and (0.7 < w/h < 1.3) and (W/4 < x + w//2 < W*3/4) and (H/4 < y + h//2 < H*3/4):
            break

    ## (4) Create mask and do bitwise-op
    mask = np.zeros(img.shape[:2],np.uint8)
    cv2.drawContours(mask, [cnt],-1, 255, -1)
    dst = cv2.bitwise_and(img, img, mask=mask)

    ## Display it
    # cv2.imwrite("dst.png", dst)
    # cv2.imshow("dst.png", dst)
    # cv2.waitKey()
    
    return( dst )
    
def createLineIterator(P1, P2, img):
    """
    Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

    Parameters:
        -P1: a numpy array that consists of the coordinate of the first point (x,y)
        -P2: a numpy array that consists of the coordinate of the second point (x,y)
        -img: the image being processed

    Returns:
        -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])     
    """    
    #define local variables for readability
    imageH = img.shape[0]
    imageW = img.shape[1]
    P1X = P1[0]
    P1Y = P1[1]
    P2X = P2[0]
    P2Y = P2[1]

    #difference and absolute difference between points
    #used to calculate slope and relative location between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    #predefine numpy array for output based on distance between points
    # itbuffer = np.empty(shape=(np.maximum(dYa,dXa),3),dtype=np.float32)
    itbuffer = np.empty(shape=(np.maximum(dYa,dXa),2),dtype=np.float32)  
    itbuffer.fill(np.nan)

    #Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X: #vertical line segment
        itbuffer[:,0] = P1X
        if negY:
            itbuffer[:,1] = np.arange(P1Y - 1,P1Y - dYa - 1,-1)
        else:
            itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)              
    elif P1Y == P2Y: #horizontal line segment
        itbuffer[:,1] = P1Y
        if negX:
            itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
        else:
            itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
    else: #diagonal line segment
        steepSlope = dYa > dXa
        if steepSlope:
            slope = dX.astype(np.float32)/dY.astype(np.float32)
            if negY:
                itbuffer[:,1] = np.arange(P1Y-1,P1Y-dYa-1,-1)
            else:
                itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
            itbuffer[:,0] = (slope*(itbuffer[:,1]-P1Y)).astype(np.int) + P1X
        else:
            slope = dY.astype(np.float32)/dX.astype(np.float32)
            if negX:
                itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
            else:
                itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
            itbuffer[:,1] = (slope*(itbuffer[:,0]-P1X)).astype(np.int) + P1Y

    #Remove points outside of image
    colX = itbuffer[:,0]
    colY = itbuffer[:,1]
    itbuffer = itbuffer[(colX >= 0) & (colY >=0) & (colX<imageW) & (colY<imageH)]

    #Get intensities from img ndarray
    # itbuffer[:,2] = img[itbuffer[:,1].astype(np.uint),itbuffer[:,0].astype(np.uint)][:,0]

    return( itbuffer )
    

def get_coord( theta, cx, cy ):
    radius  = 1000
    y       = -radius * cos( radians( theta ) ) + cy
    x       = radius * sin( radians( theta ) ) + cx
    return( (theta, x, y) ) #pixels are from the top left

def getCentroid( img, center_x, center_y, depth = 1, max_depth = 3 ):
    if( depth >= max_depth ):
        print( '    -- Current frame center location: (' + str( center_x ) + ', ' + str( center_y ) + ')', end='' )
        return( (center_x, center_y) )
    
    #compute the area around the center point
    quarter_x   = round(len(img[0])/3,0)    #going this far left and right will give half image in total
    quarter_y   = round(len(img)/3,0)
    
    x_min       = int( max(center_x-quarter_x,0) )
    x_max       = int( min(center_x+quarter_x,len(img[0])) )
    
    y_min       = int(max(center_y-quarter_y,0))
    y_max       = int(min(center_y+quarter_y,len(img)))
    
    #filter image to area surrounding center
    sub_img     = [ x[x_min:x_max] for x in img[y_min:y_max] ]
    
    #flatten the gradients in the image to make sure all the whites are the same white
    gradient_threshold  = _getGradientThreshold( img, center_x, center_y )
    for y_indx, y in enumerate(sub_img):
        for x_indx, x in enumerate(y):
            if( max(x) < gradient_threshold ):
                sub_img[y_indx][x_indx] = np.array([0,0,0], dtype=np.uint8)
            else:
                sub_img[y_indx][x_indx] = np.array([255,255,255], dtype=np.uint8)

    
    # plt.imshow(img)
    # plt.imshow(sub_img)
    # plt.show()
    
    #calculate the centroid on the sub-image
    regions     = measure.regionprops(np.array(sub_img))
    centroid    = regions[len(regions)-1].centroid[:2]
    
    #convert centroid points back to original image scale
    center_x    = int(x_min + centroid[1])
    center_y    = int(y_min + centroid[0])
    return( getCentroid( img, center_x, center_y, depth = depth+1 ) )
    
def _getGradientThreshold( img, center_x, center_y, factor = 5 ):
    #to avoid con
    x   = np.mean([ img[y][x] for x in range( center_x - 10, center_x + 10 ) for y in range( center_y - 10, center_y + 10 ) ])
    return( x / factor )

global centers
#initial guess for the location of the centroid (sets the search area)
centers = { '20190130_355pm':   (300, 177),
            '20190114_01':      (335, 235),
          }
          
def process_video( frame_dir, thetas= [0, 90, 180, 270]):    
    jellyadii   = { str(k): [ ] for k in thetas }
    files       = os.listdir( frame_dir )
    files       = [ f for f in files if f not in [ 'outputs' ] ]                            #only process frames. remove other stuff

    sort_order  = [ f.split('_')[1] if '_' in f else f for f in files ]                     #if files named frame_1.jpg convert to 1.jpg
    sort_order  = [ int( f.split('.')[0] ) for f in sort_order ]                            #convert to int to deal with leading zeroes
    sort_order  = [ x[1] for x in sorted((e,i) for i,e in enumerate(sort_order)) ]          #get the order of this array to use to sort the filenames
    files       = [ files[s] for s in sort_order ]
    
    sections_img_file_dir   = frame_dir + '\\outputs'
    if( not os.path.exists( sections_img_file_dir ) ):
        os.mkdir( sections_img_file_dir )
    
    if( os.path.basename(frame_dir) in centers.keys() ):
        this_center_x   = centers[os.path.basename(frame_dir)][0]
        this_center_y   = centers[os.path.basename(frame_dir)][1]
    else:
        this_center_x   = 335 #initial guess for the location of the centroid (sets the search area)
        this_center_y   = 235 #initial guess for the location of the centroid (sets the search area)
    
    centers_x       = [this_center_x]*250   #seed with a prior probability from which rolling average needs to pull away
    centers_y       = [this_center_y]*250
    jelly_at_edge   = False
    for N, file in enumerate( files ):
        img                 = io.imread(frame_dir + '\\' + file)
        gradient_threshold  = _getGradientThreshold( img, this_center_x, this_center_y )

        if(N % 1000 == 0):
            print( '  ++ Frame ' + str(N) + ' of ' + str( len( files ) ) )
            #centroid calc is expensive. don't do it every frame
            #pass a starting 'guess' coordinate from the current center
            centroid        = getCentroid(img, this_center_x, this_center_y )        
            centers_x.append( centroid[0] )
            centers_y.append( centroid[1] )

            #centroid naturally moves as the jellyfish pulses, so take a long rolling average. This will only move if there is a macro move in the jellyfish
            #lag is a parameter that should be tested. currently 250.
            this_center_x       = int(np.mean(centers_x[max(0,len(centers_x)-500):]))
            this_center_y       = int(np.mean(centers_y[max(0,len(centers_y)-500):]))            
            print( ' | Rolling avg. center location: (' + str( this_center_x ) + ', ' + str( this_center_y ) + ')' )
        
        if( N % 10000 == 0 ):
            if( not os.path.exists( os.path.dirname( basePath ) + '\Results\Time Series\inProgress' ) ):
                os.mkdir( os.path.dirname( basePath ) + '\Results\Time Series\inProgress' )
                
            #write intermediate output to a temp excel location; if this is slow just tweak to write incremental data only
            xlFns.to_excel( pd.DataFrame( jellyadii ),
                            file                    = os.path.dirname( basePath ) + '\Results\Time Series\inProgress\\' + frame_dir.replace(' ', '_') + '.xlsx',
                            masterFile              = os.path.dirname( basePath ) + '\Results\Time Series\Time_Series_vMaster.xlsx',
                            allowMasterOverride     = False,
                            promptIfLocked          = True,
                            xlsxEngine              = 'xlwings', #xlsxwriter, openpyxl
                            closeFile               = True,
                            topLeftCell             = {},
                            batchSize               = 10000,
                    )
        
        coordinates = [ get_coord( t, this_center_x, this_center_y ) for t in thetas ]

        show_frames = [13, 14, 15]
        # img_clean   = locateContours ( img )
        this_img = img.copy()
        for coordinate in coordinates:
            line        = createLineIterator( np.array([this_center_x, this_center_y]), np.array([int(coordinate[1]),int(coordinate[2])]), this_img )    
            for indx, pt in enumerate( line ):
                gradient    = img[int(pt[1]), int(pt[0])][0]
                if( gradient > gradient_threshold ):
                    if( (N+1) in show_frames ): this_img[int(pt[1]), int(pt[0])] = [ 255, 0, 0 ]
                    if( indx == (len(line)-1) ):
                        print( '      --No frame found below threshold!' )
                        jelly_at_edge = True
                        return( pd.DataFrame( jellyadii ) )  #need to handle these being different lengths!!
                else:
                    jellyadii[str(coordinate[0])].append( round(sqrt((this_center_x - pt[0])**2 + (this_center_y - pt[1])**2),1) )
                    break
        if( (N+1) in show_frames ):
            print( '    -- Saving output of file: ' + file )
            plt.imshow(this_img)
            sections_img_file =  sections_img_file_dir + '\\' + file
            if os.path.isfile( sections_img_file ):
                os.remove( sections_img_file )   # Opt.: os.system("rm "+strFile)
            plt.savefig( sections_img_file )
            # plt.show()
            # plt.pause(2) # pause how many seconds
            plt.close()
        if( jelly_at_edge ):
            print( '  >> Jelly at edge at frame ' + str(N) + '. Cutting off here and analyzing time series...' )
            return( pd.DataFrame( jellyadii ) )
        # if(N>1500): break

    print( '  >> All frames processed. Analyzing time series...' )
    return( pd.DataFrame( jellyadii ) )

def pulse_init( pulse       = [ 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 1 ],
                peak_type   = 'lagged_drop',
                return_type = 'global_index',                           #local_index or global_index
               ):
    global_indexes  = pulse.index.tolist()
    pulse           = pulse.tolist()
    if( peak_type == 'pct_max' ):
        this_max    = pulse.index( max( pulse ) )
        threshold   = np.percentile( pulse, 85 )
        init        = [ k for k in pulse if k < threshold ][0]
        local_index = pulse.index( init )
        #To be continued
    elif( peak_type == 'pct_decr' ):
        p1          = pulse.copy().tolist()
        p2          = pulse.copy().tolist()
        p1.pop(0)
        p2.pop()
        delta       = [ x[1]/x[0] - 1 for x in zip( p2, p1 ) ]
        
        threshold   = max( min( delta ), -.1 )
        init        = [ k for k in delta if k <= threshold ][0]     #get the all x%+ drops and take the first
        local_index = delta.index( init ) + 1                       #find the location of that drop in the deltas
    elif( peak_type == 'lagged_drop' ):
        local_index = None
        for indx, p1 in enumerate( pulse ):
            #For 60fps use 10, for 30fps use 5
            if( indx < 5 ): continue
            # if( ( p1 / pulse[indx-10] - 1 ) < -0.1 ):                 #instead of 10 ago take max in the last 10?
            if( ( p1 / max(pulse[max(indx-5,0):indx]) - 1 ) < -0.1 ):  #technically should be going to max(indx-1,0), but save this calc since @ indx its always 1/1
                local_index = indx
                break
    else:
        return( None )

    if( return_type == 'local_index' or local_index is None ):
        return( local_index )
    else:
        return( global_indexes[ local_index ] )

def first_init( frames = [ 10, 25, 25, 26, 27, 28, 29, 29, 30, 31 ] ):
    frames_reduced  = [ x for x in frames if not pd.isnull(x) ]
    frames_reduced  = [ x for x in frames_reduced if ( x > ( np.nanmedian( frames_reduced ) - 7 ) )]    #if the pulse was way before the median its likely bad data. throw out that angle.
    if( len( frames_reduced ) ):
        first_frame     = frames.index( min( frames_reduced ) )                                         #of the remaining angles, simply pick the one that fired first.
        if( isinstance( first_frame, np.ndarray ) ): 
            return( bisect( first_frame, np.median(first_frame) ) )                                     #break ties by taking median (floored if even length)
        else:
            return( first_frame )
    else:
        return( None )

