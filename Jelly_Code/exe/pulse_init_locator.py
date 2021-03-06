import  sys, os, pdb, shutil, datetime as dt, pandas as pd, numpy as np

from skimage            import io, color, measure, draw, img_as_bool	#scikit-image
from scipy              import optimize, signal
from time               import sleep
from math               import cos, sin, radians, floor
from bisect             import bisect
from ipysankeywidget    import SankeyWidget

basePath    = os.path.dirname( os.path.dirname( __file__ ) )
sys.path.append( basePath + '\Libraries' )

import image_fns as imgFns
import excel_fns as xlFns

######################
######  INPUTS  ######
######################
thetas              = list( range( 0, 360, 5 ) )
theta_rollup        = list( range( 0, 361, 45 ) )
video_dir           = os.path.dirname( basePath ) + '\Vidz\Frames'
# video_dir       = r'F:\20190131\Vidz\Frames'
specific_vids       = [  ] #pass the name of the folder in which the frames sit (no extension and no path location, just simple file name)
min_pulse_length    = 10    #frames

######################
######  DO IT   ######
######################
if( len( specific_vids ) ):
    all_vid_dirs    = specific_vids
else:
    all_vid_dirs    = os.listdir( video_dir )
all_ts          = [ ]
first_pulses    = [ ]

for idx, dir in enumerate( all_vid_dirs ):
    if( '.' in dir ): continue

    print( ' >> Computing distances for video ' + str( idx + 1 ) + ' of ' + str( len( all_vid_dirs ) ) )
    this_frame_dir  = video_dir + '\\' + dir
    ts              = imgFns.process_video( this_frame_dir, thetas = thetas )
    ts['Avg']       = ts.mean( axis = 1 )

    #using the average line, estimate the pulse interval
    # peaks           = signal.find_peaks(ts['Avg'], prominence = 3, distance = 20, height = peak_min)[0].tolist()
    valleys           = signal.find_peaks([ 1/x for x in ts['Avg'].tolist()], distance = min_pulse_length, height = np.percentile( [1/x for x in ts['Avg'].tolist()], 75 ))[0].tolist()

    #tag each timestamp for which pulse its in
    ts['pulse_index'] = [ bisect( valleys, x ) for x in ts.index.tolist() ]

    #write output to excel
    xlFns.to_excel( {'TimeSeries': ts},
                    file                    = os.path.dirname( basePath ) + '\Results\Time Series\\TS_' + dir.replace(' ', '_') + '.xlsx',
                    masterFile              = os.path.dirname( basePath ) + '\Results\Time Series\Time_Series_vMaster.xlsx',
                    allowMasterOverride     = False,
                    promptIfLocked          = True,
                    xlsxEngine              = 'xlwings', #xlsxwriter, openpyxl
                    closeFile               = True,
                    topLeftCell             = {},
                    batchSize               = 10000,
            )
    
    print( '  >> Getting pulse initiations...' )    
    pulses      = ts[[ x for x in ts.columns.values if x not in ['Avg'] ]].groupby( 'pulse_index' ).agg( imgFns.pulse_init )
    pulse_order = pulses.apply( lambda x: imgFns.init_order( x.tolist() ), axis=1).reset_index()
    
    col_groups  = [ bisect( theta_rollup, int(x) ) for x in pulses.columns.values ]
    
    rollup        = { }
    for group in set(col_groups):
        cols            = [ pulses.columns.values[indx] for indx, col in enumerate( col_groups ) if col == group ]
        rollup[ 'rollup_' + str(group) ]   = pulses[cols].mean( axis = 1 )
        # rollup[ 'rollup_' + str(group) ]   = pulses[cols].min( axis = 1 )

    # rollup                  = pd.DataFrame( rollup )                                            #adjust the rounding here to encourage / discourage ties
    # rollup                  = pd.DataFrame( rollup ).round(0)                                   #adjust the rounding here to encourage / discourage ties
    rollup                  = ((pd.DataFrame( rollup )*2).round(0)/2)                           #adjust the rounding here to encourage / discourage ties
    # rollup                  = pd.DataFrame( rollup ).round(1)                                   #adjust the rounding here to encourage / discourage ties
    rollup_order            = rollup.apply( lambda x: imgFns.init_order( x.tolist() ), axis=1)
    rollup_order.columns    = rollup.columns
    rollup_order.reset_index( inplace = True )
    # cols                            = pulse_order.columns.tolist() + [ 'init_agg_next', 'init_agg_next_next' ]

    # pulse_order['next_index']       = pulse_order['pulse_index'] - 1
    # pulse_order['next_next_index']  = pulse_order['pulse_index'] - 2

    # pulse_order                     = pulse_order.merge( pulse_order[['next_index', 'init_agg']], left_on = ['pulse_index'], right_on = ['next_index'], how = 'left', copy = False, suffixes = ['', '_next'] )
    # pulse_order                     = pulse_order.merge( pulse_order[['next_next_index', 'init_agg']], left_on = ['pulse_index'], right_on = ['next_next_index'], how = 'left', copy = False, suffixes = ['', '_next_next'] )
    # pulse_order                     = pulse_order[ cols ]
    xlFns.to_excel( {   'First_Pulse':      pulses,         #frames at which pulse initiated
                        'First_Pulse_Rank': pulse_order,    #rank of the pulse initiation
                        'RollUp':           rollup,
                        'RollUp_Rank':      rollup_order
                    },
                    file                    = os.path.dirname( basePath ) + '\Results\First Pulse\\FP_' + dir.replace(' ', '_') + '.xlsx',
                    masterFile              = os.path.dirname( basePath ) + '\Results\First Pulse\Pulse_Order_vMaster.xlsx',
                    allowMasterOverride     = True,
                    promptIfLocked          = True,
                    xlsxEngine              = 'xlwings', #xlsxwriter, openpyxl
                    closeFile               = True,
                    topLeftCell             = {},
                    batchSize               = 10000,
            )


    #make the sankey chart
    # pulse_transition            = pulse_order[['init_agg', 'init_agg_next', 'pulse_index']].groupby(['init_agg', 'init_agg_next']).count().reset_index()
    # pulse_transition.columns    = ['source', 'target', 'value']
    # sankey_data                 = pulse_transition[ [ 'source', 'target', 'value'] ]
    # sankey_data.to_csv( os.path.dirname( basePath ) + '\Results\Sankey\sankey_vMaster.csv', mode = 'w+', index = False )

    # pulses      = first_pulses['init_agg'].tolist()
    # streaks     = { key: [ ] for key in set( pulses ) }
    # this_streak = 1
    # for indx, pulse in enumerate( pulses ):
        # if( (indx+1) == len( first_pulses ) ):
            # break
        # if( pulse == pulses[indx + 1] ):
            # this_streak+=1
        # else:
            # streaks[ pulse ].append( this_streak )
            # this_streak = 1
    # streaks_df  = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in streaks.items() ]))
    # xlFns.to_excel( streaks_df,
                    # file                    = os.path.dirname( basePath ) + '\Results\Streaks\\Streaks_' + dir.replace(' ', '_') + '.xlsx',
                    # masterFile              = os.path.dirname( basePath ) + '\Results\Streaks\Streaks_vMaster.xlsx',
                    # allowMasterOverride     = True,
                    # promptIfLocked          = True,
                    # xlsxEngine              = 'xlwings', #xlsxwriter, openpyxl
                    # closeFile               = True,
                    # topLeftCell             = {},
                    # batchSize               = 10000,
            # )
