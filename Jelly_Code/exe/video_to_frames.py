import sys, os, shutil, pdb

basePath    = os.path.dirname( os.path.dirname( __file__ ) ) #location of this exe file, up 2 directories
sys.path.append( basePath + '/Libraries' )

#inPath  	= os.path.dirname( basePath ) + '\Vidz'
inPath  = r'E:\20190201'

#outPath 	= os.path.dirname( basePath ) + '\Vidz\Frames' + '\\20190201_631pm'
#outPath = r"C:\Users\Mike's\Your team Dropbox\Michael Abrams\Fellyjish\Vidz" + '\Frames\20190201_631pm'
outPath         = r'F:\20190201\Vidz\Frames\20190201_133pm'

#make sure the right folder structure exists
if( not os.path.exists( os.path.dirname( basePath ) + '\Vidz\Frames' ) ):
	os.mkdir( os.path.dirname( basePath ) + '\Vidz\Frames' )
if( not os.path.exists( os.path.dirname( basePath ) + '\Vidz\Framed' ) ):
	os.mkdir( os.path.dirname( basePath ) + '\Vidz\Framed' )
if( not os.path.exists( outPath ) ):
	os.mkdir( outPath )

import image_fns as img
files   = os.listdir(inPath)
for inFile in files:
    if not os.path.isfile( inPath + '\\' + inFile ): continue

    print( '\n  >> Extracting: ' + inFile )
    img.video_to_frames( inPath + '\\' + inFile, outPath + '\\Frames\\' + inFile.split('.')[0] )
    
    print( '  >> Archiving full-length video... ' )
    shutil.copy2( inPath + '\\' + inFile, outPath + '\\Framed\\' + inFile )
    os.remove( inPath + '\\' + inFile )

