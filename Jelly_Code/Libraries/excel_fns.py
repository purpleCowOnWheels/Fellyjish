import sys, openpyxl, pdb, os, shutil, warnings, platform, uuid, json, re, pandas as pd
import datetime as dt, xlwings as xw, numpy as np
from pandas import DataFrame, ExcelWriter, to_datetime, read_excel
from time   import sleep

basePath = os.path.dirname( os.path.dirname(__file__) )
sys.path.append( basePath + '/Libraries')

def creation_date( path_to_file ):
    # Try to get the date that a file was created, falling back to when it was
    # last modified if that isn't possible.
    # See http://stackoverflow.com/a/39501288/1709587 for explanation.
    if platform.system() == 'Windows':
        return os.path.getctime(path_to_file)
    else:
        stat = os.stat(path_to_file)
        try:
            return stat.st_birthtime
        except AttributeError:
            # We're probably on Linux. No easy way to get creation dates here,
            # so we'll settle for when its content was last modified.
            return stat.st_mtime
  
def _clear_archive( dir, updated_since = dt.datetime.now() - dt.timedelta(days=90) ):
    if not os.path.exists(dir):
        os.makedirs(dir)
        return( 1 )
   
    dir_str = [ x for x in dir.split( '\\' ) if len(x) ]
    dir_str = '\\'.join( dir_str[(len(dir_str)-3):] )
    print( '    ++ Cleaning up ' + dir_str )
    keep_counter = 0
    for filename in os.listdir( dir ):
        date_modified = dt.datetime.fromtimestamp(creation_date(dir + filename))
        if( date_modified < updated_since ):
            print( '      -- Deleting ' + filename[:35] + ' from ' + date_modified.strftime('%Y-%m-%d') )
            try:
                os.remove( dir + filename )
            except:
                print( '      -- Unable to delete. Oh well...' )
        else:
            keep_counter = keep_counter + 1
            if( keep_counter % 100 == 0 ): print( '      -- Archived files kept: ' + str( keep_counter ) )
    return( 1 )
 
def read_excel_stable( filepath,
                       sheet_name       = 0,
                       dtype            = {},
                       pause_if_failed  = False,
                       retries          = 10,
                      ):
    counter = 0
    file_loaded = False
    #if the file is locked/RO, sleep to give user time to close it. After 10 failures, wait for user input.
    while( not file_loaded ):
        try:
            data    = read_excel( filepath, sheet_name, dtype = dtype )
            file_loaded = True
        except:
            print('  >> Waiting on ' + os.path.split(filepath)[1] + ' for 15 seconds...' )
            counter = counter + 1
            if( counter >= retries ):
                if( pause_if_failed ): pdb.set_trace()
                print( '  >> Unable to open ' + os.path.split(filepath)[1] + ' after 10 tries. Skipping...' )
                return( pd.DataFrame() )
            else:
                sleep(15)
    return( data )
    
def to_csv( df, file, naStr = 'NA', delimiter = '|', encoding='utf-8', rowNames = False, append = False):
    df_local = df.copy()
    df_local.fillna(naStr, inplace=True)
    if( not append ):
        f = open(file,"w+")
        f.write(''.join(['\"sep=', delimiter, '\"\n']))
        f.close()
    with open(file, 'a') as f:
        df_local.to_csv(f, sep=delimiter, encoding=encoding, index = rowNames, header = not append) #when appending, do not add the header row
    return(1)
 
def archiveFile(masterFile):
    print( '      -- Archiving ' + os.path.split(masterFile)[1])
    if not os.path.exists(os.path.dirname(masterFile) + '\\Archive'):
        os.makedirs(os.path.dirname(masterFile) + '\\Archive')
    if( os.path.exists(masterFile) ):
        shutil.copy(masterFile, os.path.split(masterFile)[0] + '\\Archive\\' + os.path.split(masterFile)[1].split('.')[0] + '_' + dt.date.today().strftime('%Y-%m-%d') + '_' + str(uuid.uuid4()) + '.' + os.path.split(masterFile)[1].split('.')[1])
    else:
        print( '  >> File not found: ' + masterFile + '. Continuing...')
   
    if not os.path.exists(os.path.dirname(masterFile) + '\\tmp'):
        os.makedirs(os.path.dirname(masterFile) + '\\tmp')
    return(1)
 
def to_excel_xlwings( df,
                      file,
                      masterFile        = '',
                      dropIndexCol      = True,
                      promptIfLocked    = False,
                      closeFile         = True,
                      rowNames          = False,
                      clearSheets       = 'All',
                      batchSize         = 10000,
                      topLeftCell       = {},
                      calcModeReturned  = 'manual',
                      clearFilters      = False,
                      delete_sheets     = [ ],
                     
    ):
    # df = DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    print( '      -- Opening workbook...')
   
    if( promptIfLocked and masterFile != '' ):
        file_updated = False
        counter = 0
        #if the file is locked/RO, sleep to give user time to close it. After 10 failures, wait for user input.
        while( not file_updated ):
            try:
                wb = xw.Book(masterFile)
                wb.save(masterFile)
                file_updated = True
            except:
                print('  >> Waiting on ' + os.path.split(file)[1] + ' for 15 seconds' )
                counter = counter + 1
                if( counter >= 10 ):
                    pdb.set_trace()
                sleep(15)
   
    if( masterFile == ''):
        wb = xw.Book()
    else:
        wb = xw.Book(masterFile)
    wb.app.calculation      = 'manual' #set to manual calc mode
    # wb.app.screen_updating  = False
    sheets = [x.name for x in wb.sheets]
    for sheetName, this_df in df.items():
        if sheetName not in sheets:
            #xlwings can't add sheets on-the-fly.
            print( '  >> Sheet ' + sheetName + ' not found in master. Skipping...' )
            continue
        if( not len( this_df ) ):
            print( '        xx No data for sheet ' + sheetName + '. Moving on.')
            continue
        else:
            print( '        ** Writing worksheet ' + sheetName )
            if( ( 'index' in [str(x) for x in this_df.columns.values] ) and dropIndexCol ):
                this_df.drop(['index'], axis = 1, inplace = True)
            ws = wb.sheets[sheetName]
            if( clearFilters ):
                ws.range('A1:zz10000').api.AutoFilter()               #remove any filters. Known bug in xlwings only writes to visible cells.
            if( sheetName in topLeftCell.keys() ):
                this_topLeftCell = topLeftCell[sheetName]
            else:
                this_topLeftCell = 'A1'
           
            if( this_topLeftCell == 'A1' and ( clearSheets == 'All' or sheetName in clearSheets ) ):
                ws.clear()
           
            this_row        = batchSize
            top_left_col    = re.split( r'[0-9]', this_topLeftCell )[0]
            ws.range(this_topLeftCell).options(index = rowNames).value = this_df[:batchSize]
            while( this_row < len(this_df) ):
                ws.range( top_left_col + str(this_row + 2)).options(index = rowNames, header = False).value = this_df[(this_row):(this_row + batchSize)]
                this_row = this_row + batchSize
 
    for sheetName in delete_sheets:
        print( '        ** Deleting worksheet ' + sheetName )
        wb.sheets[sheetName].delete()
   
    if( calcModeReturned == 'automatic' ):
        print( '      -- Recalculating...' )
        wb.app.calculation      = 'automatic' #reset to auto calc mode
    print( '      -- Saving as ' + os.path.split(file)[1] )
    wb.save(file)
    # wb.app.screen_updating  = True
    if( closeFile ):
        wb.close()
    return(1)
   
def to_excel_ExcelWriter( df,
                          file,
                          rowNames          = False,
                          masterFile        = '',
                          promptIfLocked    = False,
                          xlsxEngine        = 'openpyxl', #xlsxwriter
                        ):
    writer = ExcelWriter(file, engine=xlsxEngine)
    DataFrame().to_excel(writer, list(df.keys())[0])
   
    #ensure the file is not locked.
    try:
        #create the file even before we try to add any data to ensure it isn't locked
        writer.save()
    except:
        counter = 0
        file_updated = False
        #if the file is locked/RO, sleep to give user time to close it. After 10 failures, wait for user input.
        while( not file_updated ):
            try:
                writer.save()
                file_updated = True
            except:
                print('  >> Waiting on ' + os.path.split(file)[1] + ' for 15 seconds' )
                counter = counter + 1
                if( counter >= 10 ):
                    pdb.set_trace()
                sleep(15)
 
    if(masterFile != ''):
        # book = xlrd.open_workbook(masterFile)
       
        #create an archived copy of the master
        book = openpyxl.load_workbook(masterFile)
        writer.book = book
        writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
 
    for sheetName, this_df in df.items():
        if( not len( this_df ) ):
            print( '    ++ No data for sheet ' + sheetName + '. Moving on.')
            continue
 
        this_df.to_excel(writer, sheetName, index = rowNames)
    writer.save()
    return(1)
 
def to_json( sheet_dict,
              file,
              masterFile    = '',
              request_id    = '',
            ):
 
    dir         = os.path.dirname( file ).replace( '\\tmp', '' )
 
    #extract the file names from the paths
    file        = [ x for x in file.split( '\\' ) if len(x) ]
    file_name   = file[len(file)-1].split('.')[0]
   
    if( request_id == 'NA' ):
        request_id = file_name
   
    if( len( masterFile ) ):
        masterFile  = [ x for x in masterFile.split( '\\' ) if len(x) ]
        master_name = masterFile[len(masterFile)-1].split('.')[0]
    else:
        master_name = 'Dataset'
 
    output = { 'OutputFileName': file_name,
                master_name: {}
             }
 
    for sheet_name, sheet in sheet_dict.items():
        if( not len( sheet ) ):
            print( '    ++ No data for sheet ' + sheet_name + '. Moving on.')
            continue
        #Ensure no duplicate columns
        this_sheet = sheet.loc[:,~sheet.columns.duplicated()].copy()
       
        #JSON conversion fails for int-type keys. convert them to strings.
        this_sheet.columns = [ str(x) for x in this_sheet.columns.values]
        for col in this_sheet.columns.values:
            #JSON conversion can only handle certain data types. Cast them to usable alternatives.
            if( type( this_sheet[col].iloc[0] ).__name__ == 'date' or type( this_sheet[col].iloc[0] ).__name__ == 'Timestamp' ):
                this_sheet[col]  = this_sheet[col].apply( lambda x: x if x == 'NA' else x.strftime('%Y-%m-%d') )
            if( 'int' in type( this_sheet[col].iloc[0] ).__name__ ):
                this_sheet[col]  = this_sheet[col].apply( lambda x: str(x) )
        output[master_name][sheet_name] = this_sheet.to_dict( orient='records' )
 
    output_location = dir + '\\' + request_id + '.json'
    out_file        = open( output_location, 'w+')
    json.dump(output, out_file)
    out_file.close()
    return( output_location )
   
def to_excel( df,
              file,
              naStr                 = 'NA',
              rowNames              = False,
              masterFile            = '',
              allowMasterOverride   = False,
              promptIfLocked        = False,
              archiveMaster         = False,
              archiveOutfile        = False,
              xlsxEngine            = 'xlwings', #xlsxwriter, openpyxl
              closeFile             = True,
              dateCols              = {},
              topLeftCell           = {},
              clearFilters          = False,
              batchSize             = 2500,
              request_id            = 'NA',     #used in web output
              delete_sheets         = [ ],
            ):
 
    assert( ( not len( topLeftCell ) ) or xlsxEngine == 'xlwings' )    #can only support non-A1 writes in xlwings
   
    #df can be a single dataframe or a dictionary of dataframes; keys will be used as sheet names
    if(not allowMasterOverride):
        assert( masterFile != file ) #don't allow accidental overwrites
    if( archiveMaster ):
        assert(archiveFile(masterFile))
    if( archiveOutfile and (not (file == masterFile and archiveMaster) ) and os.path.exists( file )):
        assert(archiveFile(file))
 
    if( type(df).__name__ != 'dict' ):
        df = {'Sheet1': df}
   
    for sheetName, this_df in df.items():
        try:
            df[sheetName] = this_df.replace(np.nan, naStr, regex=True)
        except Exception as e:
            print( sheetName + '\n' + str(e) )
 
    #cast date columns to proper dates/
    for tabName, dateCols in dateCols.items():
        assert( tabName in df.keys() ) #make sure the dataset that needs a date conversion exists.
        for col in dateCols:
            df[tabName][col] = to_datetime(df[tabName][col], infer_datetime_format = True ).dt.date
   
    if( xlsxEngine == 'json' ):
        result = to_json( sheet_dict    = df,
                          file          = file,
                          masterFile    = masterFile,
                          request_id    = request_id,
                        )
    elif( xlsxEngine == 'xlwings' ):
        result = to_excel_xlwings(  df                  = df,
                                      file              = file,
                                      masterFile        = masterFile,
                                      promptIfLocked    = promptIfLocked,
                                      closeFile         = closeFile,
                                      rowNames          = rowNames,
                                      topLeftCell       = topLeftCell,
                                      clearFilters      = clearFilters,
                                      batchSize         = batchSize,
                                      delete_sheets     = delete_sheets,
                                  )
    else:
        result = to_excel_ExcelWriter( df               = df,
                                        file            = file,
                                        rowNames        = rowNames,
                                        masterFile      = masterFile,
                                        promptIfLocked  = promptIfLocked,
                                        xlsxEngine      = xlsxEngine
                                      )
   
    #if file is saved in a tmp location, promote it.
    promote_temp( file )
    return(result)
 
def _trimBadRowsCols(df):
    df = df[df.columns[df.apply(lambda x: sum(x.values == None) != len(x), axis=0).tolist()]]
    df = df[df.apply(lambda x: sum(x.values == None) != len(x), axis=1).tolist()]
    return(df)
   
def importWS(sourceFile, headers = False, tabName = None, trim = True):
    warnings.simplefilter("ignore")
    wb = openpyxl.load_workbook(sourceFile)
    warnings.simplefilter("default")
   
    if( tabName is None ):
        ws = wb.active
    else:
        ws = wb[tabName]
   
    #load the data
    df = DataFrame(ws.values)
   
    if( trim ):
        df = _trimBadRowsCols( df )
    if( headers ):
        df.columns = df.iloc[0]
        df = df.reindex(df.reset_index().index.drop(0))
       
    return( df )
   
def promote_temp( tmp_file_loc ):
    if( os.path.exists( tmp_file_loc ) and 'tmp' in tmp_file_loc ):
        file_loc = tmp_file_loc.replace( '\\tmp', '' )
        shutil.copy( tmp_file_loc, file_loc)
        os.remove( tmp_file_loc )
    return( 1 )

