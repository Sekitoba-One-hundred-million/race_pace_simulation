import math
import random
import numpy as np
from tqdm import tqdm

import SekitobaLibrary as lib
import SekitobaDataManage as dm

def teacher_stand( data, state = "test" ):
    data_list = []
    """
    for answer_key in data["answer"][0].keys():
        data_list.clear()
        
        for i in range( 0, len( data["answer"] ) ):
            data_check = lib.testYearCheck( data["year"][i], state )

            if not data_check == "teacher":
                continue

            data_list.append( data["answer"][i][answer_key] )

        ave_data = lib.average( data_list )
        std_data = lib.stdev( data_list )

        for i in range( 0, len( data["answer"] ) ):
            value = data["answer"][i][answer_key]

            if not value == lib.base_abort:
                data["answer"][i][answer_key] = ( value - ave_data ) / std_data
    """
    for r in tqdm( range( 0, len( data["teacher"][0] ) ) ):
        data_list.clear()
        
        for i in range( 0, len( data["teacher"] ) ):
            data_check = lib.testYearCheck( data["year"][i], state )

            if not data_check == "teacher":
                continue

            data_list.append( data["teacher"][i][r] )

        ave_data = lib.average( data_list )
        std_data = lib.stdev( data_list )

        if std_data == 0:
            continue

        for i in range( 0, len( data["teacher"] ) ):
            value = data["teacher"][i][r]

            if not value == lib.base_abort:
                data["teacher"][i][r] = ( value - ave_data ) / std_data
            else:
                data["teacher"][i][r] = 0
            

def data_check( data, answer_key, state = "test" ):
    result = {}
    result["teacher"] = []
    result["test_teacher"] = []
    result["answer"] = []
    result["test_answer"] = []
    index_list = list( range( 0, len( data["teacher"] ) ) )
    random.shuffle( index_list )

    for i in index_list:        
        current_data = data["teacher"][i]
        answer_pace = data["answer"][i][answer_key]
        data_check = lib.testYearCheck( data["year"][i], state )

        if data_check == "test":
            result["test_teacher"].append( current_data )
            result["test_answer"].append( answer_pace )
        elif data_check == "teacher":
            result["teacher"].append( current_data )
            result["answer"].append( answer_pace )
            
    return result

def score_check( data, \
                models, \
                answer_key, \
                result, \
                score_years = lib.score_years ):
    predict_data = []

    for model in models:
        predict_data.append( model.predict( np.array( data["teacher"] ) ) )

    #_, min_line_data, max_line_data = out_answer_index_create( data["answer"], data["year"], answer_key )
    score = 0
    count = 0
    
    for i in range( 0, len( predict_data[0] ) ):
        race_id = data["race_id"][i]
        year = race_id[0:4]
        lib.dicAppend( result, race_id, {} )
        p_data = 0

        for r in range( 0, len( predict_data ) ):
            p_data += predict_data[r][i]

        p_data /= len( predict_data )
        result[race_id][answer_key] = p_data

        #if p_data < min_line_data or max_line_data < p_data:
        #    continue

        if year in lib.score_years:
            score += math.pow( p_data - data["answer"][i][answer_key], 2 )
            count += 1

    score /= count
    score = math.sqrt( score )
    print( "{} score: {}".format( answer_key, score ) )
    return score
