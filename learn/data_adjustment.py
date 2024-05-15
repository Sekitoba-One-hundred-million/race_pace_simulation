import math
import numpy as np

import sekitoba_library as lib
import sekitoba_data_manage as dm

def data_check( data, answer_key ):
    result = {}
    result["teacher"] = []
    result["test_teacher"] = []
    result["answer"] = []
    result["test_answer"] = []

    for i in range( 0, len( data["teacher"] ) ):
        current_data = data["teacher"][i]
        answer_pace = data["answer"][i][answer_key]
        year = data["year"][i]

        if ( not lib.prod_check and year in lib.valid_years ) or ( lib.prod_check and year in lib.test_years):
            result["test_teacher"].append( current_data )
            result["test_answer"].append( answer_pace )
        else:
            result["teacher"].append( current_data )
            result["answer"].append( answer_pace )

    return result

def score_check( data, model, answer_key, result, score_years = lib.score_years ):
    predict_data = model.predict( np.array( data["teacher"] ) )
    score = 0
    count = 0
    
    for i in range( 0, len( predict_data ) ):
        race_id = data["race_id"][i]
        year = race_id[0:4]
        lib.dic_append( result, race_id, {} )
        result[race_id][answer_key] = predict_data[i]

        if ( ( not lib.prod_check and year in score_years ) \
             or ( lib.prod_check and year in lib.test_years ) ):
            score += math.pow( predict_data[i] - data["answer"][i][answer_key], 2 )
            count += 1

    score /= count
    score = math.sqrt( score )
    print( "{} score: {}".format( answer_key, score ) )

    return score
