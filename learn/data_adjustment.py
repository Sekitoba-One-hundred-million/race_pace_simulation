import math
import numpy as np

import sekitoba_library as lib
import sekitoba_data_manage as dm

def data_check( data ):
    result = {}
    result["teacher"] = []
    result["test_teacher"] = []
    result["answer"] = []
    result["test_answer"] = []

    for i in range( 0, len( data["teacher"] ) ):
        current_data = data["teacher"][i]
        current_answer = data["answer"][i]
        answer_pace = current_answer
        year = data["year"][i]

        if ( not lib.prod_check and year in lib.valid_years ) or ( lib.prod_check and year in lib.test_years):
            result["test_teacher"].append( current_data )
            result["test_answer"].append( answer_pace )
        else:
            result["teacher"].append( current_data )
            result["answer"].append( answer_pace )

    return result

def score_check( data, model, score_years = lib.score_years, upload = False ):
    result = {}
    predict_data = model.predict( np.array( data["teacher"] ) )
    score = 0
    count = 0
    
    for i in range( 0, len( predict_data ) ):
        race_id = data["race_id"][i]
        year = race_id[0:4]
        result[race_id] = predict_data[i]

        if ( ( not lib.prod_check and year in score_years ) \
             or ( lib.prod_check and year in lib.test_years ) ):
            score += math.pow( predict_data[i] - data["answer"][i], 2 )
            count += 1

    score /= count
    score = math.sqrt( score )
    print( "score: {}".format( score ) )

    if upload:
        dm.pickle_upload( "predict_pace_data.pickle", result )

    return score
