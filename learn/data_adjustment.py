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
        
        if year in lib.test_years:
            result["test_teacher"].append( current_data )
            result["test_answer"].append( answer_pace )
        else:
            result["teacher"].append( current_data )
            result["answer"].append( answer_pace )

    return result

def score_check( data, model, upload = False ):
    result = {}
    predict_data = model.predict( np.array( data["teacher"] ) )
    score = 0
    acc = 0
    
    for i in range( 0, len( predict_data ) ):
        result[data["race_id"][i]] = predict_data[i]

        if data["year"][i] in lib.test_years:
            score += math.pow( predict_data[i] - data["answer"][i], 2 )

    score /= len( predict_data )
    score = math.sqrt( score )
    print( "score: {}".format( score ) )

    if upload:
        dm.pickle_upload( "predict_pace_data.pickle", result )

    return score
