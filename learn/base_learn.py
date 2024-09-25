import os
import json
import math
import numpy as np
import lightgbm as lgb

from learn import data_adjustment
import SekitobaLibrary as lib
import SekitobaDataManage as dm
#from learn import simulation

def lg_main( data, answer_key, index = None ):
    params = {}
    file_name = "{}_best_params.json".format( answer_key )
    
    if os.path.isfile( file_name ) and not index == None:
        f = open( file_name, "r" )
        params = json.load( f )[index]
        f.close()
    else:
        params["learning_rate"] = 0.01
        params["num_iteration"] = 10000
        params["max_depth"] = 200
        params["num_leaves"] = 175
        params["min_data_in_leaf"] = 25
        params["lambda_l1"] = 0
        params["lambda_l2"] = 0

    lgb_train = lgb.Dataset( np.array( data["teacher"] ), np.array( data["answer"] ) )
    lgb_vaild = lgb.Dataset( np.array( data["test_teacher"] ), np.array( data["test_answer"] ) )

    lgbm_params =  {
        #'task': 'train',
        "random_state": 50,
        'boosting_type': 'gbdt',
        'objective': 'regression_l2',
        'metric': 'l2',
        'early_stopping_rounds': 30,
        'learning_rate': params["learning_rate"],
        'num_iteration': params["num_iteration"],
        'min_data_in_bin': 1,
        'max_depth': params["max_depth"],
        'num_leaves': params["num_leaves"],
        'min_data_in_leaf': params["min_data_in_leaf"],
        'lambda_l1': params["lambda_l1"],
        'lambda_l2': params["lambda_l2"]
    }

    bst = lgb.train( params = lgbm_params,
                     train_set = lgb_train,     
                     valid_sets = [lgb_train, lgb_vaild ],
                     verbose_eval = 10,
                     num_boost_round = 5000 )
        
    return bst

def importance_check( model, file_name ):
    result = []
    importance_data = model.feature_importance()
    f = open( "common/rank_score_data.txt" )
    all_data = f.readlines()
    f.close()
    c = 0

    for i in range( 0, len( all_data ) ):
        str_data = all_data[i].replace( "\n", "" )

        if "False" in str_data:
            continue

        result.append( { "key": str_data, "score": importance_data[c] } )
        c += 1

    result = sorted( result, key = lambda x: x["score"], reverse= True )

    wf = open( file_name, "w" )

    for i in range( 0, len( result ) ):
        wf.write( "{}: {}\n".format( result[i]["key"], result[i]["score"] ) )        

def main( data, state = "test" ):
    model_result = {}
    result = {}
    #data_adjustment.teacher_stand( data, state = state )
    
    for answer_key in lib.predict_pace_key_list:
        learn_data = data_adjustment.data_check( data, answer_key, state = state )
        lib.dicAppend( model_result, answer_key, [] )

        for i in range( 0, 5 ):
            model_result[answer_key].append( lg_main( learn_data, answer_key, index = i ) )

    for answer_key in lib.predict_pace_key_list:
        data_adjustment.score_check( data, model_result[answer_key], answer_key, result, score_years = lib.simu_years )
        importance_check( model_result[answer_key][0], "{}_importance.txt".format( answer_key ) )

    dm.pickle_upload( "predict_pace_data.pickle", result )
    dm.pickle_upload( lib.name.model_name(), model_result )
