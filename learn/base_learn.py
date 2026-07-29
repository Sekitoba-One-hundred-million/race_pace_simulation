import os
import json
import math
import random
import numpy as np
import lightgbm as lgb

from learn import data_adjustment
import SekitobaLibrary as lib
import SekitobaDataManage as dm
#from learn import simulation

def lg_main( data, answer_key, category_index_list, index = None ):
    params = {}
    file_name = "{}_best_params.json".format( answer_key )
    
    if os.path.isfile( file_name ) and not index is None:
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

    p = list( zip( data["teacher"], data["answer"] ) )
    test_p = list( zip( data["test_teacher"], data["test_answer"] ) )
    random.shuffle( p )
    random.shuffle( test_p )
    data["teacher"], data["answer"] = zip(*p)
    data["test_teacher"], data["test_answer"] = zip(*test_p)    
    data["teacher"] = list( data["teacher"] )
    data["answer"] = list( data["answer"] )
    data["test_teacher"] = list( data["test_teacher"] )
    data["test_answer"] = list( data["test_answer"] )

    lgbm_params =  {
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
        'lambda_l2': params["lambda_l2"],
        'device_type': 'cuda'
    }

    index_list = list( range( 0, len( data["answer"] ) ) )
    random.shuffle( index_list )

    result = {}
    n_splits = 10
    n = int( len( index_list ) / n_splits + 1 )

    for i in range( 0, n_splits ):
        s = int( n * i )
        e = min( int( n * ( i + 1 ) ), len( index_list ) )
        use_index = index_list[:s] + index_list[e:]
        race_id_list = [data["race_id"][r] for r in index_list[s:e]]
        predict_teacher = [data["teacher"][r] for r in index_list[s:e]]
        use_teacher = [data["teacher"][r] for r in use_index]
        use_answer = [data["answer"][r] for r in use_index]
        lgb_train = lgb.Dataset( np.array( use_teacher ),
                                 np.array( use_answer ),
                                 categorical_feature = category_index_list )
        lgb_vaild = lgb.Dataset( np.array( data["test_teacher"] ),
                                 np.array( data["test_answer"] ),
                                 categorical_feature = category_index_list )
        bst = lgb.train( params = lgbm_params,
                         train_set = lgb_train,     
                         valid_sets = [lgb_train, lgb_vaild ],
                         num_boost_round = 5000 )
        predict_data = bst.predict( np.array( predict_teacher ) )

        for r in range( 0, len( race_id_list ) ):
            result[race_id_list[r]] = predict_data[r]

    lgb_train = lgb.Dataset( np.array( data["teacher"] ),
                             np.array( data["answer"] ),
                             categorical_feature = category_index_list )
    lgb_vaild = lgb.Dataset( np.array( data["test_teacher"] ),
                             np.array( data["test_answer"] ),
                             categorical_feature = category_index_list )

    bst = lgb.train( params = lgbm_params,
                     train_set = lgb_train,     
                     valid_sets = [lgb_train, lgb_vaild ],
                     num_boost_round = 5000 )
        
    return bst, result

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
    l = 5
    category_index_list = lib.create_category_index( data["category"] )
    
    for answer_key in lib.predict_pace_key_list:
        learn_data = data_adjustment.data_check( data, answer_key, state = state )
        lib.dic_append( model_result, answer_key, [] )

        for i in range( 0, l ):
            model, predict_data = lg_main( learn_data, answer_key, category_index_list, index = i )
            model_result[answer_key].append( model )

            for race_id in predict_data.keys():
                lib.dic_append( result, race_id, {} )
                lib.dic_append( result[race_id], answer_key, 0 )
                result[race_id][answer_key] += predict_data[race_id]

        for race_id in result.keys():
            result[race_id][answer_key] /= l

    for answer_key in lib.predict_pace_key_list:
        data_adjustment.score_check( data, model_result[answer_key], answer_key, result, score_years = lib.simu_years )
        importance_check( model_result[answer_key][0], "{}_importance.txt".format( answer_key ) )

    dm.pickle_upload( "predict_pace_data.pickle", result )
    dm.pickle_upload( lib.name.model_name(), model_result )
