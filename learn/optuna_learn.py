import json
import optuna
import numpy as np
import lightgbm as lgb

from learn import data_adjustment
import sekitoba_library as lib
import sekitoba_data_manage as dm

simu_data = {}
use_data = {}
answer_key = ""

def objective( trial ):
    lgb_train = lgb.Dataset( np.array( use_data["teacher"] ), np.array( use_data["answer"] ) )
    lgb_vaild = lgb.Dataset( np.array( use_data["test_teacher"] ), np.array( use_data["test_answer"] ) )

    learning_rate = trial.suggest_float( 'learning_rate', 0.01, 0.1 )
    num_leaves =  trial.suggest_int( "num_leaves", 10, 100 )
    max_depth = trial.suggest_int( "max_depth", 10, 100 )
    num_iteration = trial.suggest_int( "num_iteration", 20, 200 )
    min_data_in_leaf = trial.suggest_int( "min_data_in_leaf", 1, 50 )
    lambda_l1 = trial.suggest_float( "lambda_l1", 0, 1 )
    lambda_l2 = trial.suggest_float( "lambda_l2", 0, 1 )

    lgbm_params =  {
        #'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression_l2',
        'metric': 'l2',
        'early_stopping_rounds': 30,
        'learning_rate': learning_rate,
        'num_iteration': num_iteration,
        'min_data_in_bin': 1,
        'max_depth': max_depth,
        'num_leaves': num_leaves,
        'min_data_in_leaf': min_data_in_leaf,
        'lambda_l1': lambda_l1,
        'lambda_l2': lambda_l2
    }

    model = lgb.train( params = lgbm_params,
                     train_set = lgb_train,     
                     valid_sets = [lgb_train, lgb_vaild ],
                     verbose_eval = 10,
                     num_boost_round = 5000 )

    return data_adjustment.score_check( simu_data, [ model ], answer_key, {}, score_years = lib.score_years )

def main( data ):
    global use_data
    global simu_data
    global answer_key
    #data_adjustment.teacher_stand( data, state = "optuna" )
    simu_data = data
    
    for key in lib.predict_pace_key_list:
        answer_key = key
        param_list = []
        use_data = data_adjustment.data_check( data, answer_key, state = "optuna" )

        for i in range( 0, 5 ):
            study = optuna.create_study()
            study.optimize(objective, n_trials=100)
            param_list.append( study.best_params )

        f = open( "{}_best_params.json".format( answer_key ), "w" )
        json.dump( param_list, f )
        f.close()
