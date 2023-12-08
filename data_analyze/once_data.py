import math
import copy
import sklearn
from tqdm import tqdm
from mpi4py import MPI
from statistics import stdev

import sekitoba_library as lib
import sekitoba_data_manage as dm

from sekitoba_data_create.time_index_get import TimeIndexGet
from sekitoba_data_create.before_race_score_get import BeforeRaceScore

from common.name import Name

data_name = Name()

dm.dl.file_set( "race_data.pickle" )
dm.dl.file_set( "race_info_data.pickle" )
dm.dl.file_set( "baba_index_data.pickle" )
dm.dl.file_set( "horce_data_storage.pickle" )
dm.dl.file_set( "race_day.pickle" )
dm.dl.file_set( "race_jockey_id_data.pickle" )
dm.dl.file_set( "race_trainer_id_data.pickle" )
dm.dl.file_set( "true_skill_data.pickle" )
dm.dl.file_set( "first_passing_true_skill_data.pickle" )
dm.dl.file_set( "race_money_data.pickle" )
dm.dl.file_set( "wrap_data.pickle" )
dm.dl.file_set( "predict_netkeiba_pace_data.pickle" )

class OnceData:
    def __init__( self ):
        self.race_data = dm.dl.data_get( "race_data.pickle" )
        self.race_info = dm.dl.data_get( "race_info_data.pickle" )
        self.baba_index_data = dm.dl.data_get( "baba_index_data.pickle" )
        self.horce_data = dm.dl.data_get( "horce_data_storage.pickle" )
        self.race_day = dm.dl.data_get( "race_day.pickle" )
        self.race_jockey_id_data = dm.dl.data_get( "race_jockey_id_data.pickle" )
        self.race_trainer_id_data = dm.dl.data_get( "race_trainer_id_data.pickle" )
        self.true_skill_data = dm.dl.data_get( "true_skill_data.pickle" )
        self.first_passing_true_skill_data = dm.dl.data_get( "first_passing_true_skill_data.pickle" )
        self.race_money_data = dm.dl.data_get( "race_money_data.pickle" )
        self.wrap_data = dm.dl.data_get( "wrap_data.pickle" )
        self.predict_netkeiba_pace_data = dm.dl.data_get( "predict_netkeiba_pace_data.pickle" )
        
        self.time_index = TimeIndexGet()
        self.before_race_score = BeforeRaceScore()
        
        self.data_name_list = []
        self.write_data_list = []
        self.result = { "answer": [], "teacher": [], "year": [], "race_id": [] }
        self.data_name_read()

    def data_name_read( self ):
        f = open( "common/list.txt", "r" )
        str_data_list = f.readlines()

        for str_data in str_data_list:
            self.data_name_list.append( str_data.replace( "\n", "" ) )

    def score_write( self ):
        f = open( "./common/rank_score_data.txt", "w" )

        for data_name in self.write_data_list:
            f.write( data_name + "\n" )

        f.close()

    def data_list_create( self, data_dict ):
        result = []
        write_instance = []
        
        for data_name in self.data_name_list:
            try:
                result.append( data_dict[data_name] )
                write_instance.append( data_name )
            except:
                continue

        if len( self.write_data_list ) == 0:
            self.write_data_list = copy.deepcopy( write_instance )

        return result

    def clear( self ):
        dm.dl.data_clear()
    
    def create( self, k ):
        race_id = lib.id_get( k )
        year = race_id[0:4]
        race_place_num = race_id[4:6]
        day = race_id[9]
        num = race_id[7]

        key_place = str( self.race_info[race_id]["place"] )
        key_dist = str( self.race_info[race_id]["dist"] )
        key_kind = str( self.race_info[race_id]["kind"] )      
        key_baba = str( self.race_info[race_id]["baba"] )
        ymd = { "y": int( year ), "m": self.race_day[race_id]["month"], "d": self.race_day[race_id]["day"] }

        #芝かダートのみ
        if key_kind == "0" or key_kind == "3":
            return

        if not race_id in self.race_money_data:
            return

        predict_netkeiba_pace = -1

        if race_id in self.predict_netkeiba_pace_data:
            predict_netkeiba_pace = lib.netkeiba_pace( self.predict_netkeiba_pace_data[race_id] )
        
        money_class = int( lib.money_class_get( self.race_money_data[race_id] ) )
        key_race_money_class = str( money_class )
        teacher_data = []
        answer_data = []
        diff_data = []

        pace = lib.pace_data( self.wrap_data[race_id] )
        
        if pace == None:
            return

        escape_limb_count = 0
        insert_limb_count = 0
        one_popular_limb = -1
        two_popular_limb = -1
        three_popular_limb = -1
        one_popular_odds = -1
        two_popular_odds = -1
        three_popular_odds = -1
        
        current_race_data = {}
        for data_key in self.data_name_list:
            current_race_data[data_key] = []

        for horce_id in self.race_data[k].keys():
            current_data, past_data = lib.race_check( self.horce_data[horce_id],
                                                     year, day, num, race_place_num )#今回と過去のデータに分ける
            cd = lib.current_data( current_data )
            pd = lib.past_data( past_data, current_data )

            if not cd.race_check():
                continue

            limb_math = lib.limb_search( pd )
            before_cd = pd.before_cd()
            before_diff = -1000
            before_first_passing_rank = -1000
            before_last_passing_rank = -1000
            before_id_weight = -1000
            before_popular = -1000
            before_rank = -1000
            before_speed = -1000
            before_race_score = -1000

            if not before_cd == None:
                before_diff = before_cd.diff()
                before_passing_rank = before_cd.passing_rank()
                before_id_weight = before_cd.id_weight()
                before_popular = before_cd.popular()
                before_rank = before_cd.rank()
                before_speed = before_cd.speed()
                before_race_score = self.before_race_score.score_get( before_cd, limb_math, horce_id )
                
                try:
                    before_first_passing_rank = int( before_passing_rank[0] )
                except:
                    pass

                try:
                    before_last_passing_rank = int( before_passing_rank[-1] )
                except:
                    pass

            jockey_id = ""
            trainer_id = ""
            
            try:
                jockey_id = self.race_jockey_id_data[race_id][horce_id]
            except:
                pass

            try:
                trainer_id = self.race_trainer_id_data[race_id][horce_id]
            except:
                pass

            if limb_math == 1 or limb_math == 2:
                escape_limb_count += 1
            elif limb_math == 3 or limb_math == 4:
                insert_limb_count += 1

            popular = cd.popular()
            odds = cd.odds()

            if popular == 1:
                one_popular_limb = limb_math
                one_popular_odds = odds
            elif popular == 2:
                two_popular_limb = limb_math
                two_popular_odds = odds
            elif popular == 3:
                three_popular_limb = limb_math
                three_popular_odds = odds

            horce_true_skill = 25
            jockey_true_skill = 25
            trainer_true_skill = 25
            horce_first_passing_true_skill = 25
            jockey_first_passing_true_skill = 25
            trainer_first_passing_true_skill = 25

            if race_id in self.true_skill_data["horce"] and \
              horce_id in self.true_skill_data["horce"][race_id]:
                horce_true_skill = self.true_skill_data["horce"][race_id][horce_id]

            if race_id in self.true_skill_data["jockey"] and \
              jockey_id in self.true_skill_data["jockey"][race_id]:
                jockey_true_skill = self.true_skill_data["jockey"][race_id][jockey_id]

            if race_id in self.true_skill_data["trainer"] and \
              trainer_id in self.true_skill_data["trainer"][race_id]:
                trainer_true_skill = self.true_skill_data["trainer"][race_id][trainer_id]
            
            if race_id in self.first_passing_true_skill_data["horce"] and \
              horce_id in self.first_passing_true_skill_data["horce"][race_id]:
                horce_first_passing_true_skill = self.first_passing_true_skill_data["horce"][race_id][horce_id]

            if race_id in self.first_passing_true_skill_data["jockey"] and \
              jockey_id in self.first_passing_true_skill_data["jockey"][race_id]:
                jockey_first_passing_true_skill = self.first_passing_true_skill_data["jockey"][race_id][jockey_id]
                
            if race_id in self.first_passing_true_skill_data["trainer"] and \
              trainer_id in self.first_passing_true_skill_data["trainer"][race_id]:
                trainer_first_passing_true_skill = self.first_passing_true_skill_data["trainer"][race_id][trainer_id]

            current_time_index = self.time_index.main( horce_id, pd.past_day_list() )
            speed, up_speed, pace_speed = pd.speed_index( self.baba_index_data[horce_id] )
            current_race_data[data_name.race_horce_true_skill].append( horce_true_skill )
            current_race_data[data_name.race_jockey_true_skill].append( jockey_true_skill )
            current_race_data[data_name.race_trainer_true_skill].append( trainer_true_skill )
            current_race_data[data_name.race_horce_first_passing_true_skill].append( horce_first_passing_true_skill )
            current_race_data[data_name.race_jockey_first_passing_true_skill].append( jockey_first_passing_true_skill )
            current_race_data[data_name.race_trainer_first_passing_true_skill].append( trainer_first_passing_true_skill )
            current_race_data[data_name.race_up_rate].append( pd.up_rate( key_race_money_class ) )
            current_race_data[data_name.race_speed_index].append( lib.max_check( speed ) + current_time_index["max"] )
            current_race_data[data_name.race_before_diff].append( before_diff )
            current_race_data[data_name.race_before_first_passing_rank].append( before_first_passing_rank )
            current_race_data[data_name.race_before_last_passing_rank].append( before_last_passing_rank )
            current_race_data[data_name.race_before_id_weight].append( before_id_weight )
            current_race_data[data_name.race_before_popular].append( before_popular )
            current_race_data[data_name.race_before_race_score].append( before_race_score )
            current_race_data[data_name.race_before_rank].append( before_rank )
            current_race_data[data_name.race_before_speed].append( before_speed )

        if len( current_race_data[data_name.race_up_rate] ) < 2:
            return

        N = len( current_race_data[data_name.race_up_rate] )

        t_instance = {}
        t_instance[data_name.all_horce_num] = N
        t_instance[data_name.place] = int( key_place )
        t_instance[data_name.baba] = int( key_baba )
        t_instance[data_name.dist] = int( key_dist )
        t_instance[data_name.kind] = int( key_kind )
        t_instance[data_name.place] = int( key_place )
        t_instance[data_name.money_class] = money_class
        t_instance[data_name.escape_limb_count] = escape_limb_count
        t_instance[data_name.insert_limb_count] = insert_limb_count
        t_instance[data_name.one_popular_limb] = one_popular_limb
        t_instance[data_name.two_popular_limb] = two_popular_limb
        t_instance[data_name.three_popular_limb] = three_popular_limb
        t_instance[data_name.one_popular_odds] = one_popular_odds
        t_instance[data_name.two_popular_odds] = two_popular_odds
        t_instance[data_name.three_popular_odds] = three_popular_odds
        t_instance[data_name.predict_netkeiba_pace] = predict_netkeiba_pace
            
        for data_key in current_race_data.keys():
            if not type( current_race_data[data_key] ) is list or \
              len( current_race_data[data_key] ) == 0:
                continue

            t_instance["ave_"+data_key] = sum( current_race_data[data_key] ) / N
            t_instance["max_"+data_key] = max( current_race_data[data_key] )
            t_instance["min_"+data_key] = min( current_race_data[data_key] )
            t_instance["std_"+data_key] = stdev( current_race_data[data_key] )

        t_instance[data_name.std_race_horce_true_skill] = stdev( current_race_data[data_name.race_horce_true_skill] )
        t_list = self.data_list_create( t_instance )

        self.result["answer"].append( pace )
        self.result["teacher"].append( t_list )
        self.result["year"].append( year )
        self.result["race_id"].append( race_id )
