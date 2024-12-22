import math
import copy
from tqdm import tqdm
from mpi4py import MPI

import SekitobaLibrary as lib
import SekitobaDataManage as dm
import SekitobaPsql as ps

from SekitobaDataCreate.time_index_get import TimeIndexGet
from SekitobaDataCreate.before_race_score_get import BeforeRaceScore
from SekitobaDataCreate.stride_ablity import StrideAblity
from SekitobaDataCreate.get_horce_data import GetHorceData
from SekitobaDataCreate.odds_cluster import OddsCluster
from SekitobaDataCreate.kinetic_energy import KineticEnergy

from common.name import Name

data_name = Name()

dm.dl.file_set( "race_cource_info.pickle" )
dm.dl.file_set( "race_pace_analyze_data.pickle" )

class OnceData:
    def __init__( self ):
        self.race_data = ps.RaceData()
        self.race_horce_data = ps.RaceHorceData()
        self.horce_data = ps.HorceData()

        self.kinetic_energy = KineticEnergy( self.race_data )
        self.stride_ablity = StrideAblity( self.race_data )
        self.time_index = TimeIndexGet( self.horce_data )
        self.before_race_score = BeforeRaceScore( self.race_data )
        self.race_cource_info = dm.dl.data_get( "race_cource_info.pickle" )
        self.race_pace_analyze_data = dm.dl.data_get( "race_pace_analyze_data.pickle" )
        self.jockey_judgement_param_list = [ "limb", "popular", "flame_num", "dist", "kind", "baba", "place" ]
        
        self.data_name_list = []
        self.write_data_list = []
        self.result = { "answer": [], "teacher": [], "year": [], "race_id": [], "ave": [] }
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
        data_key_list = []

        if not len( self.write_data_list ) == 0:
            data_key_list = copy.deepcopy( self.write_data_list )
        else:
            data_key_list = list( data_dict.keys() )
        
        for data_name in data_key_list:
            result.append( data_dict[data_name] )

        if len( self.write_data_list ) == 0:
            self.write_data_list = copy.deepcopy( data_key_list )

        return result

    def clear( self ):
        dm.dl.data_clear()
    
    def create( self, race_id ):
        self.race_data.get_all_data( race_id )
        self.race_horce_data.get_all_data( race_id )

        if len( self.race_horce_data.horce_id_list ) == 0:
            return

        self.horce_data.get_multi_data( self.race_horce_data.horce_id_list )
        year = race_id[0:4]
        race_place_num = race_id[4:6]
        day = race_id[9]
        num = race_id[7]

        key_place = str( self.race_data.data["place"] )
        key_dist = str( self.race_data.data["dist"] )
        key_kind = str( self.race_data.data["kind"] )      
        key_baba = str( self.race_data.data["baba"] )
        ymd = { "year": self.race_data.data["year"], \
               "month": self.race_data.data["month"], \
               "day": self.race_data.data["day"] }

        #芝かダートのみ
        if key_kind == "0" or key_kind == "3":
            return

        predict_netkeiba_pace = lib.netkeibaPace( self.race_data.data["predict_netkeiba_pace"] )
        money_class = int( lib.moneyClassGet( self.race_data.data["money"] ) )
        key_race_money_class = str( money_class )
        teacher_data = []
        answer_data = []
        diff_data = []
        horce_odds_list = []

        escape_limb_count = 0
        insert_limb_count = 0
        one_popular_limb = lib.escapeValue
        two_popular_limb = lib.escapeValue
        three_popular_limb = lib.escapeValue
        one_popular_odds = lib.escapeValue
        two_popular_odds = lib.escapeValue
        three_popular_odds = lib.escapeValue
        first_straight_dist = lib.escapeValue
        last_straight_dist = lib.escapeValue

        try:
            first_straight_dist = self.race_cource_info[key_place][key_kind][key_dist]["dist"][0]
            last_straight_dist = self.race_cource_info[key_place][key_kind][key_dist]["dist"][-1]
        except:
            pass
        
        current_race_data = {}
        
        for data_key in self.data_name_list:
            current_race_data[data_key] = []

        for horce_id in self.race_horce_data.horce_id_list:
            current_data, past_data = lib.raceCheck( self.horce_data.data[horce_id]["past_data"], ymd )
            cd = lib.CurrentData( current_data )
            pd = lib.PastData( past_data, current_data, self.race_data )

            if not cd.raceCheck():
                continue

            getHorceData = GetHorceData( cd, pd )
            before_diff = getHorceData.getBeforeDiff()
            before_first_passing_rank, before_last_passing_rank  = getHorceData.getBeforePassingRank()
            before_id_weight = getHorceData.getBeforeIdWeight()
            before_popular = getHorceData.getBeforePopular()
            before_rank = getHorceData.getBeforeRank()
            before_speed = getHorceData.getBeforeSpeed()
            before_race_score = self.before_race_score.score_get( horce_id, getHorceData )
            horce_first_up3_halon = {}
            race_first_up3_ave = -1000
            race_first_up3_min = -1000
            race_first_up3_max = -1000

            try:
                horce_first_up3_halon = self.race_data.data["first_up3_halon"][str(int(cd.horceNumber()))]
            except:
                pass

            if not len( horce_first_up3_halon ) == 0:
                race_first_up3_ave = 0
                race_first_up3_min = 1000
                race_first_up3_max = -1000
                
                for k in horce_first_up3_halon.keys():
                    race_first_up3_ave += horce_first_up3_halon[k]
                    race_first_up3_min = min( race_first_up3_min, horce_first_up3_halon[k] )
                    race_first_up3_max = max( race_first_up3_max, horce_first_up3_halon[k] )

                race_first_up3_ave /= len( horce_first_up3_halon )                    
                
            if getHorceData.limb_math == 1 or getHorceData.limb_math == 2:
                escape_limb_count += 1
            elif getHorceData.limb_math == 3 or getHorceData.limb_math == 4:
                insert_limb_count += 1

            popular = cd.popular()
            odds = cd.odds()

            if popular == 1:
                one_popular_limb = getHorceData.limb_math
                one_popular_odds = odds
            elif popular == 2:
                two_popular_limb = getHorceData.limb_math
                two_popular_odds = odds
            elif popular == 3:
                three_popular_limb = getHorceData.limb_math
                three_popular_odds = odds

            horce_odds_list.append( { "horce_id": horce_id, "odds": odds } )
            horce_true_skill = self.race_horce_data.data[horce_id]["horce_true_skill"]
            jockey_true_skill = self.race_horce_data.data[horce_id]["jockey_true_skill"]
            trainer_true_skill = self.race_horce_data.data[horce_id]["trainer_true_skill"]
            horce_first_passing_true_skill = self.race_horce_data.data[horce_id]["horce_first_passing_true_skill"]
            jockey_first_passing_true_skill = self.race_horce_data.data[horce_id]["jockey_first_passing_true_skill"]
            trainer_first_passing_true_skill = self.race_horce_data.data[horce_id]["trainer_first_passing_true_skill"]
            up3_horce_true_skill = self.race_horce_data.data[horce_id]["horce_up3_true_skill"]
            up3_jockey_true_skill = self.race_horce_data.data[horce_id]["jockey_up3_true_skill"]
            up3_trainer_true_skill = self.race_horce_data.data[horce_id]["trainer_up3_true_skill"]
            current_time_index = self.time_index.main( horce_id, pd.pastDayList() )
            speed, up_speed, pace_speed = pd.speedIndex( self.horce_data.data[horce_id]["baba_index"] )
            pace_up_rate = pd.pace_up_rate()
            stride_ablity_data = self.stride_ablity.ablity_create( cd, pd )
            past_min_first_horce_body, past_max_first_horce_body, past_ave_first_horce_body, past_std_first_horce_body = \
              getHorceData.getFirstHorceBody()

            for stride_data_key in stride_ablity_data.keys():
                current_race_data["race_"+stride_data_key].append( stride_ablity_data[stride_data_key] )

            for param in self.jockey_judgement_param_list:
                current_race_data["jockey_judgment_{}".format( param )].append(
                    self.race_horce_data.data[horce_id]["jockey_judgment"][param] )

            for pace_up_rate_key in pace_up_rate.keys():
                current_race_data[data_name.race_pace_up_rate+"_"+pace_up_rate_key].append( pace_up_rate[pace_up_rate_key] )

            current_race_data[data_name.race_horce_true_skill].append( horce_true_skill )
            current_race_data[data_name.race_jockey_true_skill].append( jockey_true_skill )
            current_race_data[data_name.race_trainer_true_skill].append( trainer_true_skill )
            current_race_data[data_name.race_horce_first_passing_true_skill].append( horce_first_passing_true_skill )
            current_race_data[data_name.race_jockey_first_passing_true_skill].append( jockey_first_passing_true_skill )
            current_race_data[data_name.race_trainer_first_passing_true_skill].append( trainer_first_passing_true_skill )
            current_race_data[data_name.race_up3_horce_true_skill].append( up3_horce_true_skill )
            current_race_data[data_name.race_up3_jockey_true_skill].append( up3_jockey_true_skill )
            current_race_data[data_name.race_up3_trainer_true_skill].append( up3_trainer_true_skill )
            current_race_data[data_name.race_up_rate].append(
                pd.up_rate( key_race_money_class, self.race_data.data["up_kind_ave"] ) )
            current_race_data[data_name.race_speed_index].append( lib.maxCheck( speed ) + current_time_index["max"] )
            current_race_data[data_name.race_up_speed_index].append( lib.maxCheck( up_speed ) )
            current_race_data[data_name.race_pace_speed_index].append( lib.maxCheck( pace_speed ) )
            current_race_data[data_name.race_before_diff].append( before_diff )
            current_race_data[data_name.race_before_first_passing_rank].append( before_first_passing_rank )
            current_race_data[data_name.race_before_last_passing_rank].append( before_last_passing_rank )
            current_race_data[data_name.race_before_id_weight].append( before_id_weight )
            current_race_data[data_name.race_before_popular].append( before_popular )
            current_race_data[data_name.race_before_race_score].append( before_race_score )
            current_race_data[data_name.race_before_rank].append( before_rank )
            current_race_data[data_name.race_before_speed].append( before_speed )
            current_race_data[data_name.race_match_up3].append( pd.matchUp3() )
            current_race_data[data_name.race_level_score].append( pd.level_score( self.race_data.data["money_class_true_skill"] ) )
            current_race_data[data_name.race_level_up3].append( pd.level_up3( self.race_data.data["money_class_true_skill"] ) )
            current_race_data[data_name.race_past_min_first_horce_body].append( past_min_first_horce_body )
            current_race_data[data_name.race_past_max_first_horce_body].append( past_max_first_horce_body )
            current_race_data[data_name.race_past_ave_first_horce_body].append( past_ave_first_horce_body )
            current_race_data[data_name.race_past_std_first_horce_body].append( past_std_first_horce_body )
            current_race_data[data_name.race_stamina].append( pd.stamina_create( getHorceData.key_limb ) )
            current_race_data[data_name.corner_diff_rank_ave].append( pd.corner_diff_rank() )
            current_race_data[data_name.race_first_up3_ave].append( race_first_up3_ave )
            current_race_data[data_name.race_first_up3_min].append( race_first_up3_min )
            current_race_data[data_name.race_first_up3_max].append( race_first_up3_max )
            current_race_data[data_name.race_kinetic_energy].append( self.kinetic_energy.create( cd, pd ) )

        N = len( current_race_data[data_name.race_up_rate] )

        if N < 2:
            return

        cluster_data = [0] * 4
        oddsCluster = OddsCluster( horce_odds_list )
        oddsCluster.clustering()

        for cl in oddsCluster.cluster.values():
            cluster_data[int(cl-1)] += 1

        t_instance = {}
        t_instance[data_name.all_horce_num] = N
        t_instance[data_name.place] = self.race_data.data["place"]
        t_instance[data_name.baba] = self.race_data.data["baba"]
        t_instance[data_name.dist] = self.race_data.data["dist"]
        t_instance[data_name.kind] = self.race_data.data["kind"]
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
        t_instance[data_name.first_straight_dist] = first_straight_dist
        t_instance[data_name.last_straight_dist] = last_straight_dist
        t_instance[data_name.odds_cluster_1] = cluster_data[0]
        t_instance[data_name.odds_cluster_2] = cluster_data[1]
        t_instance[data_name.odds_cluster_3] = cluster_data[2]
        t_instance[data_name.odds_cluster_4] = cluster_data[3]
        t_instance.update( lib.paceTeacherAnalyze( current_race_data ) )
        answer_data = {}
        one_hudred_pace = lib.oneHundredPace( self.race_data.data["wrap"] )

        if not type( one_hudred_pace ) == list:
            return

        ave_data = {}
        answer_data["pace"] = round( lib.paceData( self.race_data.data["wrap"] ), 1 )
        answer_data["pace_regression"], answer_data["before_pace_regression"], answer_data["after_pace_regression"] = \
          lib.paceRegression( one_hudred_pace )
        answer_data["pace_conv"] = lib.conv( one_hudred_pace )
        answer_data["first_up3"] = sum( one_hudred_pace[0:6] )
        answer_data["last_up3"] = sum( one_hudred_pace[int(len(one_hudred_pace)-6):len(one_hudred_pace)] )

        for k in answer_data.keys():
            answer_data[k] -= self.race_pace_analyze_data[key_kind][key_dist][k]
            ave_data[k] = self.race_pace_analyze_data[key_kind][key_dist][k]

        t_list = self.data_list_create( t_instance )

        self.result["answer"].append( answer_data )
        self.result["teacher"].append( t_list )
        self.result["year"].append( year )
        self.result["race_id"].append( race_id )
        self.result["ave"].append( ave_data )
