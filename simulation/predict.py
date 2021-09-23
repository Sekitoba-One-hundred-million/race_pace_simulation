import random

import sekitoba_library as lib
import sekitoba_data_manage as dm

#100m単位のタイム
dm.dl.file_set( "suzuka_minmax_time.pickle" )
dm.dl.file_set( "falcon_minmax_time.pickle" )

def probability_check( p_data ):
    number = -1
    count = 0
    r_check = random.random()
    
    for i in range( 0, len( p_data ) ):
        count += p_data

        if r_check < count:
            number = i
            break

    return number
        
def main( data ):
    suzuka = dm.model_load( "suzuka_test_model.pickle" )
    falcon = dm.model_load( "falcon_test_model.pickle" )
    
    straight_minmax = dm.dl.data_get( "suzuka_minmax_time.pickle" )
    corner_minmax = dm.dl.data_get( "falcon_minmax_time.pickle" )
    race_info = dm.dl.data_get( "race_info_data.pickle" )

    for k in data.keys():
        race_id = k
        key_place = str( race_info[race_id]["place"] )
        key_dist = str( race_info[race_id]["dist"] )
        key_kind = str( race_info[race_id]["kind"] )        
        key_baba = str( race_info[race_id]["baba"] )
        info_key_dist = key_dist

        if race_info[race_id]["out_side"]:
            info_key_dist += "外"

        s = 0
        c = 0
        rci_info = race_cource_info[key_place][key_kind][info_key_dist]["info"]
        rci_dist = race_cource_info[key_place][key_kind][info_key_dist]["dist"]
        
        base_time = 0
        print( key_place, key_dist, key_kind, key_baba, race_id )

        for i in range( 0, len( rci_info ) ):
            if rci_info[i] == "s":
                predict_data = falcon.forward( data[k][i] ).data
                base_time = straight_minmax["min"]
            else:
                predict_data = falcon.forward( data[k][i] ).data
                base_time = corner_minmax["min"]

            p_num = probability_check( predict_data )
            current_time = p_num / 20 + base_time
            current_time *= rci_dist[i]
            base_time += current_time
            print( current_time )

        print( base_time )
        return
