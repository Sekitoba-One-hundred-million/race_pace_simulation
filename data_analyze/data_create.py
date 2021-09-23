import numpy as np
from tqdm import tqdm

import sekitoba_library as lib
import sekitoba_data_manage as dm

dm.dl.file_set( "susuka_simu_data.pickle" )
dm.dl.file_set( "falcon_simu_data.pickle" )
dm.dl.file_set( "race_info.pickle" )
dm.dl.file_set( "race_cource_info.pickle" )

def main():
    result = {}
    
    suzuka_data = dm.dl.data_get( "susuka_simu_data.pickle" )
    falcon_data = dm.dl.data_get( "falcon_simu_data.pickle" )
    race_info = dm.dl.data_get( "race_info_data.pickle" )
    race_cource_info = dm.dl.data_get( "race_cource_info.pickle" )

    for k in tqdm( suzuka_data.keys() ):
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

        try:
            current_straight = suzuka_data[race_id]
            current_corner = falcon_data[race_id]
        except:
            continue

        result[race_id] = []
        #直線とコーナーで順にデータを挿入
        for kind in rci_info:
            if kind == "s":
                result[race_id].append( np.array( current_straight[s], dtype = np.float32 ) )
                s += 1
            else:
                result[race_id].append( np.array( current_corner[c], dtype = np.float32 ) )
                c += 1
                
            #print( rci_info, kind, s, c, len( current_straight ), len( current_corner ) )
                
    return result              
