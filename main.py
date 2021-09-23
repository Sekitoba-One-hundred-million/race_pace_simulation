from argparse import ArgumentParser

from data_analyze import data_create
from simulation import predict

def main():
    parser = ArgumentParser()
    parser.add_argument( "-i", type=int, default = False, help = "optional" )

    i_check = parser.parse_args().i
    data = data_create.main()
    predict.main( data )
    
main()
