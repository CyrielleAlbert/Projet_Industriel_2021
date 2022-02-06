from utils_measurements import run_measurements
from make_fragments import run as run_frag
from register_fragments import run as run_reg
from refine_registration import run as run_ref
from integrate_scene import run as run_int
import json 
import math
import argparse

parser = argparse.ArgumentParser(description="Reconstruction 3D et mesures par photogrammétrie")

parser.add_argument('--configPath',help='Chemin vers le fichier config de l acquisition')

parser.add_argument('--reconstruction',help='Booléen définition si il y a execution ou non de la reconstruction', action='store_true')

args= vars(parser.parse_args())

config_path = args["configPath"]

with open(config_path,'r') as file:
    config = json.load(file)

RGB_PATH = config["path_dataset"]+"image/"
DEPTH_PATH = config["path_dataset"]+"depth/"
ANGLE_VUE = config["ANGLE_VUE"]*math.pi/180
       
run_measurements(RGB_PATH,DEPTH_PATH,ANGLE_VUE)
print(args)
if args['reconstruction']==True:
    run_frag(config)
    run_reg(config)
    run_ref(config)
    #run_int(config) #ne fonctionne pas car peu de données
