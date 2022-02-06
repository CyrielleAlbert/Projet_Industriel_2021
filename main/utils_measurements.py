import numpy as np
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from PIL import Image,ImageEnhance
from os import listdir
import math
import matplotlib.patches as patches
from tensorflow.python.keras.preprocessing.image import img_to_array, load_img

def _load_midas_model(model_type):
    midas = torch.hub.load("intel-isl/MiDaS",model_type)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform
    return device,midas,transform

def _estimate_depth_from_RGB_image(img,transform,device,midas):
    input_batch = transform(img).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    output = prediction.cpu().numpy()
    return img, output


#Mesure distance Find the index of the end of the brosse
def _find_bottom_index(depth_img,window_size):
    max_std_index_window = [0,0]
    max_std_value = 0
    for i in range(depth_img.shape[0]-window_size[0]):
        for j in range(int(depth_img.shape[1]/window_size[1])):
            i_min , i_max = i, min(depth_img.shape[0],i+window_size[0])
            j_min,j_max = j*window_size[1],min(depth_img.shape[1],(j+1)*window_size[1])
            window_info = depth_img[i_min:i_max,j_min:j_max]
            std_value = np.std(window_info,axis=0)
            if np.amax(std_value) > max_std_value:
                max_std_value = np.amax(std_value)
                max_std_index_window = [i,j]
    return (max_std_index_window,max_std_value)

# inverse profondeur
def _rescale_depth_image(depth_img):
    depth_img_rescale = depth_img.copy()
    min_depth = np.amin(depth_img)
    max_depth = np.amax(depth_img)
    mediane = max_depth - min_depth
    depth_img_rescale = -depth_img_rescale+2*mediane
    return depth_img_rescale

#Remise à l'échelle réelle
def _calibrate_depth_image(img1,depth_img_rescale,angle_ouv_cam):
    height,width = depth_img_rescale.shape

    #Détection du repère 
    img = Image.fromarray(img1)
    img_ycbcr = np.array(img.convert("YCbCr"))
    img_cr = img_ycbcr[int(2/3*height):,:,2].copy()
    depth_image_calib = depth_img_rescale.copy()
    half_depth_img = depth_image_calib[int(2/3*height):,:]

    AC = half_depth_img[img_cr==np.max(img_cr)][0] #Distance prof Caméra-Scotch rouge
    AB = half_depth_img[-1,int(width/2)] #Distance prof Caméra-Sol
    OA_norm = 60 #Hauteur caméra
    OC_norm = 370 #Distance Bas téléphone-Scotch
    AC_norm = math.sqrt(OA_norm**2+OC_norm**2) #Distance profondeur Caméra-scotch théorique
    AB_norm = OA_norm/(math.sin(angle_ouv_cam)) #Distance prof Caméra-premier point au centre en bas théorique
    
    #Rapport de profondeur différents
    lambda_1 = AC_norm/AC
    lambda_2 = AB_norm/AB
    a = (lambda_1-lambda_2)/(AC-AB)
    b = lambda_2-a*AB
    depth_image_calibrated = a*np.power(depth_image_calib,2)+b*depth_image_calib
    return depth_image_calibrated

#Mesures par photogrammétrie
def _measure(depth_img,angle_ouv_cam):
    height,width = depth_img.shape
    window_size = [8,2] #window size pour détection du bas de la brosse
    max_std_index_window, max_std_value = _find_bottom_index(depth_img,window_size)
    i_min,j_min = max_std_index_window[0],max_std_index_window[1]*window_size[1]
    index_bas_brosse = i_min + int(window_size[0]/2)
    angle_bas_brosse = angle_ouv_cam * (i_min-height/2)/(height/2)
    prof_bas_brosse = depth_img[index_bas_brosse,int(width/2)]
    d1 = math.sin(angle_bas_brosse)*prof_bas_brosse
    h_cam = 60
    h_planche = 15+6
    h_telephone = h_cam + h_planche

    h_brosse = h_telephone - d1 #distance brosse-rail
    taille_brosse = math.tan(angle_ouv_cam)*depth_img[int(height/2),int(width/2)] + h_brosse #taille brosse
    
    return(h_brosse,taille_brosse)

def _save_depth_img(depth_image,depth_path,image_name):
        np.save(depth_path+image_name+".npy",depth_image)

    
def _estimate_depth_image(image_name,rgb_path,angle_vue,depth_path,img_size,transform,device,midas):
    img = load_img(rgb_path+image_name,target_size=img_size,interpolation='nearest')
    print(img)
    img_arr = np.array(img)
    _,depth_img = _estimate_depth_from_RGB_image(img_arr,transform,device,midas)
    depth_img_rescale = _rescale_depth_image(depth_img)
    print(img_arr.shape)
    depth_img = _calibrate_depth_image(img_arr,depth_img_rescale,angle_vue)
    _save_depth_img(depth_img,depth_path,image_name)
    return depth_img

def run_measurements(RGB_PATH,DEPTH_PATH,ANGLE_VUE):
    model_type = "DPT_Large"
    img_size = (480,640)   
    device, midas,transform = _load_midas_model(model_type)

    images_list = listdir(RGB_PATH)
    h_brosse = []
    taille_brosse = []
    for image in images_list: 
        print("Current image:",image)
        depth_img = _estimate_depth_image(image,RGB_PATH,ANGLE_VUE,DEPTH_PATH,img_size,transform,device,midas)
        hauteur, taille = _measure(depth_img,ANGLE_VUE)
        h_brosse.append(hauteur)
        taille_brosse.append(taille)
        print(h_brosse,taille_brosse)

    print("Distance Moyenne Brosse Rails:",np.mean(h_brosse), "Ecart-type distance brosse-rail:",np.std(h_brosse),)
    print("Hauteur Moyenne Brosse:",np.mean(taille_brosse),"Ecart-type hauteur brosse:",np.std(taille_brosse),)

