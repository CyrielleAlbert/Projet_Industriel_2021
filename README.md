# Projet_Industriel_2021
Projet Industriel 2021 - Reconstruction et mesure de profil 3D par photogrammétrie

## Auteurs: 
 - Anouk Paulmier 
 - Boubou Ba
 - Joseph Couillaud
 - Cyrielle Albert

## Pré-requis :
 - Tensorflow
 - Open3D
 - Numpy 
 - OpenCV
 - Matplotlib
 - Pandas
  
## Architecture du repo :
Les images RGB sont stockées dans les sous-dossiers "/image" des dossiers ESEO1, LeMans1 et LeMans2. 
Les images Depth estimées sont enregistrées dans les sous-dossier "/depth" des dossiers ESEO1, LeMans1 et LeMans2. 
Le dossier main contient les algorithmes d'estimation des profondeurs ainsi que de reconstruction 3D. 
Les fichiers de config se trouvent dans le dossier config. 

## Démarrer :
Dans l'invité de commande, se placer à la racine du dossier main puis lancer le fichier python ```run.py```. 
Les paramètres sont les suivants :

```--configPath```: Le chemin vers le fichier de config

```--reconstruction```: Effectue la reconstruction


## Exemple :
Pour lancer les mesures sur l'acquisition LeMans1 (Technicentre - sans lumière) ainsi que la reconstruction : 
```
cd main 
python run.py  --configPath "./../config/LeMans1.json --reconstruction"

```

Pour lancer les mesures sur l'acquisition LeMans2 sans la reconstruction :
```
cd main 
python run.py  --configPath "./../config/LeMans2.json"

```

## Infos supplémentaires :
L'algorithme de reconstruction 3D a été développé par Open3D et réadapté pour nos images.
Dans le cas d'un quelconque problème, veuillez contacter Cyrielle Albert.