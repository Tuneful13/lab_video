
1.Abrir una terminal donde tengas el archivo install.sh y ejecutar:

```
chmod +x install.sh
./install.sh
```

2. Ejecutar:
```
conda init
```

3. Abrir otra terminal en la carpeta mega.pytorch y ejecutar:
```
conda activate MEGA
```
4. Descargar los modelos de drive y pegamos los dos archivos .pth en la carpeta de /mega.pytorch:

https://drive.google.com/file/d/1W17f9GC60rHU47lUeOEfU--Ra-LTw3Tq/view

https://drive.google.com/file/d/1ZnAdFafF1vW9Lnpw-RPF1AD_csw61lBY/view

5. Descargar la carpeta de moodle (image_folder.zip) y la descomprimimos en mega.pytorch/datasets. 

4. ### Ejecutamos lo siguiente:

4.1.1. Para base:
```
python demo/demo.py base configs/vid_R_101_C4_1x.yaml R_101.pth --suffix ".JPEG" --visualize-path datasets/image_folder --output-folder visualization/base --output-video
```
4.1.2. For frames:
```
python demo/demo.py base configs/vid_R_101_C4_1x.yaml R_101.pth --suffix ".JPEG" --visualize-path datasets/image_folder --output-folder visualization/base
```


4.2.1. Para mega:
```
python demo/demo.py mega configs/MEGA/vid_R_101_C4_MEGA_1x.yaml MEGA_R_101.pth --suffix ".JPEG" --visualize-path datasets/image_folder --output-folder visualization/mega --output-video
```

4.2.2. For frames:
```
python demo/demo.py mega configs/MEGA/vid_R_101_C4_MEGA_1x.yaml MEGA_R_101.pth --suffix ".JPEG" --visualize-path datasets/image_folder --output-folder visualization/mega
```


(Estos son los cambios que hemos hecho)

### mega.pytorch/mega_core/layers/nms.py:

De:

```
(5) from apex import amp

(8) nms = amp.float_function(_C.nms)
```
A esto:

```
(5) #from apex import amp

(8) nms = _C.nms
```

### mega_core/layers/roi_align.py:

De:

```
(10) from apex import amp

(57) @amp.float_function
```

A esto:
```
(10) # from apex import amp

(57) #@amp.float_function
```

### mega_core/layers/roi_pool.py:

De:

```
(10) from apex import amp

(56) @amp.float_function
```

A esto:
```
(10) # from apex import amp

(56) # @amp.float_function
```

### demo/predictor.py:

De:

```
(611)                image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
```

A esto:
```
(611)                image, s, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
```

## Parte 2


Para poder meter como input un video poner la FLAG --video

Seleccionamos videos de UCF101 y analizamos



### v_WalkingWithDog_g10_c03.avi

```
python demo/demo.py base configs/vid_R_101_C4_1x.yaml R_101.pth --suffix ".JPEG" --visualize-path datasets/session_2/v_WalkingWithDog_g10_c03.avi --output-folder visualization/session_2/walking_dog_g10/base --output-video --video

python demo/demo.py base configs/vid_R_101_C4_1x.yaml R_101.pth --suffix ".JPEG" --visualize-path datasets/session_2/v_WalkingWithDog_g10_c03.avi --output-folder visualization/session_2/walking_dog_g10/base --video

python demo/demo.py mega configs/MEGA/vid_R_101_C4_MEGA_1x.yaml MEGA_R_101.pth --suffix ".JPEG" --visualize-path datasets/session_2/v_WalkingWithDog_g10_c03.avi --output-folder visualization/session_2/walking_dog_g10/mega --output-video --video

python demo/demo.py mega configs/MEGA/vid_R_101_C4_MEGA_1x.yaml MEGA_R_101.pth --suffix ".JPEG" --visualize-path datasets/session_2/v_WalkingWithDog_g10_c03.avi --output-folder visualization/session_2/walking_dog_g10/mega --video

```

### v_WalkingWithDog_g01_c01.avi

```
python demo/demo.py base configs/vid_R_101_C4_1x.yaml R_101.pth --suffix ".JPEG" --visualize-path datasets/session_2/v_WalkingWithDog_g01_c01.avi --output-folder visualization/session_2/walking_dog_g01/base --output-video --video

python demo/demo.py base configs/vid_R_101_C4_1x.yaml R_101.pth --suffix ".JPEG" --visualize-path datasets/session_2/v_WalkingWithDog_g01_c01.avi --output-folder visualization/session_2/walking_dog_g01/base --video

python demo/demo.py mega configs/MEGA/vid_R_101_C4_MEGA_1x.yaml MEGA_R_101.pth --suffix ".JPEG" --visualize-path datasets/session_2/v_WalkingWithDog_g01_c01.avi --output-folder visualization/session_2/walking_dog_g01/mega --output-video --video

python demo/demo.py mega configs/MEGA/vid_R_101_C4_MEGA_1x.yaml MEGA_R_101.pth --suffix ".JPEG" --visualize-path datasets/session_2/v_WalkingWithDog_g01_c01.avi --output-folder visualization/session_2/walking_dog_g01/mega --video

```

### v_HorseRiding_g10_c01.avi

```
python demo/demo.py base configs/vid_R_101_C4_1x.yaml R_101.pth --suffix ".JPEG" --visualize-path datasets/session_2/v_HorseRiding_g10_c01.avi --output-folder visualization/session_2/HorseRiding/base --output-video --video

python demo/demo.py base configs/vid_R_101_C4_1x.yaml R_101.pth --suffix ".JPEG" --visualize-path datasets/session_2/v_HorseRiding_g10_c01.avi --output-folder visualization/session_2/HorseRiding/base --video

python demo/demo.py mega configs/MEGA/vid_R_101_C4_MEGA_1x.yaml MEGA_R_101.pth --suffix ".JPEG" --visualize-path datasets/session_2/v_HorseRiding_g10_c01.avi --output-folder visualization/session_2/HorseRiding/mega --output-video --video

python demo/demo.py mega configs/MEGA/vid_R_101_C4_MEGA_1x.yaml MEGA_R_101.pth --suffix ".JPEG" --visualize-path datasets/session_2/v_HorseRiding_g10_c01.avi --output-folder visualization/session_2/HorseRiding/mega --video

```

### internet.avi

```
python demo/demo.py base configs/vid_R_101_C4_1x.yaml R_101.pth --suffix ".JPEG" --visualize-path datasets/session_2/internet.avi --output-folder visualization/session_2/internet/base --output-video --video

python demo/demo.py base configs/vid_R_101_C4_1x.yaml R_101.pth --suffix ".JPEG" --visualize-path datasets/session_2/internet.avi --output-folder visualization/session_2/internet/base --video

python demo/demo.py mega configs/MEGA/vid_R_101_C4_MEGA_1x.yaml MEGA_R_101.pth --suffix ".JPEG" --visualize-path datasets/session_2/internet.avi --output-folder visualization/session_2/internet/mega --output-video --video

python demo/demo.py mega configs/MEGA/vid_R_101_C4_MEGA_1x.yaml MEGA_R_101.pth --suffix ".JPEG" --visualize-path datasets/session_2/internet.avi --output-folder visualization/session_2/internet/mega --video

```
