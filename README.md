
1.
chmod +x install.sh

./install.sh -y


2. Hacer:
```
conda init
conda activate MEGA
```

3. ### Ejecutamos lo siguiente:
```
python demo/demo.py base configs/vid_R_101_C4_1x.yaml R_101.pth --suffix ".JPEG" --visualize-path datasets/image_folder --output-folder visualization/base --output-video
```

For frames:
```
python demo/demo.py base configs/vid_R_101_C4_1x.yaml R_101.pth --suffix ".JPEG" --visualize-path datasets/image_folder --output-folder visualization/base
```


4. Para mega:
```
python demo/demo.py mega configs/MEGA/vid_R_101_C4_MEGA_1x.yaml MEGA_R_101.pth --suffix ".JPEG" --visualize-path datasets/image_folder --output-folder visualization/mega --output-video
```

For frames:
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
