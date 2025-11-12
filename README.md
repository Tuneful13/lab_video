1.

./install.sh -y

2. Meter las carpetas en mega.pytorch

3. Hacer:
conda activate MEGA

4. Potenciales comentarios


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

### Ejecutamos lo siguiente:

*python demo/demo.py base configs/vid_R_101_C4_1x.yaml R_101.pth --suffix ".JPEG"     --visualize-path datasets/image_folder --output-video*
