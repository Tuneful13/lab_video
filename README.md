1: ./install.sh

2: Meter las carpetas en mega.pytorch

3: Hacer:
conda activate MEGA

4: Potenciales comentarios


### mega.pytorch/mega_core/layers/nms.py:

De:

```
from mega_core import _C
from apex import amp

# Only valid with fp32 inputs - give AMP the hint
nms = amp.float_function(_C.nms)
```
A esto:

```
from mega_core import _C
#from apex import amp

# Only valid with fp32 inputs - give AMP the hint
#nms = amp.float_function(_C.nms)
nms = _C.nms
```

### mega_core/layers/roi_align.py:

De:

```
from apex import amp
@amp.float_function
```

A esto:
```
# from apex import amp
#@amp.float_function
```

### mega_core/layers/roi_pool.py:

De:

```
from apex import amp
@amp.float_function
```

A esto:
```
# from apex import amp
# @amp.float_function
```

### demo/predictor.py:

De:

```
image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
```

A esto:
```
image, s, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
```

### Ejecutamos lo siguiente:

*python demo/demo.py base configs/vid_R_101_C4_1x.yaml R_101.pth --suffix ".JPEG"     --visualize-path datasets/image_folder --output-video*
