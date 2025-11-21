# MEGA Video Object Detection Lab Readme

This guide provides instructions to set up the environment, apply necessary code modifications, download required models/data, and execute the **BASE** and **MEGA** detection pipelines.

---

## 1. Setup and Environment Activation

### A. Installation

1.  Open a terminal in the directory containing the `install.sh` file and run the following commands:
    ```bash
    chmod +x install.sh
    ./install.sh
    ```
2.  Initialize Conda for environment activation:
    ```bash
    conda init
    ```
3.  Open a new terminal in the `mega.pytorch` directory and activate the created environment:
    ```bash
    conda activate MEGA
    ```

### B. Download Models and Data

1.  **Download Pre-trained Models:** Download the following two model files (`.pth`) from Google Drive and place them directly into the `/mega.pytorch` folder.
    * `BASE`: [https://drive.google.com/file/d/1W17f9GC60rHU47lUeOEfU--Ra-LTw3Tq/view]
    * `MEGA`: [https://drive.google.com/file/d/1ZnAdFafF1vW9Lnpw-RPF1AD_csw61lBY/view]
2.  **Download Session 1 Data:** Download the `image_folder.zip` file from Moodle and unzip its contents into the **`mega.pytorch/datasets`** folder.

---

## 2. Necessary Code Changes (Fixing Apex Compatibility)

The original repository requires modifications to run on modern systems due to incompatible PyTorch and Apex versions. **The following changes are already applied in the `install.sh` file:**

### `mega.pytorch/mega_core/layers/nms.py`

| Original (Lines 5, 8) | Modified |
| :--- | :--- |
| `(5) from apex import amp` | `(5) #from apex import amp` |
| `(8) nms = amp.float_function(_C.nms)` | `(8) nms = _C.nms` |

### `mega_core/layers/roi_align.py`

| Original (Lines 10, 57) | Modified |
| :--- | :--- |
| `(10) from apex import amp` | `(10) # from apex import amp` |
| `(57) @amp.float_function` | `(57) #@amp.float_function` |

### `mega_core/layers/roi_pool.py`

| Original (Lines 10, 56) | Modified |
| :--- | :--- |
| `(10) from apex import amp` | `(10) # from apex import amp` |
| `(56) @amp.float_function` | `(56) # @amp.float_function` |

### `demo/predictor.py` (Line 611)

| Original (Line 611) | Modified (For integer coordinates) |
| :--- | :--- |
| `image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2` | `image, s, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2` |

---

## 3. Execution (Session 1: Image Folder)

Execute the following commands from the activated `MEGA` environment inside the `mega.pytorch` directory.

### 3.1 BASE Detector (Single-Frame)

| Output Type | Command |
| :--- | :--- |
| **With Video Render** | `python demo/demo.py base configs/vid_R_101_C4_1x.yaml R_101.pth --suffix ".JPEG" --visualize-path datasets/image_folder --output-folder visualization/base --output-video` |
| **Frames Only** | `python demo/demo.py base configs/vid_R_101_C4_1x.yaml R_101.pth --suffix ".JPEG" --visualize-path datasets/image_folder --output-folder visualization/base` |

### 3.2 MEGA Detector (Memory-Enhanced)

| Output Type | Command |
| :--- | :--- |
| **With Video Render** | `python demo/demo.py mega configs/MEGA/vid_R_101_C4_MEGA_1x.yaml MEGA_R_101.pth --suffix ".JPEG" --visualize-path datasets/image_folder --output-folder visualization/mega --output-video` |
| **Frames Only** | `python demo/demo.py mega configs/MEGA/vid_R_101_C4_MEGA_1x.yaml MEGA_R_101.pth --suffix ".JPEG" --visualize-path datasets/image_folder --output-folder visualization/mega` |

---

## 4. Execution (Session 2: Video Input)

To process a video file instead of an image folder, use the **`--video`** flag and specify the path to the `.avi` file in `--visualize-path`.

The following commands use the specified UCF101 videos for analysis. Ensure these video files (e.g., `v_WalkingWithDog_g10_c03.avi`) are placed inside the `datasets/session_2/` folder.

### A. Video: `v_WalkingWithDog_g10_c03.avi`

| Model | Output Type | Command |
| :--- | :--- | :--- |
| **BASE** | Video Render | `python demo/demo.py base configs/vid_R_101_C4_1x.yaml R_101.pth --suffix ".JPEG" --visualize-path datasets/session_2/v_WalkingWithDog_g10_c03.avi --output-folder visualization/session_2/walking_dog_g10/base --output-video --video` |
| **BASE** | Frames Only | `python demo/demo.py base configs/vid_R_101_C4_1x.yaml R_101.pth --suffix ".JPEG" --visualize-path datasets/session_2/v_WalkingWithDog_g10_c03.avi --output-folder visualization/session_2/walking_dog_g10/base --video` |
| **MEGA** | Video Render | `python demo/demo.py mega configs/MEGA/vid_R_101_C4_MEGA_1x.yaml MEGA_R_101.pth --suffix ".JPEG" --visualize-path datasets/session_2/v_WalkingWithDog_g10_c03.avi --output-folder visualization/session_2/walking_dog_g10/mega --output-video --video` |
| **MEGA** | Frames Only | `python demo/demo.py mega configs/MEGA/vid_R_101_C4_MEGA_1x.yaml MEGA_R_101.pth --suffix ".JPEG" --visualize-path datasets/session_2/v_WalkingWithDog_g10_c03.avi --output-folder visualization/session_2/walking_dog_g10/mega --video` |

### B. Video: `v_WalkingWithDog_g01_c01.avi`

| Model | Output Type | Command |
| :--- | :--- | :--- |
| **BASE** | Video Render | `python demo/demo.py base configs/vid_R_101_C4_1x.yaml R_101.pth --suffix ".JPEG" --visualize-path datasets/session_2/v_WalkingWithDog_g01_c01.avi --output-folder visualization/session_2/walking_dog_g01/base --output-video --video` |
| **BASE** | Frames Only | `python demo/demo.py base configs/vid_R_101_C4_1x.yaml R_101.pth --suffix ".JPEG" --visualize-path datasets/session_2/v_WalkingWithDog_g01_c01.avi --output-folder visualization/session_2/walking_dog_g01/base --video` |
| **MEGA** | Video Render | `python demo/demo.py mega configs/MEGA/vid_R_101_C4_MEGA_1x.yaml MEGA_R_101.pth --suffix ".JPEG" --visualize-path datasets/session_2/v_WalkingWithDog_g01_c01.avi --output-folder visualization/session_2/walking_dog_g01/mega --output-video --video` |
| **MEGA** | Frames Only | `python demo/demo.py mega configs/MEGA/vid_R_101_C4_MEGA_1x.yaml MEGA_R_101.pth --suffix ".JPEG" --visualize-path datasets/session_2/v_WalkingWithDog_g01_c01.avi --output-folder visualization/session_2/walking_dog_g01/mega --video` |

### C. Video: `v_HorseRiding_g10_c01.avi`

| Model | Output Type | Command |
| :--- | :--- | :--- |
| **BASE** | Video Render | `python demo/demo.py base configs/vid_R_101_C4_1x.yaml R_101.pth --suffix ".JPEG" --visualize-path datasets/session_2/v_HorseRiding_g10_c01.avi --output-folder visualization/session_2/HorseRiding/base --output-video --video` |
| **BASE** | Frames Only | `python demo/demo.py base configs/vid_R_101_C4_1x.yaml R_101.pth --suffix ".JPEG" --visualize-path datasets/session_2/v_HorseRiding_g10_c01.avi --output-folder visualization/session_2/HorseRiding/base --video` |
| **MEGA** | Video Render | `python demo/demo.py mega configs/MEGA/vid_R_101_C4_MEGA_1x.yaml MEGA_R_101.pth --suffix ".JPEG" --visualize-path datasets/session_2/v_HorseRiding_g10_c01.avi --output-folder visualization/session_2/HorseRiding/mega --output-video --video` |
| **MEGA** | Frames Only | `python demo/demo.py mega configs/MEGA/vid_R_101_C4_MEGA_1x.yaml MEGA_R_101.pth --suffix ".JPEG" --visualize-path datasets/session_2/v_HorseRiding_g10_c01.avi --output-folder visualization/session_2/HorseRiding/mega --video` |

### D. Video: `internet.avi`

| Model | Output Type | Command |
| :--- | :--- | :--- |
| **BASE** | Video Render | `python demo/demo.py base configs/vid_R_101_C4_1x.yaml R_101.pth --suffix ".JPEG" --visualize-path datasets/session_2/internet.avi --output-folder visualization/session_2/internet/base --output-video --video` |
| **BASE** | Frames Only | `python demo/demo.py base configs/vid_R_101_C4_1x.yaml R_101.pth --suffix ".JPEG" --visualize-path datasets/session_2/internet.avi --output-folder visualization/session_2/internet/base --video` |
| **MEGA** | Video Render | `python demo/demo.py mega configs/MEGA/vid_R_101_C4_MEGA_1x.yaml MEGA_R_101.pth --suffix ".JPEG" --visualize-path datasets/session_2/internet.avi --output-folder visualization/session_2/internet/mega --output-video --video` |
| **MEGA** | Frames Only | `python demo/demo.py mega configs/MEGA/vid_R_101_C4_MEGA_1x.yaml MEGA_R_101.pth --suffix ".JPEG" --visualize-path datasets/session_2/internet.avi --output-folder visualization/session_2/internet/mega --video` |
