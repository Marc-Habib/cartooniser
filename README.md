# Cartoon Split Project

A small Python script to **cartoonise** all images in a `data/` folder, then split them into **80% train** and **20% test**, each containing:

- `real/` (the original images)  
- `cartoon/` (the cartoonised versions)

## Installation

1. **Clone** or download this repository.
2. **Install [Python 3.7+](https://www.python.org/downloads/)** if it is not already installed.
3. **Install dependencies** from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```
*(Optionally, use a virtual environment):*

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
# or venv\Scripts\activate    # Windows
pip install -r requirements.txt
```
## Usage

1. **Place** all your images in the `data/` folder (e.g., `data/image1.jpg`, `data/picture2.png`, etc.).
2. **Run** the script:

``` bash
python cartoon_split.py
```

This will:
- Take every image in `data/`.
- Shuffle them.
- Split 80% into `data/train/` and 20% into `data/test/`.
- For each split, create:
  - `real/` — A copy of the original images.
  - `cartoon/` — Cartoonised versions of the images.

### Resulting Folder Structure

After running, you’ll have:

``` bash
data/
├─ train/
│  ├─ real/
│  │  └─ <80% original images>
│  └─ cartoon/
│     └─ <80% cartoonised images>
└─ test/
   ├─ real/
   │  └─ <20% original images>
   └─ cartoon/
      └─ <20% cartoonised images>
```

## Example

Below is an example of a **before/after** using the default settings:

| **Original** | **Cartoonised** |
|--------------|-----------------|
| ![Raw Example](real.jpg) | ![Cartoon Example](cartoon.jpg) |

## Customisation

- Edit the `main(data_dir="data", train_ratio=0.8)` call at the bottom of `cartoon_split.py` to:
  - Change the input folder (`data_dir`).
  - Adjust the train/test split ratio (`train_ratio`).
- Adjust the **k-means** parameter (`k=8`) or **Canny** thresholds (`80, 150`) in `cartoonise_clean` if you want a different style.
- The morphological `cv2.morphologyEx()` step helps remove small specks. Tweak or remove it to see the difference.