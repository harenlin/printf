# printf
Reference Codes - [printf: Preference Modeling Based on User Reviews with Item Images and Textual Information via Graph Learning](https://dl.acm.org/doi/10.1145/3583780.3615012)


### 0. Create a virtual environment & Set up dependencies
``` shell
python3 -m venv venv
source venv/bin/activate 
pip install -r requirements.txt
```

### 1. Download Amazon Data
``` shell
cd gcn
bash ./download_amazon.sh 
```

### 2. Preprocess data & Generate review embeddings
``` shell
python3 new_preprocess.py
```

### 3. Download corresponding images
``` shell
python3 download_image.py
```

### 4. Fine tune multi-modality encoder & Generate item embeddings (CMIM)
``` shell
cd albef
bash run_fine_tune.sh
```
P.S. Remember to download pretrained weights [4M](https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/ALBEF_4M.pth)/[14M](https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/ALBEF.pth) from [ALBEF Repo](https://github.com/salesforce/ALBEF).

### 5. Generate user embeddings (RAUM)
``` shell
cd gcn
python3 -W ignore gen_user_embedding.py 
```

### 6. Train printf (EPIM)
``` shell
python3 train.py 
```

### 7. Test printf
``` shell
python3 test.py
```

##### P.S. We have set up all default values for the arg parse, please feel free to change each argument and run different experiments and datasets.
