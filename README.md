# Spatial Brain Tumor Concentration Estimation
 
With this tool, you can estimate the spatial brain tumor concentration based on MR and PET Images.

![image](https://github.com/user-attachments/assets/5f0c01f9-f82b-4188-8e4c-466af7c15028)

We optimize (blue) a 3D scalar tumor concentration estimation (yellow) by simultaneously fitting the data and regularizing on physical properties. Using this predicted tumor concentration (orange), we propose a radiotherapy plan (Clinical Target Volume (CTV), orange). We evaluate (green) our method’s ability to capture areas with later tumor recurrence.


## How it works
### Input:
- Brain tumor semgentations
- Brain tissue probability maps: white matter, gray matter, cerebrospinal fluid 
- FET-PET (optional)

### Output:
- continuous brain tumor cell distribution
- standard radiotherapy plan (CTV) with a 15mm margin around the tumor core, estimating the Estroeano guideline [1].
- proposed radiotherapy plan (CTV) with the same volume as the standard radiotherapy plan

## How to use:
    ## Installation:
    Install the required packages with:
    ```pip install -r requirements.txt```

    ## Tutorial:
    The tutorial is available in the ```tutorial.ipynb``` file.

    ## Settings:
    Advanced settings can be adjusted in the ```estimateTumorConcentration.py``` file.

## Data:
We included some example data in "data". This is the first patient of the GliODIL dataset also used in our paper. The data is already preprocessed and can be used to test the tool. The desired results are also included in the "results" folder.

The full dataset can be found at huggingface.co/datasets/m1balcerak/GliODIL (https://github.com/m1balcerak/GliODIL).

### Data format:

Segmentations follow the BraTS toolkit segmentation convention [1], with voxel values representing different tumor regions: 1.0 for necrotic core, 3.0 for edema, and 4.0 for enhancing core. The command creates a subdirectory with results in a given directory. If you do not have brain tissue segmentation, you can use the s3 tool (https://github.com/JanaLipkova/s3 or github.com/andeleyev/brain-tumor-tissue-reconstruction).

# Cite
If you use this tool, please cite the following paper:
```
@article{weidner2024spatial,
  title={Spatial Brain Tumor Concentration Estimation for Individualized Radiotherapy Planning},
  author={Weidner, Jonas and Balcerak, Michal and Ezhov, Ivan and Datchev, Andr{\'e} and Lux, Laurin and Rueckert, Lucas Zimmer and Daniel and Menze, Bj{\"o}rn and Wiestler, Benedikt},
  journal={arXiv preprint arXiv:2412.13811},
  year={2024}
}
```

# References

[1] Niyazi, M., Andratschke, N., Bendszus, M., Chalmers, A.J., Erridge, S.C., Galldiks, N., Lagerwaard, F.J., Navarria, P., af Rosenschöld, P.M., Ricardi, U., et al.: Estroeano guideline on target delineation and radiotherapy details for glioblastoma. Radiotherapy and Oncology 184, 109663 (2023)


[2] Kofler, F., Berger, C., Waldmannstetter, D., Lipkova, J., Ezhov, I., Tetteh, G., Kirschke, J., Zimmer, C., Wiestler, B., & Menze, B. H. (2020). BraTS toolkit: translating BraTS brain tumor segmentation algorithms into clinical and scientific practice. Frontiers in neuroscience, 125.


# Examples
![image](https://github.com/user-attachments/assets/3221091f-f384-493a-88ec-0d5b3cbfaf07)

Demonstration of our method on example patients. In the first row, we show the two input MR images with the tumor and the recurrence, that should be covered. Edema is shown in blue, enhancing tumor in green, and necrotic in red. Our method predicts a continuous assumption of tumor cells, as shown in the second row. This continuous concentration is thresholded to have the same volume as the standard plan (grey) to create the CTV (orange). In the last row, we compare our method to the standard plan for different patients. The same result was obtained for additional patients from the RHUH dataset.
