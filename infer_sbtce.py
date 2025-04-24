import os
import shutil
import argparse
import nibabel as nib
from spatial_brain_tumor_concentration_estimation import estimateTumorConcentration


def convert_tumorseg_labels(seg_dir):
    os.makedirs("tmp", exist_ok=True)
    temp_dir = "tmp/tumorseg_134.nii.gz"
    
    seg = nib.load(seg_dir)
    aff, header = seg.affine, seg.header
    seg_data = seg.get_fdata()
    
    # SBTC:         1: non_enhancing, 3: edema, 4: enhancing
    # BRATS (new):  1: non_enhancing, 2: edema, 3: enhancing
    seg_data[(seg_data == 2) | (seg_data == 3)] += 1
    seg_new = nib.Nifti1Image(seg_data, aff, header)
    nib.save(seg_new, temp_dir)
    
    return temp_dir


if __name__ == "__main__":

    tumorSegmentationPath = "/mlcube_io0/Patient-00000/00000-tumorseg.nii.gz"
    wmPath = "/mlcube_io0/Patient-00000/00000-wm.nii.gz"
    gmPath = "/mlcube_io0/Patient-00000/00000-gm.nii.gz"
    csfPath = "/mlcube_io0/Patient-00000/00000-csf.nii.gz"

    savePath = "tmp"
    tumorSegmentationPath_134 = convert_tumorseg_labels(tumorSegmentationPath)

    # Run without PET/recurrence
    estimateTumorConcentration.estimateBrainTumorConcentration(tumorSegmentationPath_134, wmPath, gmPath, csfPath, savePath)

    # Copy to mlcubeio1 and cleanup
    pred_file = os.path.join(savePath, "tumorImage.nii.gz")
    shutil.copy(pred_file, "/mlcube_io1/00000.nii.gz")

    shutil.rmtree(savePath)
    
