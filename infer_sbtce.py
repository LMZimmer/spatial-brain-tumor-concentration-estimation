import os
import glob
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
    # Example:
    # python infer_sbtce.py -cuda_device 0
    parser = argparse.ArgumentParser()
    parser.add_argument("-cuda_device", type=str, default="0", help="GPU id to run on.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    tumorSegmentationPath = "/mlcube_io0/Patient-00000/00000-tumorseg.nii.gz"
    wmPath = "mlcube_io0/Patient-00000/00000-wm.nii.gz"
    gmPath = "mlcube_io0/Patient-00000/00000-gm.nii.gz"
    csfPath = "mlcube_io0/Patient-00000/00000-csf.nii.gz"
    petImagePath = ""

    savePath = "/mlcube_io1/00000.nii.gz"
    tumorSegmentationPath_134 = convert_tumorseg_labels(tumorSegmentationPath)

    #NOTE: with new version the arguments should be (..., savePath, petImagePath, recurrencePath)
    #estimateTumorConcentration.estimateBrainTumorConcentration(tumorSegmentationPath, wmPath, gmPath, csfPath, petImagePath, savePath, recurrencePath)
    estimateTumorConcentration.estimateBrainTumorConcentration(tumorSegmentationPath_134, wmPath, gmPath, csfPath, petImagePath, savePath)

    os.remove(tumorSegmentationPath_134)
