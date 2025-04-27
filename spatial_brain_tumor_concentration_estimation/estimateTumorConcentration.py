import torch
import torch.nn as nn
import torch.optim as optim
import nibabel as nib
import wandb
from . import tools
import numpy as np
import os


def train(config):

    with wandb.init(config=config) as run:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("used Device: ", device)

        tumorSeg = nib.load(config["tumorSegPath"]).get_fdata()
        affine = nib.load(config["tumorSegPath"]).affine
        wm = nib.load(config["wmSegPath"]).get_fdata()
        gm = nib.load(config["gmSegPath"]).get_fdata()
        csf = nib.load(config["csfSegPath"]).get_fdata()
        if config["recurrencePath"] == "":
            recurrence = np.zeros_like(tumorSeg)
        else:
            recurrence = np.round(nib.load(config["recurrencePath"]).get_fdata())

        if config["petImagePath"] == "":
            petImageNumpy = np.zeros_like(tumorSeg)
        else:
            necrotic = np.zeros_like(tumorSeg)
            necrotic[tumorSeg == 4] = 1
            petImageNumpy = nib.load(config["petImagePath"]).get_fdata()
            petImageNumpy[necrotic > 0.001] = 0

        recurrenceCore = np.zeros_like(recurrence)
        recurrenceCore[recurrence == 1] = 1
        recurrenceCore[recurrence == 4] = 1
        recurrenceAll = np.zeros_like(recurrence)
        recurrenceAll[recurrence > 0] = 1

        tumorCore= np.zeros_like(tumorSeg)
        tumorCore[tumorSeg == 1] = 1
        tumorCore[tumorSeg == 4] = 1 

        tumorEdema = np.zeros_like(tumorSeg)
        tumorEdema[tumorSeg > 0] = 1

        image_shape = (1, 1, tumorSeg.shape[0], tumorSeg.shape[1], tumorSeg.shape[2])

        coreTarget = torch.zeros(image_shape)
        coreTarget[0, 0] [tumorCore>0] = 1.0

        edemaTarget = torch.zeros(image_shape)
        edemaTarget[0, 0][tumorEdema>0] = 1.0

        tissueThreshold = 0.1
        brainMaskNumpy = (csf + wm + gm) > tissueThreshold

        wmBinary = np.logical_and(wm >= gm, brainMaskNumpy)
        wmBinary[tumorSeg > 0] = True
        gmBinary = brainMaskNumpy.copy()
        gmBinary[wmBinary] = False

        wmTarget = torch.zeros(image_shape)
        wmTarget[0, 0][wmBinary] = 1.0

        gmTarget = torch.zeros(image_shape)
        gmTarget[0, 0][gmBinary] = 1.0

        coreTarget = coreTarget.to(device)
        edemaTarget = edemaTarget.to(device)
        wmTarget = wmTarget.to(device)
        gmTarget = gmTarget.to(device)

        if config["lambda_lossPET"] > 0:
            petImage = torch.tensor(petImageNumpy).unsqueeze(0).unsqueeze(0).to(device)
        
        brainMask = torch.tensor(brainMaskNumpy).unsqueeze(0).unsqueeze(0).to(device)

        standardPlan = tools.create_standard_plan(tumorCore, config["standardPlanDistance"])
        standardPlan[brainMaskNumpy == False] = 0
        standardPlanVolume = np.sum(standardPlan)
        standardRecurrencePlanCoverage = tools.getRecurrenceCoverage(recurrenceCore, standardPlan)
        standardRecurrencePlanCoverageAll = tools.getRecurrenceCoverage(recurrenceAll, standardPlan)

        steepnessFactor = torch.tensor(0.1).to(device).requires_grad_(True)

        thresholT1_optimize = torch.tensor(config["threshold_t1c"]).to(device).requires_grad_(True)
        thresholFlair_optimize = torch.tensor(config["threshold_flair"]).to(device).requires_grad_(True)
        
        loss_softDice_edema = tools.SoftDiceLoss(default_threshold = config["threshold_flair"])
        loss_softDice_core = tools.SoftDiceLoss(default_threshold = config["threshold_t1c"])
        waveFrontLoss_function = tools.WaveFrontLossFirstOrder(wmTarget, gmTarget)
        gradientLoss_function = tools.GradientLoss()
        lossPetFunction = tools.PetLoss()

        tumorImage = (edemaTarget.clone().detach()* config["threshold_t1c"] + coreTarget.clone().detach()*  config["threshold_flair"] + 0.01 * brainMask.clone().detach()).to(device).requires_grad_(True) 

        optimizer = optim.Adam([tumorImage, steepnessFactor], lr=config["learning_rate"])

        recurrenceCoverage = 0
        recurrenceCoverageAll = 0
        tumorThreshold = 0.2

        zeroTensor = torch.tensor([0.0]).to(device)

        # Step 4: Training loop
        for step in range(config["num_epochs"]): 
            optimizer.zero_grad()  
            
            # Apply sigmoid to constrain the output between 0 and 1
            tumorImage_activated = torch.clamp(tumorImage * brainMask, min=0.00001, max=1.0)  # Apply ReLU to constrain the output between 0 and 1
            steepnessFactor_activated = torch.clamp(steepnessFactor, min=0.00001, max=100.0)

            thresholdFlair_optimize_activated = torch.clamp(thresholFlair_optimize, min=0.0001 , max=0.5)
            thresholT1_optimize_activated = torch.clamp(thresholT1_optimize, min=0.5 , max=1)

            diceEdema = 1 - loss_softDice_edema(tumorImage_activated, edemaTarget, threshold = thresholdFlair_optimize_activated)
            diceCore = 1 - loss_softDice_core(tumorImage_activated, coreTarget, threshold = thresholT1_optimize_activated)

            lossEdema = (1- diceEdema ) * config["lambda_diceEdema"]
            lossCore = (1- diceCore) * config["lambda_diceCore"]

            if config["lambda_physics"] == 0:
                lossPhysics = zeroTensor
            else:
                lossPhysics = waveFrontLoss_function(tumorImage_activated, constantFactor = steepnessFactor_activated, mask = brainMask) * config["lambda_physics"]

            if config["lambda_gradient"] == 0:
                gradientLoss = zeroTensor
            else:
                gradientLoss = gradientLoss_function(tumorImage_activated, brainMask) * config["lambda_gradient"]

            if config["lambda_lossPET"] == 0:
                lossPet = zeroTensor
            else:
                lossPet = lossPetFunction(tumorImage_activated, petImage) * config["lambda_lossPET"]

            loss =  lossEdema + lossCore +  lossPhysics +  gradientLoss +  lossPet

            loss.backward()
            optimizer.step()

            volume = tumorImage_activated.sum() / edemaTarget.sum()

            logDict = {"loss": loss.item(), "volume": volume.item(), "lossEdema": lossEdema.item(), "lossCore": lossCore.item(), "steepnessFactor": steepnessFactor_activated.item(), "lossPhysics": lossPhysics.item(), "lossGradient": gradientLoss.item(), "lossPET": lossPet, "thresholdFlair": thresholdFlair_optimize_activated.item(), "thresholdT1c": thresholT1_optimize_activated.item()}

            if step % 10 == 0:
                print(f"Step {step}, Loss: {loss.item()}")
                numpyImage = tumorImage_activated.clone().detach().cpu().numpy()
                numpyCoreTarget = coreTarget.clone().detach().cpu().numpy()
                centerOfMask = np.unravel_index(np.argmax(numpyCoreTarget), numpyCoreTarget.shape)
                z = centerOfMask[4]
                logDict ["image"] = wandb.Image(numpyImage[0,0,:,:,z])
                _, physicsLoss = waveFrontLoss_function(tumorImage_activated, constantFactor = steepnessFactor_activated, returnVoxelwise=True)
                logDict["voxelWiseLoss"] = wandb.Image(torch.abs(physicsLoss.clone()).detach().cpu().numpy()[0,0,:,:,z])

            if step % 100 == 0 and step >= 10: 
                tumorNumpy = tumorImage_activated.clone().detach().cpu().numpy()[0,0]
                tumorThreshold = tools.find_threshold(tumorNumpy, standardPlanVolume, initial_threshold= tumorThreshold)
                recurrenceCoverage = tools.getRecurrenceCoverage(recurrenceCore , tumorNumpy > tumorThreshold)
                recurrenceCoverageAll = tools.getRecurrenceCoverage(recurrenceAll , tumorNumpy > tumorThreshold)


            logDict["recurrenceCoverage"] = recurrenceCoverage
            logDict["recurrenceCoverageAll"] = recurrenceCoverageAll
            logDict["recurrenceCoverage_improvement"] = recurrenceCoverage - standardRecurrencePlanCoverage
            logDict["recurrenceCoverageAll_improvement"] = recurrenceCoverageAll - standardRecurrencePlanCoverageAll
            logDict["tumorThreshold"] = tumorThreshold
            wandb.log(logDict)  

        #save the image
        savePathPatient = config["savePath"]
        if not savePathPatient[-1] == "/":
            savePathPatient += "/"
        os.makedirs(savePathPatient, exist_ok=True)
        tumorNumpy = tumorImage_activated.clone().detach().cpu().numpy()[0,0]
        tumorThreshold = tools.find_threshold(tumorNumpy, standardPlanVolume, initial_threshold= tumorThreshold)
        
        nib.save(nib.Nifti1Image((tumorNumpy > tumorThreshold) * 1.0, affine), savePathPatient + "/recurrencePrediction.nii.gz")
        nib.save(nib.Nifti1Image(standardPlan * 1.0, affine), savePathPatient + "/standardPlan.nii.gz")
        nib.save(nib.Nifti1Image(tumorNumpy, affine), savePathPatient + "/tumorImage.nii.gz")
        nib.save(nib.Nifti1Image(brainMask.detach().cpu().numpy()[0,0] * 1.0, affine), savePathPatient + "/brainMask.nii.gz")
        
        #_, physicsLoss = waveFrontLoss_function(tumorImage_activated, constantFactor = steepnessFactor_activated, returnVoxelwise=True, mask = None)
        #nib.save(nib.Nifti1Image(torch.abs(physicsLoss.clone()).detach().cpu().numpy()[0,0], affine), savePathPatient + "/voxelWiseLoss.nii.gz")
        

def estimateBrainTumorConcentration(tumorSegmentationPath, wmPath, gmPath, csfPath, savePath, petImagePath = "", recurrencePath = ""):

    parametersDict = {
            "tumorSegPath": tumorSegmentationPath,
            "wmSegPath": wmPath,
            "gmSegPath": gmPath,
            "csfSegPath": csfPath,
            "recurrencePath": recurrencePath,
            "petImagePath": petImagePath,
            "savePath": savePath,
            'learning_rate': 0.01,
            'lambda_physics': 0,  # was 1000
            'lambda_diceEdema': 1,
            'lambda_diceCore': 1,
            'lambda_gradient': 1000,
            'lambda_lossPET': 0 if petImagePath == "" else 1,
            'num_epochs': 500,
            'standardPlanDistance': 15,
            "threshold_flair": 0.2,
            "threshold_t1c": 0.6
            }
    train(parametersDict)


if __name__ == "__main__":

    tumorSegmentationPath = "data/data_001/segm.nii.gz"
    wmPath = "./data/data_001/t1_wm.nii.gz"
    gmPath = "./data/data_001/t1_gm.nii.gz"
    csfPath = "./data/data_001/t1_csf.nii.gz"
    petImagePath = "data/data_001/FET.nii.gz"

    recurrencePath = "data/data_001/segm_rec.nii.gz"

    savePath = "./results/estimateTumorConcentration/"

    estimateBrainTumorConcentration(tumorSegmentationPath, wmPath, gmPath, csfPath, petImagePath, savePath, recurrencePath)
