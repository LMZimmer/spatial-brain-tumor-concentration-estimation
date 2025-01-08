#%%
import torch
import numpy as np
import torch.nn as nn
import numpy as np
import nibabel as nib
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import scipy
from scipy.ndimage import distance_transform_edt


def gradient(input_tensor, onlyAssymetric = False):
    if onlyAssymetric:
        gradient_x_minus = (torch.roll(input_tensor, shifts=-1, dims=2) - input_tensor) 
        gradient_y_minus = (torch.roll(input_tensor, shifts=-1, dims=3) - input_tensor)
        gradient_z_minus = (torch.roll(input_tensor, shifts=-1, dims=4) - input_tensor)

        gradient_x_plus = (input_tensor - torch.roll(input_tensor, shifts=1, dims=2))
        gradient_y_plus = (input_tensor - torch.roll(input_tensor, shifts=1, dims=3))
        gradient_z_plus = (input_tensor - torch.roll(input_tensor, shifts=1, dims=4))

        gradient_x = gradient_x_minus.abs() + gradient_x_plus.abs()
        gradient_y = gradient_y_minus.abs() + gradient_y_plus.abs()
        gradient_z = gradient_z_minus.abs() + gradient_z_plus.abs()

        return gradient_x, gradient_y, gradient_z

    else:
        gradient_x = (torch.roll(input_tensor, shifts=-1, dims=2) -  torch.roll(input_tensor, shifts=1, dims=2)) / 2 
        gradient_y = (torch.roll(input_tensor, shifts=-1, dims=3) -  torch.roll(input_tensor, shifts=1, dims=3)) / 2
        gradient_z = (torch.roll(input_tensor, shifts=-1, dims=4) -  torch.roll(input_tensor, shifts=1, dims=4)) / 2

    return gradient_x, gradient_y, gradient_z

class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
    
    def forward(self, input_tensor, mask):
    
        gradient_x, gradient_y, gradient_z = gradient(input_tensor, True)

        # Compute the magnitude of the gradients
        gradient_magnitude_mean = torch.mean(torch.sqrt(torch.clamp(gradient_x ** 2 + gradient_y ** 2 + gradient_z ** 2, min=0.00001, max=1000.0))[input_tensor > 0.001]**2)
        
        if torch.isnan(input_tensor).any():
            print("nan in tensor")
            print(input_tensor)
            print(torch.isnan(input_tensor))
            exit()

        return gradient_magnitude_mean
        

class WaveFrontLossFirstOrder(nn.Module):
    def __init__(self, wm, gm):
        super(WaveFrontLossFirstOrder, self).__init__()
        self.wm = wm
        self.gm = gm

    def gradient_magnitude(self, input_tensor):

        gradient_x, gradient_y, gradient_z = gradient(input_tensor)

        # Compute the magnitude of the gradients
        gradient_magnitude = torch.sqrt(torch.clamp(gradient_x ** 2 + gradient_y ** 2 + gradient_z ** 2, min=0.00001, max=1000.0))

        #check if nan is in tensor
        if torch.isnan(input_tensor).any():
            print("nan in tensor")
            print(input_tensor)
            print(torch.isnan(input_tensor))
            exit()

        return gradient_magnitude

    def forward(self, predictions, constantFactor = 1, returnVoxelwise = False, mask = None):
        # Compute the gradient magnitude of the predictions
        grad_magnitude = self.gradient_magnitude(predictions)

        if mask is None:
            mask = predictions > 0.001
        else:
            mask = mask > 0.5

        # no loss on pixel at the border to the mask
        # Define a 3x3x3 kernel for 3D erosion
        kernel = torch.ones(1, 1, 3, 3, 3).float().to(mask.device)
        padded_mask = F.pad(mask, (1, 1, 1, 1, 1, 1), mode='constant', value=0)

        # Apply 3D convolution to perform erosion (with padding=0)

        eroded_mask = F.conv3d(padded_mask.float(), kernel, padding=0)

        eroded_mask = (eroded_mask == 27).float()

        #voxelLoss = predictions * (1-predictions) * constantFactor / D - grad_magnitude TODO this should be the correct one
        voxelLoss = (predictions * (1-predictions) * constantFactor  - grad_magnitude )/ constantFactor# (grad_magnitude +0.000000000001)  #/constantFactor

        voxelLoss = voxelLoss * eroded_mask
        
        loss = torch.mean(torch.abs(voxelLoss)**2) # 
        if returnVoxelwise:
            return loss, voxelLoss

        return loss

#%%
class  SoftDiceLoss(nn.Module):
    def __init__(self,  default_threshold = 0.5, epsilon=1e-6, steepness =500):
        super(SoftDiceLoss, self).__init__()
        self.epsilon = epsilon
        self.default_threshold = default_threshold
        self.steepness = steepness

    def contFunction(self, x, threshold):
        return  F.sigmoid((x - threshold)*self.steepness) 
    
    def forward(self, predictions, targets, threshold = None):
        
        if threshold == None:
            threshold = self.default_threshold

        probs = self.contFunction(predictions, threshold)
        
        # Flatten tensors
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        # Calculate the intersection and the denominator
        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()
        
        # Compute the soft dice coefficient
        dice_coeff = (2.0 * intersection + self.epsilon) / (union + self.epsilon)
        
        # Dice loss is 1 - Dice coefficient
        return 1.0 - dice_coeff


class PetLoss(nn.Module):
    def __init__(self):
        super(PetLoss, self).__init__()
    
    def forward(self, input_tensor, pet_image):

        mask = (pet_image > 0.0001 ) 

        if torch.sum(mask) < 0.000000001 or torch.sum(pet_image[mask]) < 0.001:
            return 0

        flatten_input = input_tensor[mask].view(-1)
        flatten_pet = pet_image[mask].view(-1)
        

        flatten_input_mean = flatten_input - torch.mean(flatten_input)
        flatten_pet_mean = flatten_pet - torch.mean(flatten_pet)

        correlation = torch.mean(flatten_input_mean * flatten_pet_mean) / (
            (torch.std(flatten_input) * torch.std(flatten_pet) + 0.001)
        )
    
        return 1-correlation

# %%
def create_standard_plan(core_segmentation, distance):
    
    # Calculate the Euclidean distance for areas not in the core
    distance_transform = distance_transform_edt(~ (core_segmentation >0))
    
    # Mark regions within a specific distance from the core
    dilated_core = distance_transform <= distance

    return dilated_core

def find_threshold(volume, target_volume, tolerance=0.01, initial_threshold=0.5,maxIter = 10000):

    if np.sum(volume > 0) < target_volume:
        print("Volume is too small")
        return 0

    # Define the initial threshold, step, and previous direction
    threshold = initial_threshold
    step = 0.1
    previous_direction = None

    # Calculate the current volume
    current_volume = np.sum(volume > threshold)

    # Iterate until the current volume is within the tolerance of the target volume

    while abs(current_volume - target_volume) / target_volume > tolerance:
        # Determine the current direction
        if current_volume > target_volume:
            direction = 'increase'
        else:
            direction = 'decrease'

        # Adjust the threshold
        if direction == 'increase':
            threshold += step
        else:
            threshold -= step

        # Check if the threshold is out of bounds
        if threshold < 0 or threshold > 1:
            return 1.01 #above the model

        # Update the current volume
        current_volume = np.sum(volume > threshold)

        # Reduce the step size if the direction has alternated
        if previous_direction and previous_direction != direction:
            step *= 0.5

        # Update the previous direction
        previous_direction = direction

        maxIter -= 1
        if maxIter < 0:
            print("Max Iter reached, no threshold found")
            return 0

    return threshold

def getRecurrenceCoverage(tumorRecurrence, treatmentPlan):

    if np.sum(tumorRecurrence) <=  0.00001:
        return 1

    # Calculate the intersection between the recurrence and the plan
    intersection = np.logical_and(tumorRecurrence, treatmentPlan)

    # Calculate the coverage as the ratio of the intersection to the recurrence
    coverage = np.sum(intersection) / np.sum(tumorRecurrence)
    return coverage

# relative part of the prediction that is inside recurrence
def getPredictionInRecurrence(tumorRecurrence, treatmentPlan):

    if np.sum(tumorRecurrence) <=  0.00001:
        return 0
    
    if np.sum(treatmentPlan) <= 0.00001:
        return 0
    
    # normalize sum of treatment plan to 1
    normalizedTreatmentPlan = treatmentPlan / np.sum(treatmentPlan)

    coverage = np.sum((tumorRecurrence > 0) * normalizedTreatmentPlan)

    return coverage

