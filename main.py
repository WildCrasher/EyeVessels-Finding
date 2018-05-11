import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage import color, restoration
from os import listdir
from skimage.util import img_as_ubyte
from skimage.filters import gaussian
from skimage.morphology import *
from skimage.filters import frangi, rank, threshold_mean
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve

imagePath = "stare-images"
imageExpertPath = "stare-images-expert"

#----------------------------------------
filesList = listdir(imagePath)
filesExpertList = listdir(imageExpertPath)
for i in range(len(filesList)):
    filesList[i] = os.path.join(imagePath, filesList[i])
    filesExpertList[i] = os.path.join(imageExpertPath, filesExpertList[i])

accuracySum = 0
specificitySum = 0
sensitivitySum = 0
Average = [0,0,0]

for imageIndex in range(3):

    originalImage = io.imread(filesList[imageIndex])
    expertImage = io.imread(filesExpertList[imageIndex])
    greenImage = originalImage.copy()
    greenImage[:,:,0] = 0
    greenImage[:,:,2] = 0

    image = color.rgb2gray(greenImage)

    # m = np.mean(image)
    # for i in range(len(image)):
    #     for j in range(len(image[i])):
    #         if image[i][j] > 1.7*m:
    #             image[i][j] = lastpixel
    #         else:
    #             lastpixel = image[i][j]

    for i in range(len(image)):
        for j in range(len(image[i])):
            if image[i][j] > 0.35:
                image[i][j] = lastpixel
            else:
                lastpixel = image[i][j]
    # image = opening(image, disk(5))
    # image = opening(image, disk(7))
    # image = opening(image, disk(11))

    filteredImage = image.copy()
    filteredImage = img_as_ubyte(filteredImage)
    filteredImageMorf = erosion(filteredImage,np.ones((5,5),np.uint8))
    #filteredImageGaussian = gaussian(filteredImageMorf, sigma=3)
    filteredImageGaussian = filteredImageMorf.copy()
    '''
    radius = 5
    selem = disk(radius)
    
    local_otsu = rank.otsu(filteredImageNotGaussian, selem)
    binary = filteredImageNotGaussian < local_otsu
    binary = img_as_ubyte(binary)
    '''

    imgFrangi = frangi(filteredImageGaussian, scale_range=(1, 4), scale_step=1, beta1=1, beta2=20, black_ridges=True)

    thresh = threshold_mean(imgFrangi)
    binary = imgFrangi > thresh

    finalImage = img_as_ubyte(binary)

    fig, (ax0, ax1) = plt.subplots(nrows=2,
                                        ncols=2,
                                        sharex=True,
                                        sharey=True)

    ax0[0].imshow(originalImage, cmap="gray")
    ax0[1].imshow(imgFrangi, cmap="gray")
    ax1[0].imshow(finalImage, cmap="gray")
    ax1[1].imshow(expertImage, cmap="gray")

    confusionMatrix = confusion_matrix(np.asarray(expertImage).flatten()/255, np.asarray(finalImage).flatten()/255)
    accuracy = accuracy_score(np.asarray(expertImage).flatten()/255, np.asarray(finalImage).flatten()/255)
    specificity, sensitivity, _ = roc_curve(np.asarray(expertImage).flatten()/255, np.asarray(finalImage).flatten()/255)
    accuracySum += accuracy
    specificitySum += 1 - specificity[1]
    sensitivitySum += sensitivity[1]
    Average[imageIndex] = ((1 - specificity[1]) + sensitivity[1])/2

    print("Confusion matrix:")
    print(confusionMatrix)
    print("Accuracy: " + str(accuracy))
    print("Specificity: " + str((1 - specificity[1])))
    print("Sensitivity: " + str(sensitivity[1]))
    print("Average of sensitivity and specificity: " + str(Average[imageIndex]))
    print("\n")

plt.show()

print("Average Accuracy: " + str(accuracySum/3))
print("Average Specificity: " + str(specificitySum/3))
print("Average Sensitivity: " + str(sensitivitySum/3))
print("Average of global accuracy: " + str(np.sum(Average)/3))