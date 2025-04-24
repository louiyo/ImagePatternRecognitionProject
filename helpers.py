# DATA LOADING
import os 
from PIL import Image

# SEGMENTATION
import numpy as np
import cv2
import skimage
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import scipy.stats as stats

# FEATURE EXTRACTION
import torch
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from skimage.filters import gabor_kernel

# CLASSIFICATION
from sklearn.cluster import KMeans

from yellowbrick.cluster import KElbowVisualizer
######################################### DATA LOADING #########################################

def load_input_image(image_index ,  folder ="train" , path = "data_project"):
    
    filename = "train_{}.png".format(str(image_index).zfill(2))
    path_solution = os.path.join(path,folder , filename )
    
    im= Image.open(os.path.join(path,folder,filename)).convert('RGB')
    im = np.array(im)
    return im

def save_solution_puzzles(image_index , solved_puzzles, outliers  , folder ="train" , path = "data_project"  ,group_id = 0):
    
    path_solution = os.path.join(path,folder + "_solution_{}".format(str(group_id).zfill(2)))
    if not  os.path.isdir(path_solution):
        os.mkdir(path_solution)

    print(path_solution)
    for i, puzzle in enumerate(solved_puzzles):
        filename =os.path.join(path_solution, "solution_{}_{}.png".format(str(image_index).zfill(2), str(i).zfill(2)))
        Image.fromarray(puzzle).save(filename)

    for i , outlier in enumerate(outliers):
        filename =os.path.join(path_solution, "outlier_{}_{}.png".format(str(image_index).zfill(2), str(i).zfill(2)))
        Image.fromarray(outlier).save(filename)

def solve_and_export_puzzles_image(image_index , folder = "train" , path = "data_project"  , group_id = "00"):
    """
    Wrapper funciton to load image and save solution
            
    Parameters
    ----------
    image:
        index number of the dataset

    Returns
    """

      # open the image
    image_loaded = load_input_image(image_index , folder = folder , path = path)
    #print(image_loaded)
    
   
    ## call functions to solve image_loaded
    solved_puzzles = [ (np.random.rand(512,512,3)*255).astype(np.uint8)  for i in range(2) ]
    outlier_images = [ (np.random.rand(128,128,3)*255).astype(np.uint8) for i in range(3)]
    
    save_solution_puzzles (image_index , solved_puzzles , outlier_images , folder = folder ,group_id =group_id)

    return image_loaded , solved_puzzles , outlier_images

######################################### SEGMENTATION #########################################

def preprocess_img(img):
    """Preprocesses the image for segmentation. Uses Canny edge detection, morphological closing 
    and hole filling.

    Args:
        img (ndarray): input image

    Returns:
        ndarray: preprocessed image (same size as input image), in grayscale
    """
    
    img= img.copy()
    # convert to grayscale
    im_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # detect edges
    im_gray = cv2.Canny(im_gray, 35, 40)    

    # close
    im_gray = cv2.morphologyEx(im_gray, cv2.MORPH_CLOSE, np.ones((3, 3)), iterations=6)
    
    # fill the holes in the image
    im_gray = ndimage.binary_fill_holes(im_gray).astype(np.uint8)*255  

    return im_gray

def are_rectangles_overlapping(contour1, contour2):
    """Checks if two contours have overlapping bounding rectangles.

    Args:
        contour1 (cv2.contour): contour 1
        contour2 (cv2.contour): contour 2

    Returns:
        bool: True if the bounding rectangles overlap, False otherwise
    """
    rect1 = cv2.minAreaRect(contour1)
    rect2 = cv2.minAreaRect(contour2)

    return cv2.rotatedRectangleIntersection(rect1, rect2)[0] != cv2.INTERSECT_NONE

def are_rectangles_close(contour1, contour2, threshold):
    """
    Checks if two contours have bounding rectangles that are close enough. This is used to merge
    contours that are close to each other but do not overlap.

    Args:
        contour1 (cv2.contour): Contour 1
        contour2 (cv2.contour): Contour 2
        threshold (int): The euclidean distance between the centers of the bounding rectangles at which we consider them close enough.

    Returns:
        bool: True if the bounding rectangles are close, False otherwise
    """
    rect1 = cv2.minAreaRect(contour1)
    rect2 = cv2.minAreaRect(contour2)

    center1 = rect1[0]
    center2 = rect2[0]

    distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

    return distance <= threshold


def merge_contours(contours):
    """
    Merges contours that are close to each other or overlapping. This extends the list of contours by 
    adding the concatenated contours. The new list of contours is then cleaned to remove small contours by 
    another function.

    Args:
        contours (list or array): list or array containing all the contours obtained for an image

    Returns:
        list: list of contours after merging
    """
    out = []
    for c in contours:
        for i, cnt2 in enumerate(contours):
            if c is not cnt2 and len(c) > 5 and len(cnt2) > 5:              
                if are_rectangles_close(c, cnt2, 100) or are_rectangles_overlapping(c, cnt2):
                    c = np.concatenate((c,cnt2))
        out.append(c)
    # we return the new contour and the list of merged contours so that we can skip them for the rest of the iterations
    return clean_contours(out)


def clean_contours(contours):
    """
    Cleans the list of contours by removing small contours and contours that 
    overlap after merging (drops duplicates).

    Args:
        contours (list): list of contours (in the context it is used, this list contains all the contours after merging)

    Returns:
        list: cleaned list of contours
    """
    # Sort contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Create a mask to keep track of visited contours
    mask = np.ones((len(contours),), dtype=bool)

    # Iterate through each contour
    for i, contour in enumerate(contours):
        if mask[i]:
            # remove small contours
            if len(contour) < 100:
                mask[i] = False
                continue

            # Iterate through remaining contours
            for j in range(i+1, len(contours)):
                # Check if the bounding rectangles are overlapping
                if are_rectangles_overlapping(contour, contours[j]):
                    mask[j] = False

    # Return the list of non-overlapping contours
    return [contour for i, contour in enumerate(contours) if mask[i]]

def find_pieces(im):
    """
    Implements all of the preprocessing and segmentation steps to find the puzzle pieces in an image.
    It then extracts the puzzle pieces from the image and returns them as a list of images (128x128).

    Args:
        im (ndarray): raw image

    Returns:
        list: list of the puzzle pieces (each being 128x128)
    """

    img= im.copy()
    
    contours, _ = cv2.findContours(preprocess_img(img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = merge_contours(contours)
    puzzle_pieces = []

    for c in contours:
        (x, y), (w, h), angle  = cv2.minAreaRect(c)
        if w < h:
            angle += 90
        rotation_matrix = cv2.getRotationMatrix2D((x, y), angle, 1)

        # Rotate the image to make the square level again
        rotated_image = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))

        # Crop the rotated image to 128x128 pixels
        square_size = max(w, h)
        square_center = (int(x), int(y))
        square_half_size = int(square_size / 2)
        square_top_left = (square_center[0] - square_half_size, square_center[1] - square_half_size)
        square_bottom_right = (square_center[0] + square_half_size, square_center[1] + square_half_size)
        cropped_image = rotated_image[square_top_left[1]:square_bottom_right[1], square_top_left[0]:square_bottom_right[0]]
        if cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0:
            print("Skipping piece with width or height of 0")
            continue
        puzzle_pieces.append(cv2.resize(cropped_image, (128, 128)))
    
    return puzzle_pieces


######################################### FEATURE EXTRACTION #########################################

def preprocess_piece(image):
    """Preprocess puzzle pieces so that they can be used as input for the ResNet50 model.

    Args:
        image (ndarray, 128x128): input puzzle piece

    Returns:
        torch.tensor: tensor of the preprocessed puzzle piece, ready to be used as input for the ResNet50 model
    """

	# swap the color channels from BGR to RGB and transpose it to have the channels first
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.transpose(image,(2,0,1))

    return torch.from_numpy(image).unsqueeze(0)

def get_features(puzzle_pieces):
    """
    Extracts the features from the puzzle pieces using the ResNet50 model. We use the activations of 
    the first convolutional block (conv1). We then normalize the activations and apply truncated SVD to reduce the
    dimensionality of the features (better suited for sparce data).

    Args:
        puzzle_pieces (_type_): _description_

    Returns:
        _type_: _description_
    """

	# Initialize the Weight Transforms
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(progress=True, weights=weights)
    preprocess = weights.transforms()

    images = torch.concatenate([preprocess(preprocess_piece(puzzle_pieces[i])) for i in range(len(puzzle_pieces))] ,dim=0)

    # Apply it to the input image
    img_transformed = preprocess(images)

    model.eval()
    plt.show()

    activation = {}
    def getActivation(name):
        # the hook signature
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    handle1 = model.conv1.register_forward_hook(getActivation('conv1'))
    handle2 = model.layer1.register_forward_hook(getActivation('layer1'))
    #handle3 = model.layer2.register_forward_hook(getActivation('layer2'))
    #handle4 = model.layer3.register_forward_hook(getActivation('layer3'))

    # forward pass
    output = model(img_transformed)

    # scale the activations
    for key in activation.keys():
        activation[key] = activation[key].reshape(activation[key].shape[0], -1)
        activation[key] = StandardScaler().fit_transform(activation[key])

        # Diemnsionality reduction
        svd = TruncatedSVD(n_components=1000)
        activation[key] = svd.fit_transform(activation[key])

    # concatenate the activations -> we get a matrix of shape (n_samples, n_features * n_activation_layers)
    activation = np.concatenate([activation[key] for key in activation.keys()], axis=1)
    
    handle1.remove()
    handle2.remove()
    #handle3.remove()
    #handle4.remove()

    return activation


def extract_gabor_features(puzzle_pieces):
    """Extracts the Gabor features from an image.

    Args:
        image (ndarray): input image

    Returns:
        ndarray: Gabor features of the input image
    """

    features = []

    for image in puzzle_pieces:
        # convert to grayscale
        im_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # define the range of theta and sigma
        thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        sigmas = [1, 2, 5]  
        frequencies = [0.05, 0.3]        

        features_per_piece = []
        # loop over the thetas
        for theta in thetas:
            # loop over the sigmas
            for sigma in sigmas:
                # loop over the wavelengths
                for frequency in frequencies:
                    # apply the gabor filter
                    gabor = np.real(gabor_kernel(frequency=frequency, theta=theta,
                                          sigma_x=sigma, sigma_y=sigma))
                    filtered = ndimage.convolve(im_gray, gabor, mode='wrap')
                    
                    # compute the features
                    variance = np.var(filtered)
                    features_per_piece.append(variance)
                    
                    # kurtozis
                    kurtosis = stats.kurtosis(filtered.flatten())
                    features_per_piece.append(kurtosis)

                    # skewness
                    skewness = stats.skew(filtered.flatten())
                    features_per_piece.append(skewness)

                    # min, max, mean
                    features_per_piece.append(np.min(filtered))
                    features_per_piece.append(np.max(filtered))
                    features_per_piece.append(np.mean(filtered))

                    # local energy
                    local_energy = np.sum(np.power(filtered, 2))
                    features_per_piece.append(local_energy)
                    
                    # mean amplitude
                    mean_amplitude = np.sum(np.abs(filtered))
                    features_per_piece.append(mean_amplitude)

                    # power spectrum
                    spectrum = np.abs(np.fft.fft2(filtered))
                    features_per_piece.append(np.sum(spectrum**2))
        
        features.append(features_per_piece)

    features = np.array(features)

    return features

def extract_histogram_features(puzzle_pieces, bins=64):
    """Generates features based on the binned histogram values of the RGB channel of the images.

    Args:
        puzzle_pieces (list(np.array)): list of 128x128 puzzle pieces
    """

    features = []

    for image in puzzle_pieces:

        hist_R = np.histogram(image[:,:,0], bins=bins)[0]
        hist_G = np.histogram(image[:,:,1], bins=bins)[0]
        hist_B = np.histogram(image[:,:,2], bins=bins)[0]

        features.append(np.concatenate([hist_R, hist_G, hist_B]))

    features = np.array(features)
    
    return features

def get_clusters(puzzle_pieces):

    X1 = extract_gabor_features(puzzle_pieces)
    X2 = extract_histogram_features(puzzle_pieces)
    X = np.concatenate((X1, X2), axis=1)

    # replace NaNs with 0s
    X = np.nan_to_num(X)
    k_desired = elbow_method(X) 
    # normalize the features
    X = StandardScaler().fit_transform(X)
    
    # perform k-means clustering
    kmeans = KMeans(n_clusters= k_desired, random_state=0).fit(X)

    filtered_clusters = []
    filtered_outliers = []

    for i in range(k_desired):
        labels = np.where(kmeans.labels_ == i)[0]
        cluster_size = len(labels)
        if cluster_size in (9,12,16):
            filtered_clusters.append([puzzle_pieces[j] for j in labels])

    outlier_indices = []
    for i in range(k_desired):
        labels = np.where(kmeans.labels_ == i)[0]
        cluster_size = len(labels)
        outlier_indices_2 = []
        if cluster_size not in (9,12,16):
            cluster_to_reduce = labels 
            while cluster_size not in (0,9,12,16):
                cluster_center = kmeans.cluster_centers_[i]
                distances = []
                for k in cluster_to_reduce:
                    distances.append(np.linalg.norm(X[k] - cluster_center))
                outlier_index = cluster_to_reduce[np.argmax(distances)]
                indice = np.where(cluster_to_reduce == outlier_index)[0]
                indice_2 = np.where(labels == outlier_index)[0]
                cluster_to_reduce = np.delete(cluster_to_reduce, indice)
                outlier_indices.append(outlier_index)
                outlier_indices_2.append(indice_2)
                cluster_size = cluster_size-1
            if cluster_size != 0:
                filtered_arr = np.delete(labels, outlier_indices_2)
                filtered_clusters.append([puzzle_pieces[j] for j in filtered_arr])
    filtered_outliers.append([puzzle_pieces[j] for j in outlier_indices])
        
    return filtered_clusters , filtered_outliers


def elbow_method(data, max_k = 4):
    distortions = []
    for k in range(1, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)  # Sum of squared distances to nearest centroid

    # Calculate the difference in distortions
    distortions_diff = np.diff(distortions)

    # Find the index of the elbow point (where the difference starts to level off)
    elbow_index = np.argmax(distortions_diff) + 1

    # Return the appropriate value of k
    return elbow_index


