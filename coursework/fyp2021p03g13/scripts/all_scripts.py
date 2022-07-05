######################################################################################
##################################### IMPORTS ########################################
######################################################################################

# ENV
import os

# ISIC API
import requests
import glob

# Main data structure(s)
import pandas as pd
import numpy as np

# For pi value
import math

# For API calls
import time
import json

# Building a data set
import shutil
import zipfile
from io import BytesIO

# Data visualization
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

# Data normalization
from sklearn import preprocessing

# Feature extraction
from skimage import morphology
from skimage.transform import rotate
from scipy.spatial.distance import cdist
from scipy.stats.stats import mode

# Color 'brightness' (value) spread
from skimage.segmentation import slic # Segments image using k-means clustering in Color-(x,y,z) space
from skimage.measure import regionprops
from skimage.color import rgb2hsv # Converts RGB color to HSV (Hue, Saturation, Value)

# Proper split of training and test data
from sklearn.model_selection import train_test_split

# Feature selection
from sklearn.feature_selection import chi2, mutual_info_classif, SelectKBest

# Model building
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# For model evaluation
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score


######################################################################################
##################################### GIVEN FUNCTIONS ################################
######################################################################################
def measure_area_perimeter(mask):
    # Measure area: the sum of all white pixels in the mask image
    area = np.sum(mask)

    # Measure perimeter: first find which pixels belong to the perimeter.
    struct_el = morphology.disk(1)
    mask_eroded = morphology.binary_erosion(mask, struct_el)
    image_perimeter = mask - mask_eroded

    # Now we have the perimeter image, the sum of all white pixels in it
    perimeter = np.sum(image_perimeter)

    return area, perimeter

######################################################################################
##################################### ISIC API #######################################
######################################################################################

class ISICApi(object):

    """
    Copied from (in order to be able to use the API):
    https://www.isic-archive.com/#!/topWithHeader/onlyHeaderTop/apiDocumentation.
    """

    def __init__(self, hostname='https://isic-archive.com',
                 username=None, password=None):
        self.baseUrl = f'{hostname}/api/v1'
        self.authToken = None

        if username is not None:
            if password is None:
                password = input(f'Password for user "{username}":')
            self.authToken = self._login(username, password)

    def _makeUrl(self, endpoint):
        return f'{self.baseUrl}/{endpoint}'

    def _login(self, username, password):
        authResponse = requests.get(
            self._makeUrl('user/authentication'),
            auth=(username, password)
        )
        if not authResponse.ok:
            raise Exception(f'Login error: {authResponse.json()["message"]}')

        authToken = authResponse.json()['authToken']['token']
        return authToken

    def get(self, endpoint):
        url = self._makeUrl(endpoint)
        headers = {'Girder-Token': self.authToken} if self.authToken else None
        return requests.get(url, headers=headers)

    def getJson(self, endpoint):
        return self.get(endpoint).json()

    def getJsonList(self, endpoint):
        endpoint += '&' if '?' in endpoint else '?'
        LIMIT = 50
        offset = 0
        while True:
            resp = self.get(
                f'{endpoint}limit={LIMIT:d}&offset={offset:d}'
            ).json()
            if not resp:
                break
            for elem in resp:
                yield elem


def downloadIsicImageSegment(savePath, image_name, db_id, show_status = False):

    """
    Implemented based on and uses some of the code from:
    https://www.isic-archive.com/#!/topWithHeader/onlyHeaderTop/apiDocumentation.

    Downloads the specified segment to the specified folder (name of the image is set to be image_id)
    using the public ISIC API.
    :db_id: internal id --> see raw data sets
    :savePath: string, for instance 'images/'
    :show_status: Do you want the status message to be displayed
    :return: boolean --> True = success, False --> fail
    """

    # Initialize the API object; no login is necessary since we want to query just public data
    api = ISICApi()

    # If you specify a path where there are directories which do not
    # exist yet, this piece of code creates them
    # --> this avoids the error when you will be saving the given image within the directory
    if not os.path.exists(savePath):
        os.makedirs(savePath)

    # Make an API calls to get the needed data
    try:

        # Get correct if for segmentation
        seg_id = api.getJson('segmentation?imageId='+ db_id)[0]["_id"]

        # SEGMENTATION
        segmentFileResp = api.get('segmentation/%s/mask' % seg_id)

        # Checks that the operation went ok --> 200 Status code, if not, catches the error
        segmentFileResp.raise_for_status()

    except requests.HTTPError as exception:
        if show_status:
            print(f"Something went wrong for the image {image_name}.\nHere is the error detail: {exception}")
        return False


    # Save the data to the specified folder
    segmentFileOutputPath = os.path.join(savePath, '%s_segmentationMask.jpg' % image_name)
    # * SEGMENT
    with open(segmentFileOutputPath, 'wb') as segmentFileOutputStream:
        for chunk in segmentFileResp:
            segmentFileOutputStream.write(chunk)

    # Inform about the process
    if show_status:
        print(f"Data for image {image_name} were successfuly downloaded.")
    
    return True


def deleteImageData(folderPath):

    """
    Deletes all image data within the specified folder. (Image data = .jpg or .png)
    :image_id: Use the ISIC format; e.g.: ISIC_0012086 --> can be found within CSVs
    :folderPath: string, for instance 'images/'
    :return: None
    """

    jpg_files = glob.glob(f'{folderPath}*.jpg')
    png_files = glob.glob(f'{folderPath}*.png')
    all_files = jpg_files + png_files
    for f in all_files:
        os.remove(f)


def downloadChunkIsicImageDataAsZip(image_ids):

    """
    Download the given chunk of images using ISIC API.
    Extract the images to the given folder and returns its names along with the
    paths to the particular files.
    :image_ids: list
    """


    # Get the api object
    api = ISICApi()

    # Get the json array of image ids
    json_array = json.dumps(image_ids)

    # Create  requests
    # * Create the request for images
    endpoint_images = f'image/download?include=images&imageIds={json_array}'    

    # Send requests
    # * Send the request to the given end point
    resp_images = api.get(endpoint_images)
    print(f"Status for downloading chunk with images: {resp_images.status_code}")

    # Parse requests into a zip file
    # * Parse the response into a zip file
    bytes = BytesIO(resp_images.content)
    zip_file = zipfile.ZipFile(bytes)

    # Extract all
    zip_filepaths = zip_file.namelist()
    zip_filename = zip_filepaths[0].split(os.sep)[0]
    zip_file.extractall()
    zip_file.close()

    return zip_filename, zip_filepaths


def findImagePathInZip(image_name, zip_filepaths):

    """
    Returns a path to the given file within zip folder.
    """

    for path in zip_filepaths:
        if image_name in path:
            return path


def testCorrectnessOfImages(all_data):

    """
    The goal of this function is to download an image and corresponding mask 
    and then you can compare to data which you for example manually downloaded.
    :all_data: pandas dataframe with all ISIC 2017 data
    :return: None, it just saves the given images to the directory you are in
    """

    # Generate a random record
    random_record = all_data.sample(n=1)
    
    # Parse info from the random record
    image_name = random_record["image_id"].iloc[0]
    db_id = random_record["db_id"].iloc[0]

    zip_filename, _ = downloadChunkIsicImageDataAsZip([db_id])
    downloadIsicImageSegment(f"{zip_filename}/", image_name, db_id, show_status = True)


######################################################################################
################################# DATA FILTERING #####################################
######################################################################################


def getImageMetaData(all_data):

    """
    Get following meta data about the images:
    1. Image database ID (internal ID which allows you to query images)
    2. Image size X
    3. Image size Y
    :all_data: pandas DF with all 2017 ISIC DATA
    :return: all_data DF with new columns inluding the meta data
    """

    # Initiliza api
    api = ISICApi()

    # Create the request url
    r_url = f'image?limit=16072&sort=name&sortdir=1&detail=true'

    # Get the response
    resp = api.getJson(r_url)

    # Extract the needed data from the resp for each image
    image_true_ids = [None]*all_data.shape[0]
    size_x = [None]*all_data.shape[0]
    size_y = [None]*all_data.shape[0]

    for meta_info in resp:

        # Extract needed info
        image_size_info = meta_info["meta"]["acquisition"]
        image_size_x = image_size_info["pixelsX"]
        image_size_y = image_size_info["pixelsY"]
        im_name = meta_info['name']
        im_id = meta_info['_id']

        # Assign the info to corresponding place if possible and relevant
        row_index = all_data["image_id"][all_data["image_id"] == im_name].index
        if len(row_index) == 1:
            image_true_ids[row_index[0]] = im_id
            size_x[row_index[0]] = image_size_x
            size_y[row_index[0]] = image_size_y
        elif len(row_index) > 1:
            print(f"You should check image id: {im_name}. It looks like there is a duplicate.")
            continue

    
    # Assign the values to the df
    all_data["db_id"] = image_true_ids
    all_data["size_x"] = size_x
    all_data["size_y"] = size_y

    return all_data


def isMaskTooCloseToBorder(mask, threshold):

    """
    Outputs if the image is coinciding with the border or is close to the border within specified threshold.
    :mask: 2D numpy array with binary values
    :threshold: number of columns from the edge of the image
    :return: boolean
    """

    # Get the height and width of the mask
    border_height, border_width = mask.shape

    # Check left and right side
    if 1 in mask[:,:threshold] or 1 in mask[:, -threshold:]:
        return True

    # Check bottom and up
    if 1 in mask[:threshold] or 1 in mask[-threshold:]:
        return True
    
    # If image is not too close within any of the borders --> return False
    return False


def normalizeFeatures(features):

    """
    Normalizes features to a common scale using Sklearn preproccessing module.
    :features: pandas dataframe with selected features
    :return: dataframe with scaled features
    """

    # Fit scaler on the data
    scaler = preprocessing.StandardScaler().fit(features)

    # Apply the scaler to the data
    new_features = scaler.transform(features) # Returns 2D numpy array

    # Transform the numpy array back to DF
    new_features = pd.DataFrame(data = new_features, columns = list(features.columns))

    # Put back the binary values for melanoma and keratosis
    if "melanoma" in list(features.columns):
        new_features["melanoma"] = features["melanoma"]
    else:
        new_features["keratosis"] = features["keratosis"]

    return new_features


######################################################################################
############################### BUILDING THE DATASET #################################
######################################################################################


def sampleFromAllData(all_data, N, cancer_frac, cancer_type):

    """
    The goal of this function is to take a sample from ISIC 2017 datasets in arandom way.
    The sampling procedure works as follows:
    1. Get needed images for given cancer type
    2. Fill rest of the images with HEALTHY data
    :all_data: dataframe with all ISIC 2017 data
    :N: size of the dataset
    :cancer_frac: fraction of N which you want to be cancerous data
    :cancer_type: Name of cancer which you want to sample for
    :return: new DF with sampled data and only column with given cancer
    """

    # Setup the parameters
    cancer_size = int(N*cancer_frac)
    healthy_size = N - cancer_size

    # Get cancerous data first
    cancer_mask = all_data[cancer_type] == 1
    cancer = all_data[cancer_mask].sample(n=cancer_size, replace=False, random_state=1)

    # Second, get healthy data
    healthy_mask = (all_data["melanoma"] == 0) & (all_data["seborrheic_keratosis"] == 0)
    healthy = all_data[healthy_mask].sample(n=healthy_size, replace=False, random_state=1)

    # Put together the dataframes
    result = cancer.append(healthy, ignore_index = True)

    # Drop the column with the other cancer type
    if cancer_type == "melanoma":
        result = result.drop(['seborrheic_keratosis'], axis=1)
    else:
        result = result.drop(['melanoma'], axis=1)

    return result

def splitDataIntoChunks(data, chunk_size):

    """
    Splits the given sequence into chunks with given size.
    :data: 1D pandas series
    :chunk_size: int
    :return: A nested list where each item is a pandas series
    """

    result = [data.iloc[start:min(len(data), start + chunk_size)] for start in range(0, len(data), chunk_size)]
    return result


def buildClassifierInput(sampled_data, chunk_size = 100, temp_img_fold = "imageData/", path_to_save = "model_input.csv"):

    """
    Build a dataset which will server as an input to the classifier. --> the output csv file is written on the disk after the process.
    :sampled_data: pandas dataframe
    :chunk_size: By how many chunks do you want to download data from ISIC API --> 100 recommended, 300 max
    :temp_img_fold: path to the folder where you want to store temporarily the images --> will be deleted afterwards.
    :return: None
    """

    # Split the data into chunks
    chunks = splitDataIntoChunks(sampled_data["db_id"], chunk_size)

    # Cancer type
    if "melanoma" in list(sampled_data.columns):
        cancer_type = "melanoma"
    else:
        cancer_type = "keratosis" 

    # Prepare a dataframe where you will store the resulting data
    # * Define columns
    columns = ["image_id", cancer_type, "compactness", "assymetry", "border_irr", "hue_sd", "satur_sd", "value_sd", "iqr_val"]
    # * Create DF
    dataset_df = pd.DataFrame(columns=columns)

    # Fill the dataframe with the given records about images
    for chunk in chunks:
        
        # Convert chunk to proper image_ids
        image_db_ids = [i for i in chunk]

        # Download image data localy so you can access it
        zip_filename, zip_filepaths = downloadChunkIsicImageDataAsZip(image_db_ids)

        for image_db_id in chunk:

            # Get all available data about given record
            record_mask = sampled_data["db_id"] == image_db_id
            record = sampled_data[record_mask]
            image_id = record["image_id"].iloc[0]
            if cancer_type == "melanoma":
                is_cancer = record["melanoma"].iloc[0]
            else:
                is_cancer = record["seborrheic_keratosis"].iloc[0]

            # Load the image to NTB
            # * Image
            file_image = findImagePathInZip(image_id, zip_filepaths)
            image = plt.imread(file_image)

            # Load the mask to NTB
            # * Call the API and download the mask locally
            worked_ok = downloadIsicImageSegment(temp_img_fold, image_id, image_db_id, show_status = True)
            if not worked_ok:
                continue

            # * Mask
            file_mask = temp_img_fold + image_id + '_segmentationMask.jpg'
            mask = plt.imread(file_mask)

            # Make sure the image has relevant properties for us
            arePropertiesValid = ~isMaskTooCloseToBorder(mask, threshold = 50)

            if arePropertiesValid:
                try:
                    # Extract the features
                    features = [getShapeCompactness(mask), computeAsymmetry(mask), getBorderIrregularity(mask)] + getColorFeatures(image, mask)

                    # Add record to the given dataframe
                    new_record = pd.DataFrame([[image_id, is_cancer] + features], columns = columns)
                    dataset_df = dataset_df.append(new_record, ignore_index=True)
    
                except Exception as e:
                    print(e)
            
            # Delete the segment data corresponding to given chunk
            deleteImageData(temp_img_fold)

        # Download the folder with image data corresponding to given chunk
        shutil.rmtree(f"{zip_filename}/")
    
    # Save the dataset
    dataset_df.to_csv(path_to_save, index = False)

    # Delete the temporary img folder
    os.rmdir(temp_img_fold)



######################################################################################
################################ ASYMMETRY COMPUTING #################################
######################################################################################


def rotateImage(image, angle, center):

    """
    Rotates the given image by given angle in clockwise direction. Center is set to center of image by default. 
    See skimage documentation: https://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.rotate
    :image: 2D numpy array
    :angle: degrees
    :return: new image - 2D numpy array
    """

    rotated_image = rotate(image, -angle, center = center)
    return rotated_image


def findCentroid(mask):

    """
    Finds centroid using the mean of x and y coordinates.
    If confused regards to axis in numpy, see this answer (scroll to the bottom for visual):
    https://stackoverflow.com/questions/17079279/how-is-axis-indexed-in-numpys-array
    :mask: 2D array containing binary values where 1s signify the selected region
    :return: xCoord, yCoord
    """

    # X-coordinates
    region_column_indices = np.where(np.sum(mask, axis = 0) > 0)[0]
    left, right = region_column_indices[0], region_column_indices[-1]
    xCoord = (left + right)//2

    # Y-coordinates
    region_row_indices = np.where(np.sum(mask, axis = 1) > 0)[0]
    top, bottom = region_row_indices[0], region_row_indices[-1]
    yCoord = (top + bottom)//2

    return xCoord, yCoord


def halveTheRegionHorizontally(yCoord, mask):

    """
    Splits the image into two halves horizontally. The horizontal "line" is set to go through the y-th Coordinate.
    :yCoord: index
    :mask: 2D binary numpy array
    :return: 2x 2D numpy array with exact same dimensions representing the two halves.
    """

    # Get the halves
    upper = mask[:yCoord]
    lower = mask[yCoord:]

    # Make sure both halves have the same amount of rows
    n_rows_upper = upper.shape[0]
    n_rows_lower = lower.shape[0]

    # Lower half needs more rows
    if  n_rows_upper > n_rows_lower:

        # Get inputs for transformation
        row_difference = n_rows_upper - n_rows_lower
        n_columns = lower.shape[1]
        additional_rows = [[0]*n_columns for _ in range(row_difference)]

        # Stacks row-wise lower and then additional rows --> notice the order is important since we want to add new rows to the bottom
        lower = np.vstack((lower, additional_rows))

    # Upper half needs more rows
    elif n_rows_upper < n_rows_lower:

        # Get inputs for transformation
        row_difference = n_rows_lower - n_rows_upper
        n_columns = upper.shape[1]
        additional_rows = [[0]*n_columns for _ in range(row_difference)]

        # Same logic as above, notice here that we are choosing first additional rows and then upper
        upper = np.vstack((additional_rows, upper))
    
    # Flip the lower along the x-axis, so it can be then compared directly without any further transformation
    lower = np.flip(lower, axis = 0)

    return lower, upper



def computeAsymmetry(mask):

    """
    Computes the asymmetry of the region by following procedure:
    1. Finds the midpoint of lesion using the mean of x and y coordinates
    Then rotates the images by specified angles and for each rotation:
    2. Splits the region in half using the above coordinates (horizontally)
    3. Subtracts the two halves from each other, sums the differences and computes 
    this difference relative to the size of the lesion.
    Finally, out of all computations, we take the minimum value and return it as the asymmetry.
    :mask: 2D binary numpy array
    :return: horizontal_asymmetry (normalized by division by lesion area)
    """

    # Total area of lesion
    lesion_area = np.sum(mask)
    
    # Get center
    xCoord, yCoord = findCentroid(mask)
    center = [xCoord, yCoord]

    # Specify the angles for rotation
    angles = [i for i in range(30, 181, 30)]

    # Get the asymmetry results for each rotation
    asymmetry_results = []
    for angle in angles:

        # Rotation
        rotated_mask = rotateImage(mask, angle, center)

        # Horizontal split
        bottom, top = halveTheRegionHorizontally(yCoord, rotated_mask)
        horizontal_asymmetry = abs(np.sum((bottom - top)))/lesion_area

        # Save the result
        asymmetry_results.append(horizontal_asymmetry)

    return min(asymmetry_results)



######################################################################################
################################ BORDER IRREGULARITY #################################
######################################################################################

def getShapeCompactness(mask):

    """
    Measures shape's compactness. In other words, it measures how the given shape is similar to a perferct circle
    whose compactness value is 1. The less similar the higher the compatness value.
    :mask: 2D numpy binary array
    :return: Compactness values (float): (1, +inf)
    """

    # Get area and perimiter
    area, perimiter = measure_area_perimeter(mask)

    result = (perimiter**2)/(4*math.pi*area)
    return result


def getBorderOfTheShape(mask):

    """
    Returns a 2D numpy array which represents a Grayscale mapping where 1 is only at position (x, y) where the border is
    :mask: 2D numpy array with binary values
    :return: 2D numpy array as mentioned above
    """

    # Get perimeter object: 2D numpy array which represents a Grayscale mapping
    # where 1 is only at position (x, y) where the border is
    border_width_pixels = 1
    struct_el = morphology.disk(border_width_pixels)
    mask_eroded = morphology.binary_erosion(mask, struct_el)
    image_perimeter = mask - mask_eroded

    return image_perimeter


def findSplitIndex(coordinates, splitThreshold):

    """
    Finds the index based on which the split should be made between the two segments.
    The logic is that there has to be a certain difference between y-coordinates which will then
    decide about the split.
    :coordinates: Nested list where each item is a tuple representing a point
    :splitThreshold: What is the minimum difference between y coordinates which will make the cut
    :return: Index within the segment list where to make the cut
    """

    result = None
    for coordinate_index in range(1, len(coordinates)):
        y1 = coordinates[coordinate_index - 1][1]
        y2 = coordinates[coordinate_index][1]
        if (y1 - y2) > splitThreshold:
            result = coordinate_index
            return result
    return result


def splitInitialSegment(initial_segment, splitThreshold):

    """
    Splits the initial segment such that it separates it into two halves based
    on the difference between y-coordinates. If the y-coordinates are all close to each other,
    which happends at the start and end, it then takes the half.
    :initial_segment: Nested list where each item is a tuple representing a point
    :splitThreshold: What is the minimum difference between y coordinates which will make the cut
    :return: Two nested lists with coordinates for each segment
    """

    # Sort the given segment according to y-axis
    coordinates = sorted(initial_segment, key= lambda coordinate: coordinate[1], reverse = True)

    # Find the index within coordinates which allow you to make the split
    split_index = findSplitIndex(coordinates, splitThreshold)

    # If split index not found, then it means that the segment is continuous on the border
    # We then split the segment in the half
    if not split_index:
        split_index = len(coordinates)//2
    
    # Make the split and return the two new segments 
    segment1, segment2 = coordinates[:split_index], coordinates[split_index:]

    return segment1, segment2


def getSegments(border, N):

    """
    Splits the given border into N segments with equal size.
    Note: Segments which are last might be of a smaller size compate to the others, but it is ensured that
    in case the size of last segment would be too small compare to the other, it is then removed and ignored.
    :border: 2D binary numpy array
    :N: Number of segments
    :return: Nested list where each inner list contains tuples (x, y) which each represent a point within the given segment
    """

    # Get positions of the white pixels
    y_coords, x_coords = np.where(border == 1)

    # Create a list of obtained positions where each item will look like: (x_coord, y_coord)
    coordinates = [(x_coord, y_coord) for x_coord, y_coord in zip(x_coords, y_coords)]

    # Get the initial segmentation --> N/2 --> goes from left to right
    initial_n_segments = N//2
    initial_size_segments = len(coordinates)//initial_n_segments

    # Get the initial segments
    coordinates_sorted_x = sorted(coordinates, key=lambda coordinate: coordinate[0]) # Sort by x-coordinate
    initial_segments = [coordinates_sorted_x[start: min(start + initial_size_segments, len(coordinates_sorted_x))] for start in range(0, len(coordinates_sorted_x), initial_size_segments)]

    # Filter out segments which are drastically smaller than the rest --> more specifically, look at the end cut
    if len(initial_segments[-1]) < abs((initial_size_segments - 50)):
        initial_segments.pop()

    # Get the relevant final segments
    segments_up = []
    segments_below = []
    for initial_segment in initial_segments:
        segment_below, segment_up = splitInitialSegment(initial_segment, splitThreshold = 25)
        segments_up.append(segment_up)
        segments_below.append(segment_below)

    # Get the final result
    final_segments = segments_up + [segments_below[i] for i in range(len(segments_below) - 1, -1, -1)]

    return final_segments

def testSegmentation(border, N):

    """
    Use the function getBorderOfTheShape to get the border and then specify
    into how many segments you want split the border into. It will
    then be ploted as subplots for each segment.
    """

    # Compute the segments
    segments = getSegments(border, N)

    # Get axes for subplots
    if N%3:
        number_of_rows = N//3 + 1
    else:
        number_of_rows = N//3
    fig, axarr = plt.subplots(number_of_rows, 3)
    fig.set_size_inches(18.5, 12.5)

    # Show progress of segments
    index = 0
    for row in axarr:
        for ax in row:
            if index < len(segments):
                new_image = np.zeros(border.shape)
                for segment in segments[:(index + 1)]:
                    for coordinate in segment:
                        x, y = coordinate
                        new_image[y, x] = 1
                ax.imshow(new_image, cmap = "binary");
            else:
                ax.set_axis_off()
            
            index += 1


def getEuclideanDistance(point1, point2):

    """
    Computes an Euclidean distance between two points.
    """

    distance = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    return distance


def getStandardScore(value, mean, sd):
    """
    Computes a standard score which tells you by how many standard deviations given value is above or below the mean.
    For reference, see here: https://en.wikipedia.org/wiki/Standard_score
    :value: value from the given sample dataset
    :mean: mean of the given sample dataset
    :sd: sd of the given sample dataset
    :return: Z-score (float)
    """

    z_score = (value - mean)/sd
    return z_score


def getZScoresOverThreshold(segment, centroid, threshold):

    """
    Computes how many pixels are above the given threshold relative to the length
    of segment. 
    :segment: Nested list with points
    :centroid: tuple
    :threshold: float, signifies number of standard deviations from the mean of distances for the given point within the segment
    :return: float
    """

    # Get distances and their mean and SD
    distances = [getEuclideanDistance(point, centroid) for point in segment]
    distances_mean = np.mean(np.array(distances))
    distances_sd = np.std(np.array(distances))

    # Get z scores whose value is over the given threshold
    all_z_scores = [getStandardScore(value, distances_mean, distances_sd) for value in distances]
    selected_z_scores = list()
    for z_score in all_z_scores:
        if abs(z_score) > threshold:
            selected_z_scores.append(z_score)

    # Normalize the count of such z scores by number of points within the segment
    result = len(selected_z_scores)/len(segment)
    return result

def getBorderIrregularity(mask):
    
    # Get border
    border = getBorderOfTheShape(mask)

    # Compute the center of the mask
    centroid = findCentroid(mask)

    # Compute the area of the lesion
    lesion_area = np.sum(mask)

    # Split the border in N segments
    segments = getSegments(border, 50)

    # Compute the metric of irregularity
    results_per_segment = [getZScoresOverThreshold(segment, centroid, 1.9) for segment in segments]
    result = np.std(np.array(results_per_segment))*100
    
    return result


######################################################################################
################################### COLOR VARIATION ##################################
######################################################################################


def find_topbottom(mask):
    '''
    Function to get top / bottom boundaries of lesion using a binary mask.
    :mask: Binary image mask as numpy.array
    :return: top, bottom as int
    '''
    region_row_indices = np.where(np.sum(mask, axis = 1) > 0)[0]
    top, bottom = region_row_indices[0], region_row_indices[-1]
    return top, bottom


def find_leftright(mask):
    '''
    Function to get left / right boundaries of lesion using a binary mask.
    :mask: Binary image mask as numpy.array
    :return: left, right as int
    '''

    region_column_indices = np.where(np.sum(mask, axis = 0) > 0)[0]
    left, right = region_column_indices[0], region_column_indices[-1]
    return left, right



def lesionMaskCrop(image, mask):
    '''
    This function masks and crops an area of a color image corresponding to a binary mask of same dimension.

    :image: RGB image read as numpy.array
    :mask: Corresponding binary mask as numpy.array
    '''
    # Getting top/bottom and left/right boundries of lesion
    top, bottom = find_topbottom(mask)
    left, right = find_leftright(mask)

    # Masking out lesion in color image
    im_masked = image.copy()
    im_masked[mask==0] = 0 # color 0 = black

    # Cropping image using lesion boundaries
    im_crop = im_masked[top:bottom+1,left:right+1]

    return(im_crop)



def rgb_to_hsv(r, g, b):

    """
    Credit for the entire function goes to: 
    https://www.w3resource.com/python-exercises/math/python-math-exercise-77.php
    """
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (df/mx)*100
    v = mx*100
    return h, s, v


def getColorFeatures(image, mask):

    """
    TODO: Add rest of the description

    This function computes the color brightness variations of an image, quantified as the IQR. This method 
    uses SLIC segmentation to select superpixels for grathering average regional color intensities. 
    These averages are converted to HSV to measure the spread of brightness ('Value') across all regions.

    :image: RGB image read as numpy.array
    :mask: Corresponding binary mask as numpy.array
    :return: list with extracted features
    """

    # Mask and crop image to only contain lesion
    im_lesion = lesionMaskCrop(image, mask)

    # Get SLIC boundaries
    segments = slic(im_lesion, n_segments=250, compactness=50, sigma=1, start_label=1)

    # Fetch RegionProps - this includes min/mean/max values for color intensity
    regions = regionprops(segments, intensity_image=im_lesion)

    # Access mean color intensity for each region
    mean_intensity = [r.mean_intensity for r in regions]

    # Get only segments with color in them
    color_intensity = []
    for mean in mean_intensity:
        if sum(mean) != 0:
            color_intensity.append(mean)

    # Convert RGB color means to HSV
    color_mean_hsv = [rgb_to_hsv(col_int[0], col_int[1], col_int[2]) for col_int in color_intensity]

    # Extract values for each channel
    color_mean_hue = [hsv[0] for hsv in color_mean_hsv]
    color_mean_satur = [hsv[1] for hsv in color_mean_hsv]
    color_mean_value = [hsv[2] for hsv in color_mean_hsv]

    # Compute different features based on the above values
    # * Compute SD for hue
    hue_sd = np.std(np.array(color_mean_hue))

    # * Compute SD for satur
    satur_sd = np.std(np.array(color_mean_satur))

    # * Compute SD for value
    value_sd =np.std(np.array(color_mean_value))

    # * Computing IQR range for color values
    q1 = np.quantile(color_mean_value, 0.25, interpolation='midpoint')
    q3 = np.quantile(color_mean_value, 0.75, interpolation='midpoint')
    iqr_val = q3 - q1
    
    return [hue_sd, satur_sd, value_sd, iqr_val]


######################################################################################
######################## Models building and Evaluation ##############################
######################################################################################


def splitDataIntoTrainTest(X, y):

    """
    Wrapper around the scikit function train_test_split.
    The goal of this function is to split properly a given dataset into training and test data.
    :X: DF containing only features
    :y: pd series containing binary values
    :return: dataframes and series splitted according to the given criteria.
    """

    # Split the given data according to given criteria
    # * random state --> for reproducibility
    # * stratify --> makes sure that the distribution of cancer is in each equal
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle=True, stratify=y)

    # Return the result
    return X_train, X_test, y_train, y_test



def featureScores(X_train, y_train, k):
    """
    This fucntion returns the selector object and importance scores and for each feature 
    for a univariate, filter-based feature selection.
    :X_train: Set of input variables for testing
    :y_train: Set of output variables for testing
    :k: Number of features to be selected (integer)
    """
        
    selector = SelectKBest(mutual_info_classif, k=k) # Selecting top k features using mutual information for feature scoring
    selector.fit(X_train, y_train) # fit selector to data

    # Retrieve scores
    scores = selector.scores_

    return scores, selector


def crossValidate(X_train, y_train, clfs):

    # Prepare cross-validation
    # * Specify K - industry standard is 5 - 10
    K = 5
    cv = StratifiedShuffleSplit(n_splits=K, test_size=0.2, random_state=1)

    # Build a dataframe where you will save the results of cross-validation
    # * Define metrics that will be measured
    metrics = ["accuracy", "precision", "recall", "roc_auc"]
    header = ["classifier_name"] + metrics
    # * Build the empty dataframe
    results = pd.DataFrame(columns = header)

    # Compute the results
    for name, clf in clfs.items():

        # Get the results for each metric as dict
        result_dict = cross_validate(clf, X_train, y_train, cv=cv, scoring = metrics)

        # Condense the results using their mean and save it to the list
        result = []
        for metric_name in metrics:
            key =  f"test_{metric_name}"
            value = np.mean(result_dict[key])
            result.append(value)
        
        # Create a new record representing the given classifier and corresponding metrics
        new_record = pd.DataFrame([[name] + result], columns = header)
        results = results.append(new_record, ignore_index=True)

    return results


def evaluateTestData(X_test, y_true, clfs):

    # Build a dataframe where you will save the results
    # * Define metrics that will be measured
    metrics = ["accuracy", "precision", "recall"]
    header = ["classifier_name"] + metrics
    # * Build the empty dataframe
    results = pd.DataFrame(columns = header)

    # Compute the results
    for name, clf in clfs.items():
        
        # Get the results
        y_pred = clf.predict(X_test)

        # Compute the metrics
        # * Accuracy
        acc = accuracy_score(y_true, y_pred)
        
        # * Precision
        prec = precision_score(y_true, y_pred)

        # * Recall
        rec = recall_score(y_true, y_pred)

        result = [acc, prec, rec]

        # Create a new record representing the given classifier and corresponding metrics
        new_record = pd.DataFrame([[name] + result], columns = header)
        results = results.append(new_record, ignore_index=True)
    
    return results