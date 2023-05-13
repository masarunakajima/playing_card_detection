import cv2 as cv
from os import path
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil
import yaml
import argparse
import urllib.request
import tarfile

def draw_hull(img, hull, gray=False):
    img_copy = img.copy()
    if gray:
        color = 255
    else:
        color = (255,0,0)
    if len(hull) >= 3:
        pts = np.array(hull, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv.polylines(img_copy, [pts], True, color, 4)
    return img_copy


def plot_images(images):
    sizes = [image.shape for image in images]
    w = np.max([size[1] for size in sizes])
    h = np.max([size[0] for size in sizes])
    aspect_ratios = [size[1]*1.0/size[0] for size in sizes]
    fig, ax = plt.subplots(1, len(images), figsize=(3*len(images), 3), sharex=True, sharey=True)
    for i, image in enumerate(images):
        ax[i].imshow(image)
        ax[i].set_xlim(0, w)
        ax[i].set_ylim(0, h)
        ax[i].invert_yaxis()
        #ax[i].axis('off')
    plt.show()

# function for creating two corners of a rectangle based on mask
def get_corners(mask):
    mask = mask.reshape(mask.shape[0], mask.shape[1])
    # get the coordinates of the pixels in the mask
    coords = np.argwhere(mask)
    # get the bounding box of those pixels
    y_min = np.min(coords[:,0])
    y_max = np.max(coords[:,0])
    x_min = np.min(coords[:,1])
    x_max = np.max(coords[:,1])
    return (x_min, y_min), (x_max, y_max)


def get_card_images(dir_path, new_height=200):
    # get files in the directory
    files = os.listdir(dir_path)
    # load all images in the directory
    # with red and blue switched
    rank_suits = [f.split(".")[0] for f in files]

    images = [cv.imread(path.join(dir_path, file)) for file in files]

    # get the max width and height
    max_width = max([img.shape[1] for img in images])
    max_height = max([img.shape[0] for img in images])

    # make the card to same size
    for i in range(len(images)):
        images[i] = cv.copyMakeBorder(images[i], 0, 0, 0, 
                                      max_width - images[i].shape[1], 
                                      cv.BORDER_CONSTANT, value=(255, 255, 255))

    # set the maximum height 
    new_height = 200

    # resize the images
    for i in range(len(images)):
        h, w = images[i].shape[:2]
        scale = new_height * 1.0 / h
        images[i] = cv.resize(images[i], (int(w*scale), int(h*scale)))
    
    return images, rank_suits

def get_convex_hull(images, min_area=100, box_width=68, box_height=150, verbose=False):

    xy0 = (0,0)
    xy1 = (box_width, box_height)


    # Get convex hulls within the bounding box for each card
    # and plot the convex hulls
    images_hull = []
    images_all = []
    convex_hulls = []
    for i in range(len(images)):
        img_copy = images[i].copy()
        # restrict to the bounding box
        img = img_copy[xy0[1]:xy1[1], xy0[0]:xy1[0]]
        # convert to grayscale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # take the negative
        gray = 255 - gray
        # threshold the image
        ret, thresh = cv.threshold(gray, 80, 255, 0)
        # find contours
        contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # get convex hull for each contour
        all_hulls = [cv.convexHull(c) for c in contours]
        # draw all hulls
        img_hulls = images[i].copy()
        for hull in all_hulls:
            img_hulls = draw_hull(img_hulls, hull)
        images_all.append(img_hulls)
        # get the areas
        areas = [cv.contourArea(c) for c in contours]
        # get the indexes of contours with areas larger than min_area
        indexes = [i for i in range(len(areas)) if areas[i] > min_area]
        # combined the selected contours
        cnt = np.concatenate([contours[i] for i in indexes])
        # get the convex hull
        hull = cv.convexHull(cnt)
        convex_hulls.append(hull)
        # draw the convex hull
        img_hull = draw_hull(img_copy, hull)
        images_hull.append(img_hull)

    if verbose:
        plt.figure(figsize=(15, 7))
        for i in range(13):
            for j in range(4):
                plt.subplot(4, 13, i + j*13 + 1)
                plt.imshow(images_all[i + j*13])
                plt.axis('off')


        plt.figure(figsize=(15, 7))
        for i in range(13):
            for j in range(4):
                plt.subplot(4, 13, i + j*13 + 1)
                plt.imshow(images_hull[i + j*13])
                plt.axis('off')

    return convex_hulls

def get_rois(convex_hulls):
    rois = []
    for i in range(len(images)):
        height, width = images[i].shape[:2]
        convex_full = convex_hulls[i]
        mask = np.zeros((height, width, 1), dtype=np.uint8)
        if len(convex_full) >= 3:
            pts = np.array(convex_full, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv.fillPoly(mask, [pts], 1)

        rois.append(mask)
    return rois


def rotate_image(img, angle, roi, mask=np.array([])):

    # Get the height and width of the image
    height, width = img.shape[:2]

    # Define the rotation angle in degrees and the rotation point

    rot_point = (width // 2, height // 2)

    # Calculate the rotation matrix using the rotation point and angle
    rot_matrix = cv.getRotationMatrix2D(rot_point, angle, 1.0)

    # Calculate the dimensions of the rotated image
    cos = abs(rot_matrix[0, 0])
    sin = abs(rot_matrix[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))

    # Adjust the rotation matrix to take into account the new image dimensions
    rot_matrix[0, 2] += (new_width / 2) - rot_point[0]
    rot_matrix[1, 2] += (new_height / 2) - rot_point[1]

    # Apply the rotation to the image using the new dimensions
    rotated_img = cv.warpAffine(img, rot_matrix, (new_width, new_height))

    # rotate roi
    rotated_roi = cv.warpAffine(roi, rot_matrix, (new_width, new_height))
    rotated_roi = rotated_roi.reshape((new_height, new_width, 1)).astype(np.uint8)

    # get mask of rotated rectangle (black background)
    if len(mask) == 0:
        mask = np.ones((height, width, 1), dtype=np.uint8)
        rotated_mask = cv.warpAffine(mask, rot_matrix, (new_width, new_height))
    else:
        rotated_mask = cv.warpAffine(mask, rot_matrix, (new_width, new_height))
    rotated_mask = rotated_mask.reshape((new_height, new_width, 1)).astype(np.uint8)
    # mask = np.zeros_like(rotated_img)
    # cv.rectangle(mask, (0,0), (width,height), (255,255,255), -1)
    # mask = cv.warpAffine(mask, rot_matrix, (new_width, new_height))
    # mask = (mask==(255,255,255)).astype(np.uint8)
    

    return rotated_img, rotated_roi, rotated_mask


## Blur the image
def blur_image(img, kernel_size=(5, 5)):
    # Blur the image
    blurred_img = cv.blur(img, ksize=kernel_size)

    return blurred_img

# Define a function to scale the image
def scale_image(img, roi, mask=np.array([]), fx=1.5, fy=1.5):
    
    # Scale the image
    scaled_img = cv.resize(img, None, fx=fx, fy=fy, interpolation=cv.INTER_CUBIC)
    scaled_height, scaled_width = scaled_img.shape[:2]


    if len(mask) == 0:
        scaled_mask = np.ones((scaled_height, scaled_width, 1), dtype=np.uint8) 
    else:
        scaled_mask = cv.resize(mask, None, fx=fx, fy=fy, interpolation=cv.INTER_CUBIC) 
    scaled_mask = scaled_mask.reshape((scaled_height, scaled_width, 1)).astype(np.uint8)
    scaled_roi = cv.resize(roi, None, fx=fx, fy=fy, interpolation=cv.INTER_CUBIC)
    scaled_roi = scaled_roi.reshape((scaled_height, scaled_width, 1)).astype(np.uint8)
    return scaled_img, scaled_roi, scaled_mask

def augment_image(img, roi, mask=np.array([]), angle_range=(0,360), kernel_range=(1,10), fx_range=(0.3,0.6), fy_range=(0.3,0.6)):
    # blur image
    kernel_size = (np.random.randint(kernel_range[0], kernel_range[1]), np.random.randint(kernel_range[0], kernel_range[1]))
    img = blur_image(img, kernel_size=kernel_size)    
    # scale image
    fx = np.random.uniform(fx_range[0], fx_range[1])
    fy = np.random.uniform(fy_range[0], fy_range[1])
    img, roi, mask = scale_image(img, roi, mask=mask, fx=fx, fy=fy)


    # rotate image
    angle = np.random.randint(angle_range[0], angle_range[1])
    img, roi, mask = rotate_image(img, angle, roi, mask=mask)  



    return img, roi, mask


def get_background_images(bg_directory):
    ## This part takes about 10 - 20 minutes to run

    if not path.exists(bg_directory):
        print("downloading background images...")
        print("This will take about 10 - 20 minutes to run.")
        url = 'https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.tar.gz'
        filename = 'dtd-r1.0.1.tar.gz'
        urllib.request.urlretrieve(url, filename)

        with tarfile.open(filename, 'r:gz') as tar:
            tar.extractall()

        print("Background images downloaded!")

    bg_files = glob(path.join(bg_directory,"*", "*.jpg"))
    return bg_files


# Define a function to randomly place an image on a background image
def place_image(bg_img, rois_bg,  img, roi, img_mask, threshold = 0.2):
    # get the height and width of the background image
    bg_height, bg_width = bg_img.shape[:2]
    # get the height and width of the image to be placed
    img_height, img_width = img.shape[:2]
    # get the randome location of the image to be placed
    x = np.random.randint(0, bg_width - img_width)
    y = np.random.randint(0, bg_height - img_height)

    # rescale the image mask
    img_mask_scale = np.zeros((bg_height, bg_width, 1))
    img_mask_scale[y:y+img_height, x:x+img_width,:] = img_mask
    img_mask_scale = img_mask_scale.astype(np.uint8)


    # check if the image overlaps with any of the rois
    areas = []
    new_rois = []
    for i in range(len(rois_bg)):
        roi_bg = rois_bg[i]['roi']
        covered_area = rois_bg[i]['covered_area']
        total_area = rois_bg[i]['total_area']
        new_roi = roi_bg * (1-img_mask_scale)
        area = np.sum(roi_bg * img_mask_scale)
        areas.append(area)
        new_rois.append(new_roi)
        if (covered_area + area)*1.0/total_area > threshold:
            return bg_img, rois_bg, False
    
    # update the covered area
    for i in range(len(rois_bg)):
        rois_bg[i]['covered_area'] += areas[i]
        rois_bg[i]['roi'] = new_rois[i]


    # place the image on the background
    img_scale = np.zeros_like(bg_img)
    img_scale[y:y+img_height, x:x+img_width] = img
    bg_img = bg_img * (1-img_mask_scale)
    bg_img += img_scale * img_mask_scale

    # create a mask from roi
    roi_bg = {}

    roi_mask = np.zeros((bg_height, bg_width, 1))
    roi_mask[y:y+img_height, x:x+img_width] = roi
    roi_mask = roi_mask.astype(np.uint8)
    roi_bg['roi'] = roi_mask
    roi_bg['covered_area'] = 0
    roi_bg['total_area'] = np.sum(roi_mask)
    rois_bg.append(roi_bg)

    return bg_img, rois_bg, True

def place_cards(bg_file, images, rois, iteration= 20, threshold = 0.2, angle_range=(-45,45), kernel_range=(1,2), dev_ratio = 0.7):
    bg_img = cv.imread(bg_file)
    height, width = bg_img.shape[:2]
    img_height, img_width = images[0].shape[:2]
    fraction =  min(height, width) * 1.0 / max(img_height, img_width) /6
    fx_range = (fraction - fraction * dev_ratio, fraction + fraction * dev_ratio)
    fy_range = (fraction - fraction * dev_ratio, fraction + fraction * dev_ratio)
    rois_bg = []
    image_idx = []
    threshold = 0.1
    for _ in range(iteration):
        # randomly select an image
        num_images = len(images)
        # randome index 
        index = np.random.randint(0, num_images)
        img = images[index]
        roi = rois[index]
        # augment the image
        scaled_img, roi, mask = augment_image(img, roi, angle_range=angle_range, kernel_range=kernel_range, fx_range=fx_range, fy_range=fy_range)

        bg_img, rois_bg, success = place_image(bg_img, rois_bg, scaled_img, roi, mask, threshold=threshold)
        if success:
            image_idx.append(index)

    boxes = []
    for roi in rois_bg:
        xy0, xy1 = get_corners(roi['roi'])
        boxes.append((xy0, xy1))

    return bg_img, boxes, image_idx

# function to return label texts
def get_label_text(bg_img, image_idx, boxes):
    h, w = bg_img.shape[:2]
    label_text = ""
    for i in range(len(image_idx)):
        (x_0, y_0), (x_1, y_1) = boxes[i]
        # convert them to relative coordinates
        x_0 = x_0*1.0/w
        y_0 = y_0*1.0/h
        x_1 = x_1*1.0/w
        y_1 = y_1*1.0/h
        # get center coordinates
        x_c = (x_0 + x_1)/2
        y_c = (y_0 + y_1)/2
        # get width and height
        rect_width = x_1 - x_0
        rect_height = y_1 - y_0

        label_text += "{} {} {} {} {}\n".format(image_idx[i], x_c, y_c, rect_width, rect_height)
    return label_text


# functioin for generating data
def generate_data(directory, num_data, bg_files, images, rois, iteration= 10,
                  threshold = 0.2, angle_range=(0,360), 
                  kernel_range=(1,5), dev_ratio = 0.5):
    # create the directory if it does not exist
    if path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

    # get the number of background images
    num_bg = len(bg_files)
    # create images directory
    images_dir = path.join(directory, "images")
    if not path.exists(images_dir):
        os.makedirs(images_dir)
    # create labels directory
    labels_dir = path.join(directory, "labels")
    if not path.exists(labels_dir):
        os.makedirs(labels_dir)
    
    # iterate over the number of data
    # start progress bar
    
    for i in tqdm(range(num_data)):
        # randomly select a background image file
        bg_file = bg_files[np.random.randint(0, num_bg)]
        # place the cards
        bg_img, boxes, image_idx = place_cards(bg_file, images, rois, iteration= iteration,threshold=threshold, 
                                               angle_range=angle_range, kernel_range=kernel_range, dev_ratio = dev_ratio)
        # save the image
        cv.imwrite(path.join(images_dir, "{}.jpg".format(i)), bg_img)
        # get the label text
        label_text = get_label_text(bg_img, image_idx, boxes)
        # save the label text
        with open(path.join(labels_dir, "{}.txt".format(i)), "w") as f:
            f.write(label_text)


def generate_dataset(data_dir, bg_files, images, rois, n_train, n_valid, n_test, **kargs):
    if not path.exists(data_dir):
        os.makedirs(data_dir)

    train_dir = path.join(data_dir, "train")
    print("Generating training data in: {}".format(train_dir) )
    generate_data(train_dir, n_train, bg_files, images, rois, **kargs)

    valid_dir = path.join(data_dir, "valid")
    print("Generating validation data in: {}".format(valid_dir) )   
    generate_data(valid_dir, n_valid, bg_files, images, rois,  **kargs)

    test_dir = path.join(data_dir, "test")
    print("Generating test data in: {}".format(test_dir) )
    generate_data(test_dir, n_test, bg_files, images, rois, **kargs)

# Create data.yaml file
def create_yaml_text(data_dir, classes):
    data_yaml = dict(
        train = path.join('../'+path.basename(data_dir)+ '/train/images'),
        val = path.join('../'+path.basename(data_dir)+ '/valid/images'),
        nc = len(classes),
        names = classes
    )
    return data_yaml


# main function
if __name__ == "__main__":

    # Create the parser
    parser = argparse.ArgumentParser(description='Generates training data.')

    # Add command-line options
    parser.add_argument('-i', '--inputdir', default='rank_suit_wsop', 
                        help='Specify input directory.')
    parser.add_argument('-b', '--bgdir', default='dtd-r1/images', 
                        help='Specify background image directory.')
    parser.add_argument('-o', '--outdir', default='data', help='Specify output directory.')
    parser.add_argument('-v', '--verbose', default=True, action='store_true', 
                        help='Enable verbose mode.')
    parser.add_argument('-t', '--ntrain', default=10, help='Number of train images.')
    parser.add_argument('-l', '--nvalid', default=10, help='Number of valid images.')
    parser.add_argument('-e', '--ntest', default=10, help='Number of test images.')


    kargs = {}
    kargs['angle_range'] = (-45,45)
    kargs['iteration'] = 20
    kargs['kernel_range'] = (1,2)
    kargs['dev_ratio'] = 0.5

    # Parse the command-line arguments
    args = parser.parse_args()
    
    # get the card images
    images, rank_suits = get_card_images(args.inputdir)

    # get the convex hulls around the rank and the suit
    convex_hulls = get_convex_hull(images, verbose=args.verbose)

    # Convert the convex hulls (polygons) to masks
    rois = get_rois(convex_hulls)

    # get back ground image file paths
    bg_files = get_background_images(args.bgdir)
   
    # generate the dataset
    generate_dataset(args.outdir, bg_files, images, rois, args.ntrain, args.nvalid, args.ntest, 
                  **kargs)

    # create data.yaml file
    data_yaml = create_yaml_text(args.outdir, rank_suits)
    data_yaml_path = path.join(args.outdir, "data.yaml")
    with open(data_yaml_path, "w") as f:
        yaml.dump(data_yaml, f)