import cv2
from train import processFiles, trainSVM
from detector import Detector

import warnings
warnings.filterwarnings("ignore")

# Replace these with the directories containing your
# positive and negative sample images, respectively.
pos_dir = "samples/vehicles"
neg_dir = "samples/non-vehicles"

# Replace this with the path to your test video file.
video_file = "videos/project_video.mp4"


def experiment1():
    """
    Train a classifier and run it on a video using default settings
    without saving any files to disk.
    """
    # TODO: You need to adjust hyperparameters
    # Extract HOG features from images in the sample directories and 
    # return results and parameters in a dict.
    feature_data = processFiles(pos_dir=pos_dir, neg_dir=neg_dir, recurse=True,
                output_file=False, output_filename="result_video",
                color_space="bgr", channels=[0, 1, 2],
                hog_features=True, hist_features=True, spatial_features=True,
                hog_lib="sk", size=(48, 48), hog_bins=8, pix_per_cell=(8, 8),
                cells_per_block=(2, 2), block_stride=None, block_norm="L2",
                transform_sqrt=True, signed_gradient=False, hist_bins=8,
                spatial_size=(8, 8))


    # Train SVM and return the classifier and parameters in a dict.
    # This function takes the dict from processFiles() as an input arg.
    classifier_data = trainSVM(feature_data=feature_data)


    # TODO: You need to adjust hyperparameters of loadClassifier() and detectVideo()
    #       to obtain a better performance

    # Instantiate a Detector object and load the dict from trainSVM().
    detector = Detector(
        init_size=(64, 64),
        x_overlap=0.24, y_step=0.007,
        x_range=(0.5, 0.93), y_range=(0.55, 0.85),
        scale=1.2).loadClassifier(classifier_data=classifier_data)
  
    # Open a VideoCapture object for the video file.
    cap = cv2.VideoCapture(video_file)

    # Start the detector by supplying it with the VideoCapture object.
    # At this point, the video will be displayed, with bounding boxes
    # drawn around detected objects per the method detailed in README.md.
    detector.detectVideo(video_capture=cap,
                num_frames=4, threshold=47,
                min_bbox=(32, 32),
                show_video=True, draw_heatmap=True, draw_heatmap_size=0.3,
                write=True, write_fps=24)


# def experiment2
#    ...

if __name__ == "__main__":
    experiment1()
    # experiment2() may you need to try other parameters
    # experiment3 ...


