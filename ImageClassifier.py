import os
from sklearn.model_selection import train_test_split
from ImageProcessingSupport import *
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import time
import pickle

color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32)  # Spatial binning dimensions
spatial_feat = True  # Spatial features on or off
hist_feat = False  # Histogram features on or off
hog_feat = True  # HOG features on or off
y_start_stop = [None, None]  # Min and max in y to search in slide_window()

def train_the_classifier():
    # Read in cars and notcars
    vroot = 'datasets\\vehicles'
    nvroot = 'datasets\\non-vehicles'
    cars = []
    notcars = []
    for path, subdirs, files in os.walk(vroot):
        for name in files:
            cars.append(os.path.join(path, name))
    for path, subdirs, files in os.walk(nvroot):
        for name in files:
            notcars.append(os.path.join(path, name))

    # Reduce the sample size because
    # The quiz evaluator times out after 13s of CPU time
    sample_size = 5000
    cars = cars[0:sample_size]
    notcars = notcars[0:sample_size]
    car_features = extract_features(cars, color_space=color_space,
                                    spatial_size=spatial_size,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space,
                                       spatial_size=spatial_size,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,hog_feat=hog_feat)
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)
    print('Using:', orient, 'orientations', pix_per_cell,
          'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    return svc, X_scaler


def get_classifier():
    if os.path.isfile('classifier.pkl'):
        with open('classifier.pkl', mode='rb') as f:
            data = pickle.load(f)
        svc = data['svc']
        X_scaler = data['X_scaler']
        print('Classifier file found and Loaded.')
    else:
        svc, X_scaler = train_the_classifier()
        data1 = {'svc': svc,
                 'X_scaler': X_scaler}
        output = open('classifier.pkl', 'wb')
        pickle.dump(data1, output)
        print('Classifier file saved.')

    return svc , X_scaler