import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize
from scipy.spatial.distance import cdist
from nltk import word_tokenize
from skimage import feature as ft
import numpy

# Load data and show some images.
data = pickle.load(open('mscoco_small.p'))
train_data = data['train']
val_data = data['val']

def preparedata(data, Qnumber):

    features = np.zeros((len(data['images']), 729), dtype=np.float)
    for (counter, image_id) in enumerate(data['images']):
        image = imread('mscoco/%s' % image_id)
        #tiny_image = imresize(image, (16, 16), interp='nearest')
        if Qnumber == 1:
            tiny_image = imresize(image, (16, 16), interp='nearest')
            features[counter, :] = tiny_image.flatten().astype(np.float) / 255
        if Qnumber == 3:
            features[counter, :] = ft.hog(image, orientations=9, pixels_per_cell=(38, 38), cells_per_block=(3, 3),
                              block_norm='L2-Hys', visualize=False, visualise=None, transform_sqrt=False,
                              feature_vector=True, multichannel=None)  # return HOG map

        if (1 + counter) % 10000 == 0:
            print('Computed features for %d data' % (1 + counter))
    return features


def retrieve(sampleTestImageId,train_features,val_features):

    # Retrieve the feature vector for this image.
    sampleImageFeature = val_features[sampleTestImageId: sampleTestImageId + 1, :]
    # Compute distances between this image and the training set of images.
    distances = cdist(sampleImageFeature, train_features, 'correlation')
    # Compute ids for the closest images in this feature space.
    nearestNeighbors = np.argsort(distances[0, :])  # Retrieve the nearest neighbors for this image.
    return nearestNeighbors

def bleu(sampleTestImageId,nearestNeighbors):
    reference = [w.lower() for w in word_tokenize(val_data['captions'][sampleTestImageId])]
    candidate = [w.lower() for w in word_tokenize(train_data['captions'][nearestNeighbors[0]])]

    print ('ref', reference)
    print ('cand', candidate)

    bleu_score = float(len(set(reference) & set(candidate))) / len(candidate)
    return bleu_score


def main():

    scores = []
    #print len(val_data['captions'])


    #get features
    train_features = preparedata(train_data,3)
    val_features = preparedata(val_data,3)

    for i in range(len(val_data['captions'])):
        score = bleu(i,retrieve(i,train_features,val_features))
        scores.append(score)

    print np.sum(scores)/len(scores)

    #Question 2


if __name__ == '__main__':
    main()

