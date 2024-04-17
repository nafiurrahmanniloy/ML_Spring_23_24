import numpy as np
import matplotlib.pyplot as plt
import data_utils
import download

def download_data():
    # Download CIFAR-10 data if not downloaded already
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    download_dir = "./data"
    download.maybe_download_and_extract(url, download_dir)

# Class to initialize and apply K-nearest neighbor classifier
class KNearestNeighbor(object):
    def __init__(self):
        pass

    # Method to initialize classifier with training data
    def train(self, X, y):
        self.X_train = X
        self.y_train = y
    
    # Method to predict labels of test examples using 'compute_distances' and 'predict_labels' methods.
    def predict(self, X, k=1, num_loops=0):
        if num_loops == 0:
            dists = self.compute_distances(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)
        return self.predict_labels(dists, k=k)

    # Method to compute Euclidean distances from each text example to every training example  
    def compute_distances(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train)) 
        #my code
        dists = np.sqrt(np.sum(np.square(self.X_train), axis=1) + np.sum(np.square(X), axis=1)[:, np.newaxis] - 2 * np.dot(X, self.X_train.T))
        pass
        return dists

    # Method to predict labels of test examples using chosen value of k given Euclidean distances obtained from 'compute_distances' method.
    def predict_labels(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        #my code
        for i in range(num_test):
            closest_y = []
            sorted_dist = np.argsort(dists[i])
            closest_y = list(self.y_train[sorted_dist[0:k]])
            pass
            y_pred[i]= (np.argmax(np.bincount(closest_y)))
            pass
        return y_pred
    
def visualize_data(X_train, y_train):
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)
    samples_per_class = 7
    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(y_train == y)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X_train[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.show()

if __name__ == "__main__":
    # Download CIFAR10 data and store it in the current directory if you have not done it.
    #download_data()
    cifar10_dir = './data/cifar-10-batches-py'

    # Load training and testing data from CIFAR10 dataset
    X_train, y_train, X_test, y_test = data_utils.load_CIFAR10(cifar10_dir)

    # Checking the size of the training and testing data
    print('Training data shape: ', X_train.shape)
    print('Training labels shape: ', y_train.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)

    # Memory error prevention by subsampling data. We sample 10000 training examples and 1000 test examples.
    num_training = 10000
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]

    num_test = 1000
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # reshaping data and placing into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))

    print(X_train.shape, X_test.shape) # X_train should be (10000, 3072) and X_test should be (1000, 3072)

    # Performing KNN
    classifier = KNearestNeighbor()

    # 1) Initialize classifier with training data
    classifier.train(X_train, y_train)

    # 2) Use classifier to compute distances from each test example in X_test to every training example
    dists = classifier.compute_distances(X_test)
    
    # 3) Use classifier to predict labels of each test example in X_test using k=5 
    y_test_pred = classifier.predict_labels(dists, k=5)

    num_correct = np.sum(y_test_pred == y_test) # number of test examples correctly predicted, where y_test_pred
                                                # should contain labels predicted by classifier
    accuracy = float(num_correct) / num_test
    print('Got %d / %d correct with k=5 => accuracy: %f' % (num_correct, num_test, accuracy))
    # Accuracy above should be ~ 29-30%

    # Perform 5-fold cross validation to find optimal k from choices below
    num_folds = 5
    k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

    X_train_folds = []
    y_train_folds = []

    X_train_folds = np.array_split(X_train, num_folds)
    y_train_folds = np.array_split(y_train, num_folds)
    
    k_to_accuracies = {} # dictionary to hold validation accuracies for each k 

    for k in k_choices:
        k_to_accuracies[k] = [] # each key, k, should hold its list of 5 validation accuracies
        for num_knn in range(0,num_folds):
                        
            # Split training data into validation fold and training folds

            X_test = X_train_folds[num_knn]
            y_test = y_train_folds[num_knn]
            X_train = X_train_folds
            y_train = y_train_folds
        
            temp = np.delete(X_train,num_knn,0)
            X_train = np.concatenate((temp),axis = 0)
            y_train = np.delete(y_train,num_knn,0)
            y_train = np.concatenate((y_train),axis = 0)

            # Initialize classifier with training folds and compute distances between 
            # examples in validation fold and training folds
            classifier = KNearestNeighbor()
            classifier.train(X_train, y_train)
            
            # Compute distances and predict labels for the validation set
            dists = classifier.compute_distances(X_test)
            y_test_pred = classifier.predict_labels(dists, k)
                    
        # Compute accuracy and append to the list for the current k
        num_correct = np.sum(y_test_pred == y_test)
        accuracy = float(num_correct) / num_test
        #print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
        k_to_accuracies[k].append(accuracy)

    # Print the accuracies for varying values of k
    print("Printing our 5-fold accuracies for varying values of k:")
    print()
    for k in sorted(k_to_accuracies):
        for accuracy in k_to_accuracies[k]:
            print('k = %d, accuracy = %f' % (k, accuracy))
    
    # Print the average accuracies for each k
    for k in sorted(k_to_accuracies):
        print('k = %d, avg. accuracy = %f' % (k, np.mean(k_to_accuracies[k])))
    
    # Plotting the accuracies
    for k in k_choices:
        accuracies = k_to_accuracies[k]
        plt.scatter([k] * len(accuracies), accuracies)

    # plot the trend line with error bars that correspond to standard deviation

    accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
    accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
    plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
    plt.title('Cross-validation on k')
    plt.xlabel('k')
    plt.ylabel('Cross-validation accuracy')
    plt.show()

    # Choose the best value of k based on cross-validation results
    # Choosing best value of k based on cross-validation results
    
    best_k = 10
    print("Best value of k based on cross-validation:", best_k)
    classifier = KNearestNeighbor()
    classifier.train(X_train, y_train)
    y_test_pred = classifier.predict(X_test, k=best_k)

    # Computing and displaying the accuracy for best k found during cross-validation
    num_correct = np.sum(y_test_pred == y_test)
    accuracy = float(num_correct) / num_test
    print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))