## Kaggle Titanic in OpenCV for Processing

I'm using the [Kaggle Titanic Challenge](http://www.kaggle.com/c/titanic-gettingStarted) as a platform for adding machine learning support to my [OpenCV for Processing](https://github.com/atduskgreg/opencv-processing) library.

OpenCV supports:

* Naive Bayes
* Adaptive Neural Nets
* K-Nearest Neighbor
* AdaBoost
* SVM
* Random Decision Forests

and a couple of other techniques.

So far, I've wrapped:

* KNN
* AdaBoost
* Random Decision Forests
* SVM

with an in-progress Naive Bayes as well (needs debugging).

Thanks to [Rune Madsen](http://github.com/runemadsen) for starting the RDF implementation.

_I've also included an implementation with my libsvm-based [Processing-SVM](https://github.com/atduskgreg/processing-svm) library for comparison with OpenCV's svm implementation. For some reason libsvm produces signficantly better results with the same parameters._