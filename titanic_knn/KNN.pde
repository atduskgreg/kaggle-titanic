import gab.opencv.*;
import java.util.Arrays;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.TermCriteria;
import org.opencv.ml.CvKNearest;

class Sample {
  double[] featureVector;
  int label;

  Sample(double[] featureVector, int label) {
    this.featureVector = featureVector;
    this.label = label;
  }
  Sample(int featureVectorSize) {
    featureVector = new double[featureVectorSize];
  }

  void setLabel(int label) {
    this.label = label;
  }
}

class KNN {  
  CvKNearest classifier;
  ArrayList<Sample> trainingSamples;

  KNN() {
    trainingSamples = new ArrayList<Sample>();
  }

  void addTrainingSample(double[] featureVector, int label) {
    addTrainingSample(new Sample(featureVector, label));
  }

  void addTrainingSample(Sample sample) {
    trainingSamples.add(sample);
  }

  void addTrainingSamples(Sample[] samples) {
    trainingSamples =  new ArrayList<Sample>(Arrays.asList(samples));
  }  

  void addTrainingSamples(ArrayList<Sample> samples) {
    trainingSamples.addAll(samples);
  }

  void train() {  
    Mat trainingMat = new Mat(trainingSamples.size(), trainingSamples.get(0).featureVector.length, CvType.CV_32FC1);
    Mat labelMat = new Mat( trainingSamples.size(), 1, CvType.CV_32FC1);

    // load samples into training and label mats. 
    for (int i = 0; i < trainingSamples.size(); i++) {
      Sample trainingSample = trainingSamples.get(i);

      //trainingMat.put(0, i, trainingSample.featureVector);
      for (int j = 0; j < trainingSample.featureVector.length; j++) {              
        trainingMat.put(i, j, trainingSample.featureVector[j]);
      }

      labelMat.put(i, 0, trainingSample.label);
    }

    classifier = new CvKNearest();
    classifier.train(trainingMat, labelMat);
  }

  // Use this function to get a prediction, after having trained the algorithm.
  double predict(Sample sample) {
    Mat predictionTraits = new Mat(1, sample.featureVector.length, CvType.CV_32FC1);
    predictionTraits.put(0, 0, sample.featureVector);

    return classifier.find_nearest(predictionTraits, 4, new Mat(), new Mat(), new Mat());
  }
}

