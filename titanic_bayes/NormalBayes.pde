import gab.opencv.*;
import java.util.Arrays;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.TermCriteria;
import org.opencv.ml.CvNormalBayesClassifier;

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

class NormalBayes {  
  CvNormalBayesClassifier classifier;
  ArrayList<Sample> trainingSamples;

  NormalBayes() {
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

      for (int j = 0; j < trainingSample.featureVector.length; j++) {              
        trainingMat.put(i, j, trainingSample.featureVector[j]);
      }

      labelMat.put(i, 0, trainingSample.label);
    }
    
    classifier = new CvNormalBayesClassifier();
    classifier.train(trainingMat, labelMat, new Mat(), new Mat(), false);

  }

  // Use this function to get a prediction, after having trained the algorithm.

  float predict(Sample sample) {
    // create a mat for the prediction
    Mat predictionTraits = new Mat(1, sample.featureVector.length, CvType.CV_32FC1);

    predictionTraits.put(0, 0, sample.featureVector);

    return classifier.predict(predictionTraits);
  }
}

