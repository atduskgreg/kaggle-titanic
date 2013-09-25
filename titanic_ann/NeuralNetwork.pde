import gab.opencv.*;
import java.util.Arrays;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.TermCriteria;
import org.opencv.ml.CvANN_MLP;
import org.opencv.ml.CvANN_MLP_TrainParams;
import org.opencv.core.Size;
import org.opencv.core.Scalar;

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

class NeuralNetwork {  
  CvANN_MLP classifier;

  ArrayList<Sample> trainingSamples;

  NeuralNetwork() {
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

    CvANN_MLP_TrainParams params = new CvANN_MLP_TrainParams();

//    params.set_train_method(CvANN_MLP_TrainParams.BACKPROP );
//    params.set_bp_dw_scale(0.01);
//    params.set_bp_moment_scale(0.05);
//    params.set_term_crit(new TermCriteria(TermCriteria.EPS + TermCriteria.COUNT, 10000, 0.1));


    

    Mat layersMat = new Mat(3, 1, CvType.CV_32SC1);
    layersMat.put(0,0, 7);
    layersMat.put(1,0, 15);
//    layersMat.put(2,0, 30);
//     layersMat.put(3,0, 10);

    layersMat.put(2,0, 1);
    
classifier = new CvANN_MLP(layersMat);

    classifier.train(trainingMat, labelMat, new Mat(), new Mat(), params, CvANN_MLP_TrainParams.BACKPROP);


  }

  // Use this function to get a prediction, after having trained the algorithm.

  double predict(Sample sample) {
    Mat predictionTraits = new Mat(1, sample.featureVector.length, CvType.CV_32FC1);
    predictionTraits.put(0, 0, sample.featureVector);

    Mat result = new Mat();
    classifier.predict(predictionTraits, result);

    return result.get(0,0)[0];
  }
}

