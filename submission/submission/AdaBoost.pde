import org.opencv.ml.CvBoost;
import org.opencv.ml.CvBoostParams;
import org.opencv.core.Range;

class AdaBoost {  
  CvBoost classifier;

  ArrayList<Sample> trainingSamples;

  AdaBoost() {
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

    Mat varType = new Mat(trainingMat.width()+1, 1, CvType.CV_8U );
    varType.setTo(new Scalar(0)); // 0 = CV_VAR_NUMERICAL.
    varType.put(trainingMat.width(), 0, 1); // 1 = CV_VAR_CATEGORICAL;

    // Begin magic numbers...
    // TODO: make this setable.

    CvBoostParams params = new CvBoostParams();
    params.set_boost_type(CvBoost.DISCRETE);
    params.set_weight_trim_rate(0);
//    params.set_weak_count(50000);
    params.set_cv_folds(3);
   

    classifier = new CvBoost();
    classifier.train(trainingMat, 1, labelMat, new Mat(), new Mat(), varType, new Mat(), params, false);
//    classifier.prune(new Range(0, (int)(params.get_weak_count() * 0.2)));
    
    
  }

  // Use this function to get a prediction, after having trained the algorithm.

  double predict(Sample sample) {
    // create a mat for the prediction
    Mat predictionTraits = new Mat(1, sample.featureVector.length, CvType.CV_32FC1);
    predictionTraits.put(0, 0, sample.featureVector);

    return classifier.predict(predictionTraits);
  }
}
