import psvm.*;

class Libsvm {  
  SVM model;
  ArrayList<Sample> trainingSamples;
  PApplet parent;

  Libsvm(PApplet parent) {
    trainingSamples = new ArrayList<Sample>();
    this.parent = parent;
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

  float[][] samplesToArrays(ArrayList<Sample> samples) {
    float[][] result = new float[samples.size()][samples.get(0).featureVector.length];


    return result;
  }

  float[] doubleToFloat(double[] input){
    float[] result = new float[input.length];
    for(int i = 0; i < input.length; i++){
      result[i] = (float)input[i];
    }
    
    return result;
  }
  
  void train() {  

    float[][] trainingVectors = new float[trainingSamples.size()][trainingSamples.get(0).featureVector.length];
    int[] labels = new int[trainingSamples.size()];
    
    for(int i = 0; i < trainingSamples.size(); i++){
      trainingVectors[i] = doubleToFloat(trainingSamples.get(i).featureVector);
      labels[i] = trainingSamples.get(i).label;
    }
    
    model = new SVM(parent);

    model.params.kernel_type = SVM.RBF_KERNEL;

    SVMProblem problem = new SVMProblem();
    problem.setNumFeatures(2);
    problem.setSampleData(labels, trainingVectors);
    model.train(problem);
  }

  // Use this function to get a prediction, after having trained the algorithm.
  double predict(Sample sample) {
    return model.test(doubleToFloat(sample.featureVector));
  }
}
