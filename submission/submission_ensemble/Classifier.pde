class Classifier {
  PApplet parent;
  ArrayList<Sample> trainingSamples;

  Classifier(PApplet parent) {
    this.parent = parent;
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
  
  void reset(){
    trainingSamples.clear();
  }
  
  // implemented in sub-class
  void train(){}
  
  // implemented in sub-class
  double predict(Sample sample){
    return 0.0;
  }
}
