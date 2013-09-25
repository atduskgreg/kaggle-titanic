
import gab.opencv.*;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.TermCriteria;

import java.util.Map;
import java.util.Arrays;

Classifier[] classifiers;

boolean trainingMode = true;
String trainingFile = "train_minimal.csv";

Sample[] samplesFromTable(Table table, boolean isTraining) {
  Sample[] result = new Sample[table.getRowCount() - 1];
  int numFeatures = 4;

  if (numFeatures != table.getColumnCount()-2) {
    throw new RuntimeException("numFeatures an input data don't match! numFeatures: " + numFeatures + " colCount: " + (table.getColumnCount()-2));
  }

  for (int row = 1; row < table.getRowCount(); row++) {
    Sample sample = new Sample(numFeatures);

    sample.setRecordId(table.getInt(row, 0));

    int lastCol;
    if (isTraining) {
      // col0 is passengerId and colTheLast is survived
      lastCol = table.getColumnCount()-1;
    } 
    else {
      lastCol = table.getColumnCount();
    }

    for (int col = 1; col < lastCol; col++) {
      sample.featureVector[col-1] = table.getFloat(row, col);
    }

    if (isTraining) {
      sample.setLabel(table.getInt(row, table.getColumnCount()-1));
    }

    result[row-1] = sample;
  }

  return result;
}

void populateClassifiers() {
  classifiers[0] = new AdaBoost(this);
  classifiers[1] = new RandomForest(this);
  classifiers[2] = new Libsvm(this);
  classifiers[3] = new KNN(this);
}

void setup() {
  // necessary to initialize opencv
  OpenCV opencv = new OpenCV(this, 0, 0);

  Table trainingData = loadTable(trainingFile);

  Sample[] allSamples = samplesFromTable(trainingData, true);

  classifiers = new Classifier[4];
  populateClassifiers();

  if (trainingMode) {
    HashMap<String, Float> averages = crossfold(3, allSamples);

    println();
    println("===Average Results===");
    println("Accuracy: " + averages.get("accuracy"));
    println("Precision: " + averages.get("precision"));
    println("Recall: " + averages.get("recall"));
    println("F-measure: " + averages.get("fmeasure"));
  } 
  else {
    //    classifier = new AdaBoost();
    //    classifier.addTrainingSamples(allSamples);
    //    classifier.train();
    for (int i = 0; i < classifiers.length; i++) {
      classifiers[i].addTrainingSamples(allSamples);
      classifiers[i].train();
    }


    // HERE: run test data through the ruby script!

    Table testingData = loadTable("test_minimal.csv");
    //    Sample[] testSamples = samplesFromTable(testingData, false);
    //
    //    String[] rows = new String[testSamples.length];
    //
    //    for (int i = 0; i < testSamples.length; i++) {
    //      int prediction = (int)classifier.predict(testSamples[i]);
    //      rows[i] = testSamples[i].recordId + "," + prediction;
    //    }
    //
    //    saveStrings("predictions.csv", rows);
  }
}

HashMap crossfold(int nFolds, Sample[] samples) {
  HashMap<String, Float> result = new HashMap<String, Float>();
  result.put("accuracy", 0.0);
  result.put("precision", 0.0);
  result.put("recall", 0.0);
  result.put("fmeasure", 0.0);

  ArrayList<ArrayList<Sample>> folds = new ArrayList<ArrayList<Sample>>();
  for (int i = 0; i < nFolds; i++) {
    folds.add(new ArrayList<Sample>());
  }

  for (int i = 0; i < samples.length; i++) {
    int fold = (int)random(0, nFolds);

    folds.get(fold).add(samples[i]);
  }

  for (int i = 0; i < folds.size(); i++) {
    ArrayList<Sample> testing = folds.get(i);
    ArrayList<Sample> training = new ArrayList<Sample>();

    for (int j = 0; j < folds.size(); j++) {
      if (j != i) {
        training.addAll(folds.get(j));
      }
    }

    println();
    println("Executing fold " + (i+1) + "...");
    ClassificationResult score = executeFold(training, testing);

    println("training size: " + training.size() + " testing size: " + testing.size());
    result.put("accuracy", result.get("accuracy") + score.getAccuracy());
    result.put("precision", result.get("precision") + score.getPrecision());
    result.put("recall", result.get("recall") + score.getRecall());
    result.put("fmeasure", result.get("fmeasure") + score.getFMeasure());
  }

  result.put("accuracy", result.get("accuracy")/nFolds);
  result.put("precision", result.get("precision")/nFolds);
  result.put("recall", result.get("recall")/nFolds);
  result.put("fmeasure", result.get("fmeasure")/nFolds);

  return result;
}

ClassificationResult executeFold(ArrayList<Sample> training, ArrayList<Sample> testing) {

  ClassificationResult score = new ClassificationResult();

populateClassifiers();
  for (int i = 0; i < classifiers.length; i++) {
    println("Training classifier " + i);
    classifiers[i].addTrainingSamples(training);
    classifiers[i].train();
  }

  for (Sample sample : testing) {
    float averagePrediction = 0.0; 

    for (int j = 0; j < classifiers.length; j++) {
      double prediction = classifiers[j].predict(sample);
      averagePrediction += prediction;
    }

    averagePrediction = averagePrediction/classifiers.length;

    println("ave predict: " +averagePrediction);

    int jointPrediction;
    if (averagePrediction >= 0.3) {
      jointPrediction = 1;
    } 
    else {
      jointPrediction = 0;
    }

    println("ap: " + averagePrediction +  " jp: "+ jointPrediction + " label: " + sample.label);
    //double prediction = classifier.predict(sample);
    score.addResult((int)jointPrediction == 1, (int)jointPrediction == sample.label);
  }

  println();
  println("==========");
  println("Accuracy: "+ score.getAccuracy() +" Precision: " + score.getPrecision() + " Recall: " + score.getRecall() + " F-measure: " + score.getFMeasure());
  println("==========");

  println();

  return score;
}

void draw() {
}
