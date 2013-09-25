import java.util.Map;

RandomForest classifier;

boolean trainingMode = false;
String trainingFile = "train_minimal.csv";

Sample[] samplesFromTable(Table table, boolean isTraining) {
  Sample[] result = new Sample[table.getRowCount() - 1];
  int numFeatures = 4;

  // training has an extra column for the label
  if (isTraining) {
    if (numFeatures != table.getColumnCount()-2) {
      throw new RuntimeException("numFeatures an input data don't match! numFeatures: " + numFeatures + " colCount: " + (table.getColumnCount()-2));
    }
  } 
  else {
    if (numFeatures != table.getColumnCount()-1) {
      throw new RuntimeException("numFeatures an input data don't match! numFeatures: " + numFeatures + " colCount: " + (table.getColumnCount()-2));
    }
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

void setup() {
  // necessary to initialize opencv
  OpenCV opencv = new OpenCV(this, 0, 0);

  Table trainingData = loadTable(trainingFile);
  Sample[] trainingSamples = samplesFromTable(trainingData, true);

  if (trainingMode) {


    HashMap<String, Float> averages = crossfold(3, trainingSamples);

    println();
    println("===Average Results===");
    println("Accuracy: " + averages.get("accuracy"));
    println("Precision: " + averages.get("precision"));
    println("Recall: " + averages.get("recall"));
    println("F-measure: " + averages.get("fmeasure"));
  } 
  else {
    classifier = new RandomForest();
    classifier.addTrainingSamples(trainingSamples);
    classifier.train();
    
    Table testingData = loadTable("test_minimal.csv");
    Sample[] testSamples = samplesFromTable(testingData, false);   

    String[] rows = new String[testSamples.length+1];
    // add header
    rows[0] = "PassengerId,Survived";

    for (int i = 0; i < testSamples.length; i++) {
      int prediction = (int)classifier.predict(testSamples[i]);
      rows[i+1] = testSamples[i].recordId + "," + prediction;
    }

    saveStrings("predictions.csv", rows);
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

  classifier = new RandomForest();
  classifier.addTrainingSamples(training);
  classifier.train();

  for (Sample sample : testing) {
    double prediction = classifier.predict(sample);
    score.addResult((int)prediction == 1, (int)prediction == sample.label);
  }

  println("Accuracy: "+ score.getAccuracy() +" Precision: " + score.getPrecision() + " Recall: " + score.getRecall() + " F-measure: " + score.getFMeasure());

  return score;
}

void draw() {
}
