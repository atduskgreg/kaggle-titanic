import psvm.*;

SVM model;
float[][] trainingVectors;
float[][] testVectors;
int[] labels;
int[] sort;

int numFeatures = 7;

// COLUMN LABELS: 
// PassengerId  Survived  Pclass  Name  Sex  Age  SibSp  Parch  Ticket  Fare  Cabin  Embarked
//  0            1        2        3    4    5    6      7      8        9    10      11
// Skip: PassengerId, Name, Ticket, Cabin
// Total number of cols in the feature vector: 12-4-1 = 7

// TODO: do this right!

// loads into existing trainingVectors, testVectors 
void loadSamplesFromTable(Table table, boolean isTraining) {

  int offset = 0;
  if (!isTraining) {
    offset = -1;
  }

  int currTrain = 0;  
  int currTest = 0;

  // skip the top row which is labels
  for (int row = 1; row < table.getRowCount(); row++) {



    if (sort[row-1] == 0) {
      trainingVectors[currTrain][0] = table.getInt(row, 2+offset);
    } 
    else {
      testVectors[currTest][0] = table.getInt(row, 2+offset);
    }

    int sex;
    if (table.getString(row, 4+offset) == "male") {
      sex = 0;
    } 
    else {
      sex = 1;
    }

    if (sort[row-1] == 0) {
      trainingVectors[currTrain][1] = sex;
    } 
    else {
      trainingVectors[currTest][1] = sex;
    }

    if (sort[row-1] == 0) {
      trainingVectors[currTrain][2] = table.getFloat(row, 5+offset);
    }
    else {
      testVectors[currTest][2] = table.getFloat(row, 5+offset);
    }

    if (sort[row-1] == 0) {
      trainingVectors[currTrain][3] = table.getInt(row, 6+offset);
    }
    else {  
      testVectors[currTest][3] = table.getInt(row, 6+offset);
    }

    if (sort[row-1] == 0) {
      trainingVectors[currTrain][4] = table.getInt(row, 7+offset);
    }
    else {
      testVectors[currTest][4] = table.getInt(row, 7+offset);
    }

    if (sort[row-1] == 0) {
      trainingVectors[currTrain][5] = table.getFloat(row, 9+offset);
    }
    else {
      testVectors[currTest][5] = table.getFloat(row, 9+offset);
    }

    String embarked = table.getString(row, 11+offset);
    int embarkedCode;

    if (embarked == "S") {
      embarkedCode = 0;
    } 
    else if (embarked == "C") {
      embarkedCode = 1;
    } 
    else { // "Q"
      embarkedCode = 2;
    }


    if (sort[row-1] == 0) {
      trainingVectors[currTrain][6] = embarkedCode;
    }
    else {
      testVectors[currTest][6] = embarkedCode;
    }

    if (isTraining && sort[row-1] == 0) {
      labels[currTrain] = table.getInt(row, 1);
      currTrain++;
    } else{
      currTest++;
    }
  }
}

SVM classifier;


void setup() {
  Table trainingData = loadTable("train.csv");


  float percentTraining = 0.66;

  int numTraining = 0;
  int numTesting = 0;

  sort = new int[trainingData.getRowCount()-1];

  for (int i = 0; i < trainingData.getRowCount()-1; i++) {
    if (random(0, 1) < percentTraining) {
      numTraining++;
      sort[i] = 0;
    } 
    else {
      numTesting++;
      sort[i] = 1;
    }
  }

  println(numTraining + " " + numTesting + " " + (float)numTraining/sort.length);

  trainingVectors = new float[numTraining][numFeatures];
  testVectors = new float[numTesting][numFeatures];
  labels = new int[numTraining];

  loadSamplesFromTable(trainingData, true);
    
  model = new SVM(this);
  SVMProblem problem = new SVMProblem();
  problem.setNumFeatures(numFeatures);
  problem.setSampleData(labels, trainingVectors);
  model.train(problem);

  //  OpenCV opencv = new OpenCV(this, 0, 0);
  //  classifier = new SVM();
  //  classifier.addTrainingSamples(training);
  //  classifier.train();

  //  int numCorrect = 0;
  //  for(int i = 0; i < testVectors.length; i++){
  //    double prediction = classifier.predict(testVectors[i]);
  //    println("Prediction: " + (int)prediction);
  //
  //    if ((int) prediction == sample.label) {
  //      numCorrect++;
  //    }
  //  }
  //
  //  println("Score: " + numCorrect + "/" + testing.size() + " (" + ((float)numCorrect/testing.size()) + "%)" );
}

void draw() {
}

