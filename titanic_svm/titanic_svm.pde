
// COLUMN LABELS: 
// PassengerId  Survived  Pclass  Name  Sex  Age  SibSp  Parch  Ticket  Fare  Cabin  Embarked
//  0            1        2        3    4    5    6      7      8        9    10      11
// Skip: PassengerId, Name, Ticket, Cabin
// Total number of cols in the feature vector: 12-4-1 =7


Sample[] samplesFromTable(Table table, boolean isTraining) {
  Sample[] result = new Sample[table.getRowCount() - 1];

  int offset = 0;
  if (!isTraining) {
    offset = -1;
  }

  // skip the top row which is labels
  for (int row = 1; row < table.getRowCount(); row++) {

    Sample sample = new Sample(7);

    if (isTraining) {
      sample.setLabel(table.getInt(row, 1));
    }

    sample.featureVector[0] = table.getInt(row, 2+offset);

    int sex;
    if (table.getString(row, 4+offset) == "male") {
      sex = 0;
    } 
    else {
      sex = 1;
    }

    sample.featureVector[1] = sex;
    sample.featureVector[2] = table.getFloat(row, 5+offset);
    sample.featureVector[3] = table.getInt(row, 6+offset);
    sample.featureVector[4] = table.getInt(row, 7+offset);
    sample.featureVector[5] = table.getFloat(row, 9+offset);

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


    sample.featureVector[6] = embarkedCode;

    result[row-1] = sample;
  }

  return result;
}

SVM classifier;


void setup() {
  Table trainingData = loadTable("train.csv");

  Sample[] allSamples = samplesFromTable(trainingData, true);  

  // create random partition
  ArrayList<Sample> training = new ArrayList<Sample>();
  ArrayList<Sample> testing = new ArrayList<Sample>();

  float percentTraining = 0.66;

  for (int i = 0; i < allSamples.length; i++) {
    if (random(0, 1) < percentTraining) {
      training.add(allSamples[i]);
    } 
    else {
      testing.add(allSamples[i]);
    }
  }

  println(training.size() + " " + testing.size() + " " + (float)training.size()/(training.size() + testing.size()));

  OpenCV opencv = new OpenCV(this, 0, 0);
  classifier = new SVM();
  classifier.addTrainingSamples(training);
  classifier.train();

  int numCorrect = 0;
  for (Sample sample : testing) {
    double prediction = classifier.predict(sample);
    println("Prediction: " + (int)prediction);

    if ((int) prediction == sample.label) {
      numCorrect++;
    }
  }

  println("Score: " + numCorrect + "/" + testing.size() + " (" + ((float)numCorrect/testing.size()) + "%)" );
}

void draw() {
}

