import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.NominalPrediction;
import weka.core.FastVector;
import weka.core.Instances;

public class WekaTest {
	
	public static BufferedReader readDataFile(String filename){
		
		BufferedReader inputReader = null;
		
		try{
			inputReader = new BufferedReader(new FileReader(filename));
		}catch(FileNotFoundException ex){
			System.out.println("File not found:" + filename);
		}
		
		return inputReader;
	}
	
	
	public static Evaluation classify(Classifier model , Instances trainningSet ,
			Instances testingSet) throws Exception{
		
		Evaluation evaluation = new Evaluation(testingSet);
		model.buildClassifier(trainningSet);
		evaluation.evaluateModel(model, testingSet);
		
		return evaluation;
	}
	
	
	public static double calculateAccuracy(FastVector predictions){
		
		double correct = 0;
		
		for(int i = 0; i < predictions.size(); i++){
			
			NominalPrediction np = (NominalPrediction)predictions.elementAt(i);
			
			if(np.predicted() == np.actual()){
				correct++;
			}
			
		}
		
		return 100 * correct / predictions.size();
	}
	
	public static Instances[][] crossValidationSplit(Instances data , int numberOfFolds){
		
		Instances[][] split = new Instances[2][numberOfFolds];
		
		for(int i = 0; i < numberOfFolds; i++){
			split[0][i] = data.trainCV(numberOfFolds , i);
			split[1][i] = data.testCV(numberOfFolds , i);
		}
		
		return split;
	}
	
	public static void main(String[] args) throws Exception{
		
		BufferedReader datafile = readDataFile("weather.txt");
		
		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		
		//Do 10-split cross validation
		Instances[][] split = crossValidationSplit(data, 10);
		
		//
		
	}
}
