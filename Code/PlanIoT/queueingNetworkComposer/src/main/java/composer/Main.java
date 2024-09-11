package composer;

import java.io.File;

import analysis.SimulationResultsWriter;
import jmt.engine.simDispatcher.DispatcherJSIMschema;

public class Main {

	/* args: [0]: json specifications (input file)
	   		 [1]: dataset path (output file)
	   		 [2]: simulation duration (in sec)
	   		 [3]: alias 
	*/
	public static void main(String[] args) throws Exception {
		String inputFile = args[0];
		String outputFile = args[1];
		int simulationDuration = Integer.valueOf(args[2]);
		String alias = args[3];
		//TODO add checks for arguments
		
		QueueingNetworkComposer composer = new QueueingNetworkComposer();
		System.out.println("Composing the queueing network ...");
		String jsimgFile = composer.composeNetwork(inputFile, simulationDuration);
		System.out.println("Created simulation file.");
		System.out.println("Running the simulation ...");
		DispatcherJSIMschema djss = new DispatcherJSIMschema(jsimgFile);
		djss.setSimulationMaxDuration(simulationDuration*1000);
		djss.solveModel();
		File simResultFile = djss.getOutputFile();
		System.out.println("Simulation done.");
		
		System.out.println("Writing results to csv ...");
		SimulationResultsWriter writer = new SimulationResultsWriter();
		writer.readXML(simResultFile.getCanonicalPath());
		String metricsFile = inputFile.split(".json")[0] + "_" + alias + ".csv";
		writer.writeToCsv(metricsFile);
		System.out.println("Done writing to csv.");
		
		System.out.println("Adding results to dataset");
		writer.addResultsToDataset(outputFile, alias);
		System.out.println("Done.");
		
	}

}
