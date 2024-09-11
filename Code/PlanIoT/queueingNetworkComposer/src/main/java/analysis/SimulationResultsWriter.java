package analysis;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import javax.xml.XMLConstants;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

import com.opencsv.CSVWriter;

public class SimulationResultsWriter {
	
	HashMap<String, Double> responseTimes;
	HashMap<String, Double> throughputs;
	HashMap<String, Double> dropRates;
	double utilization;
	
	public SimulationResultsWriter() {
		responseTimes = new HashMap<String, Double>();
		throughputs = new HashMap<String, Double>();
		dropRates = new HashMap<String, Double>();
		utilization = 0;
	}
	
	
	//Read the results from the XML file produced by JMT
	public void readXML(String inputFile){
		DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();		
		try {
          // optional, but recommended
          // process XML securely, avoid attacks like XML External Entities (XXE)
          dbf.setFeature(XMLConstants.FEATURE_SECURE_PROCESSING, true);

          // parse XML file
          DocumentBuilder db = dbf.newDocumentBuilder();

          Document doc = db.parse(new File(inputFile));

          // optional, but recommended
          // http://stackoverflow.com/questions/13786607/normalization-in-dom-parsing-with-java-how-does-it-work
          doc.getDocumentElement().normalize();

          // get metrics
          NodeList list = doc.getElementsByTagName("measure");

          for (int temp = 0; temp < list.getLength(); temp++) {

              Node node = list.item(temp);

              if (node.getNodeType() == Node.ELEMENT_NODE) {
                  Element element = (Element) node;
                  
                  String classType = element.getAttribute("class");
                  String measureType = element.getAttribute("measureType");
                  String station = element.getAttribute("station");
                  Double meanValue = Double.parseDouble(element.getAttribute("meanValue"));
                  
                  String subscription = "";
                  if (!classType.equals("")) {
                	  String[] parts = classType.split("_");
                      String topic = parts[1];
      				  String app = parts[3];
      				  subscription = topic + "_" + app;
                  }
                  if (measureType.equals("Response Time"))
                	  responseTimes.put(subscription, meanValue);
                  else if (measureType.equals("Throughput"))
                	  throughputs.put(subscription, meanValue);
                  else if (measureType.equals("Drop Rate"))
                	  dropRates.put(subscription, meanValue);
                  else if (measureType.equals("Utilization") && classType.equals("") && station.equals("outputQueue"))
                	  utilization = meanValue;
              }
          }

	      } catch (Exception e) {
	          e.printStackTrace();
	      }
	}
	
	//Write results into a CSV file to create the dataset of performance metrics
	public void writeToCsv(String outputFile) {
		File file = new File(outputFile);
		try {
			FileWriter fileWriter = new FileWriter(file);
			CSVWriter writer = new CSVWriter(fileWriter);
			
			String[] header = {"topic", "app", "response_time", "throughput", "drop_rate"};
			writer.writeNext(header);
			
			for (String key : responseTimes.keySet()) {
				String[] parts = key.split("_");
				String topic = parts[0];
				String app = parts[1];
				
				String responseTime = responseTimes.get(key).toString();
				String throughput = throughputs.get(key).toString();
				String dropRate = dropRates.get(key).toString();
				
				String[] data = {topic, app, responseTime, throughput, dropRate};
				writer.writeNext(data);
			}
			
			writer.close();
			
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void addResultsToDataset(String datasetPath, String alias) throws IOException {
		File datasetFile = new File (datasetPath);
		if (!datasetFile.exists() && !datasetFile.isDirectory()) {
			File file = new File(datasetPath);
			FileWriter fileWriter = new FileWriter(file);
			CSVWriter writer = new CSVWriter(fileWriter);
			String[] header = {"topic", "app", alias};
			writer.writeNext(header);
			
			for (String subscription : responseTimes.keySet()) {
				Double responseTime = responseTimes.get(subscription);
				String[] parts = subscription.split("_");
				String topic = parts[0];
				String app = parts[1];
				String[] data = new String[] {topic, app, responseTime.toString()};
				writer.writeNext(data);
			}
			writer.close();
		}
		
		else {
			String fileName = datasetPath.split(".csv")[0] + "%tmp.csv";
			File tmpFile = new File(fileName);
			FileWriter fileWriter = new FileWriter(tmpFile);
			CSVWriter writer = new CSVWriter(fileWriter);
			

			File file = new File(datasetPath);
			FileReader fileReader = new FileReader(file);
			BufferedReader reader = new BufferedReader(fileReader);
			String line = null;
			int counter = 0;
			while ((line=reader.readLine()) != null) {
				ArrayList<String> data = new ArrayList<String>(Arrays.asList(line.split(",")));
				if (counter == 0) {
					data.add(alias);
					String[] array = data.toArray(new String[0]);
					for (int i = 0; i < array.length; i++)
						array[i] = array[i].replace("\"", "");					
					writer.writeNext(array);
					counter++;
				}
				else {
					String topic = data.get(0).replace("\"", "");
					String app = data.get(1).replace("\"", "");
					String subscription = topic + "_" + app;
					if (responseTimes.containsKey(subscription)) {
						Double responseTime = responseTimes.get(subscription);
						data.add(responseTime.toString());
						String[] array = data.toArray(new String[0]);
						for (int i = 0; i < array.length; i++)
							array[i] = array[i].replace("\"", "");	
						writer.writeNext(array);	
					}
				}
			}
			reader.close();
			writer.close();
			file.delete();
			File newFile = new File(fileName.replace("%tmp", ""));
			tmpFile.renameTo(newFile);
		}
	}
}
