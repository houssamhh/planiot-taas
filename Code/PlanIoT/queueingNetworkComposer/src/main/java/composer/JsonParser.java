package composer;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;

import org.json.JSONArray;
import org.json.JSONObject;

import iotSystemComponents.Actuator;
import iotSystemComponents.Application;
import iotSystemComponents.Broker;
import iotSystemComponents.IoTdevice;
import iotSystemComponents.Subtopic;
import iotSystemComponents.Topic;
import iotSystemComponents.VirtualSensor;

public class JsonParser {

	public String inputFile;
	
	public static HashMap<String, IoTdevice> iotDevices = new HashMap<String, IoTdevice>();
	public static HashMap<String, VirtualSensor> virtualSensors = new HashMap<String, VirtualSensor>();
	public static HashMap<String, Actuator> actuators = new HashMap<String, Actuator>();
	public static HashMap<String, Application> applications = new HashMap<String, Application>();
	public static HashMap<String, Topic> topics = new HashMap<String, Topic>();
	public static HashMap<String, Subtopic> subtopics = new HashMap<String, Subtopic>();
	public static HashMap<String, ArrayList<String>> topics_subtopicsList = new HashMap<String, ArrayList<String>>();
	public static HashMap<String, String> subtopicsClassSwitches = new HashMap<String, String>();	
	public static HashMap<String, Double> topicsRate = new HashMap<String, Double>();	
	public static ArrayList<Broker> brokers = new ArrayList<Broker>();
	public static double systemBandwidth = 0;
	public static String bandwidthPolicy = "";
	public static String priorityPolicy = "";
	
	public final double GLOBAL_MESSAGE_SIZE = 52428800.0;		//50 MB
	public double CHANNEL_LOSS_AN = 0;
	public double CHANNEL_LOSS_RT = 0;
	public double CHANNEL_LOSS_TS = 0;
	public double CHANNEL_LOSS_VS = 0;
	public int BROKER_CAPACITY = 0;
	
	public JsonParser () {
	}
	
	public void readJSON(String file) throws IOException {
    	String text = new String(Files.readAllBytes(Paths.get(file)), StandardCharsets.UTF_8);
		String jsonString = text; //assign your JSON String here
		JSONObject obj = new JSONObject(jsonString);
		
		//add iot devices
		JSONArray arr = obj.getJSONArray("IoTdevices");
		for (int i = 0; i < arr.length(); i++)
		{
		    String deviceId = arr.getJSONObject(i).getString("deviceId");
		    String deviceName = arr.getJSONObject(i).getString("deviceName");
		    double publishFrequency = arr.getJSONObject(i).getInt("publishFrequency");
		    double messageSize = arr.getJSONObject(i).getInt("messageSize");
		    String distribution = arr.getJSONObject(i).getString("distribution");
		    JSONArray array = arr.getJSONObject(i).getJSONArray("publishesTo");
		    ArrayList<String> publishedTopics = new ArrayList<String>();
		    for (Object o : array) {
		    	publishedTopics.add((String) o);
		    	if (!topicsRate.containsKey((String) o))
		    			topicsRate.put((String) o, (messageSize * publishFrequency) / GLOBAL_MESSAGE_SIZE);
		    	else {
		    		double rate = topicsRate.get((String) o);
		    		rate += (messageSize * publishFrequency) / GLOBAL_MESSAGE_SIZE;
		    		topicsRate.put((String) o, rate);
		    	}
		    }
		    
		    IoTdevice device = new IoTdevice(deviceId, deviceName, publishFrequency, messageSize, distribution, publishedTopics);
		    iotDevices.put(deviceId, device);
		}
	
		
		//add virtual sensors
		arr = obj.getJSONArray("virtualSensors");
		for (int i = 0; i < arr.length(); i++) 
		{
		    String deviceId = arr.getJSONObject(i).getString("deviceId");
		    String deviceName = arr.getJSONObject(i).getString("deviceName");
		    String applicationCategory = arr.getJSONObject(i).getString("applicationCategory");
		    double publishFrequency = arr.getJSONObject(i).getInt("publishFrequency");
		    double messageSize = arr.getJSONObject(i).getInt("messageSize");
		    int processingRate = arr.getJSONObject(i).getInt("processingRate");
		    int priority = arr.getJSONObject(i).getInt("priority");
		    String processingDistribution = arr.getJSONObject(i).getString("processingDistribution");
		    JSONArray array = arr.getJSONObject(i).getJSONArray("subscribesTo");
		    ArrayList<String> publishedTopics = new ArrayList<String>();
		    ArrayList<String> subscribedTopics = new ArrayList<String>();
		    for (Object o : array) {
		    	publishedTopics.add((String) o);
		    	if (!topicsRate.containsKey((String) o))
		    			topicsRate.put((String) o, messageSize * publishFrequency);
		    	else {
		    		double rate = topicsRate.get((String) o);
		    		rate += messageSize * publishFrequency;
		    		topicsRate.put((String) o, rate);
		    	}
		    }
		    array = arr.getJSONObject(i).getJSONArray("publishesTo");
		    for (Object o : array) {
		    	subscribedTopics.add((String) o);
		    	if (!topicsRate.containsKey((String) o))
	    			topicsRate.put((String) o, (messageSize * publishFrequency) / GLOBAL_MESSAGE_SIZE);
		    	else {
		    		double rate = topicsRate.get((String) o);
		    		rate += (messageSize * publishFrequency) / GLOBAL_MESSAGE_SIZE;
		    		topicsRate.put((String) o, rate);
		    	}
		    }
		    VirtualSensor sensor = new VirtualSensor(deviceId, deviceName, applicationCategory, publishFrequency, messageSize,
		    		processingRate, priority, processingDistribution, subscribedTopics, publishedTopics);
		    virtualSensors.put(deviceId, sensor);
		}
		
		//add actuators
		arr = obj.getJSONArray("actuators");
		for (int i = 0; i < arr.length(); i++) 
		{
		    String deviceId = arr.getJSONObject(i).getString("deviceId");
		    String deviceName = arr.getJSONObject(i).getString("deviceName");
		    String applicationCategory = arr.getJSONObject(i).getString("applicationCategory");
		    int publishFrequency = arr.getJSONObject(i).getInt("publishFrequency");
		    int messageSize = arr.getJSONObject(i).getInt("messageSize");
		    JSONArray array = arr.getJSONObject(i).getJSONArray("subscribesTo");
		    ArrayList<String> subscribedTopics = new ArrayList<String>();
		    double bwNeeded = 0;
		    for (Object o : array) {
		    	subscribedTopics.add((String) o);
		    	bwNeeded += topicsRate.get((String) o);
		    }
		    	
		    Actuator actuator = new Actuator(deviceId, deviceName, applicationCategory, publishFrequency, messageSize, subscribedTopics);
		    actuators.put(deviceId, actuator); 
		}
		
		//add applications
		arr = obj.getJSONArray("applications");
		for (int i = 0; i < arr.length(); i++) 
		{
		    String applicationId = arr.getJSONObject(i).getString("applicationId");
		    String applicationName = arr.getJSONObject(i).getString("applicationName");
		    String applicationCategory = arr.getJSONObject(i).getString("applicationCategory");
		    int priority = arr.getJSONObject(i).getInt("priority");
		    int processingRate = arr.getJSONObject(i).getInt("processingRate");
		    String processingDistribution = arr.getJSONObject(i).getString("processingDistribution");
		    JSONArray array = arr.getJSONObject(i).getJSONArray("subscribesTo");
		    ArrayList<String> subscribedTopics = new ArrayList<String>();
		    double bwNeeded = 0;
		    for (Object o : array) {
		    	subscribedTopics.add((String) o);
		    	bwNeeded += topicsRate.get((String) o);
		    }
		    	
		    Application application = new Application(applicationId, applicationName, applicationCategory, processingRate, 
		    		processingDistribution, subscribedTopics, priority);
		    applications.put(applicationId, application);
		}
		
		//add topics
		arr = obj.getJSONArray("topics");
		for (int i = 0; i < arr.length(); i++) 
		{
		    String topicId = arr.getJSONObject(i).getString("topicId");
		    String topicName = arr.getJSONObject(i).getString("topicName");
		    int topicPriority = arr.getJSONObject(i).getInt("priority");
		    JSONArray array = arr.getJSONObject(i).getJSONArray("subscribers");
		    ArrayList<String> subscribers = new ArrayList<String>();
		    for (Object o : array)
		    	subscribers.add((String) o);
		    array = arr.getJSONObject(i).getJSONArray("publishers");
		    ArrayList<String> publishers = new ArrayList<String>();
		    for (Object o : array)
		    	publishers.add((String) o);
		    Topic topic = new Topic(topicId, topicName, publishers, subscribers);
		    topic.topicPriority = topicPriority;
		    topics.put(topicId, topic);
		}
		
		//add brokers
		arr = obj.getJSONArray("broker");
		for (int i = 0; i < arr.length(); i++) 
		{
		    String brokerId = arr.getJSONObject(i).getString("brokerId");
		    String brokerName = arr.getJSONObject(i).getString("brokerName");
		    int bufferSize = arr.getJSONObject(i).getInt("bufferSize");
		    int processingRate = arr.getJSONObject(i).getInt("processingRate");
		    JSONArray array = arr.getJSONObject(i).getJSONArray("topics");
		    ArrayList<String> topics = new ArrayList<String>();
		    for (Object o : array)
		    	topics.add((String) o);
		    Broker broker = new Broker(brokerId, brokerName, bufferSize, processingRate, topics);
		    brokers.add(broker);
		}
		
		double systemBandwidth = obj.getDouble("systemBandwidth");
		JsonParser.systemBandwidth = systemBandwidth;
		String bandwidthPolicy = obj.getString("bandwidthPolicy");
		JsonParser.bandwidthPolicy = bandwidthPolicy;
		String priorityPolicy = obj.getString("priorityPolicy");
		JsonParser.priorityPolicy = priorityPolicy;
		double channelLossAN = obj.getDouble("commChannelLossAN");
		CHANNEL_LOSS_AN = channelLossAN;
		double channelLossRT = obj.getDouble("commChannelLossRT");
		CHANNEL_LOSS_RT = channelLossRT;
		double channelLossTS = obj.getDouble("commChannelLossTS");
		CHANNEL_LOSS_TS = channelLossTS;
		double channelLossVS = obj.getDouble("commChannelLossVS");
		CHANNEL_LOSS_VS = channelLossVS;
		int brokerCapacity = obj.getInt("brokerCapacity");
		BROKER_CAPACITY = brokerCapacity;
    }
}
