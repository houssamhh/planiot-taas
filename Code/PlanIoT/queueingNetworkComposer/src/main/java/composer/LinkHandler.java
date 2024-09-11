package composer;

import java.util.HashMap;

//import iotSys.broker.JsonParser;
import iotSystemComponents.Actuator;
import iotSystemComponents.Application;
import iotSystemComponents.Broker;
import iotSystemComponents.IoTdevice;
import iotSystemComponents.Topic;
import iotSystemComponents.VirtualSensor;
import jmt.gui.common.definitions.CommonModel;

public class LinkHandler {

	public void setConnections(CommonModel jmtModel, JsonParser parser, HashMap<String, String> subtopicsClassSwitches) {
    	Broker broker = parser.brokers.get(0);
    	HashMap<String, IoTdevice> iotDevices = parser.iotDevices;
    	HashMap<String, Topic> topics = parser.topics;
    	HashMap<String, Application> applications = parser.applications;
    	HashMap<String, VirtualSensor> virtualSensors = parser.virtualSensors;
    	HashMap<String, Actuator> actuators = parser.actuators;
    	
    	//connecting sources to topic class switches
    	for (IoTdevice device : iotDevices.values()) {
    		for (String topic : device.publishedTopics)
    			jmtModel.setConnected(jmtModel.getStationByName(device.deviceName), jmtModel.getStationByName(topic + "_switch"), true);
    	}
    	
    	//connecting class switches to the input queue
    	for (Topic topic : topics.values())
    		jmtModel.setConnected(jmtModel.getStationByName(topic.topicName + "_switch"), broker.brokerName, true);
    	
    	//connecting the input queue to topic forks
    	for (Topic topic : topics.values())
    		jmtModel.setConnected(jmtModel.getStationByName("input"), jmtModel.getStationByName(topic.topicName + "_fork"), true);
    	
    	//connecting the topic forks (for subtopics) to subtopics class switches
    	for (Topic topic : topics.values()) 
    		for (String subTopicName : topic.subTopics) {
    			String classSwitchName = subtopicsClassSwitches.get(subTopicName);
    			jmtModel.setConnected(jmtModel.getStationByName(topic.topicName + "_fork"), jmtModel.getStationByName(classSwitchName), true);
    		}
	
    	//connecting the subtopics class switches to the output queue
    	for (Topic topic : topics.values()) 
    		for (String subTopicName : topic.subTopics) {
    			String classSwitchName = subtopicsClassSwitches.get(subTopicName);
    			jmtModel.setConnected(jmtModel.getStationByName(classSwitchName), jmtModel.getStationByName("outputQueue"), true);
    		}
    			
    	//connecting the subtopics class switches to the class switch before dropping sink    	
    	for (Topic topic : topics.values()) 
    		for (String subTopicName : topic.subTopics) {
    			String classSwitchName = subtopicsClassSwitches.get(subTopicName);
    			jmtModel.setConnected(jmtModel.getStationByName(classSwitchName), jmtModel.getStationByName("topics_class_switch_dropping"), true);
    		}
    	
    	//connecting class switch for dropping to topic joins for dropping
    	jmtModel.setConnected(jmtModel.getStationByName("topics_class_switch_dropping"), jmtModel.getStationByName("dropping_join"), true);
    	jmtModel.setConnected( jmtModel.getStationByName("dropping_join"), jmtModel.getStationByName("dropping_sink"), true);
    			
    	
    	
    	for (Actuator actuator : actuators.values())
    		for (String topic : actuator.subscribedTopics)
    			jmtModel.setConnected(jmtModel.getStationByName(topic + "_fork"), jmtModel.getStationByName(actuator.deviceId + "_outputQueue"), true);
    	
    	for (VirtualSensor virtualSensor : virtualSensors.values())
    		for (String topic : virtualSensor.subscribedTopics)
    			jmtModel.setConnected(jmtModel.getStationByName(topic + "_fork"), jmtModel.getStationByName(virtualSensor.deviceId + "_outputQueue"), true);
    	
//    	connecting apps to
//    	1. output queue
//    	2. topics class switch
    	for (Application app : applications.values()) {
    		jmtModel.setConnected(jmtModel.getStationByName("outputQueue"), jmtModel.getStationByName(app.applicationName), true);
    		jmtModel.setConnected(jmtModel.getStationByName(app.applicationName), jmtModel.getStationByName("topics_class_switch"), true);
    	}
    	
    	//connecting the topics class switch to topic joins
    	jmtModel.setConnected(jmtModel.getStationByName("topics_class_switch"), jmtModel.getStationByName("topics_join"), true);
    	
//    	connecting topic joins to topic sinks
    	jmtModel.setConnected(jmtModel.getStationByName("topics_join"), jmtModel.getStationByName("topics_sink"), true);
    	
//    	connecting actuators to
//    	1. output queue
//    	2. topics class switch
    	for (Actuator actuator : actuators.values()) {
    		jmtModel.setConnected(jmtModel.getStationByName("output queue"), jmtModel.getStationByName(actuator.deviceId), true);
    		jmtModel.setConnected(jmtModel.getStationByName(actuator.deviceId), jmtModel.getStationByName("topics_class_switch"), true);
    	}
    	
//    	connecting virtual sensors to 
//    	1. output queue
//    	2. topic forks
//    	3. class switch
    	for (VirtualSensor sensor : virtualSensors.values()) {
    		for (String topic : sensor.subscribedTopics) {
    			jmtModel.setConnected(jmtModel.getStationByName("output queue"), jmtModel.getStationByName(sensor.deviceId + "_processing"), true);
    			jmtModel.setConnected(jmtModel.getStationByName(sensor.deviceName + "_processing"), jmtModel.getStationByName(topic + "_join"), true);
    		}
    			
    		for (String topic : sensor.publishedTopics)
    			jmtModel.setConnected(jmtModel.getStationByName(sensor.deviceName + "_source"), jmtModel.getStationByName(topic + "_switch"), true);
    	}

    	//connecting class switches to broker
    	for (Topic topic : topics.values())
    		jmtModel.setConnected(jmtModel.getStationByName(topic.topicName + "_switch"), jmtModel.getStationByName(broker.brokerName), true);
    }
}
