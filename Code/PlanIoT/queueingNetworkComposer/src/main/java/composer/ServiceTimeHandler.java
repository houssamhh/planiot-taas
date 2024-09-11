package composer;

import java.util.ArrayList;
import java.util.HashMap;

import iotSystemComponents.Application;
import iotSystemComponents.Broker;
import iotSystemComponents.Subtopic;
import iotSystemComponents.Topic;
import iotSystemComponents.VirtualSensor;
import jmt.gui.common.definitions.CommonModel;
import jmt.gui.common.distributions.Deterministic;
import jmt.gui.common.distributions.Exponential;

public class ServiceTimeHandler {

    public void setInputQueueServiceTime(CommonModel jmtModel, Broker broker, HashMap<String, Topic> topics) {
    	Exponential exponential = new Exponential();
    	exponential.setMean(1.0/broker.processingRate);
    	for (Topic topic : topics.values())
    		jmtModel.setServiceTimeDistribution(jmtModel.getStationByName(broker.brokerName), jmtModel.getClassByName(topic.topicName + "_class"), exponential);
    }
      
    public void setOutputQueueServiceTime(CommonModel jmtModel, NetworkResourcesManager networkManager,
    		HashMap<String, Subtopic> subtopics) {
    	Object outputQueue = jmtModel.getStationByName("outputQueue");
    	if (!networkManager.allocationPolicy.equals("none")) {
    		jmtModel.setStationNumberOfServers(outputQueue, 4);
    	}
		for (Subtopic subtopic : subtopics.values()) {
			if (subtopic.category.equals("RT")) {
				Deterministic serviceTimeDistribution = new Deterministic();
				if (networkManager.allocationPolicy.equals("none")) {
					serviceTimeDistribution.setMean(1/(networkManager.allocatedBw_RT * (1048576.0 / networkManager.globalMessageSize)));
				}
				else
					serviceTimeDistribution.setMean(1/(networkManager.allocatedBw_RT));
				jmtModel.setServiceTimeDistribution(outputQueue, jmtModel.getClassByName(subtopic.name + "_class"), serviceTimeDistribution);
				
			}
			else if (subtopic.category.equals("AN")) {
				Deterministic serviceTimeDistribution = new Deterministic();
				if (networkManager.allocationPolicy.equals("none"))
					serviceTimeDistribution.setMean(1/(networkManager.allocatedBw_AN * (1048576.0/networkManager.globalMessageSize)));
				else 
					serviceTimeDistribution.setMean(1/(networkManager.allocatedBw_AN));
				jmtModel.setServiceTimeDistribution(outputQueue, jmtModel.getClassByName(subtopic.name + "_class"), serviceTimeDistribution);
			}
			else if (subtopic.category.equals("TS")) {
				Deterministic serviceTimeDistribution = new Deterministic();
				if (networkManager.allocationPolicy.equals("none"))
					serviceTimeDistribution.setMean(1/(networkManager.allocatedBw_TS * (1048576.0/networkManager.globalMessageSize)));
				else
					serviceTimeDistribution.setMean(1/(networkManager.allocatedBw_TS));
				jmtModel.setServiceTimeDistribution(outputQueue, jmtModel.getClassByName(subtopic.name + "_class"), serviceTimeDistribution);
			}
			else if (subtopic.category.equals("VS")) {
				Deterministic serviceTimeDistribution = new Deterministic();
				if (networkManager.allocationPolicy.equals("none"))
					serviceTimeDistribution.setMean(1/(networkManager.allocatedBw_VS * (1048576.0/networkManager.globalMessageSize)));
				else 
					serviceTimeDistribution.setMean(1/(networkManager.allocatedBw_VS));
				jmtModel.setServiceTimeDistribution(outputQueue, jmtModel.getClassByName(subtopic.name + "_class"), serviceTimeDistribution);
			}
		}
    }
    
    public void setApplicationsServiceTime(CommonModel jmtModel, HashMap<String, Application> applications) {
    	for (Application app : applications.values()) {
    		Exponential exponential = new Exponential();
    		exponential.setMean(1.0/app.processingRate);
    		for (String subscribedTopic : app.subscribedTopics) {
    			jmtModel.setServiceTimeDistribution(jmtModel.getStationByName(app.applicationName), jmtModel.getClassByName(subscribedTopic + "_class"),
    					exponential);
    		}
    	}
    }
    
    public void setVirtualSensorsServiceTime(CommonModel jmtModel, HashMap<String, VirtualSensor> virtualSensors) {
    	for (VirtualSensor sensor : virtualSensors.values()) {
    		Exponential exponential = new Exponential();
    		exponential.setMean(1.0/sensor.processingRate);
    		for (String subscribedTopic : sensor.subscribedTopics) {
    			jmtModel.setServiceTimeDistribution(jmtModel.getStationByName(sensor.deviceName + "_processing"), jmtModel.getClassByName(subscribedTopic + "_class"),
    					exponential);
    		}
    	}
    }
    
}
