package composer;

import java.util.HashMap;

import iotSystemComponents.Application;
import iotSystemComponents.Subtopic;
import iotSystemComponents.Topic;
import iotSystemComponents.VirtualSensor;
import jmt.gui.common.definitions.CommonModel;
import jmt.gui.common.routingStrategies.ProbabilityRouting;

public class PriorityHandler {

	public void convertToJmtPriorities(CommonModel jmtModel, HashMap<String, Application> applications, HashMap<String, VirtualSensor> virtualSensors, HashMap<String, Subtopic> subtopics) {
		int maxPriority = 0;
		int minPriority = 100;
		for (Application app : applications.values()) {
			if (app.priority > maxPriority)
				maxPriority = app.priority;
			if (app.priority < minPriority)
				minPriority = app.priority;
		}	
		
		for (VirtualSensor virtualSensor : virtualSensors.values()) {
			if (virtualSensor.priority > maxPriority)
				maxPriority = virtualSensor.priority;
			if (virtualSensor.priority < minPriority)
				minPriority = virtualSensor.priority;
		}
		
		Object classSwitch = jmtModel.getStationByName("topics_class_switch");
		for (Subtopic subtopic : subtopics.values()) {
    		ProbabilityRouting probaRouting = new ProbabilityRouting();
    		probaRouting.getValues().put(jmtModel.getStationByName(subtopic.parentTopicName + "_join"), 1d);
    		jmtModel.setRoutingStrategy(classSwitch, jmtModel.getClassByName(subtopic.name + "_class"), probaRouting);
    	}
		
		for (Application app : applications.values())
			app.jmtPriority = maxPriority - app.priority + minPriority;
	
		for (VirtualSensor virtualSensor : virtualSensors.values())
			virtualSensor.jmtPriority = maxPriority - virtualSensor.priority + minPriority;
	}
	
    public void convertTopicPrioritiesToJmtPriorities(CommonModel jmtModel, HashMap<String, Topic> topics, HashMap<String, Subtopic> subtopics) {
    	int maxPriority = 0;
		int minPriority = 100;
		for (Topic topic : topics.values()) {
			if (topic.topicPriority > maxPriority)
				maxPriority = topic.topicPriority;
			if (topic.topicPriority < minPriority)
				minPriority = topic.topicPriority;
		}		
		Object classSwitch = jmtModel.getStationByName("topics_class_switch");
		for (Subtopic subtopic : subtopics.values()) {
    		ProbabilityRouting probaRouting = new ProbabilityRouting();
    		probaRouting.getValues().put(jmtModel.getStationByName(subtopic.parentTopicName + "_join"), 1d);
    		jmtModel.setRoutingStrategy(classSwitch, jmtModel.getClassByName(subtopic.name + "_class"), probaRouting);
    	}
		
		for (Topic topic : topics.values()) {
			topic.jmtPriority = maxPriority - topic.topicPriority + minPriority;
		}
			
    }
}
