package composer;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;

import analysis.SimulationResultsWriter;
//import iotSys.broker.App;
//import iotSys.broker.JsonParser;
import iotSystemComponents.*;
import jmt.engine.simDispatcher.DispatcherJSIMschema;
import jmt.gui.common.definitions.CommonModel;
import jmt.gui.common.xml.XMLWriter;

public class QueueingNetworkComposer {
	 
	 public String composeNetwork(String inputFile, int simulationDuration) throws Exception{
		 JsonParser parser = new JsonParser();
		 parser.readJSON(inputFile);
		 
		 String priorityPolicy = JsonParser.priorityPolicy;
		 
		 CommonModel jmtModel = new CommonModel();
		 IoTdeviceHandler iotDeviceHandler = new IoTdeviceHandler();
		 HashMap<String, IoTdevice> iotDevices = parser.iotDevices;
		 iotDeviceHandler.addSources(jmtModel, iotDevices.keySet());
		 
		 BrokerHandler brokerHandler = new BrokerHandler();
		 brokerHandler.addInputQueue(jmtModel, "input");
		 brokerHandler.addOutputQueue(jmtModel, "outputQueue");
		 
		 ApplicationHandler applicationHandler = new ApplicationHandler();
		 HashMap<String, Application> applications = parser.applications;
		 applicationHandler.addApplications(jmtModel, applications.values());
	 
		 VirtualSensorHandler virtualSensorHandler = new VirtualSensorHandler();
		 HashMap<String, VirtualSensor> virtualSensors = parser.virtualSensors;
		 virtualSensorHandler.addVirtualSensor(jmtModel, virtualSensors.values());
		 
		 TopicHandler topicHandler = new TopicHandler();
		 HashMap<String, Topic> topics = parser.topics;
		 topicHandler.addTopicsForks(jmtModel, topics.keySet());
		 topicHandler.addTopicsJoin(jmtModel);
		 topicHandler.addTopicsSink(jmtModel);
		 topicHandler.addTopicsClassSwitches(jmtModel, topics.keySet());
		 
		 ClassHandler classHandler = new ClassHandler();
		 classHandler.addClassesForIoTdevices(jmtModel, iotDevices.values(), parser.GLOBAL_MESSAGE_SIZE);
		 classHandler.addClassesForVirtualSensors(jmtModel, virtualSensors.values(), parser.GLOBAL_MESSAGE_SIZE);
		 classHandler.addClassesForTopics(jmtModel, topics.values());
		 
		 DroppingHandler droppingHandler = new DroppingHandler();
		 droppingHandler.addDroppingSink(jmtModel);
		 
		 topicHandler.addSubTopics(jmtModel, topics.values(), applications, virtualSensors, priorityPolicy);
		 
		 HashMap<String, Subtopic> subtopics = topicHandler.subtopics;
		 PriorityHandler priorityHandler = new PriorityHandler();
		 priorityHandler.convertToJmtPriorities(jmtModel, applications, virtualSensors, subtopics);
		 priorityHandler.convertTopicPrioritiesToJmtPriorities(jmtModel, topics, subtopics);
		 
		 classHandler.addClassesForSubtopics(jmtModel, subtopics);
		 
		 topicHandler.setClassSwitchMatrix(jmtModel, topics);
		 topicHandler.setClassSwitchMatrixForSubtopics(jmtModel, subtopics);
		 topicHandler.setTopicsClassSwitchMatrix(jmtModel, subtopics);
		 topicHandler.setTopicsClassSwitchDroppingMatrix(jmtModel, subtopics);
		 
		 RoutingHandler routingHandler = new RoutingHandler();
		 routingHandler.setTopicsClassSwitchRouting(jmtModel, subtopics);
		 routingHandler.setApplicationsRouting(jmtModel, applications);
		 routingHandler.setInputQueueRouting(jmtModel, topics);
		 routingHandler.setOutputQueueRouting(jmtModel, subtopics);
		 routingHandler.setVirtualSensorsRouting(jmtModel, virtualSensors);
		 routingHandler.addDroppingRouting(jmtModel, subtopics, topicHandler.subtopicsClassSwitches,
				 parser.CHANNEL_LOSS_AN, parser.CHANNEL_LOSS_RT, parser.CHANNEL_LOSS_TS, parser.CHANNEL_LOSS_VS);
		 
		 NetworkResourcesManager networkManager = new NetworkResourcesManager(parser.systemBandwidth, parser.bandwidthPolicy, 
				 parser.GLOBAL_MESSAGE_SIZE);
		 networkManager.allocateResources();
		 
		 
		 LinkHandler linkHandler = new LinkHandler();
		 linkHandler.setConnections(jmtModel, parser, topicHandler.subtopicsClassSwitches);
		 
		 FiniteCapacityRegionHandler fcrHandler = new FiniteCapacityRegionHandler();
		 Object fcrObj = fcrHandler.setFiniteCapacityRegion(jmtModel, parser.BROKER_CAPACITY, subtopics);
		 
		 ServiceTimeHandler serviceTimeHandler = new ServiceTimeHandler();
		 serviceTimeHandler.setInputQueueServiceTime(jmtModel, parser.brokers.get(0), topics);
		 serviceTimeHandler.setOutputQueueServiceTime(jmtModel, networkManager, subtopics);
		 serviceTimeHandler.setApplicationsServiceTime(jmtModel, applications);
		 serviceTimeHandler.setVirtualSensorsServiceTime(jmtModel, virtualSensors);
		 
		 PerformanceMetricsHandler performanceMetricsHandler = new PerformanceMetricsHandler();
		 double confInterval = 0.95;
		 double relErr = 0.05;
		 performanceMetricsHandler.setPerformanceMetrics(jmtModel, subtopics, fcrObj, confInterval, relErr);
		 
		 
		 int i = inputFile.lastIndexOf(".");
		 String jsimgFilePath =  inputFile.substring(0, i) + ".jsimg";
		 File jsimFile = new File(jsimgFilePath);
		 XMLWriter.writeXML(jsimFile, jmtModel);
		 
		 return jsimgFilePath;
		 
		 
	 }
}
