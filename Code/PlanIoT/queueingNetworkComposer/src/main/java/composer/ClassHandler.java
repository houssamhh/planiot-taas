package composer;

import java.util.Collection;
import java.util.HashMap;

import iotSystemComponents.IoTdevice;
import iotSystemComponents.Subtopic;
import iotSystemComponents.Topic;
import iotSystemComponents.VirtualSensor;
import jmt.gui.common.CommonConstants;
import jmt.gui.common.Defaults;
import jmt.gui.common.definitions.CommonModel;
import jmt.gui.common.distributions.Deterministic;
import jmt.gui.common.distributions.Exponential;

public class ClassHandler {

	public void addClassesForIoTdevices(CommonModel jmtModel, Collection<IoTdevice> iotDevices, double globalMessageSize) {
		for (IoTdevice device : iotDevices) {			
			if (device.distribution.equals("exponential")) {
				Exponential exp = new Exponential();
				double arrivalRate = (device.publishFrequency * device.messageSize);
				double mean = (double) 1.0 / (arrivalRate / globalMessageSize);
				exp.setMean(mean);
				String className = device.deviceId + "_class";
				jmtModel.addClass(className, CommonConstants.CLASS_TYPE_OPEN, Defaults.getAsInteger("classPriority"), null, exp);
				jmtModel.setClassRefStation(jmtModel.getClassByName(className), jmtModel.getStationByName(device.deviceId));
			}
			else if (device.distribution.equals("deterministic")) {
				Deterministic determinstic = new Deterministic();
				double arrivalRate = (device.publishFrequency * device.messageSize);
				double mean = (double) 1.0 / (arrivalRate / globalMessageSize);
				determinstic.setMean(mean);
				String className = device.deviceId + "_class";
				jmtModel.addClass(className, CommonConstants.CLASS_TYPE_OPEN, Defaults.getAsInteger("classPriority"), null, determinstic);
				jmtModel.setClassRefStation(jmtModel.getClassByName(className), jmtModel.getStationByName(device.deviceId));
			}
		}
	}
	
	public void addClassesForVirtualSensors(CommonModel jmtModel, Collection<VirtualSensor> virtualSensors, double globalMessageSize) {
		for (VirtualSensor virtualSensor : virtualSensors) {
			if (virtualSensor.processingDistribution.equals("exponential")) {
				Exponential exp = new Exponential();
				double arrivalRate = (virtualSensor.publishFrequency * virtualSensor.messageSize);
				double mean = (double) 1.0 / (arrivalRate / globalMessageSize);
				exp.setMean(mean);
				String className = virtualSensor.deviceId + "_source_class";
				jmtModel.addClass(className, CommonConstants.CLASS_TYPE_OPEN, Defaults.getAsInteger("classPriority"), null, exp);
				jmtModel.setClassRefStation(jmtModel.getClassByName(className), jmtModel.getStationByName(virtualSensor.deviceId + "_source"));
			}
			else if (virtualSensor.processingDistribution.equals("deterministic")) {
				Deterministic deterministic = new Deterministic();
				double arrivalRate = (virtualSensor.publishFrequency * virtualSensor.messageSize);
				double mean = (double) 1.0 / (arrivalRate / globalMessageSize);
				deterministic.setMean(mean);
				String className = virtualSensor.deviceId + "_source_class";
				jmtModel.addClass(className, CommonConstants.CLASS_TYPE_OPEN, Defaults.getAsInteger("classPriority"), null, deterministic);
				jmtModel.setClassRefStation(jmtModel.getClassByName(className), jmtModel.getStationByName(virtualSensor.deviceId + "_source"));
			}
		}
	}
	
	public void addClassesForTopics(CommonModel jmtModel, Collection<Topic> topics) {
		for (Topic topic : topics) {
			Exponential exp = new Exponential();
			String className = topic.topicId + "_class";
			Object classObject = jmtModel.addClass(className, CommonConstants.CLASS_TYPE_OPEN, Defaults.getAsInteger("classPriority"), null, exp);
			jmtModel.setClassRefStation(classObject, CommonConstants.STATION_TYPE_CLASSSWITCH);
		}
	}
		
	public void addClassesForSubtopics(CommonModel jmtModel, HashMap<String, Subtopic> subtopics) {		
		for (Subtopic subtopic : subtopics.values()) {
			Exponential exp = new Exponential();
			String className = subtopic.name + "_class";
			Object classObject = new Object();
			classObject = jmtModel.addClass(className, CommonConstants.CLASS_TYPE_OPEN, subtopic.jmtPriority, null, exp);
			jmtModel.setClassRefStation(classObject, CommonConstants.STATION_TYPE_CLASSSWITCH);
		}	
	}
	
}
