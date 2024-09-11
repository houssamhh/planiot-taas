package iotSystemComponents;

import java.util.ArrayList;

public class Actuator {

	public String deviceId;
	public String deviceName;
	public String applicationCategory;
	public int publishFrequency;
	public int messageSize;
	public ArrayList<String> subscribedTopics;
	
	public Actuator(String deviceId, String deviceName, String applicationCategory, int publishFrequency, int messageSize, ArrayList<String> subscribedTopics) {
		this.deviceId = deviceId;
		this.deviceName = deviceName;
		this.applicationCategory = applicationCategory;
		this.publishFrequency = publishFrequency;
		this.messageSize = messageSize;
		this.subscribedTopics = (ArrayList<String>) subscribedTopics.clone();
	}
}
