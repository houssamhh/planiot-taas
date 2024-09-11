package iotSystemComponents;

import java.util.ArrayList;

public class IoTdevice {

	public String deviceId;
	public String deviceName;
	public double publishFrequency;  //in msgs/sec
	public double messageSize; //in bytes
	public String distribution;
	public ArrayList<String> publishedTopics;
	
	
	public IoTdevice(String deviceId, String deviceName, double publishFrequency, double messageSize, String distribution, ArrayList<String> publishedTopics) {
		this.deviceId = deviceId;
		this.deviceName = deviceName;
		this.publishFrequency =publishFrequency;
		this.messageSize = messageSize;
		this.distribution = distribution;
		this.publishedTopics = (ArrayList<String>) publishedTopics.clone();
		
	}
	
}
