package iotSystemComponents;

import java.util.ArrayList;

public class Broker {

	public String brokerId;
	public String brokerName;
	public int bufferSize;
	public int processingRate;
	public ArrayList<String> topics;
	
	public Broker(String brokerId, String brokerName, int bufferSize, int processingRate, ArrayList<String> topics) {
		this.brokerId = brokerId;
		this.brokerName = brokerName;
		this.bufferSize = bufferSize;
		this.processingRate = processingRate;
		this.topics = (ArrayList<String>) topics.clone();
	}
	
}
