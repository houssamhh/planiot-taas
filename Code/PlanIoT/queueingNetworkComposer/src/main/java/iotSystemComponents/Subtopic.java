package iotSystemComponents;

public class Subtopic {

	public String name;
	public String parentTopicName;
	public String appName;
	public String category;
	public int priority;
	public int jmtPriority;
	
	
	public Subtopic(String name, String parentTopicName, String appName, String category, int priority, int jmtPriority) {
		this.name = name;
		this.parentTopicName = parentTopicName;
		this.appName = appName;
		this.category = category;
		this.priority = priority;
		this.jmtPriority = jmtPriority;
	}
}
