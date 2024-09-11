package composer;

import java.util.HashMap;

import iotSystemComponents.Subtopic;
import jmt.gui.common.definitions.CommonModel;
import jmt.gui.common.definitions.SimulationDefinition;

public class PerformanceMetricsHandler {

	public void setPerformanceMetrics(CommonModel jmtModel, HashMap<String, Subtopic> subtopics, Object FINITE_CAPACITY_REGION_OBJECT,
								 double CONFIDENCE_INTERVAL, double RELATIVE_ERROR) {
	    	
	    	//Measure the response time for each subtopic in the fcr
	    	for (Subtopic subtopic : subtopics.values())
	    		jmtModel.addMeasure(SimulationDefinition.MEASURE_RP, FINITE_CAPACITY_REGION_OBJECT, jmtModel.getClassByName(subtopic.name + "_class"), CONFIDENCE_INTERVAL, RELATIVE_ERROR, true);
				
			
			//Measure fcr throughput for each class of topics
			for (Subtopic subtopic : subtopics.values())
				jmtModel.addMeasure(SimulationDefinition.MEASURE_X, FINITE_CAPACITY_REGION_OBJECT,  jmtModel.getClassByName(subtopic.name + "_class"), CONFIDENCE_INTERVAL, RELATIVE_ERROR, true);
			
			//Measure total fcr throughput
			jmtModel.addMeasure(SimulationDefinition.MEASURE_X, FINITE_CAPACITY_REGION_OBJECT, "", CONFIDENCE_INTERVAL, RELATIVE_ERROR, true);
			
			//Measure fcr dropping for each class of topics
			for (Subtopic subtopic : subtopics.values())
				jmtModel.addMeasure(SimulationDefinition.MEASURE_DR, FINITE_CAPACITY_REGION_OBJECT, jmtModel.getClassByName(subtopic.name + "_class"), CONFIDENCE_INTERVAL, RELATIVE_ERROR, true);
			
			//Measure utilization of fcr for each class
			for (Subtopic subtopic : subtopics.values())
				jmtModel.addMeasure(SimulationDefinition.MEASURE_U, FINITE_CAPACITY_REGION_OBJECT, jmtModel.getClassByName(subtopic.name + "_class"), CONFIDENCE_INTERVAL, RELATIVE_ERROR, true);
			
			//Measure total fcr utilization
			jmtModel.addMeasure(SimulationDefinition.MEASURE_U, FINITE_CAPACITY_REGION_OBJECT, "", CONFIDENCE_INTERVAL, RELATIVE_ERROR, true);
			
	    	//Measure the utilization of the output queue
			jmtModel.addMeasure(SimulationDefinition.MEASURE_U, jmtModel.getStationByName("outputQueue"), "", CONFIDENCE_INTERVAL, RELATIVE_ERROR, true);
			for (Subtopic subtopic : subtopics.values()) {
				jmtModel.addMeasure(SimulationDefinition.MEASURE_U, jmtModel.getStationByName("outputQueue"), jmtModel.getClassByName(subtopic.name + "_class"), CONFIDENCE_INTERVAL, RELATIVE_ERROR, true);
			}
	}
}
