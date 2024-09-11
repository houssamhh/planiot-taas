package composer;

import java.util.HashMap;

import iotSystemComponents.Subtopic;
import jmt.gui.common.CommonConstants;
import jmt.gui.common.definitions.CommonModel;

public class FiniteCapacityRegionHandler {

	public Object setFiniteCapacityRegion(CommonModel jmtModel, int brokerCapacity, HashMap<String, Subtopic> subtopics) {
		Object fcrObject = jmtModel.addBlockingRegion("finite_capacity_region", CommonConstants.INFINITE_CAPACITY);
		jmtModel.addRegionStation(fcrObject, jmtModel.getStationByName("input"));
		jmtModel.addRegionStation(fcrObject, jmtModel.getStationByName("outputQueue"));
		if (brokerCapacity > 0) {
			jmtModel.setRegionCustomerConstraint(fcrObject, brokerCapacity);
			for (Subtopic subtopic : subtopics.values()) {
				String subtopicname = subtopic.name;
				jmtModel.setRegionClassCustomerConstraint(fcrObject, jmtModel.getClassByName(subtopicname + "_class"), brokerCapacity);
				jmtModel.setRegionClassDropRule(fcrObject, jmtModel.getClassByName(subtopicname + "_class"), true);
			}
		}
		return fcrObject;
	}
}
