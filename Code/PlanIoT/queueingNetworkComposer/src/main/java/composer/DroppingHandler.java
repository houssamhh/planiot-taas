package composer;

import jmt.gui.common.CommonConstants;
import jmt.gui.common.definitions.CommonModel;

public class DroppingHandler {
	
	public void addDroppingSink(CommonModel jmtModel) {
		jmtModel.addStation("dropping_sink", CommonConstants.STATION_TYPE_SINK);
		jmtModel.addStation("topics_class_switch_dropping", CommonConstants.STATION_TYPE_CLASSSWITCH);
		jmtModel.addStation("dropping_join", CommonConstants.STATION_TYPE_JOIN);
    }
}
