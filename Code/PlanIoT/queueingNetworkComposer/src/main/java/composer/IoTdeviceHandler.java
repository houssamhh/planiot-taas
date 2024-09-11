package composer;

import java.util.Set;

import jmt.gui.common.CommonConstants;
import jmt.gui.common.definitions.CommonModel;

public class IoTdeviceHandler {

	
	public void addSources(CommonModel jmtModel, Set<String> devicesIds) {
    	for (String deviceId : devicesIds)
    		jmtModel.addStation(deviceId, CommonConstants.STATION_TYPE_SOURCE);
    }
}
