package composer;

import java.util.Collection;
import java.util.HashMap;

import iotSystemComponents.VirtualSensor;
import jmt.gui.common.CommonConstants;
import jmt.gui.common.definitions.CommonModel;

public class VirtualSensorHandler {

	public HashMap<String, Integer> subsPerAppCategory = new HashMap<String, Integer>();
			
    public void addVirtualSensor(CommonModel jmtModel, Collection<VirtualSensor> virtualSensors) {
    	for (VirtualSensor virtualSensor : virtualSensors) {
    		Object processingObj = jmtModel.addStation(virtualSensor.deviceId+ "_processing", CommonConstants.STATION_TYPE_SERVER);
    		Object sourceObj = jmtModel.addStation(virtualSensor.deviceName + "_source", CommonConstants.STATION_TYPE_SOURCE);
    		Object sinkObj = jmtModel.addStation(virtualSensor.deviceName + "_sink", CommonConstants.STATION_TYPE_SINK);
    		
    		String category = virtualSensor.applicationCategory;
    		int nbOfSubs = subsPerAppCategory.get(category);
    		nbOfSubs++;
    		subsPerAppCategory.put(category, nbOfSubs);
    		jmtModel.setConnected(processingObj, sinkObj, true);
    	}
    }
    
    public VirtualSensorHandler() {
    	subsPerAppCategory.put("AN", 0);
    	subsPerAppCategory.put("RT", 0);
    	subsPerAppCategory.put("TS", 0);
    	subsPerAppCategory.put("VS", 0);
    }
}
