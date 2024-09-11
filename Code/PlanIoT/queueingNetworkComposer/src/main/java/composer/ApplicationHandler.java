package composer;

import java.util.Collection;
import java.util.HashMap;

import iotSystemComponents.Application;
import jmt.gui.common.CommonConstants;
import jmt.gui.common.definitions.CommonModel;

public class ApplicationHandler {

	public HashMap<String, Integer> subsPerAppCategory = new HashMap<String, Integer>();
	
    public void addApplications(CommonModel jmtModel, Collection<Application> applications) {
    	for (Application app : applications) {
    		jmtModel.addStation(app.applicationName, CommonConstants.STATION_TYPE_SERVER);
    		String category = app.applicationCategory;
    		int nbOfSubs = subsPerAppCategory.get(category);
    		nbOfSubs++;
    		subsPerAppCategory.put(category, nbOfSubs);
    	}		
    }
    
    public ApplicationHandler() {
    	subsPerAppCategory.put("AN", 0);
    	subsPerAppCategory.put("RT", 0);
    	subsPerAppCategory.put("TS", 0);
    	subsPerAppCategory.put("VS", 0);
    }
}
