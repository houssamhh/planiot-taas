package composer;

public class NetworkResourcesManager {

	public String allocationPolicy;
	public double totalResources;
	public double allocatedBw_AN;
	public double allocatedBw_RT;
	public double allocatedBw_TS;
	public double allocatedBw_VS;
	
	public double globalMessageSize;
	
	public NetworkResourcesManager(double totalResources, String allocationPolicy, double globalMessageSize) {
		this.totalResources = totalResources;
		this.allocationPolicy = allocationPolicy;
		this.globalMessageSize = globalMessageSize;
	}
	
	public void allocateResources() {
		if (allocationPolicy.equals("none")) {
			allocatedBw_AN = totalResources;
			allocatedBw_RT = totalResources;
			allocatedBw_TS = totalResources;
			allocatedBw_VS = totalResources;
		}
		
		else if (allocationPolicy.equals("shared")) {
			allocatedBw_AN = totalResources / 4;
			allocatedBw_RT = totalResources / 4;
			allocatedBw_TS = totalResources / 4;
			allocatedBw_VS = totalResources / 4;
		}

	}
}
