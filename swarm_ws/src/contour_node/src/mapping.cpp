#include <ros/ros.h>
#include <wvu_swarm_std_msgs/obstacle.h>
#include <wvu_swarm_std_msgs/map_levels.h>
#include <contour_node/level_description.h>
#include <wvu_swarm_std_msgs/nuitrack_data.h>
#include <nuitrack_bridge/nuitrack_data.h>
#include <contour_node/gaussian_object.h>

#include <math.h>

#define DEBUG 0
#define TEST_EQU 0
#define TEST_NUI 1

#if DEBUG
#include <iostream>
#endif

// Global variables for nuitrack data
wvu_swarm_std_msgs::nuitrack_data g_nui;
geometry_msgs::Point leftProjected, rightProjected;
levelObject* g_selected = nullptr;

// Find x,y where the line passing between alpha and beta intercepts an xy plane at z
geometry_msgs::Point findZIntercept(geometry_msgs::Point _alpha,
		geometry_msgs::Point _beta, double _zed)
{
	/* THEORY
	 * Equation of line: r(t) = v*t+v0
	 * Direction vector: v = (xa - xb, ya - yb, za - zb)
	 * Offset vector: v0 = (xa, ya, za)
	 * Plug in z to r(t), solve for t, use t to solve for x and y/
	 */

	geometry_msgs::Point ret;

	// Check if no solution
	if (_alpha.z == _beta.z)
	{
		printf("\033[1;31mhand_pointer: \033[0;31mNo solution for intercept\033[0m\n");
		ret.x = 0;
                ret.y = 0;
                ret.z = 0;
	}
	else
	{
		double t = (_zed - _alpha.z) / (_alpha.z - _beta.z);
		double x = _alpha.x * (t + 1) - _beta.x * t;
		double y = _alpha.y * (t + 1) - _beta.y * t;

		ret.x = x;
		ret.y = y;
		ret.z = _zed;
	}

	return ret;
}

static wvu_swarm_std_msgs::map_levels overall_map;

void newObs(wvu_swarm_std_msgs::obstacle obs)
{
	while (overall_map.levels.size() <= obs.level)
	{
#if DEBUG
		std::cout << "\033[30;43mAdded level: " << overall_map.levels.size()
				<< "\033[0m" << std::endl;
#endif
		overall_map.levels.push_back(wvu_swarm_std_msgs::map_level());
	}
	bool found = false;

	for (size_t i = 0;
			i < overall_map.levels.at(obs.level).functions.size() && !found; i++)
	{
		if (overall_map.levels.at(obs.level).functions.at(i).name.compare(
				obs.characteristic.name) == 0)
		{
			overall_map.levels.at(obs.level).functions[i] = obs.characteristic;
			found = true;
		}
	}

	if (!found)
		overall_map.levels[obs.level].functions.push_back(obs.characteristic);

	if (obs.level != map_ns::COMBINED)
	{
		wvu_swarm_std_msgs::obstacle comb;
		obs.characteristic.amplitude *= 1 - ((obs.level % 2) * 2);
		comb.characteristic = obs.characteristic;
		comb.level = map_ns::COMBINED;

		newObs(comb);
	}
}

void nuiCallback(wvu_swarm_std_msgs::nuitrack_data nui)
{
    // Copy the message
    g_nui = nui;

    // Find projections of hands onto table
    leftProjected = findZIntercept(g_nui.leftWrist, g_nui.leftHand, 0.0);
    rightProjected = findZIntercept(g_nui.rightWrist, g_nui.rightHand, 0.0);
}

wvu_swarm_std_msgs::gaussian gausToRos(gaussianObject* _gaus)
{
    wvu_swarm_std_msgs::gaussian ret;
    
    ret.amplitude = _gaus->getAmplitude();
    ret.name = _gaus->getName();
    ret.selected = _gaus->isSelected();
    ret.ellipse.offset_x = _gaus->getOrigin().first;
    ret.ellipse.offset_y = _gaus->getOrigin().second;
    ret.ellipse.theta_offset = _gaus->getTheta();
    ret.ellipse.x_rad = _gaus->getRadii().first;
    ret.ellipse.y_rad = _gaus->getRadii().second;
    
    return ret;
}

double getDistance(geometry_msgs::Point *_hand, levelObject *_target)
{
    double ret = pow(_hand->x - _target->getOrigin().first, 2)
                    + pow(_hand->y - _target->getOrigin().second, 2);
    return sqrt(ret);
}

void findSelection(double _maxDist, std::vector<levelObject*> _map)
{
    double currentMinimum = _maxDist, thisDist;
    g_selected = nullptr;
    
    for(levelObject* i : _map)
    {
        // Check if this distance is less than threshold
        thisDist = getDistance(&leftProjected, i);
        if(thisDist < currentMinimum) {
            // Decrease minimum, we want closest point
            currentMinimum = thisDist;
            g_selected = i;
        }
    }
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "mapping");
	ros::NodeHandle n;

	ros::Publisher map_pub = n.advertise < wvu_swarm_std_msgs::map_levels
			> ("/map_data", 1000);
        ros::Publisher left_pub = n.advertise<geometry_msgs::Point>("/nui_bridge/hand_1", 1000);
        ros::Publisher right_pub = n.advertise<geometry_msgs::Point>("/nui_bridge/hand_2", 1000);
	ros::Subscriber n_obs = n.subscribe("/add_obstacle", 1000, newObs);
	ros::Subscriber nuiSub = n.subscribe("/nuitrack_bridge", 1000, nuiCallback);

	///////////////////////////////////////////////////////////////////
	// default map setup
#if DEBUG
	std::cout << "Adding equation" << std::endl;
#endif

#if TEST_EQU
	wvu_swarm_std_msgs::ellipse el;
	el.x_rad = 5;
	el.y_rad = 2;
	el.theta_offset = M_PI_4;

	wvu_swarm_std_msgs::gaussian gaus;
	gaus.ellipse = el;
	gaus.ellipse.offset_x = 0;
	gaus.ellipse.offset_y = 0;
	gaus.amplitude = 20;
	gaus.name = "Bob";

	wvu_swarm_std_msgs::obstacle obs;
	obs.characteristic = gaus;
	obs.level = map_ns::TARGET;

	newObs(obs);

	el.x_rad = 4;
	el.y_rad = 7;
	el.theta_offset = 0;

	gaus.ellipse = el;
	gaus.ellipse.offset_x = 10;
	gaus.ellipse.offset_y = 0;
	gaus.amplitude = 10;
	gaus.name = "Jeff";

	obs.characteristic = gaus;
	obs.level = map_ns::TARGET;

	newObs(obs);
#endif
        
#if TEST_NUI
        std::vector<levelObject*> worldMap;
        
        levelObject* ptr;
        
        ptr = new gaussianObject(0, 0, "Gary", 10, 20, M_PI / 4.0, 10, map_ns::COMBINED);
        worldMap.push_back(ptr);
        
        ptr = new gaussianObject(50, 0, "Larry", 5, 5, 0, 10, map_ns::COMBINED);
        worldMap.push_back(ptr);
#endif 
        
#if DEBUG
	std::cout << "\033[30;42mdone adding equation\033[0m" << std::endl;
#endif
	// end default map setup
	///////////////////////////////////////////////////////////////////

	ros::Rate rate(60);

#if TEST_EQU
	int tick = 0;
#endif
        
        geometry_msgs::Point* anchor = nullptr; // Where did the user's hand start when they grabbed a feature?

	while (ros::ok())
	{
#if TEST_EQU
		tick++;
		tick %= 1000;

		el.x_rad = 5;
		el.y_rad = 2;
		el.theta_offset = tick * M_PI / 100;

		gaus.ellipse = el;
		gaus.ellipse.offset_x = 0;
		gaus.ellipse.offset_y = 0;
		gaus.amplitude = 20;
		gaus.name = "Bob";

		obs.characteristic = gaus;
		obs.level = map_ns::TARGET;
		newObs(obs);
#endif
                
#if TEST_NUI
                // Populate ros map from our map
                wvu_swarm_std_msgs::map_level m;
                for(levelObject* i : worldMap) {
                    m.functions.push_back(gausToRos((gaussianObject*)i));
                }
                
                overall_map.levels.clear();
                overall_map.levels.push_back(wvu_swarm_std_msgs::map_level());
                overall_map.levels.push_back(wvu_swarm_std_msgs::map_level());
                overall_map.levels.push_back(wvu_swarm_std_msgs::map_level());
                overall_map.levels.push_back(m);
#endif

		map_pub.publish(overall_map);
                
                // If user's left hand (RIGHT) is open, points are free to move
                if(!g_nui.rightClick)
                {
                    // Reset anchor since nothing is being manipulated
                    anchor = nullptr;
                    
                    // Check if something is selected, if so lock point to it
                    findSelection(10.0, worldMap);
                    if(g_selected != nullptr)
                    {
                        leftProjected.x = g_selected->getOrigin().first;
                        leftProjected.y = g_selected->getOrigin().second;
                    }
                    
                    // If nothing is selected, move points freely
//                    if(leftProjected.x == 0.0 && leftProjected.y == 0.0 && leftProjected.z == 0.0)
                        left_pub.publish(leftProjected);
//                    if(rightProjected.x == 0.0 && rightProjected.y == 0.0 && rightProjected.z == 0.0)
                        right_pub.publish(rightProjected);
                }
                // If user's left hand (RIGHT) is closed, maybe modify objects
                else
                {
                    ROS_INFO("Hand closed!");
                    // If nothing had been selected before hand was closed, do nothing
                    if(g_selected != nullptr)
                    {
                        ROS_INFO("%s", g_selected->getName().c_str());
                        // If object was just grabbed, set the anchor
                        if(anchor == nullptr) {
                            anchor = new geometry_msgs::Point;
                            anchor->x = g_nui.leftHand.x;
                            anchor->y = g_nui.leftHand.y;
                            anchor->z = g_nui.leftHand.z;
                        }
                        
                        // Manipulate the object
                        g_selected->nuiManipulate(g_nui.leftHand.x - anchor->x,
                                g_nui.leftHand.y - anchor->y, g_nui.leftHand.z - anchor->z);
                        
                        // Update anchor so it's one iteration behind hand
                        anchor->x = g_nui.leftHand.x;
                        anchor->y = g_nui.leftHand.y;
                        anchor->z = g_nui.leftHand.z;
                        
                        ROS_INFO("Anchor moved!");
                    }
                }
                
                

		ros::spinOnce();
		rate.sleep();
	}
}
