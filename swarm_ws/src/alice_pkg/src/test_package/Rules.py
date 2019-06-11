'''
Rules doesn't save any data. Instead, it is a collection of functions, each of which will get called sequentially
They will always return a tuple of the form (direction, speed) and accept at least a tolerance

Need to:
Stay near the center of the swarm
Don't hit other robots
Avoid collions with objects
Match speed with other robots
Match angle with other robots

if abs(avg_thing - my_thing) * tolerance > something:
'''
import math
import cmath

ROBOT_SIZE = 7

class Rules:

    '''
    Helper method for maintaining spacing
    '''
    def findCenter(self, robots):
        x = y = 0
        for robot in robots:
            x += robot[1] * math.cos(robot[0])
            y += robot[1] * math.sin(robot[0])
        distance = (x**2 + y**2) ** 0.5
        if x != 0:
        	direction = math.atan(y/x)
        else:
        	direction = 0
        return direction, distance
    

    def maintainSpacing(self, robots, tolerance):
        direction, distance = self.findCenter(robots)
        if (4 * ROBOT_SIZE / tolerance < distance): #This is mostly arbitrary and will need to be adjusted in simulation
            print("Maintaining spacing")
            return  (direction, 1 - math.sin(direction))
        else:
            return None

    def avoidRobots(self, robots, tolerance):
        for robot in robots:
            if ((robot[0] < math.pi/12 * tolerance) or (robot[0] > 11 * math.pi/12 - math.pi/12 *tolerance)) and (robot[1] < ROBOT_SIZE / 2 * tolerance): #This is mostly arbitrary and will need to be adjusted in simulation
                print("Avoiding robots")
                return ((robot[0] + math.pi/2)%math.pi, 1)
        return None

    def avoidObstacles(self, obstacles, tolerance):
        if obstacles == None:
            return None
        for obstacle in obstacles:
            if ((obstacle[0] < math.pi/12 * tolerance) or (obstacle[0] > 11 * math.pi/12 - math.pi/12 *tolerance)) and (obstacle[1] < ROBOT_SIZE / 2 * tolerance):
                print("Avoiding obstacles")
                return ((obstacle[0] + math.pi/2)%math.pi, 1/(1 + tolerance))
        return None
    
    '''
    Helper for matchSpeed
    '''
    def findSpeedMean(self, robots):
        speed_list = [robot[2] for robot in robots]
        return sum(speed_list)/len(speed_list)
        
    def matchSpeed(self, robots, speed, tolerance):
    	if len(robots) == 0:
    		return None
        avg_speed = self.findSpeedMean(robots)
        if abs(speed - avg_speed) * tolerance > 0.5:
            print("Matching speed")
            return (0, avg_speed)
        return None
'''
    This seems extraneous
    Helper for matchAngle
      
    def findAngleMedian(self, robots):
        angle_list = [robot[3] for robot in robots]
        return statistics.median(angle_list)
    
    def matchAngle(self, robots, tolerance):
        avg_angle = self.findAngleMedian(robots)
        if abs(avg_angle - math.pi/2) * tolerance > math.pi/2:
            print("Matching angle")
            return (avg_angle, 1)
        return None
'''          
        
