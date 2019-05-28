'''
Rules doesn't save any data. Instead, it is a collection of functions, each of which will get called sequentially
They will always return a tuple of the form (direction, speed) and accept at least a tolerance

Need to:
Stay near the center of the swarm
Don't hit other robots
Avoid collions with objects
Match speed with other robots
Match angle with other robots
'''
import math
from statistics import median

ROBOT_SIZE = 7

class Rules:

    '''
    Helper method for maintaining spacing
    '''
    def findCenters(robots):
        direction = 0
        distance = 0
        for robot in robots:
            direction = direction + robot[0]/len(robots)
            distance = distance + robot[1]/len(robots)
        return direction, distance

    def maintainSpacing(robots, tolerance):
        direction, distance = findCenters(robots)
        if (tolerance * ROBOT_SIZE / 3 < distance): #This is mostly arbitrary and will need to be adjusted in simulation
            return  (direction, 1 - math.sin(direction))
        else:
            return None

    def avoidRobots(robots, tolerance):
        for robot in robots:
            if (robot[1] > tolerance * ROBOT_SIZE / 5): #This is mostly arbitrary and will need to be adjusted in simulation
                return ((robot[0] + math.pi/2)%math.pi, 1)
        return None

    def avoidObjects(obstacles, tolerance):
        for obstacle in obstacles:
            if ((obstacle[1] > math.pi/12) or obstacle[1] > 11*math.pi/12) and (obstacle[0] > tolerance * ROBOT_SIZE / 4):
                return ((robot[0] + math.pi/2)%math.pi, 1/(1 + tolerance))
        return None

    '''
    Helper for matchSpeed
    '''
    def findSpeedMedian(robots):
        speed_list = []
        for robot in robots:
            speed_list += robot[2]
        return statistics.median(speed_list)
        
    def matchSpeed(robots, speed, tolerance):
        avg_speed = findSpeedMedian(robots)
        if not (speed - tolerance * 0.2 < avg_speed < speed + tolerance * 0.2):
            return (0, avg_speed)
        return None

    '''
    Helper for matchAngle
    '''       
    def findAngleMedian(robots):
        angle_list = []
        for robot in robots:
            angle_list += robot[0]
        return statistics.median(angle_list)
    
    def matchAngle(robots, tolerance):
        avg_angle = findAngleMedian(robots)
        if not ((math.pi - tolerance * math.pi/12) < avg_angle or avg_angle < tolerance * math.pi/12):
            return (avg_angle, 1)
        return None
            
        
