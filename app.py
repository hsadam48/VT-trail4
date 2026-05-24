import queue
import random
from functools import reduce
import time
import os

# Run the simulation how many times?
Trials = 1000
FilePath = "./Specific_Runs/"
if not os.path.exists(FilePath):
    os.makedirs(FilePath)

# Constants
###############
ArrivalSeparation = (1, 30)  # The time between passenger arrivals (min, max) seconds
DOORTIME = 5                 # The amount of time it takes to open or close the doors of the elevator
LOADINGTIME = 3              # The time it takes to get in or out of an elevator
STARTTIME = time.perf_counter() # Used for judging time to complete trials
###############
ITERCOUNTER = 0


# CLASS passenger has four variables
# arrival - the arrival time in seconds
# elevatorTime - the amount of time spent in an elevator
# destination - the destination floor
# floor - the person's current floor
class Passenger:
    def __init__(self, arrivaltime, destinationfloor, startingfloor, passengernumber):
        self.arrival = arrivaltime
        self.destination = destinationfloor
        self.floor = startingfloor
        self.number = passengernumber
        self.elevatorTime = 0


# CLASS elevator handles dispatching and processing loop logic
class Elevator:
    def __init__(self, queuearray, group_idx, floor_specification, elevator_groups, floors_count, capacity):
        self.counter = 0
        self.unused = 0
        self.doors = 0  # 0: open, 1: closed
        self.stops = 0
        self.state = 'w'  # states: loading 'l', waiting 'w', going up 'u', going down 'd', exiting 'e'
        self.queue = queuearray
        self.capacity = capacity
        
        # Pull configurations based on group index mapping
        self.availablefloors = elevator_groups[floor_specification[group_idx] - 1]
        self.feedqueue = None  # Will be mapped explicitly during run execution setup
        self.basefloor = min(self.availablefloors)
        self.currentfloor = self.basefloor
        self.topfloor = max(self.availablefloors)
        
        for _ in range(floors_count):
            self.queue.append(queue.LifoQueue(self.capacity))

    def SuperReset(self):
        self.counter = 0
        self.unused = 0
        self.doors = 0
        self.state = 'w'
        self.stops = 0
        self.currentfloor = self.basefloor

    def countunused(self):
        self.unused += 1

    def changestate(self, s):
        self.state = s

    def opendoors(self):
        self.doors = 0

    def closedoors(self):
        self.doors = 1

    def reset(self):
        self.counter = 0

    def count(self):
        self.counter += 1

    def passengers(self):
        return sum([e.qsize() for e in self.queue])


# Create the queue on floor 1
class FloorQueue:
    def __init__(self, FLOORS, S):
        self.availablefloors = FLOORS
        self.downstairs = queue.Queue()
        self.servicefloor = S
        self.loading = 0


def QUEUESIZE(F):
    total = 0
    for i in F:
        if i.servicefloor == 1:
            total += i.downstairs.qsize()
    return total


def TotalAvailableFloors(floors_list, elevator_groups):
    T = list(floors_list)
    for F in floors_list:
        for Q in elevator_groups:
            if F == min(Q):
                T.extend(Q)
    return list(set(T))


def floorchooser(prob_distribution):
    r = random.random()
    cumulative = 0.0
    for idx, prob in enumerate(prob_distribution):
        cumulative += prob
        if r <= cumulative:
            return idx + 2
    return len(prob_distribution) + 1


# Iterative simulation matrix runner
for numfloors in [(50, [8, 10, 12], [10, 6, 4]), (75, [10, 12, 14], [10, 5, 3])]:
    floors = numfloors[0]
    
    # Structural configuration profiles setup
    split_groups = [list(range(1, floors // 2 + 1)), list(range(floors // 2, floors + 1))]
    evenodd_groups = [
        [1] + [x * 2 + 1 for x in range(1, (floors + 1) // 2) if (x * 2 + 1) <= floors],
        [1] + [x * 2 for x in range(1, floors // 2 + 1) if (x * 2) <= floors]
    ]
    normal_groups = [list(range(1, floors + 1))]
    
    for EGROUPS in [("split", split_groups), ("evenodd", evenodd_groups), ("normal", normal_groups)]:
        ElevatorGroups = EGROUPS[1]
        
        for els in numfloors[1]:
            FloorSpecification = [1] * (els // 2) + [len(ElevatorGroups)] * (els - (els // 2))
            NumElevators = els
            
            for TransitionTime in numfloors[2]:
                for Elevator_Wait_Time in [15, 10, 5]:
                    for CAPACITY in [20, 16, 12]:
                        
                        FloorProbability = [1] * (floors - 1)
                        FileName = f"{floors}_{EGROUPS[0]}_{els}_{TransitionTime}_{Elevator_Wait_Time}_{CAPACITY}"
                        
                        Storage = open(f"{FilePath}data{FileName}.csv", "w")
                        Stats = open(f"{FilePath}statdata{FileName}.csv", "w")
                        
                        f_header = ""
                        for i in range(NumElevators):
                            f_header += f",Elevator {i+1} Usage %,Elevator {i+1} Stops"
                        
                        Stats.write("Max Waiting Time,Mean Waiting Time,Max Elevator Time,Mean Elevator Time,Max Total Time,Mean Total Time" + f_header + ",Max Queue Size\n")
                        Storage.write("Waiting Time in Queue,Time Spent in Elevator,Delivery Time,Destination Floor,Enter Time\n")
                        
                        FLOORQUEUES = []
                        for group in ElevatorGroups:
                            FLOORQUEUES.append(FloorQueue(TotalAvailableFloors(group, ElevatorGroups), min(group)))
                        
                        S_sum = float(sum(FloorProbability))
                        FloorProbability = [x / S_sum for x in FloorProbability]
                        
                        E = []
                        QUEUES = []
                        for i in range(NumElevators):
                            QUEUES.append([])
                            elevator_instance = Elevator(QUEUES[i], i, FloorSpecification, ElevatorGroups, floors, CAPACITY)
                            # Link correct loading zone queue
                            elevator_instance.feedqueue = FLOORQUEUES[FloorSpecification[i] - 1]
                            E.append(elevator_instance)
                            
                        for iterations in range(Trials):
                            Time_Spent_In_Elevator = []
                            Total_Time = []
                            Waiting_Time = []
                            Destination_Floor = []
                            Enter_Time = []
                            stoptime = 0
                            
                            # Generating deterministic runtime timeline schedules
                            curr_time_idx = 0
                            ArrivalTimes = []
                            while curr_time_idx < 4800:
                                R = random.randint(ArrivalSeparation[0], ArrivalSeparation[1])
                                if curr_time_idx + R > 4800:
                                    break
                                ArrivalTimes.append(curr_time_idx + R)
                                curr_time_idx += R
                                
                            ArrivalTimesSet = set(ArrivalTimes)
                            endlen = len(ArrivalTimesSet)
                            MaxSize = 0
                            PassengerCount = 0
                            
                            for sim_time in range(0, 8000):
                                # Passenger arrival check loop
                                if sim_time in ArrivalTimesSet:
                                    PassengerCount += 1
                                    person = Passenger(sim_time, floorchooser(FloorProbability), 1, PassengerCount)
                                    
                                    for I in FLOORQUEUES:
                                        if person.destination in I.availablefloors and I.servicefloor == person.floor:
                                            I.downstairs.put(person)
                                            break
                                            
                                    Size = QUEUESIZE(FLOORQUEUES)
                                    if Size > MaxSize:
                                        MaxSize = Size
                                        
                                # Simulation terminal checks conditions
                                if len(Time_Spent_In_Elevator) == endlen and all(e.state == 'w' for e in E) and all(not e.doors for e in E):
                                    stoptime = sim_time
                                    break
                                    
                                # Core Lift Bank Iteration System Loop Logic
                                for elevator in E:
                                    if elevator.state == 'w':
                                        if elevator.doors == 0:
                                            elevator.count()
                                            if elevator.counter == DOORTIME:
                                                elevator.closedoors()
                                                elevator.reset()
                                        elif elevator.feedqueue.downstairs.empty():
                                            if elevator.passengers() > 0:
                                                elevator.count()
                                                if elevator.counter == Elevator_Wait_Time:
                                                    elevator.changestate('u')
                                                    elevator.reset()
                                            else:
                                                elevator.countunused()
                                        else:
                                            if not elevator.feedqueue.loading:
                                                elevator.changestate('l')
                                                elevator.reset()
                                                elevator.feedqueue.loading = 1
                                            else:
                                                if elevator.passengers() > 0:
                                                    elevator.count()
                                                    if elevator.counter == Elevator_Wait_Time:
                                                        elevator.changestate('u')
                                                        elevator.reset()
                                                else:
                                                    elevator.countunused()
                                                    
                                    elif elevator.state == 'l':
                                        if elevator.feedqueue.downstairs.empty():
                                            elevator.changestate('w')
                                            elevator.reset()
                                            elevator.feedqueue.loading = 0
                                        else:
                                            elevator.count()
                                            if elevator.counter == LOADINGTIME:
                                                person = elevator.feedqueue.downstairs.get()
                                                person.elevatorTime = sim_time
                                                
                                                buff = elevator.availablefloors[0]
                                                for floor_stop in elevator.availablefloors:
                                                    if person.destination >= floor_stop:
                                                        buff = floor_stop
                                                    else:
                                                        break
                                                        
                                                elevator.queue[buff - 1].put(person)
                                                elevator.reset()
                                                
                                                if elevator.passengers() == elevator.capacity:
                                                    elevator.changestate('u')
                                                    elevator.reset()  # Reset step counter to clear travel time drift
                                                    elevator.feedqueue.loading = 0
                                                elif elevator.feedqueue.downstairs.empty():
                                                    elevator.changestate('w')
                                                    elevator.reset()
                                                    elevator.feedqueue.loading = 0
                                                    
                                    elif elevator.state == 'u':
                                        if not elevator.doors:
                                            elevator.count()
                                            if elevator.counter == DOORTIME:
                                                elevator.closedoors()
                                                elevator.reset()
                                        else:
                                            elevator.count()
                                            if elevator.counter == TransitionTime:
                                                elevator.currentfloor += 1
                                                elevator.reset()
                                                elevator.changestate('e')
                                                
                                    elif elevator.state == 'e':
                                        if elevator.queue[elevator.currentfloor - 1].empty():
                                            if elevator.topfloor > elevator.currentfloor and elevator.passengers():
                                                elevator.changestate('u')
                                                elevator.reset()
                                            else:
                                                elevator.changestate('d')
                                                elevator.reset()
                                        else:
                                            if elevator.doors:
                                                elevator.count()
                                                if elevator.counter == DOORTIME:
                                                    elevator.reset()
                                                    elevator.opendoors()
                                                    elevator.stops += 1
                                            else:
                                                elevator.count()
                                                if elevator.counter == LOADINGTIME:
                                                    person = elevator.queue[elevator.currentfloor - 1].get()
                                                    person.floor = elevator.currentfloor
                                                    elevator.reset()
                                                    
                                                    if person.destination == person.floor:
                                                        Enter_Time.append(person.elevatorTime)
                                                        Time_Spent_In_Elevator.append(sim_time - person.elevatorTime)
                                                        Total_Time.append(sim_time - person.arrival)
                                                        Waiting_Time.append(person.elevatorTime - person.arrival)
                                                        Destination_Floor.append(person.destination)
                                                    else:
                                                        for I in FLOORQUEUES:
