#!/usr/bin/env python
# coding: utf-8

# # Vechile Detection and Counting 

# In[102]:


import cv2
import numpy as np;
import os
import re
from time import sleep
import time
import matplotlib.pyplot as plt


# In[103]:


# Video Capture  
cap = cv2.VideoCapture("Lane1.mp4")

object_detection = cv2.createBackgroundSubtractorKNN()  

detect = []
vechile = 0


# In[104]:


#Reading individual frame to know the dimension of the capturing frame  
ret, frame1 = cap.read()

height = frame1.shape[0]
width =  frame1.shape[1]

print(height, width)


# In[105]:


#Finding the centroid of the largest contour corresponding to each vechile 
def centroid(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy


# In[106]:


''' Frame by Frame Processing for detection and counting '''
while True:
    
    ret, frames = cap.read()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(3,3),5)
    img_sub = object_detection.apply(blur)
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilated = cv2.morphologyEx(dilat,cv2.MORPH_CLOSE, kernel)
    dilated = cv2.morphologyEx(dilated,cv2.MORPH_CLOSE, kernel)
    
    #mask = object_detection.apply(frames)
    #ret, thresh = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)
    
    #kernel = np.ones((5, 5),np.uint8)
    #dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    #cv2.line(frames, (100,300), (450,300), (0,0,255), 2)
    #cv2.line(frames, (180,200), (380,200), (0,0,255), 2)
    
    conts, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.line(frames, (25,550), (1200,550), (0,0,255), 2)
    
    for c in conts:    
        if cv2.contourArea(c)<150:
            continue
            
        (x,y,w,h) = cv2.boundingRect(c)
        validar_contorno = (w >= 80) and (h >= 80)
        if not validar_contorno:
            continue
        cv2.rectangle(frames,(x,y),(x+w,y+h),(0,255,0),2)
        
        center = centroid(x, y, w, h)
        detect.append(center)
        cv2.circle(frames, center, 3, (0,0,255),-1)
    
        for (x,y) in detect:
            if (y<556) and (y>544):
                vechile+=1;
                cv2.line(frames, (100,300), (450,300), (0,0,255), 2)
                #cv2.line(frames, (180,200), (380,200), (0,0,255), 2)
                detect.remove((x,y))
                print("Car is detected"+str(vechile))
        
        cv2.putText(frames, "Total Vechile :"+str(vechile), (50,50), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0),  2)
        
        print(vechile)
     
    cv2.imshow("Processed_Video",  frames)
    key = cv2.waitKey(30)
    
    if(key==27):
        break


# In[107]:


cap.release()
cv2.destroyAllWindows()


# # Traffic Management Algorithm

# In[82]:


import random
import math
import time
import sys
import os
import threading


# In[83]:


''' Assigning default values for signal times '''

defaultRed = 150
defaultYellow = 5
defaultGreen = 20
defaultMinimum  = 10
defaultMaximum = 60
noOfsignals = 4
signals = []
timeElapsed = 0


# In[84]:


#For changing te time of simulation the simTime is changed
simTime = 300


# In[85]:


currentGreen = 0      #States which signal is green
nextGreen = (currentGreen+1)%noOfsignals
currentYellow = 0     #states whether the yellow signal is on or off


# In[86]:


''' Average time for different types vechile to pass the intersetion is preset'''

carTime = 2
bikeTime = 1
rickshawTime = 2.25
busTime = 2.5
truckTime = 2.5

''' Initializing the count of the vechile at the traffi signal'''

noOfCars = 0
noOfBikes = 0
noOfBus = 0
noOfTrucks = 0
noOfRickshaws = 0
noOfLanes = 2

detectionTime = 5     # Red signal time at which cars will be detected at a signal

speeds = {'car':2.25, 'bus':1.8, 'truck':1.8, 'rickshaw':2, 'bike':2.5}  # Average speeds of vehicles


# In[87]:


# Coordinates of start
x = {'right':[0,0,0], 'down':[755,727,697], 'left':[1400,1400,1400], 'up':[602,627,657]}    
y = {'right':[348,370,398], 'down':[0,0,0], 'left':[498,466,436], 'up':[800,800,800]}

vehicles = {'right': {0:[], 1:[], 2:[], 'crossed':0}, 'down': {0:[], 1:[], 2:[], 'crossed':0}, 'left': {0:[], 1:[], 2:[], 'crossed':0}, 'up': {0:[], 1:[], 2:[], 'crossed':0}}
vehicleTypes = {0:'car', 1:'bus', 2:'truck', 3:'rickshaw', 4:'bike'}
directionNumbers = {0:'right', 1:'down', 2:'left', 3:'up'}

# Coordinates of signal image, timer, and vehicle count
signalCoods = [(530,230),(810,230),(810,570),(530,570)]
signalTimerCoods = [(530,210),(810,210),(810,550),(530,550)]
vehicleCountCoods = [(480,210),(880,210),(880,550),(480,550)]
vehicleCountTexts = ["0", "0", "0", "0"]

# Coordinates of stop lines
stopLines = {'right': 590, 'down': 330, 'left': 800, 'up': 535}
defaultStop = {'right': 580, 'down': 320, 'left': 810, 'up': 545}
stops = {'right': [580,580,580], 'down': [320,320,320], 'left': [810,810,810], 'up': [545,545,545]}

mid = {'right': {'x':705, 'y':445}, 'down': {'x':695, 'y':450}, 'left': {'x':695, 'y':425}, 'up': {'x':695, 'y':400}}
rotationAngle = 3

# Gap between vehicles
gap = 15    # stopping gap
gap2 = 15   # moving gap


# In[88]:


class TrafficSignal:
    def __init__(self, red, yellow, green, minimum, maximum):
        self.red = red
        self.yellow = yellow
        self.green = green
        self.minimum = minimum
        self.maximum = maximum
        self.signalText = "30"
        self.totalGreenTime = 0


# In[89]:


''' Initialization of signals with default values '''

def initialize():
    ts1 = TrafficSignal(0, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts1)
    ts2 = TrafficSignal(ts1.red+ts1.yellow+ts1.green, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts2)
    ts3 = TrafficSignal(defaultRed, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts3)
    ts4 = TrafficSignal(defaultRed, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts4)
    repeat()


# In[91]:


''' Set Time as per the case that when the number of vechile at a particular lane is more te Green signal alloted to the lane
    will also be for more time'''

def setTime():
    global noOfCars, noOfBikes, noOfBuses, noOfTrucks, noOfRickshaws, noOfLanes
    global carTime, busTime, truckTime, rickshawTime, bikeTime
    os.system("say detecting vehicles, "+directionNumbers[(currentGreen+1)%noOfSignals])
    noOfCars, noOfBuses, noOfTrucks, noOfRickshaws, noOfBikes = 0,0,0,0,0
    
    # The two weeler have a differnt lane as per the reference  hance handelled differnetly
    for j in range(len(vehicles[directionNumbers[nextGreen]][0])):
        vehicle = vehicles[directionNumbers[nextGreen]][0][j]
        if(vehicle.crossed==0):
            vclass = vehicle.vehicleClass
            # print(vclass)
            noOfBikes += 1
            
    for i in range(1,3):
        for j in range(len(vehicles[directionNumbers[nextGreen]][i])):
            vehicle = vehicles[directionNumbers[nextGreen]][i][j]
            if(vehicle.crossed==0):
                vclass = vehicle.vehicleClass
                # print(vclass)
                if(vclass=='car'):
                    noOfCars += 1
                elif(vclass=='bus'):
                    noOfBuses += 1
                elif(vclass=='truck'):
                    noOfTrucks += 1
                elif(vclass=='rickshaw'):
                    noOfRickshaws += 1
    
    greenTime = math.ceil(((noOfCars*carTime) + (noOfRickshaws*rickshawTime) + (noOfBuses*busTime) + (noOfTrucks*truckTime)+ (noOfBikes*bikeTime))/(noOfLanes+1))
    print('Green Time: ',greenTime)
    if(greenTime<defaultMinimum):
        greenTime = defaultMinimum
    elif(greenTime>defaultMaximum):
        greenTime = defaultMaximum
    # greenTime = random.randint(15,50)
    signals[(currentGreen+1)%(noOfSignals)].green = greenTime
    


# In[92]:


def repeat():
    global currentGreen, currentYellow, nextGreen
    while(signals[currentGreen].green>0):   # while the timer of current green signal is not zero
        printStatus()
        updateValues()
        if(signals[(currentGreen+1)%(noOfSignals)].red==detectionTime):    # set time of next green signal 
            thread = threading.Thread(name="detection",target=setTime, args=())
            thread.daemon = True
            thread.start()
            # setTime()
        time.sleep(1)
    currentYellow = 1   # set yellow signal on
    vehicleCountTexts[currentGreen] = "0"
    # reset stop coordinates of lanes and vehicles 
    for i in range(0,3):
        stops[directionNumbers[currentGreen]][i] = defaultStop[directionNumbers[currentGreen]]
        for vehicle in vehicles[directionNumbers[currentGreen]][i]:
            vehicle.stop = defaultStop[directionNumbers[currentGreen]]
    while(signals[currentGreen].yellow>0):  # while the timer of current yellow signal is not zero
        printStatus()
        updateValues()
        time.sleep(1)
    currentYellow = 0   # set yellow signal off
    
    # reset all signal times of current signal to default times
    signals[currentGreen].green = defaultGreen
    signals[currentGreen].yellow = defaultYellow
    signals[currentGreen].red = defaultRed
       
    currentGreen = nextGreen # set next signal as green signal
    nextGreen = (currentGreen+1)%noOfSignals    # set next green signal
    signals[nextGreen].red = signals[currentGreen].yellow+signals[currentGreen].green    # set the red time of next to next signal as (yellow time + green time) of next signal
    repeat()     


# In[93]:


''' Print the signal timers on cmd '''

def printStatus():                                                                                           
    for i in range(0, noOfSignals):
        if(i==currentGreen):
            if(currentYellow==0):
                print(" GREEN TS",i+1,"-> r:",signals[i].red," y:",signals[i].yellow," g:",signals[i].green)
            else:
                print("YELLOW TS",i+1,"-> r:",signals[i].red," y:",signals[i].yellow," g:",signals[i].green)
        else:
            print("   RED TS",i+1,"-> r:",signals[i].red," y:",signals[i].yellow," g:",signals[i].green)
    print()


# In[101]:


''' Update values of the signal timers after every second '''
def updateValues():
    for i in range(0, noOfSignals):
        if(i==currentGreen):
            if(currentYellow==0):
                signals[i].green-=1
                signals[i].totalGreenTime+=1
            else:
                signals[i].yellow-=1
        else:
            signals[i].red-=1

