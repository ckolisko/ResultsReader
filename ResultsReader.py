
from typing import Optional

import pandas as pd
from enum import Enum
import numpy as np
import math
import os
from pathlib import Path
import time
import datetime
import shutil
import matplotlib.pyplot as plt
import jax.numpy as jnp
import copy
import warnings


heatCorrectionValues = {
    "TYE665":{
        "Cyt5":{ # Determined from 500 nM of TYE665 fluorophore on CYT5 machine (06/21/23)
            "AmplitudeOvershoot": 0.35,
            "InvTau": 0.09
        }
    }
}
kVal = 0.09
AmplitudeOvershoot = 0.35

class ResultsReader:
    
    timeUnitDict = {"s":"seconds", "m":"minutes", "h":"hours", "d":"days"}
    
    # Make a results reader. Filename is the name of the file reading the data.
    # Uses break size to divide file data into list of data lists.
    def __init__(self, filename:str, outputFolderName:str = "OutputResultsReader", breakSize:int = 120, tempIn:float = 37.0, tempOut:float= 21.0, heatCorrect:bool=True, timeUnit:str = "s", csv:bool=False):
        
        if self.timeUnitDict.get(timeUnit) is None:
            self.timeUnit= timeUnit
        else:
            self.timeUnit = self.timeUnitDict[timeUnit]
        
        # Folder initialization.
        self.logStep = 0
        self.logSubStep = 1
        self.runnerName = outputFolderName
        os.mkdir(outputFolderName)
        logFolder = (outputFolderName + "/logs")
        os.mkdir(logFolder)

        # Want to start off by reading parsing the plate reader's results, adding copy to folder.
        self.filename = filename
        rawData:pd.DataFrame 
        if (csv):
            rawData = pd.read_csv(filename)
        else :
            rawData = pd.read_table(filename, header=0)
        shutil.copy(filename, ("./" + outputFolderName + "/rawData.txt"))

        self.logfiles = [logFolder + "/globalLog.txt"]
        self.__initialLogs(outputFolderName, breakSize, timeUnit, heatCorrect, tempOut, tempIn)

        self.tempIn = tempIn
        self.tempOut = tempOut
        self.highVals = {} # These are per column
        self.lowVals = {} # These are per column

        # Assume first column is time.
        timeVals:pd.Series = rawData.iloc[:,0]

        # Put the column names into the dicionary.
        for i in rawData.columns:
            if (rawData.columns.get_loc(i) != 0):
                self.highVals[i] = -1
                self.lowVals[i] = -1
        
        # Clean off the 0's at the end.
        zeroIndex:int = self._getZeroIndex(timeVals)
        if (zeroIndex != -1):
            timeVals = timeVals[0:zeroIndex]
            rawData = rawData[0:zeroIndex]
            
        # Get rid of asterisks.
        rawData = rawData.replace(r'\*', '', regex=True)
                
        rawData = rawData.astype(float)
        
        # Remove wells with all 0's for data.
        rawData = self.__removeAllZeroWells(rawData)
        
        self.headers = rawData.columns
        
        # Get the index of each time break in rawData.
        timeBreakInd:list[int] = self.__createTimeBreaks(timeVals, breakSize)

        self.appendLog("Global time break indicies: " + str(timeBreakInd), 0)
        
        # Now, Lets get temp val over time.
        tempVals:list[float] = self.__findTempVals(timeVals, timeBreakInd, tempIn, tempOut)
        
        if (heatCorrect):
            # Overlay the temp vals with the raw data for a second.
            rawData.insert(1, "Temps", tempVals)

            # Lets use the temps to correct flourecense.
            self.heatCorrectHelper(rawData, tempOut, tempIn)
            rawData = rawData.drop("Temps", axis=1)


        # Break raw data into dataframes, then each dataframe into dataframe for each well.
        temporaryData:list[pd.DataFrame] = []
        if (len(timeBreakInd) == 0):
            timeBreakInd.append(len(timeVals - 1))
        for i in range (len(timeBreakInd)):
            # for the first one, go from 0 - ind first.
            if (i == 0):
                temporaryData.append (rawData[0:timeBreakInd[i]])
            if (timeBreakInd[i] == timeBreakInd[-1]):
                # For the last time break, go to end of rawData.
                data = rawData[timeBreakInd[i]:]
                if (len(data) != 0):
                    temporaryData.append (rawData[timeBreakInd[i]:])
            else : 
                # Otherwise, get all data between timeBreakInd[i] and timeBreakInd[i + 1]
                temporaryData.append(rawData[timeBreakInd[i]:timeBreakInd[i + 1]])

        # Break each entry into seperate dataframes.
        # Add log entries for each well.
        self.Data:list[list[pd.DataFrame]] = []
        columnNames = rawData.columns        

        # map columnName to log index
        self.columnNameLogMap = {}
        for i in range(len(columnNames) - 1):
            self.Data.append([])
            self.logfiles.append(logFolder + "/well" + columnNames[i+1] + "Log.txt")
            self.columnNameLogMap[columnNames[i+1]] = i+1
            for j in range(len(temporaryData)): # going down time frames
                # Create dataframe that is time + well info
                well:pd.DataFrame = pd.concat([temporaryData[j][columnNames[0]],  temporaryData[j][columnNames[i+1]]], axis=1, keys=[columnNames[0], columnNames[i + 1]])
                self.Data[i].append(well)
                
    def __str__(self):
        headers = self.headers

        outputStr = f"Results from {self.filename}\n{"#" * 40}\n" \
        f"Output file: {self.runnerName}\n" \
        f"{len(headers)} Data Headers: {headers}\n" \
        f"Outside Temperature: {self.tempOut}°C\n" \
        f"Inside Temperature: {self.tempIn}°C\n\n" \

        for well in self.Data: # Adds in the end of one interval and the start of the next at each time break
            outputStr += f"Well: {well[0].columns[1]}\n"
            outputStr += f"Time Intervals:\n"
            for timeSpan in well:
                outputStr += f"[{timeSpan.iloc[0,0]},{timeSpan.iloc[-1,0]}]\n"
            outputStr += f"\n"
                
        return outputStr

    # Append to a log file. Default is 0, which is global log
    def appendLog (self, string:str, log:int = 0):
        with open(self.logfiles[log], "a") as f:
            current_time = time.time() # epoch time float
            # current_time = time.asctime() # [Sun Jan 25 18:13:05 2026]

            # Need substeps for the sub logs.
            substep = ""
            if log == 0:
                self.logStep += 1
                self.logSubStep = 1
            else:
                substep = "." + str(self.logSubStep)
                self.logSubStep += 1

            f.write(str(self.logStep) + substep + " [" + str(current_time) + "] " + string + "\n")
    
    # Helper function that prints out some initial logs. Cleans up main init function.
    def __initialLogs(self, runnerName, breakSize, timeUnit, heatCorrect, tempOut = 0, tempIn = 0):
        self.appendLog("Execution log of " + runnerName + ".py", 0)

        startTime = time.time()

        dt_object = datetime.datetime.fromtimestamp(startTime, datetime.timezone.utc)
        self.appendLog(dt_object.strftime('%Y-%m-%d %H:%M:%S'), 0)

        self.appendLog("Initializing results reader", 0)
        
        if (timeUnit != "s" and timeUnit != "m" and timeUnit != "h"):
            raise Exception("Invalid timeUnit provided")
        
        self.appendLog("Time unit: " + str(timeUnit), 0)
        self.appendLog("Time break size: " + str(breakSize), 0)
        self.appendLog("Heat corrections: " + str(heatCorrect), 0)
        if (heatCorrect):
            self.appendLog("Ambiant temperature: " + str(tempOut), 0)
            self.appendLog("Plate Reader temperature: " + str(tempIn), 0)
            self.appendLog("kval: " + str(kVal), 0)
            self.appendLog("Amplitude overshoot: " + str(AmplitudeOvershoot), 0)

    # This method is to clean off the 0's at the end of the data table.
    # -1 on no zero index found
    @staticmethod
    def _getZeroIndex(timeVals:pd.Series)->int:
        # If we go from a positive number to 0, we are in 0 territory. Cut them off
        inPositive:bool = False        
        for i in range (len(timeVals)):
            if timeVals[i] > 0:
                inPositive = True
            if timeVals[i] == 0 and inPositive:
                return i
        # If we made it to the end, return -1 as index.
        return -1

    #This method breaks the time series whenever there is a gap in time greater then breakVal
    @staticmethod
    def __createTimeBreaks(timeVals:pd.Series, breakVal:int)->list[int]:
        if len(timeVals)<=1: # Only one or less data points, thus no breaks.
            return []
        retList:list[int] = []
        for i in range (len(timeVals) - 1):
            if timeVals[i+1] - timeVals[i] > breakVal:
                retList.append(i+1)
        return retList
    
    # Removes all of the wells from raw data that are all 0's
    def __removeAllZeroWells(self, rawData: pd.DataFrame) -> pd.DataFrame:
        
        wellChecker = rawData.any()
        for i in rawData.columns:
            if (not wellChecker[i]):
                self.appendLog(("Removed well " + str(i) + ", Contained only zero/null data"), 0)
                rawData = rawData.drop(columns=[i])
                
        return rawData
        
    # Save the modified data under a certain file name in the folder. Default is "modifiedData"
    def saveData(self, filename:str | None = None):
        if filename is None:
            filename = "./"+self.runnerName + "/modifiedData.csv"
        else :
            filename = "./"+self.runnerName + "/" + filename
        for i in range(len(self.Data)):
            for j in range(len(self.Data[i])):
                self.Data[i][j].to_csv(filename , mode='a', index=False,)


    # t0 - time leaving. t1 - time entering plate reader. t2 - curTime. k - number, flor - curFlor val. 
    # What is the mult by A, some unit of flor per temp. 
    # Amp overshoot, 
    def __calcTempAtMeasurement(self, t0:int, t1:int, k:float, temp0:float, tempAmb:float):
        # First, lets get the temp of well at t1.
        # We assume start temp of substance is plate reader temp.
        # From hot to cold,
        tempCurrent = tempAmb + ((temp0 - tempAmb) * (math.exp(-k * ((t1 - t0)/60))))
        # What is the A value, to convert 
        # tempCurrent * A
        return tempCurrent
    

    # Helper function that, given the timeVals and timeBreak Indices returns temp vals. 
    # TODO: Instant read at time of 0, so it will begin heating to 37.
    def __findTempVals (self, timeVals:pd.Series, timeBreakInd:list[int], tempIn:float, tempOut:float) :
        tempList = []
        curBreakNum = 0

        # Initial temp is out temp.
        tempList.append(tempOut)

        for i in range(1, len(timeVals)): # For each time val
            # Check if at time break, then we're outside break

            if curBreakNum < len(timeBreakInd) and i == timeBreakInd[curBreakNum]:
                # Calculate new temperature
                newTemp = self.__calcTempAtMeasurement(timeVals[i-1], timeVals[i], kVal, tempList[-1], tempOut)
                curBreakNum += 1
            else : # Else, we're inside
                newTemp = self.__calcTempAtMeasurement(timeVals[i-1], timeVals[i], kVal * (3/2), tempList[-1], tempIn)
            tempList.append(newTemp)
        return tempList
    
    
    # Method takes raw data with format time, tempval, well floresence and applies the heat corrections.
    @staticmethod
    def heatCorrectHelper(rawData:pd.DataFrame, ambTemp, inTemp):
        # want to consider a 35% decrease in flor from amb heat to full heat
        
        wellNames =  rawData.columns[2:].to_list()
        tempFromHeated:pd.DataFrame = (rawData["Temps"] - inTemp) * -1
        # This puts it as percentage to room temp [0-1], then mult by amp for amp per row. decrease by 0 at full, .35 at roomtemp
        correction:pd.DataFrame = 1 - (tempFromHeated / (inTemp - ambTemp) *  AmplitudeOvershoot)
        rawData[wellNames] = rawData[wellNames].mul(correction, axis=0)


    # Helper function for adding the time break for a well given a time.
    @staticmethod
    def __TimeBreakWell(time:int, well:list[pd.DataFrame]):
        for i in range(len(well)): # For each time series in the well.            
            timeVals:pd.Series = well[i].iloc[:,0]
            if (time > timeVals.iloc[0] and time <= timeVals.iloc[len(timeVals)-1] ):
                for j in range(len(timeVals)-1):
                    if (time > timeVals.iloc[j] and time <= timeVals.iloc[j + 1]): # Now, we can actually split.
                        data2 = well[i][j+1:]
                        well[i] = well[i][0:j+1]
                        well.insert(i+1, data2)
                        return

    
    # This method adds an arbitrary time break given a time value. Given time is included in next time break if lands on a specified time. 
    def addTimeBreak(self, time:int, columnName = ""):
        # Go well by well
        dataLen:int = len(self.Data)
        for i in range (dataLen):
            if columnName == "" or self.Data[i][0].columns[1] == columnName: 
                self.__TimeBreakWell(time, self.Data[i])
                if columnName != "":
                    self.appendLog("Added time break at " + str(time), self.columnNameLogMap[columnName])
        if columnName == "":
            self.appendLog("Added time break at " + str(time), 0)
            
            
        # Helper to find the time frame in a well and apply function to that time frame.
    def __findDataFrameHelp(self, well:list[pd.DataFrame], time:int, functionArg):
        timeVal2:int
        numBreaks = len(well) 
        for i in range(numBreaks): # For each time break
            if i == numBreaks - 1: # If at last frame, do last val + 1
                timeVal2 = well[i].iloc[len(well[i]) - 1 , 0] + 1
            else : # Else, first val of next break
                timeVal2 = well[i+1].iloc[0,0] 
            timeVals:pd.Series = well[i].iloc[:,0]
            if (time >= timeVals.iloc[0] and time < timeVal2 ):
                # If this is the time frame that has the time, do the function.
                ret = functionArg(well[i])
                if ret is not None:
                    well[i] = ret
                # If the operation leaves dataframe empty, remove it.
                if len(well[i]) == 0:
                    del well[i]
                return
            
    # Applies function to a time frame based on index of time frame.
    def __findDataFrameByIndex(self, well:list[pd.DataFrame], index:int, functionArg):
        # If this is the time frame that has the time, do the function.
        ret = functionArg(well[index])
        if ret is not None:
            well[index] = ret
        # If the operation leaves dataframe empty, remove it.
        if len(well[index]) == 0:
            del well[index]
        return

    
    # This method finds the data frame containing a given time. Executes the function given on that data frame.
    def findDataFrame(self, indOrTime:int, functionArg, columnName = "", useRealTime = False):

        for i in range (len(self.Data)- 1, -1, -1): # For each well, we do it
            # If the well is not the right columnName, don't do it.
            if columnName == "" or self.Data[i][0].columns[1] == columnName: 
                if (useRealTime): #search by time
                    self.__findDataFrameHelp(self.Data[i], indOrTime, functionArg)
                else :
                    self.__findDataFrameByIndex(self.Data[i], indOrTime, functionArg)
                # If function deleted well, delete it from data.
                if len(self.Data[i]) == 0:
                    del self.Data[i]
        
        # Given a dataframe, deletes it.
    def __voidDataFrameFunc(self, dataframe:pd.DataFrame):
        # If this dataframe has a high or low val series set, nullify that value and warn.
        name = dataframe.columns[1]
        if self.lowVals[name] != -1 or self.highVals[name] != -1: 
            Warning("High or low value was calculated before this dataframe was deleted")
        
        # Delete the dataframe, replace with blank
        dataframe = pd.DataFrame()
        return dataframe
        
    # This function can delete a singular specified data frame given an index or time within.
    # For internal use.
    def __voidTimeBreak (self, indOrTime:int, columnName:str = "", useRealTime = False):
        self.findDataFrame(indOrTime, self.__voidDataFrameFunc,columnName, useRealTime)
        
        
    # returns the dataframes that contain the start time, end time, and inbetween.  
    @staticmethod
    def __findDataFramesTime(well:list[pd.DataFrame], startBound:int, endBound:int) -> list[pd.DataFrame]:
        if (endBound < startBound):
            raise Exception("Provided start bound is less than provided end bound.")
        dataFrameList = []
        numBreaks = len(well) 
        for i in range(numBreaks): # For each time break
            t1 = well[i].iloc[0,0]
            t2 = well[i].iloc[-1,0]
            # if both s and e are less than t1 or >= t2, dont include.
            if ( not((startBound < t1 and endBound < t1) or (startBound >= t2 and endBound >= t2))):
                dataFrameList.append(well[i])
        return copy.deepcopy(dataFrameList)
    
    # Find the indicies so direct refs can be found later
    @staticmethod
    def __findDataFramesIndiciesByTime(well:list[pd.DataFrame], startBound:int, endBound:int) -> list[int]:
        if (endBound < startBound):
            raise Exception("Provided start bound is less than provided end bound.")
        dataFrameIndexList = []
        numBreaks = len(well) 
        for i in range(numBreaks): # For each time break
            t1 = well[i].iloc[0,0]
            t2 = well[i].iloc[-1,0]
            # if both s and e are less than t1 or >= t2, dont include.
            if ( not((startBound < t1 and endBound <= t1) or (startBound > t2 and endBound >= t2))):
                dataFrameIndexList.append(i)
        return dataFrameIndexList


    # function that prints statements for user actions, low, high
    def __create_log_user_action_index(self, columnName, startIndex:int, endIndex:int, operationMessage:str):
        self.appendLog(operationMessage + " Index range: " + str(startIndex) + " to " + str(endIndex), self.columnNameLogMap[columnName])

    # function that prints statements for user actions, low, high
    def __create_log_user_action_time(self, columnName, startTime:int, endTime:int, operationMessage:str):
        self.appendLog(operationMessage + " Time range: " + str(startTime) + " to " + str(endTime), self.columnNameLogMap[columnName])        

    # Given a dataframe, this get the x highest percentile value. 
    def __getPercentileValue(self, dataframe:pd.DataFrame, percentile:float) -> tuple[int, float]:
        sortedDf: pd.DataFrame = dataframe.sort_values(by=dataframe.columns[1], ascending=True)
        indexPercentile:int = round((len(sortedDf.iloc[:,1]) - 1) * percentile)
        time = sortedDf.iloc[indexPercentile,0]
        data = sortedDf.iloc[indexPercentile,1]
        return (time, data)
       
    # Gets the numeric values for the start and end bounds if none was provided.
    @staticmethod
    def __getStartEndTime(well:list[pd.DataFrame], startBound:int|None, endBound:int|None) -> tuple[int, int]:
        if startBound is None:
            startBound = well[0].iloc[0,0]
        if endBound is None:
            endBound = well[-1].iloc[-1, 0] + 1
        return (startBound, endBound)

    @staticmethod
    def __getStartEndIndex(well:list[pd.DataFrame], startBound:int|None, endBound:int|None) -> tuple[int, int]:
        if startBound is None:
            startBound = 0
        if endBound is None:
            endBound = len(well)
        if (startBound == endBound):
            raise ValueError("Start bound and end bound are both " + str(startBound) + ", meaning slice will have no data.")
        return (startBound, endBound)

    # Return a dataframe that trims any data from time before start time
    @staticmethod
    def __trimStart(data:pd.DataFrame, startTime)-> pd.DataFrame:
        return data[data["Time"] >= startTime]
    
    # Return a dataframe that trims any data from time after end time
    @staticmethod
    def __trimEnd(data:pd.DataFrame, endTime)-> pd.DataFrame:
        return data[data["Time"] < endTime]
        
    def __getListOfWellsByNames(self, columnName):
        listOfWells = []
        if isinstance(columnName, str): # List of one well.
            listOfWells.append(self.getWellFrame(columnName))
        elif isinstance(columnName, list): # List of some wells
            if not all(isinstance(wellName, str) for wellName in columnName):
                raise Exception("Non string provided to list of well names.")
            for name in columnName:
                listOfWells.append(self.getWellFrame(name))
        else: # List of all wells
            listOfWells = self.Data
            
        return listOfWells
    
        # Get the xth percentile value and time from a well / well list / for all wells by time intervals.
    def __getPercentileValuesByTimeInterval(self, startBound: Optional[int] = None, endBound: Optional[int] = None, columnName: Optional[str | list[str]] = None, percentile: float = 50):
        fractionalPercentile = percentile / 100
        # Get all dataframes that satisfy these column names
        listOfWells = self.__getListOfWellsByNames(columnName)
             
        listOfValues = []
        # Find all the dataframes that fall into these start and end bounds.
        for well in listOfWells:
            (newStartBound, newEndBound) = self.__getStartEndTime(well, startBound, endBound)
            curWellName = well[0].columns[1]
            curWellDataframes:list[pd.DataFrame] = self.__findDataFramesTime(well, newStartBound, newEndBound)
            
            # If no dataframes satisfy, don't do the operation
            if len(curWellDataframes) == 0:
                raise Exception("No data found that fits in time range " + str(newStartBound) + " - " + str(newEndBound))
            # Trim the start and end time spans to allign with bounds
            
            curWellDataframes[0] = self.__trimStart(curWellDataframes[0], newStartBound)
            curWellDataframes[-1] = self.__trimEnd(curWellDataframes[-1], newEndBound)
            
            # Find lowest val in the well and set
            (timeVal, dataVal) = self.__getPercentileValue(pd.concat(curWellDataframes), fractionalPercentile)
            listOfValues.append((curWellName,timeVal, dataVal, newStartBound, newEndBound))
            
        return listOfValues
            
    def __voidDataByTimeInterval(self, startBound: Optional[int] = None, endBound: Optional[int] = None, columnName: Optional[str | list[str]] = None):
        # Get all dataframes that satisfy these column names
        listOfWells = self.__getListOfWellsByNames(columnName)
             
        listOfValues = []
        # Find all the dataframes that fall into these start and end bounds.
        for well in range(len(listOfWells)):
            
            (newStartBound, newEndBound) = self.__getStartEndTime(listOfWells[well], startBound, endBound)
            curWellName = listOfWells[well][0].columns[1]
            curWellDataframesIndicies:list[int] = self.__findDataFramesIndiciesByTime(listOfWells[well], newStartBound, newEndBound)
            
            # If no dataframes satisfy, don't do the operation
            if len(curWellDataframesIndicies) == 0:
                raise Exception("No data found that fits in time range " + str(newStartBound) + " - " + str(newEndBound))

            # Trim data off past start and before end time spans.  
            if len(curWellDataframesIndicies) == 1: # special case, cutting out the middle of a single time span
                # Just cut out the middle section
                self.addTimeBreak(newStartBound, curWellName)
                self.addTimeBreak(newEndBound, curWellName)
                self.__voidTimeBreak(newStartBound, curWellName, True)
                listOfValues.append((curWellName, newStartBound, newEndBound))
                
            else :
                
                listOfWells[well][curWellDataframesIndicies[0]] = listOfWells[well][curWellDataframesIndicies[0]][listOfWells[well][curWellDataframesIndicies[0]]["Time"] < newStartBound]
                listOfWells[well][curWellDataframesIndicies[-1]] = listOfWells[well][curWellDataframesIndicies[-1]][listOfWells[well][curWellDataframesIndicies[-1]]["Time"] >= newEndBound]
            
                # Delete all those inbetween wells as well
                for i in range(len(curWellDataframesIndicies) - 1 , -1, -1):
                    if i == len(curWellDataframesIndicies) - 1 or i == 0:
                        if (len(listOfWells[well][curWellDataframesIndicies[i]]) == 0):
                            del listOfWells[well][curWellDataframesIndicies[i]]
                    else :
                        name = listOfWells[well][curWellDataframesIndicies[i]].columns[1]
                        if self.lowVals[name] != -1 or self.highVals[name] != -1: 
                            Warning("High or low value for this well was calculated before this dataframe was deleted")
                        listOfWells[well][curWellDataframesIndicies[i]] = pd.DataFrame()
                        del listOfWells[well][curWellDataframesIndicies[i]]
                        listOfValues.append((curWellName, newStartBound, newEndBound))
        return listOfValues
    
    def showDataSeriesByTime(self, startBound: Optional[int] = None, endBound: Optional[int] = None, columnName: Optional[str | list[str]] = None):
        # Get all dataframes that satisfy these column names
        listOfWells = self.__getListOfWellsByNames(columnName)
        
        propCycle = plt.rcParams['axes.prop_cycle']
        colors = propCycle.by_key()['color']
        minStartBound  = None

        # Bools which indicate the types of plots we will need.
        hasHighLow:bool = False
        noHighLow:bool = False
        for i in range(len(listOfWells)):
            if self.highVals[listOfWells[i][0].columns[1]] == -1 or self.lowVals[listOfWells[i][0].columns[1]] == -1:
                noHighLow = True                
            else:
                hasHighLow = True

        twoPlots = False
        if (hasHighLow and noHighLow):
            twoPlots = True
             
        # Find all the dataframes that fall into these start and end bounds.
        for i in range(len(listOfWells)):
            (newStartBound, newEndBound) = self.__getStartEndTime(listOfWells[i], startBound, endBound)
            if minStartBound is None:
                minStartBound = newStartBound
            elif minStartBound > newStartBound:
                minStartBound = newStartBound
                
            curWellName = listOfWells[i][0].columns[1]
            curWellDataframes:list[pd.DataFrame] = self.__findDataFramesTime(listOfWells[i], newStartBound, newEndBound)
            
            # If no dataframes satisfy, don't do the operation
            if len(curWellDataframes) == 0:
                raise Exception("No data found that fits in time range " + str(newStartBound) + " - " + str(newEndBound))
            # Trim the start and end time spans to allign with bounds
            curWellDataframes[0] = self.__trimStart(curWellDataframes[0], newStartBound)
            curWellDataframes[-1] = self.__trimEnd(curWellDataframes[-1], newEndBound)            
            
            # Determine plot type.
            normed = True
            if self.lowVals[curWellDataframes[0].columns[1]] == -1 or self.highVals[curWellDataframes[0].columns[1]] == -1:
                normed = False
            
            # If we have two plots, select the subplot and norm data as needed.
            if twoPlots:
                if not normed:
                    plt.subplot(2, 1, 1)
                else:
                    curWellDataframes = self.__normData(self.lowVals[curWellName], self.highVals[curWellName], curWellDataframes, False)
                    plt.subplot(2, 1, 2)
                    
            # If not two plots, still check for norm.
            elif normed:
                curWellDataframes = self.__normData(self.lowVals[curWellName], self.highVals[curWellName], curWellDataframes, False)

            for j in range (len(curWellDataframes)):
                if j == 0:
                    plt.plot(curWellDataframes[j]["Time"], curWellDataframes[j][curWellName], color = colors[i % (len(colors))], label= curWellName)
                else:
                    plt.plot(curWellDataframes[j]["Time"], curWellDataframes[j][curWellName], color = colors[i % (len(colors))])

            # If two plots, need to set both labels and such.
            if twoPlots:
                plt.subplot(2, 1, 1)
                plt.xlim(left=minStartBound)
                plt.grid(True)
                plt.legend()
                plt.xlabel("Time ( " + str(self.timeUnit) + " )")
                plt.ylabel('Fluorescence')
                plt.title("Fluorescence over time")

                plt.subplot(2, 1, 2)            
                plt.xlim(left=minStartBound)
                plt.grid(True)
                plt.legend()
                plt.xlabel("Time ( " + str(self.timeUnit) + " )")
                plt.ylabel('Fluorescence')
                plt.subplots_adjust(hspace=0.5, wspace=0.4)
                plt.title("Normalized Fluorescence over time")

            # Otherwise, just the one.
            else :
                if not normed:
                    plt.xlim(left=minStartBound)
                    plt.grid(True)
                    plt.legend()
                    plt.xlabel("Time ( " + str(self.timeUnit) + " )")
                    plt.ylabel('Fluorescence')
                    plt.title("Fluorescence over time")
                    
                else:
                    plt.xlim(left=minStartBound)
                    plt.grid(True)
                    plt.legend()
                    plt.xlabel("Time ( " + str(self.timeUnit) + " )")
                    plt.ylabel('Fluorescence')
                    plt.title("Normalized Fluorescence over time")
                
        plt.show()
                
        return
    
    
    def setHighValuesByTimeInterval(self, startBound: Optional[int] = None, endBound: Optional[int] = None, columnName: Optional[str | list[str]] = None, percentile: float = 50):
        valsList = self.__getPercentileValuesByTimeInterval(startBound, endBound, columnName, percentile)
        for well in valsList:
            self.highVals[well[0]] = well[2]
            self.__create_log_user_action_time(well[0], well[3], well[4], ("Found high value " + str(well[2]) + " at time " + str(well[1]) + ", percentile " + str(percentile) + "% ."))
    
    def setLowValuesByTimeInterval(self, startBound: Optional[int] = None, endBound: Optional[int] = None, columnName: Optional[str | list[str]] = None, percentile: float = 50):
        valsList = self.__getPercentileValuesByTimeInterval(startBound, endBound, columnName, percentile)
        for well in valsList:
            self.lowVals[well[0]] = well[2]
            self.__create_log_user_action_time(well[0], well[3], well[4], ("Found low value " + str(well[2]) + " at time " + str(well[1]) + ", percentile " + str(percentile) + "% ."))
            
    # Voids data with timespan provided. Will create a time break in the data 
    def voidTimeSpansByTimeInterval(self, startBound: Optional[int] = None, endBound: Optional[int] = None, columnName: Optional[str | list[str]] = None):
        valsList = self.__voidDataByTimeInterval(startBound, endBound, columnName)
        for well in valsList:
            self.lowVals[well[0]] = well[2]
            self.__create_log_user_action_time(well[0], well[1], well[2], ("Voided data."))            


    # Get the percentile value tuple for a well based on its start and end indicies
    def __getPercentileValuesByBreakInterval(self, startBound: Optional[int] = None, endBound: Optional[int] = None, columnName: Optional[str | list[str]] = None, percentile: float = 50):
        fractionalPercentile = percentile / 100
        # Get all dataframes that satisfy these column names
        listOfWells = self.__getListOfWellsByNames(columnName)
                     
        listOfValues = []
        # Find all the dataframes that fall into these start and end indicies.
        for well in listOfWells:
            curWellName = well[0].columns[1]
            (newStartBound, newEndBound) = self.__getStartEndIndex(well, startBound, endBound)
            
            curWellDataframes:list[pd.DataFrame] = well[newStartBound:newEndBound]
            
            # If no dataframes satisfy, don't do the operation
            if len(curWellDataframes) == 0:
                raise Exception("No data found that fits in time range " + str(newStartBound) + " - " + str(newEndBound))
            
            # Find lowest val in the well and set
            (timeVal, dataVal) = self.__getPercentileValue(pd.concat(curWellDataframes), fractionalPercentile)
            listOfValues.append((curWellName,timeVal, dataVal, newStartBound, newEndBound))
            
        return listOfValues
    
    def __voidDataByIndex(self, startBound: Optional[int] = None, endBound: Optional[int] = None, columnName: Optional[str | list[str]] = None):
        # Get all dataframes that satisfy these column names
        listOfWells = self.__getListOfWellsByNames(columnName)

        listOfValues = []
        # Void all the dataframes that fall into these start and end indicies.
        for well in listOfWells:
            curWellName = well[0].columns[1]

            (newStartBound, newEndBound) = self.__getStartEndIndex(well, startBound, endBound)
            
            for i in range (newEndBound - 1, newStartBound - 1, -1):
                self.__voidTimeBreak(i , curWellName, False)
            listOfValues.append((curWellName, newStartBound, newEndBound))
        
        return listOfValues
    
    def showDataSeriesByIndex(self, startBound: Optional[int] = None, endBound: Optional[int] = None, columnName: Optional[str | list[str]] = None):
        # Get all dataframes that satisfy these column names
        listOfWells = self.__getListOfWellsByNames(columnName)
        
        propCycle = plt.rcParams['axes.prop_cycle']
        colors = propCycle.by_key()['color']
        minStartBound = None
        
                
        # Bools which indicate the types of plots we will need.
        hasHighLow:bool = False
        noHighLow:bool = False
        for i in range(len(listOfWells)):
            if self.highVals[listOfWells[i][0].columns[1]] == -1 or self.lowVals[listOfWells[i][0].columns[1]] == -1:
                hasHighLow = True
            else:
                noHighLow = True

        twoPlots = False
        if (hasHighLow and noHighLow):
            twoPlots = True

             
        for i in range(len(listOfWells)):
            (newStartBound, newEndBound) = self.__getStartEndIndex(listOfWells[i], startBound, endBound)
            curWellName = listOfWells[i][0].columns[1]
            
            if minStartBound is None:
                minStartBound = listOfWells[i][newStartBound].iloc[0,1]
            elif minStartBound > listOfWells[i][newStartBound].iloc[0,1]:
                minStartBound = listOfWells[i][newStartBound].iloc[0,1]
                
            curWellDataframes = listOfWells[i][newStartBound:newEndBound]
                
            # Determine plot type.
            normed = True
            if self.lowVals[curWellDataframes[0].columns[1]] == -1 or self.highVals[curWellDataframes[0].columns[1]] == -1:
                normed = False
            
            # If we have two plots, select the subplot and norm data as needed.
            if twoPlots:
                if not normed:
                    plt.subplot(2, 1, 1)
                else:
                    curWellDataframes = self.__normData(self.lowVals[curWellName], self.highVals[curWellName], curWellDataframes, False)
                    plt.subplot(2, 1, 2)            
            
            for j in range (newStartBound, newEndBound):
                if j == 0:
                    plt.plot(curWellDataframes[j]["Time"], curWellDataframes[j][curWellName], color = colors[i % (len(colors))], label= curWellName)
                else:
                    plt.plot(curWellDataframes[j]["Time"], curWellDataframes[j][curWellName], color = colors[i % (len(colors))])

        # If two plots, need to set both labels and such.
        if twoPlots:
            plt.subplot(2, 1, 1)
            plt.xlim(left=minStartBound)
            plt.ylim(bottom=0)
            plt.grid(True)
            plt.legend()
            plt.xlabel("Time ( " + str(self.timeUnit) + " )")
            plt.ylabel('Fluorescence')
            plt.title("Fluorescence over time")

            plt.subplot(2, 1, 2)            
            plt.xlim(left=minStartBound)
            plt.ylim(bottom=0)
            plt.grid(True)
            plt.legend()
            plt.xlabel("Time ( " + str(self.timeUnit) + " )")
            plt.ylabel('Fluorescence')
            plt.subplots_adjust(hspace=0.5, wspace=0.4)
            plt.title("Normalized Fluorescence over time")

        # Otherwise, just the one.
        else :
            if not normed:
                plt.xlim(left=minStartBound)
                plt.ylim(bottom=0)
                plt.grid(True)
                plt.legend()
                plt.xlabel("Time ( " + str(self.timeUnit) + " )")
                plt.ylabel('Fluorescence')
                plt.title("Fluorescence over time")
                
            else:
                plt.xlim(left=minStartBound)
                plt.ylim(bottom=0)
                plt.grid(True)
                plt.legend()
                plt.xlabel("Time ( " + str(self.timeUnit) + " )")
                plt.ylabel('Fluorescence')
                plt.title("Normalized Fluorescence over time")
                
        plt.show()
        return

    def setHighValuesByBreakInterval(self, startBound: Optional[int] = None, endBound: Optional[int] = None, columnName: Optional[str | list[str]] = None, percentile: float = 50):
        valsList = self.__getPercentileValuesByBreakInterval(startBound, endBound, columnName, percentile)
        for well in valsList:
            self.highVals[well[0]] = well[2]
            self.__create_log_user_action_index(well[0], well[3], well[4], ("Found low value " + str(well[2]) + " at time " + str(well[1]) + ", percentile " + str(percentile) + "% ."))

    def setLowValuesByBreakInterval(self, startBound: Optional[int] = None, endBound: Optional[int] = None, columnName: Optional[str | list[str]] = None, percentile: float = 50):
        valsList = self.__getPercentileValuesByBreakInterval(startBound, endBound, columnName, percentile)
        for well in valsList:
            self.lowVals[well[0]] = well[2]
            self.__create_log_user_action_index(well[0], well[3], well[4], ("Found low value " + str(well[2]) + " at time " + str(well[1]) + ", percentile " + str(percentile) + "% ."))
    
    def voidTimeSpansByIndex(self, startBound: Optional[int] = None, endBound: Optional[int] = None, columnName: Optional[str | list[str]] = None):
        valsList = self.__voidDataByIndex(startBound, endBound, columnName)
        for well in valsList:
            self.__create_log_user_action_index(well[0], well[1], well[2], ("Voided data."))
        
    # Uses the other well to set the high values of the wells with names provided. 
    def setHighUsingDifferentWell(self, settingWell:str, columnName: Optional[str | list[str]] = None):
        # Get the high val of the well.
        if self.highVals[settingWell] == -1:
            raise Exception("High value for well: " + str(settingWell) + "is not set.")
        highVal = self.highVals[settingWell]
        listOfWellsNames:list[str] = []
        if isinstance(columnName, str): # List of one well.
            listOfWellsNames.append(columnName)
        elif isinstance(columnName, list): # List of some wells
            listOfWellsNames = columnName
        else: # List of all wells
            for i in self.Data:
                listOfWellsNames.append(i[0].columns[1])
                        
        for name in listOfWellsNames:
            self.highVals[name] = highVal
            self.appendLog(("Used low val from well " + settingWell + "to set value of well" + name), self.columnNameLogMap[name])        

        
    # Remove all wells named columnName. Also, remove high low val entry from dict.
    def removeWell(self, columnName:str):
        for i in range(len(self.Data)-1, -1, -1):
            if self.Data[i][0].columns[1] == columnName:
                del self.highVals[columnName]
                del self.lowVals[columnName]
                
                self.appendLog(("Removed well: " + columnName))
                self.Data.pop(i)
    
    # Get the well dataframes based on index or string name.
    # When normed data is true, norms data for which high and low vals are set.
    # Otherwise, just returns the normal plate reader concentation values.
    def getWellFrame (self, well:str, normedDataBool = True, fluConcInverse = False) -> list[pd.DataFrame]:
        
        targetWellIndex:int|None = None
        for i in range(len(self.Data)):
            for j in self.Data[i]:
                if ((j.columns)[1] == well):
                    targetWellIndex = i
                    
        if targetWellIndex is None:
            raise Exception("Did not enter valid well name")
        
        
        rawFrame = self.Data[targetWellIndex]
        if normedDataBool and (self.lowVals[well] != -1) and (self.highVals[well] != -1):
            # Return a normalized copy while preserving internally stored raw plate-reader values.
            return self.__normData(self.lowVals[well], self.highVals[well], rawFrame, fluConcInverse)
        if normedDataBool:
            warnings.warn("Tried to get normed data without having high and low values set.", UserWarning)

        # Return deep copies to prevent caller-side mutation of internal state.
        return [i.copy(deep=True) for i in rawFrame]
    
    # Get the full list of times associated with a particular well.
    def getWellTimes (self, well:int|str) -> list[pd.Series]:
        data = self.getWellFrame(well, False)
        # Now we have the list of lists of frames. Drop the data.
        timeList = [] 
        for i in range(len(data)):
            timeList.append(data[i]["Time"])
        return timeList
    
    # Same as get well times, but now it is as a jax array.
    def getWellTimesAsJaxList(self, well:int|str) -> list[jnp.ndarray] :
        data:list[pd.Series] = self.getWellTimes(well)
        return self.__wellToJaxList(data)
    
    
    # Get the list of only data from the dataframes.
    def getWellData(self, well:int|str, normedData = True, fluConcInverse = False) -> list[pd.Series]:
        data:list[pd.DataFrame] = self.getWellFrame(well, normedData, fluConcInverse)
        # Now we have the list of lists of frames. Drop the data.
        dataList = []
        name = data[0].columns[1]
        for i in range(len(data)):
            dataList.append(data[i][name])
        return dataList
    
    # Same as get well data, but now it is as a jax array.
    def getWellDataAsJax(self, well:int|str, normedData = True, fluConcInverse = False) -> jnp.ndarray :
        data:list[pd.Series] = self.getWellData(well, normedData, fluConcInverse)
        return self.__wellToJax(data)    
    
    # Converts a list of series into a unified jax array.
    @staticmethod
    def __wellToJax(well:list[pd.Series]) -> jnp.ndarray:
        tempList:list[list[float]] = []
        for series in well:
            tempList += series.to_list()
        return jnp.array(tempList)

    # Converts a list of series into a unified jax array.
    @staticmethod
    def __wellToJaxList(well:list[pd.Series]) -> list[jnp.ndarray]:
        tempList:list[list[float]] = []
        for series in well:
            tempVar = series.to_list()
            tempList.append(jnp.array(tempVar))
        return tempList

    
    # Adding signal helper method, converts fluorescence to fractional datapoints (concentrations if linear relationship assumed). 
    @staticmethod
    def __normData (lowVal:float, highVal:float, frame:list[pd.DataFrame], fluConcInverse:bool) -> list[pd.DataFrame]:
        newFrame = [i.copy(deep=True) for i in frame]
        
        wellName = frame[0].columns[1]
        valDif = highVal - lowVal
        for i in range(len(frame)):
            if not fluConcInverse :
                newFrame[i][wellName] = (frame[i][wellName] - lowVal) / valDif 
            else :
                newFrame[i][wellName] = ((frame[i][wellName] - highVal) / valDif) * -1
        return newFrame
            

##########################################################
#---------------------RUNNER SCRIPT----------------------# 
##########################################################

if __name__ == "__main__":
    filename = "./test.txt"

    
    # Tmperature, in degrees celsius.
    
    plateReaderTemp = 37
    roomtemp = 20
    timeUnit = "m" # "s", "m", "h"
    RR = ResultsReader("./test.txt", breakSize=120, tempIn=plateReaderTemp, tempOut = roomtemp, heatCorrect = True, timeUnit = "s")

    print(RR)
    
    RR.showDataSeriesByTime(startBound=20000, endBound=40000, columnName=["K8", "K9"])
    RR.showDataSeriesByIndex(startBound=0, endBound=3)
    RR.setLowValuesByTimeInterval(columnName="K8")
    RR.setHighValuesByTimeInterval(columnName="K8")
    RR.showDataSeriesByIndex()
    RR.showDataSeriesByTime()


    RR.setLowValuesByTimeInterval(startBound=0, percentile=0, columnName="K8")
    RR.setLowValuesByTimeInterval(startBound=0, percentile=0, columnName="K9")
    RR.setHighValuesByTimeInterval(startBound=0, percentile=100, columnName="K8")
    RR.setHighValuesByTimeInterval(startBound=0, percentile=100, columnName= "K9")
    RR.setHighValuesByBreakInterval(startBound=1, endBound=2, percentile=100)    
    RR.setLowValuesByBreakInterval(percentile=0)    
    RR.setLowValuesByTimeInterval(startBound=39, percentile=0, columnName=["K8", "K9"])
    RR.setHighValuesByTimeInterval(startBound=39, percentile=100, columnName=["K8", "K9"])
    
    RR.voidTimeSpansByTimeInterval(startBound=360, endBound=480 ,columnName=["K8", "K9"])
    RR.voidTimeSpansByTimeInterval(startBound=240, endBound=2012, columnName="K10")
    RR.voidTimeSpansByTimeInterval(startBound=3152)
    RR.voidTimeSpansByTimeInterval(endBound=240, columnName="K8")
    RR.voidTimeSpansByTimeInterval(startBound= 300, endBound=301, columnName="K8")
    RR.voidTimeSpansByTimeInterval(startBound= 240, endBound=241, columnName="K8")

    RR.voidTimeSpansByIndex(columnName="K8")    
    RR.voidTimeSpansByIndex(startBound=1, endBound=2, columnName=["K9", "K10"])
    
    
    RR.removeWell("K9")
    
#    RR.showDataSeriesByTime()
#    RR.showDataSeriesByIndex()

    RR.saveData("modifiedData.csv")
    