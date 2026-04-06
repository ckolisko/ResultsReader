
import ResultsReader as RR

# Getting file name without extention
filename = "./testingCSV2.txt"

# Temperature, in degrees celsius.

plateReaderTemp = 37
roomtemp = 20
timeUnit = "s" # "s", "m", "h"

resultsObj:RR.ResultsReader = RR.ResultsReader(filename, "OutputFolder", 120, plateReaderTemp, roomtemp, heatCorrect = True, timeUnit = timeUnit)

resultsObj.showDataSeriesByTime() 
resultsObj.removeWell("K9")
resultsObj.addTimeBreak(10000)

woi = "L11"
resultsObj.addTimeBreak(30000, columnName=woi)
resultsObj.voidTimeSpansByIndex(startBound = 3, endBound=4, columnName=woi)
resultsObj.showDataSeriesByTime() 

resultsObj.setLowValuesByBreakInterval(endBound = 1, percentile=10)
resultsObj.setHighValuesByTimeInterval(startBound = 90 * 60, percentile=90)
resultsObj.setHighUsingDifferentWell(settingWell="L11", columnName=["K8", "K10"]) # 90 minutes to the end of the time series, somewhere around 110 min

resultsObj.showDataSeriesByTime() 

test = resultsObj.getWellData(woi)

resultsObj.saveData("./testOutput.csv")