
import ResultsReader as RR

# The name of the table/CSV file containing the plate data.
filename = "./testingCSV2.txt"

# Time break size is the number of seconds between data point reads before an automatic time break is created.
# These automatic time breaks are meant to represent times whent the plate reader has been taken out, and factors in to heat correction.
# Since our machine records data every 60 seconds, we can set our time break size to be anything above that.
timeBreakSize = 120

# Temperature, in degrees celsius. Only needs to be set if using heatCorrections.
plateReaderTemp = 37
roomtemp = 20
timeUnit = "s" # "s", "m", "h"


# We are going to be using a table, as is the default output of a cytation 5 machine, so we set csv to false.
csv = False
resultsObj:RR.ResultsReader = RR.ResultsReader(filename, "OutputFolder", timeBreakSize, plateReaderTemp, roomtemp, heatCorrect = True, timeUnit = timeUnit, csv = csv)

# This will show the 
resultsObj.showDataSeriesByTime() 

resultsObj.removeWell("K9")
resultsObj.addTimeBreak(10000)

woi = "L11"
resultsObj.addTimeBreak(30000, columnName=woi)
resultsObj.voidTimeSpansByIndex(startBound = 3, endBound=4, columnName=woi)
resultsObj.showDataSeriesByTime() 
resultsObj.voidTimeSpansByTimeInterval(endBound=180) 

resultsObj.setLowValuesByBreakInterval(endBound = 1, percentile=10)
resultsObj.setHighValuesByTimeInterval(startBound = 90 * 60, percentile=90)
resultsObj.setHighUsingDifferentWell(settingWell="L11", columnName=["K8", "K10"]) # 90 minutes to the end of the time series, somewhere around 110 min

resultsObj.showDataSeriesByTime() 

test = resultsObj.getWellData(woi)

resultsObj.saveData("./testOutput.csv")