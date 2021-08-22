import numpy as np
from typing import List, Callable, Union
import ctypes
import os
import subprocess
import re

def tempDataFn():
    x1 = [x for x in range(5, 10)]
    x2 = [x for x in range(20, 40, 4)]
    y1 = [3 * x1 + x2 + 2 for (x1, x2) in zip(x1, x2)]
    y2 = [2 * x1 + 2 * x2 + 17 for (x1, x2) in zip(x1, x2)]
    # return list(zip(x1, x2)), list(zip(y1, y2))
    return [[5, 20], [6, 24], [7, 28], [8, 32], [9, 36]], [27, 32, 37, 42, 47]


def forceErrorAsInputDataFn():
    return [["not a number", "still not"]], 4

# Helper function to run a command in the command line
def runInCmdLine(cmd: str):
    return subprocess.run(cmd.split())

eqAndVarSizes = {}
offsetCount = {}
def batchIterator(inputDataFn: Callable, einkornFileName: str, inputList: List[str], loss: str,
                  # Batch specifications
                  stepSize: float = 0.01, numEpochs: int = 1000, minGradSquaredSum: float = 0,
                  minLossSquared: float = 0,
                  # Optimization algorithms
                  momentumProportion: float = 0, usesNAG: bool = True,
                  learningSchedule: Callable[[float, int], float] = lambda x,y: x,
                  # Train/test printed feedback
                  lossPrintRate: int = 100, testDataFn: Callable = lambda: [], testPrintRate: int = 0,
                  epochTestBatches: int = 1, finalTestBatches: int = 1,
                  # File IO (predominantly with saving/loading weights)
                  outputFileName: str = None, prevWeightFileName: str = None):
    """
    batchIterator is a Python function that implements basic gradient descent and other optimization algorithms onto a
    tensor computation specified in Einkorn. It has 4 required inputs, and the rest of the inputs can be grouped into
    categories: batch specifications, optimization algorithms, train/test printed feedback, and file IO.

    :param inputDataFn: A function with no inputs whose returned value(s) match inputList in number and modes.
        This is the method for inputting data (both the inputs and the true outputs), and must return object(s) of
            the following types: A single number, an n-dimensional list of numbers, or
                an n-dimensional NumPy array of numbers
        This function must return object(s) as specified by the inputList, matching in both order and dimensions.
    :param einkornFileName: A string that is the filepath to an Einkorn file
        NOTE: While undeclared modes are allowed in Einkorn ("x :: Mode ..."), they are not allowed in this function.
    :param inputList: A list of strings of the form "<input>[<mode 1>,...,<mode n>]".
        - Must match exactly with the returned value(s) from inputDataFn.
        - Example: ["X[n,p]", "y[p]", "z[]", "a[p,n]"]
    :param loss: A string that is the name of a tensor specification in the Einkorn file.
        The underlying calculation of loss (the input string) as defined in the Einkorn file will be the loss function
            that the gradients are found with respect to.
    :param stepSize: A float with a default value of 0.01.
         Sometimes referred to as alpha in mathematics, stepSize is the
    :param numEpochs: An int with a default value of 1000.
        numEpochs is the maximum number of iterations possible (see minGradSquaredSum and minLossSquared for exceptions).
        NOTE: The epoch counter is not 0th-indexed, but is actually 1st-indexed.
    :param minGradSquaredSum: A float whose default value is 0.
        If the sum of the gradients squared is less than minGradSquaredSum, batchIterator will break out early.
    :param minLossSquared: A float whose default value is 0.
        If the loss squared is less than minLoss, batchIterator will break out of its iterations early.
    :param momentumProportion: A float whose default value is 0.
        A coefficient that is the impact that any previous state will have on the next iteration.
        ______________________________________________
    :param usesNAG: A boolean whose default value is True.
        If usesNAG is true, then Nesterov Accelerated Gradient is used.______________________________________________
    :param learningSchedule: A function that takes in a float and an int and returns a float; default is returning the first argument.
        The function will be used to update stepSize as iterations proceed, and is used in the form:
            stepSize = learningSchedule(stepSize, epoch)
        - NOTE: If learningSchedule is not its default, the stepSize that is first applied to gradients will already
            have been altered by the learningSchedule before it is actually used, as the learningSchedule update
            occurs before gradients are calculated and used.
    :param lossPrintRate: An int whose default is 100.
        Every lossPrintRate times, the loss from the current training batch will be printed.
    :param testDataFn: A function with no inputs whose returned values match inputList in number and modes, and
        whose default return is an empty list.
        Functions similar to inputDataFn, but it can instead be used to give a separate test dataset to be used
            during/after training.
    :param testPrintRate: An int whose default is 0.
        Every testPrintRate times, the loss from epochTestBatches test batches will be printed.
    :param epochTestBatches: An int whose default is 1.
        Specifies the number of test batches used for every test done during training.
    :param finalTestBatches: An int whose default is 1.
        Specifies the number of test batches used for the final test done after training.
    :param outputFileName: A string that is a filepath to where the weights will be written to.
        ______________________________________________
        They will be written in the same form as the returned dictionary, but the values will be in list form as
            opposed to NumPy arrays.
    :param prevWeightFileName: A string that is a filepath to where the weights will be loaded from.
        ______________________________________________
    :return: A dictionary containing the training weight names and their respective values after iterations are done.
    """
    # Before running batchIterator, I will assume that the user has already run cabal build and compile.sh?
    # At some point, can set it up so that, at minimum, they must have set up the filepaths in compile.sh (and exportInC.c?)

    # Use assertions to ensure valid values for basic parameters:
    assert epochTestBatches >= 0, 'epochTestBatches cannot be negative'
    assert finalTestBatches >= 0, 'finalTestBatches cannot be negative'

    # File names to be written to; if the user has a file of these names, they will be written over.
    kernelFileCC = "exportedKernels.cpp"
    kernelFileO = kernelFileCC[:-3] + "o"
    kernelFileSO = kernelFileO[:-1] + "so"

    # Convert inputList to a string that will be broken up by Haskell (C can't handle Python/Haskell lists)
    justInputNamesList = []
    modesOfInputs = {}

    splitInputList = re.split("\[|\]", "".join(inputList))
    for i in range(0, len(splitInputList)-1, 2):
        justInputNamesList.append(splitInputList[i])
        modesOfInputs[splitInputList[i]] = splitInputList[i+1].split(",")
    inputStrList = ":".join(justInputNamesList)

    # Uses a shell script and the four required inputs to generate a file that contains the kernel functions
    if os.path.exists(kernelFileCC):
        os.remove(kernelFileCC)
    runInCmdLine("bash runWArgs.sh "+inputStrList+" "+loss+" "+einkornFileName+" "+kernelFileCC)

    # Attempt to read in the file; if that fails, then some error happened with runWArgs.sh
    try:
        f = open(kernelFileCC, "r")
        descString = f.readline()
        if descString == '':
            raise FileNotFoundError
        f.close()
    except FileNotFoundError as err:
        print("FileNotFoundError: Auto-generated kernel file could not be found\n"
              "\t   Potential causes: invalid Einkorn syntax or incorrect filepaths in shell script")
        raise err
    except OSError as err:
        print("OS error on file parsing: {0}".format(err))
        raise

    # Compile the C++ output code that is auto-generated from Einkorn's export.hs into a shared library,
    # then load the shared library
    # NOTE: Edit this if the kernels need to be loaded in a different way?
    def loadKernels():
        runInCmdLine("g++ -c " + kernelFileCC + " -o " + kernelFileO)
        runInCmdLine("g++ -shared -o " + kernelFileSO + " " + kernelFileO)
        return ctypes.CDLL(kernelFileSO)
    kernels = loadKernels()

    # Read in a string that describes the DAG, tensor types (both of Float/Int and input/weight) and dimensions
    print("FOR DEBUGGING ONLY - descString:", descString)
    (modeDesc, sizeDesc, childDesc, typeDesc, weightToGradientDesc, offsetDesc) = re.split("\|", descString)

    # Create a modeSize dictionary that tracks all of the sizes of the modes (to be used in creation of variables later)
    # Set the default value to a size of 1 in case an element is not given any modes (which means it is ~ 1-dimensional)
    modeSize = {'': 1}
    splitModeDesc = re.split(",|:", modeDesc[3:])
    for i in range(0, len(splitModeDesc), 2):
        modeSize[splitModeDesc[i][1:-1]] = int(splitModeDesc[i + 1])

    # Create an eqAndVarSizes dict to store mode names and the data type
    # TBD: Actually check that everything is the correct data type?
    splitSizeDesc = re.split(":|\{|\}", sizeDesc)
    for i in range(0, len(splitSizeDesc) - 1, 3):
        eqAndVarSizes[splitSizeDesc[i]] = {"type": splitSizeDesc[i + 1], "modes": splitSizeDesc[i + 2].split(",")}

    # Create a childDepDict dict that maps an equation to all of its dependencies (inputs and weights are not included)
    splitChildDeps = re.split("\[|\]", childDesc[:-1])
    childDepDict = {}
    for i in range(0, len(splitChildDeps), 2):
        childDepDict[splitChildDeps[i]] = splitChildDeps[i + 1].split(",")

    # Create a typeDict that maps 1 of the 4 categories: inputs, weights, gradients, outputs; onto each value calculated
    splitType = re.split("\{|\}", typeDesc[:-1])
    typeDict = {}
    for i in range(0, len(splitType), 2):
        typeDict[splitType[i]] = splitType[i + 1].split(",")

    # Create a weightToGradient dict that maps each weight value to the gradient equation name used for gradient descent
    weightToGradient = {}
    splitWeightToGradient = re.split(":|;", weightToGradientDesc)
    for i in range(0, len(splitWeightToGradient), 2):
        weightToGradient[splitWeightToGradient[i]] = splitWeightToGradient[i + 1]

    splitOffsetDesc = re.split(":|;", offsetDesc)
    for i in range(0, len(splitOffsetDesc), 2):
        offsetCount[splitOffsetDesc[i]] = int(splitOffsetDesc[i+1])

    # This is a dictionary that will store all of the values as they are being calculated/updated.
    eqAndVarVals = {}

    # This is a dictionary that will track whether a value/equation has been calculated.
    # If it has been found, or is not something to be recalculated every time, it will be True; else, False
    hasBeenCalculatedThisIteration = {}

    # Initialize all of the trainable weights to a random numpy array of the correct dimensions (from modeSizes)
    for weightVar in typeDict["weights"]:
        eqAndVarVals[weightVar] = np.ascontiguousarray(np.random.random([modeSize[dim] for dim in eqAndVarSizes[weightVar]["modes"]]), dtype=np.float32)
        hasBeenCalculatedThisIteration[weightVar] = True

    if prevWeightFileName != None:
        f = open(prevWeightFileName, "r")
        prevWeightDictStr = f.read()
        if prevWeightDictStr == '':
            raise FileNotFoundError("prevWeightFileName was either not found or a blank file")
        f.close()
        splitPrevWeightDictStr = re.split("\{|'|:|}", prevWeightDictStr)

        # Change the way that vals are stored, must contain mode info
        for i in range(2, len(splitPrevWeightDictStr)-1, 3):
            name, modesList = re.split("\[", splitPrevWeightDictStr[i][:-1].strip())
            modesList = modesList.split(",")

            # Throw a warning if a new weight is introduced (will not directly conflict with anything)
            if name not in eqAndVarVals.keys():
                raise Warning("prevWeightFileName names a weight that is not defined in the Einkorn file:", name)

            # Check if the modes used in the weight loaded match the modes in
            if sorted(modesList) != sorted(eqAndVarSizes[name]['modes']):
                raise IOError("The modes specified in the input list for {input} "
                              "do not match the modes used in the einkorn file\n"
                              "\t modes in input List for {input}: {inputModes}\n"
                              "\t modes in einkorn for {input}: {einkornModes}"
                              .format(input=name,
                                      inputModes=modesList,
                                      einkornModes=eqAndVarSizes[name]['modes']))

            # Read in all of the numerical values into a 1D list
            tempSplit = re.split("\[|\]|,", splitPrevWeightDictStr[i+2].strip())
            oneDWeight = []
            for x in tempSplit:
                if x.strip() != '':
                    oneDWeight.append(float(x))

            # Convert the 1D list to the shape specified by desc in prevWeightFileName
            npWeights = np.ascontiguousarray(oneDWeight)
            npWeights.shape = tuple([modeSize[dim] for dim in modesList])

            # Transpose from desc in prevWeightFileName -> shape specified in Einkorn (see getInputs())
            if npWeights.shape != tuple([modeSize[dim] for dim in eqAndVarSizes[name]['modes']]):
                try:
                    inputDims = "".join(modesList)
                    finalDims = "".join(eqAndVarSizes[name]['modes'])
                    npWeights = np.ascontiguousarray(np.einsum(inputDims + '->' + finalDims, npWeights))
                except:
                    raise IOError("Could not convert input data into form as specified in Einkorn file")
            # prevWeightDict[name] = np.array(npWeights)
            # Throw an error if the shapes do not match between previously-found weight and current random weights
            # Check if alphabetized modes match instead
            # if eqAndVarVals[name].shape != weight.shape:
            #     raise IOError("prevWeightFileName defines trainable weight {name} to be of shape {prevShape}, "
            #                   "but {name} is defined to be of shape {currShape} in the Einkorn file input of {einFile}"
            #                   .format(name=name, prevShape=weight.shape, currShape=eqAndVarVals[name].shape,
            #                           einFile=einkornFileName))
            # Reshape oneD to specified modes in prevWeightFile
            # Reshape the prevWeightFile to match required shape as specified by Einkorn
            eqAndVarVals[name] = npWeights

    # Set all of the inputs to an empty NumPy array
    for var in typeDict["inputs"]:
        # eqAndVarVals[var] = np.ascontiguousarray(np.random.random([modeSize[dim] for dim in eqAndVarSizes[var]["modes"]]), dtype=np.float32)
        hasBeenCalculatedThisIteration[var] = True

    # Initialize the rest as placeholders to be calculated later
    for var in typeDict["outputs"] + typeDict["gradients"]:
        eqAndVarVals[var] = np.ascontiguousarray(np.random.random([modeSize[dim] for dim in eqAndVarSizes[var]["modes"]]), dtype=np.float32)
        hasBeenCalculatedThisIteration[var] = False

    for input, modesList in modesOfInputs.items():
        if sorted(modesList) != sorted(eqAndVarSizes[input]['modes']):
            raise IOError("The modes specified in the input list for {input} "
                          "do not match the modes used in the einkorn file\n"
                          "\t modes in input List for {input}: {inputModes}\n"
                          "\t modes in einkorn for {input}: {einkornModes}"
                          .format(input=input,
                                  inputModes=modesList,
                                  einkornModes=eqAndVarSizes[input]['modes']))

    # Helper function to get the inputs and ensure that they are in the correct shape
    def getInputs(dataFn: Callable):
        inputs = dataFn

        # Automatically fails if there is not the correct number of inputs
        if len(inputs) != len(justInputNamesList):
            raise IOError("Discrepancy in number of specified inputs between input and input data function")

        for i, inputName in enumerate(justInputNamesList):
            npInputs = np.ascontiguousarray(inputs[i])
            if npInputs.shape != tuple([modeSize[dim] for dim in modesOfInputs[inputName]]):
                raise IOError("The size of input {input} returned from the input data function "
                              "does not match the specified input dimensions.\n"
                              "\t Size expected: {reqdShape}\n"
                              "\t Size returned from input data function: {inputShape}"
                              .format(input=inputName,
                                      inputShape=npInputs.shape,
                                      reqdShape=tuple([modeSize[dim] for dim in modesOfInputs[inputName]])))

            # If the modes specified are not in the same order, transposes the input data to match Einkorn specs
            if npInputs.shape != tuple([modeSize[dim] for dim in eqAndVarSizes[inputName]['modes']]):
                try:
                    inputDims = "".join(modesOfInputs[inputName])
                    finalDims = "".join(eqAndVarSizes[inputName]['modes'])
                    npInputs = np.ascontiguousarray(np.einsum(inputDims+'->'+finalDims, npInputs))
                except:
                    raise IOError("Could not convert input data into form as specified in Einkorn file")
            eqAndVarVals[justInputNamesList[i]] = npInputs
        return

    # Where actual iterations/epochs/batches occur, using above inner helper functions
    locStepSize = stepSize
    for epoch in range(1, numEpochs + 1):
        # Get the input data, checking to make sure that it is all the correct size (see getInputs)
        getInputs(inputDataFn)

        # Use the learningSchedule parameter to update stepSize
        locStepSize = learningSchedule(locStepSize, epoch)

        # Use grad functions (that are linked to weights) to change the weighted matrices
        for weight, gradient in weightToGradient.items():
            try:

                momTempVal = momentumProportion * eqAndVarVals[gradient] if epoch > 1 else 0

                if usesNAG:
                    eqAndVarVals[weight] = eqAndVarVals[weight] - momTempVal
                recursiveCaller(kernels, gradient, childDepDict, eqAndVarVals, hasBeenCalculatedThisIteration)
                eqAndVarVals[weight] = eqAndVarVals[weight] - (momTempVal + locStepSize * eqAndVarVals[gradient])
            except RuntimeWarning:
                print("Gradient was too extreme at that point, potentially due to stepSize being too large")

        # If it is time to return the loss, return the loss
        if lossPrintRate != 0 and epoch % lossPrintRate == 0:
            recursiveCaller(kernels, loss, childDepDict, eqAndVarVals, hasBeenCalculatedThisIteration)
            print("With epoch", epoch, "a loss of", np.sum(eqAndVarVals[loss]), "was found")

        # Check if the gradient is sufficiently small (sum over all of the squared gradients)
        if minGradSquaredSum != 0:
            totalChange = 0
            for gradient in typeDict['gradients']:
                totalChange += np.sum(eqAndVarVals[gradient] ** 2)
            if stepSize * totalChange < minGradSquaredSum:
                break

        # Check if the loss is sufficiently small (sum over the loss squared)
        if minLossSquared != 0 and np.sum(eqAndVarVals[loss] ** 2) < minLossSquared:
            break

        # If it is time to print the loss on a test, return the loss
        if testPrintRate != 0 and epoch % testPrintRate == 0:
            if testDataFn == (lambda: []):
                raise IOError("You have defined testPrintRate, but testDataFn is undefined")
            totalLoss = 0
            for _ in range(epochTestBatches):
                getInputs(testDataFn)
                recursiveCaller(kernels, loss, childDepDict, eqAndVarVals, hasBeenCalculatedThisIteration)
                totalLoss += np.sum(eqAndVarVals[loss])
            print("With epoch", epoch, "a test of", epochTestBatches, "batch(es) had a total loss of", totalLoss,
                  "for an "
                  "average loss of", totalLoss / epochTestBatches)

        # Set all of the calculated values to None again
        for resettingVar in typeDict["outputs"] + typeDict["gradients"]:
            hasBeenCalculatedThisIteration[resettingVar] = False


        # hasSmallGradient = runIteration(epoch, stepSize)
        # if hasSmallGradient:
        #     break

    if testDataFn != (lambda:[]) and finalTestBatches > 0:
        totalLoss = 0
        for _ in range(finalTestBatches):
            inputs = inputDataFn
            getInputs(testDataFn)
            recursiveCaller(kernels, loss, childDepDict, eqAndVarVals, hasBeenCalculatedThisIteration)
            totalLoss += np.sum(eqAndVarVals[loss])
        print("For the test phase of", epochTestBatches, "batch(es), had a total loss of", totalLoss, "for an "
         "average loss of", totalLoss / epochTestBatches)


    outputDict = {name: (eqAndVarVals[name]).tolist() for name in typeDict["weights"]}
    # Write to outputFileName (if it's not None)
    if outputFileName != None:
        f = open(outputFileName, "w")
        # fileOutput = ""
        # for weight in typeDict["weights"]:
            # fileOutput += weight+": "+str(eqAndVarVals[weight])+"\n"
        writtenOutputDict = dict()
        for name, matrix in outputDict.items():
            writtenOutputDict[name+"["+",".join(eqAndVarSizes[name]["modes"])+"]"] = matrix

        f.write(str(writtenOutputDict))
        f.close()
    return outputDict


def recursiveCaller(kernels: ctypes.cdll.LoadLibrary, eqToFind: str, childDepDict: {}, eqAndVarVals: {},
                    hasBeenCalculatedThisIteration: {}):
    # Check that all of the dependencies have been calculated already. If not, calculate them.
    argList = []
    for dependency in childDepDict[eqToFind]:
        if not hasBeenCalculatedThisIteration[dependency]:
            recursiveCaller(kernels, dependency, childDepDict, eqAndVarVals, hasBeenCalculatedThisIteration)
        # tempDep = npToCArr(eqAndVarVals[dependency], eqAndVarSizes[dependency]['type'])
        argList.append(npToCArr(eqAndVarVals[dependency], eqAndVarSizes[dependency]['type']))

    # Create an array to store the output value, then reshape the output to the right dimensions and save it in eqAndVarVals
    temp = npToCArr(eqAndVarVals[eqToFind], eqAndVarSizes[eqToFind]['type'])
    offsets = [ctypes.c_int(0)] * offsetCount[eqToFind]
    getattr(kernels, eqToFind)(*argList, temp, *offsets)
    npTemp = np.ascontiguousarray(temp)
    npTemp.shape = eqAndVarVals[eqToFind].shape
    eqAndVarVals[eqToFind] = npTemp

    hasBeenCalculatedThisIteration[eqToFind] = True
    return


def npToCArr(arr: np.ndarray, type: str):
    flatArr = arr.flatten()
    if type == "Float":
        carray = (ctypes.c_float * len(flatArr))(*flatArr)
    elif type == "Int":
        carray = (ctypes.c_int * len(flatArr))(*flatArr)
    else:
        print("Failed to create a C Array w/type =", type)
        carray = null
    return carray


# Things to add/fix:
# Change structure so that npToCArr is called on initialization instead of on iteration?
    # Would then need to convert back with output
    # Everything is represented in C++ as a 1D array anyway, just need to transpose then convert
# Add optimization algos (RMSProp, Adam, Adagrad)
# More error handling
# Add way to add prev gradients back into the function (specify modes?)
    # Need to change the way that output is done to print it as its proper modes


numEpochs = 10
batchIterator(tempDataFn(), "linReg.einkorn", ["X[n,p]", "y[n]"], "error",
              stepSize=0.01, numEpochs=1, lossPrintRate=numEpochs/10, testDataFn=tempDataFn(),
              testPrintRate=numEpochs/10, momentumProportion=.01, prevWeightFileName="testoutput.txt") #, outputFileName="testoutput.txt")
