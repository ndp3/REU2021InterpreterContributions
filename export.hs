{-# OPTIONS_GHC -Wall                 #-}
{-# LANGUAGE ForeignFunctionInterface #-}

module EinkornLite.GradDesc.Export where

import Foreign.C.String

import qualified Data.Text as T
import Data.Set ( Set, unions, member )
import qualified Data.Set as Set
import qualified Data.Map as Map
import qualified Data.List as List
import Data.Maybe ( fromJust )

import Control.Monad.Except

-- import Debug.Trace

import EinkornLite.Barb.KernelGen as KernelGen ( compileKernel )
import EinkornLite.Parse.Parser ( parse )
import EinkornLite.Parse.Lexer ( scanTokens )
import EinkornLite.Expr as Expr ( Expr, childrenOf, incidentModes )
import EinkornLite.Parse.Types
    
import EinkornLite.Parse.ParserUtils ( compileToBaseExpr, getCSizeTable, atomsOf )
-- import qualified EinkornLite.Data.Graph ( Id )

-- import qualified EinkornLite.Data.Graph ( Id )

import EinkornLite.Types
import EinkornLite.Misc ( sepBy, (|>) )
import EinkornLite.Korn ( ScalarType(STInt, STFloat, STBool)  )


-- foreign export ccall compileMultC :: CString -> CString -> CString -> [IO ()]
-- -- In order for an append writer to work, do I need to somehow clear the file in advance?
-- compileMultC :: CString -> CString -> CString -> IO ()
-- compileMultC [] _ _ = return ()
-- compileMultC cEinFormulaFileName cFilename cDescStr = do
--     let indDescStr:restDestStr = T.splitOn (T.pack ",") (T.pack cDescStr)
--     (compileC cEinFormula cFilename cDescStr) : compileMultC xs cFilename cDescStr


-- convert CString -> String -> input for compileKernel (ModeSizes -> String -> Expr)
-- save the output from running compileKernel
-- write the output to the filename input

foreign export ccall compileC :: CString -> CString -> CString -> IO ()
compileC :: CString -> CString -> CString -> IO ()
compileC cEinFormula cFilename cDescStr = do


    -- Convert CStrings to Strings (can now be used as inputs for others?)
    -- haskEinFormulaFileName <- peekCString cEinFormulaFileName
    -- haskEinFormula <- readFile haskEinFormulaFileName
    haskEinFormula <- peekCString cEinFormula
    -- haskEinFormula <- return $ unlines $ [
    --     "i :: Mode 10",
    --     "j :: Mode 10",
    --     "k :: Mode 10",
    --     "X :: Float{i,j}",
    --     "Y :: Float{i,k}",

    --     "Out = A{i,j} + A[i.j,j.i];"
    --   ]
    haskFilename <- peekCString cFilename
    descStr <- peekCString cDescStr
    -- Have confirmed that I can run normal String fns on these versions post-peek


    -- Need to figure out how to parse a String into ModeSizes, String, and Expr
    exprMaybe <- getExprAndSizes haskEinFormula descStr
    case exprMaybe of
        Nothing -> putStrLn "error: couldn't parse 'Out' from 'expToBarb.einkorn'"
        Just (expr,sizes) -> do
            -- save the output from running compileKernel
            let cProperFormula = KernelGen.compileKernel sizes descStr expr

            -- write the output to the filename: TBC: make it an appendFile instead (w/iterations?)
            -- make the initialization write an empty str to the file
            writeFile haskFilename cProperFormula

    -- TBR: Just a printing string to help w/debugging
    putStrLn "Done w/ compile_c from export.hs"



-- This has been slightly altered from code that is used in a few exps files.
-- Basically, it takes in a string that represents the Einkorn calculation, 
-- and finds the inputs that need to be passed into compileKernel in an 
-- IO (Maybe ()) wrapper.
getExprAndSizes :: String -> String -> IO (Maybe (Expr.Expr, ModeSizes))
getExprAndSizes src descStr = do
  let doThis = do
        -- I am still confused how compileToBaseExpr works
        -- Is there more that I need to add to make the string input variable?
        expr <- scanTokens src >>= parse >> compileToBaseExpr descStr
        mszs <- getCSizeTable
        return (expr, mszs)
  return $ fst $ runP doThis emptyPState

-- Function uses the Data.Text library to find the first word in a string, 
    -- separated by an equal sign
-- Can be used to isolate the equality of the input equation (and then 
    -- be used as the description string in compileC)
-- getFirstWord :: String -> String
-- getFirstWord formula = T.unpack $ T.strip $ head $ T.splitOn (T.pack "=") (T.pack formula)

-- pCExpr: If Expr && is a tensor --> out/loss, KAtomType: singleton -> input, 
        -- call atomsOf (in parserUtils) Expr to determine if it's a fn to be thrown out, if atomsOf returns 1 item in the list it's out/loss
            -- 
-- typeOfExpr :: pCExpr -> String 
-- typeOfExpr pcexpr = 
--     case pcexpr of 
--         Just (Left e) | length (atomsOf e) == 1     -> "out/loss"
--         Just (Right x) | length x == 1          -> "input"
--         otherwise                              -> "function"

-- PState -> compileToBaseExpr (base Expr will have leaves of input variables)
-- get 

throwErrorE :: String -> ExceptT () IO ()
throwErrorE err = do
    liftIO $ putStrLn err
    throwError ()

processHelper :: String -> String -> String -> String -> IO String
processHelper inputs loss src outputFileName =

    

        
    let 
        process' :: ExceptT () IO String
        process' = do
            
            let pStateFromSrc = snd $ runP (scanTokens src >>= parse) emptyPState
                -- convert inputs into actual list of inputs (parse based on ";")
                inputList = map T.unpack $ T.splitOn (T.pack ":") (T.pack inputs)
                (trainingVars, outputs) = getGradientsAndOutputs inputList pStateFromSrc
            -- Throws an error if pStateFromSrc is invalid in order to stop the code earlier
            case errorInfo pStateFromSrc of
                Nothing -> return ()
                Just err -> throwErrorE $ "Error from initial Einkorn file: "++err
            let
                -- Adds the gradient functions for each weight (Einkorn input not in inputs list)
                setOfAllIdentifiers = getAllIdentifiers pStateFromSrc
                -- Checks if there is already a function named grad++trainable; if so, try again with ++"_"
                    -- Technically, will fail if there are trainable weights u, v s.t. u.name == v.name ++ "_"
                makeGradFnName trainable | retVal `member` setOfAllIdentifiers  = makeGradFnName (trainable++"_" )
                                         | otherwise                            = retVal
                    where retVal = "grad"++trainable
                gradients = map makeGradFnName trainingVars                  
                
                makeGrad (trainableVar, gradFn) = gradFn++" = "++loss++" @ "++trainableVar++";"
                gradSrc = unlines $ map makeGrad trainingVarsAndGradients
                    where trainingVarsAndGradients = zip trainingVars gradients
                pStatePlusGradients = snd $ runP (scanTokens gradSrc >>= parse) pStateFromSrc

            case errorInfo pStatePlusGradients of
                Nothing -> return ()
                Just err -> throwErrorE $ "Error from adding gradients into Einkorn state "++ 
                        "(potentially due to unusual <variable>_ name): "++ err
            
            let
                baseExpressionsPlusTable = do
                    let items = outputs ++ gradients
                    baseExprs <- mapM compileToBaseExpr items
                    mszs <- getCSizeTable
                    return (zip items baseExprs, mszs)
                (baseExpressionsPlusTableMaybe, pStateFinal) = runP baseExpressionsPlusTable pStatePlusGradients
                (nameBaseExprs, modeSizes) = fromJust baseExpressionsPlusTableMaybe
                
                -- Creates the kernels as strings (to be written into outputFileName)
                compileKernelHelper (name,baseExpr) = compileKernel modeSizes name baseExpr
                kernels = unlines $ map compileKernelHelper nameBaseExprs
                
                -- Below is all of the string creation to be read in and handled by Python
                allVars = inputList++trainingVars++gradients++outputs
                -- Maps mode:size (ex: n:5,p:2)
            let makeModeSizesStr (modeName,int) = show modeName ++ ":" ++ show int
                modeSizesDescStr = sepBy "," id $ map makeModeSizesStr (Map.assocs modeSizes)

                -- Maps a tensor to its type and its mode(s)
                tensorTypeDescStr = concatMap stringToTensorTypeToStringHelper allVars
                    where stringToTensorTypeToStringHelper x = stringToTensorTypeToString x pStateFinal

                -- Representation of the DAG: maps each non-input to all of its dependencies
                makeItemString (name,baseExpr) = name++"["++sepBy "," id (List.sort children)++"]"
                    where children = Set.toAscList $ childrenOf baseExpr
                childDependencies = concatMap makeItemString nameBaseExprs

                --  Maps all functions and tensors to their type as represented in Python
                dictOfTypes = "inputs{" ++ sepBy "," id inputList ++ "}weights{"++ sepBy "," id trainingVars ++
                    "}gradients{"++ sepBy "," id gradients ++"}outputs{" ++ sepBy "," id outputs++"}"
                
                -- Maps each trainable weight to its gradient function
                trainableGradDict = sepBy ";" id $ map makeTrainableGradDict trainingVarsAndGradients
                    where 
                        trainingVarsAndGradients = zip trainingVars gradients
                        makeTrainableGradDict (trainable, gradient) = trainable++":"++gradient

                -- Maps each kernel function to its number of offsets (needed as inputs to call kernel function)
                offsetDescMaker nameBaseExpr = sepBy ";" id $ map findOffsetCount nameBaseExpr
                    where findOffsetCount (name, baseExpr) = name++":"++show (numOffsets baseExpr)
                offsetDescStr = offsetDescMaker nameBaseExprs
                
                -- Headers needed in the C++ code to actually compile
                headers = unlines [
                    "#include <cmath>",
                    "using std::pow;"
                    ]
                
                -- Strings in order to create the kernels in outputFileName
                descStr = modeSizesDescStr++"|"++tensorTypeDescStr++"|"++childDependencies++"|"++dictOfTypes++
                      "|"++trainableGradDict++"|"++offsetDescStr
                childDepPlusKernels = unlines [ 
                    -- Commented descriptive string at top to be read in easily and not prevent compilation.
                    "// "++descStr, 
                    headers, 
                    "extern \"C\" {",
                    kernels, 
                    "}"
                    ]
            
            case errorInfo pStateFinal of
                Nothing -> return ()
                Just err -> throwErrorE err
            -- Actually writes the auto-generated string to the outputFile
            liftIO $ writeFile outputFileName childDepPlusKernels
            return descStr
    in do
        maybeOutput <- runExceptT process'
        case maybeOutput of
            -- Probably need to change the return type of processHelper to an Either/Maybe?
            Left () -> return "Could not return a string\n" 
            Right childDepencies -> return childDepencies

foreign export ccall process :: CString -> CString -> CString -> CString -> IO CString
process :: CString -> CString -> CString -> CString -> IO CString
process inputsC lossC srcFileC outputFileNameC = do
        -- Convert each CString to a String, then pass into processHelper (to help error handling)
        inputs <- peekCString inputsC
        loss <- peekCString lossC
        srcFile <- peekCString srcFileC
        src <- readFile srcFile
        outputFileName <- peekCString outputFileNameC
        out <- processHelper inputs loss src outputFileName
        newCString out

getGradientsAndOutputs :: [String] -> PState -> ([String], [String])
getGradientsAndOutputs inputList pState =
  let concreteExpressionsList = Map.toList $ pCExpr pState
      einkornInputs  = filter isEinkornExprInput concreteExpressionsList
      einkornOutputs = filter isEinkornExprOutput concreteExpressionsList
      isEinkornExprInput (name,eitherObject) =
        case eitherObject of
          Left _ | name `elem` inputList -> error "Inputs cannot be defined!"
          Left _ -> False              -- something declared must not be an input              
          Right [_] -> True            -- only one atom? must be a tensor                      
          Right _ -> False             -- more than one atom? must be a function               
      isEinkornExprOutput (_,eitherObject) =
        case eitherObject of
          Left e | length (atomsOf e) == 1 -> True -- tensor and defined                       
          Left _ -> False                          -- not a tensor but defined                 
          Right _ -> False                         -- not defined                              
      gradients = map fst einkornInputs List.\\ inputList
      outputs = map fst einkornOutputs
   in (gradients, outputs) 


getInputType :: String -> PState -> TensorType
getInputType name pState = 
  let table = pCExpr pState
   in case Map.lookup name table of
        Nothing -> error $ "This variable is not found in the Einkorn file: "++name
        Just (Left e) -> 
          case atomsOf e of
            [AtomTensor st ms] -> TensorType st ms
            _                  -> error "also not good"
        Just (Right [AtomTensor st ms]) -> TensorType st ms
        _ -> error "Also shouldn't happen!"

getAllIdentifiers :: PState -> Set Id
getAllIdentifiers state =
    let allModes = pModes state
        allCs    = pCExpr state |> Map.keysSet
        allQs    = pQExpr state |> Map.keysSet
        allKorns = pKorns state |> Map.keysSet
        allIds = unions [allModes, allCs, allQs, allKorns]
    in allIds

numOffsets :: Expr.Expr -> Int
numOffsets e = Set.size (incidentModes e)

stringToTensorTypeToString :: String -> PState -> String 
stringToTensorTypeToString name pState = 

    let tensorType = getInputType name pState
        (TensorType tensorTypes  modes) = tensorType
        findType :: ScalarType -> String
        findType types  | types == STInt   = "Int"
                        | types == STFloat = "Float"
                        | types == STBool  = "Bool"
                        | otherwise        = error "Wrong type passed in"
        thisType = findType tensorTypes
        -- case types of
        --     -- Nothing -> error "The ScalarType of a tensor should not be undefined!"
        --     STInt -> let type = "Int"--{"++concat modes++"}"
        --     STFloat -> type = "Float"--{"++sepBy "," id (Set.toList modes)++"}"
        --     STBool -> type = "Bool"--{"++concat modes++"}"
            -- Just _ -> error "The TensorType didn't match any of the types"
    in name ++ ":" ++ thisType ++ "{" ++ sepBy "," id (Set.toList modes) ++ "}"
-- tensorTypeToString (TensorType types modes) = "Int{"++concat modes++"}"

