Steps for initial setup:

1. Build the most recent cabal module.
1a. Navigate to where einkorn-lite is stored on your device.
1b. Run 'cabal build gradDesc.so'

2. Set the EinkornLiteCabalFolder variable in runWArgs.sh
2a. Find the auto-generated directory that is the result of step 1.
2b. Navigate inside of this directory.
2c. Navigate inside of the directory labelled 'build.'
2d. Continue with this navigation, all the way until you reach a directory that looks similar to 'einkorn-lite-0.1.0.0'.
2e. Navigate inside of the directory that is named something similar to 'einkorn-lite-0.1.0.0'.
2f. Copy the path to the directory that is named something similar to 'einkorn-lite-0.1.0.0' (should be the current working directory).
2g. Set EinkornLiteCabalFolder equal to this directory path.

3. Set the GHCVersion variable in runWArgs.sh
3a. Find the version of ghc used (Somewhere within the newly-defined EinkornLiteCabalFolder, there is a directory of the form 'ghc-#.#.#'. The version is those three numbers (any of them could be more than one digit long) and the decimals that separate them). 
3b. Set GHCVersion to the version of ghc used within the cabal build.

4. Ensure that your path to .ghcup is correct
4a. Run the command "which ghc"
4b. If the output is not of the form "usr/.ghcup/bin/ghc", then set PathToGHCUP to the path that leads to .ghcup (everything before .ghcup)



<!-- You do not need to read below this, this was a prior attempt/guide in creating the README.md -->
Verbose descriptions (not needed but could be useful if the instructions are confusing):
TODO: Check Daniel's initial version (on Slack) to check naming convention differences default vs. nix

Before using batchIterator(), you must first ensure that the Einkorn File you have created is valid and that you are using the most recent version of Einkorn.

Before you can actually use batchIterator(), the correct filepath variables must be defined in runWArgs.sh. Before those variables can be defined properly, the files must actually exist. Therefore, you must run 'cabal build gradDesc.so' or 'cabal build', as either of these commands will create all of the OS-specific files that are not already imported. Personally, I would recommend cabal build gradDesc.so, as that will only build what is explicitly required for batchIterator as opposed to building the entire Einkorn library.

After running 'cabal build gradDesc.so' properly, you should see some new directory, potentially named 'dist-newstyle'. If you do not see some new directory created, then you have either already created the directory upon previous builds of cabal or you have run 'cabal build gradDesc.so' incorrectly.

Assuming you have found your new directory, navigate into it. You should then see multiple directories, one of which is named 'build'. Navigate into that working directory, then continue to navigate down (through what should just be single-directory choices) until you reach a file of the form 'einkorn-lite-#.#.#.#'. At this point, navigate into that directory, then copy the current working directory and set the variable "EinkornLiteCabalFolder" in line 4 of runWArgs.sh to this copied directory path.

If done correctly, line 4 should look something like the following:
    "export EinkornLiteCabalFolder=~/einkornlite/dist-newstyle/build/x86_64-osx/ghc-8.10.4/einkorn-lite-0.1.0.0"
