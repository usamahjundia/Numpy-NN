# Useful (albeit buggy) tools

## Pattern Creator
File : patterncreator.py
Requirement : 
1. numpy
2. tkinter (maybe already installed with python, idk)
3. Python 3

Usage :
python patterncreator.py &lt;number of rows&lt; &gt;number of cols&lt;

The GUI has 4 buttons :
1. Reset
2. Submit
3. Save
4. Clear

The program has an array, keeping all saved pattern.
Clicking on the grids in the GUI will decide what pattern to write. White = -1, black = 1. Clicking on the same grid will revert the change.

Reset will reset the current pattern.

Once you finished designing one pattern, you can click Submit to add the pattern into the array of patterns.

Once you finished with all the patterns, you can click on Save to save the patterns. Currently the file written will be a numpy array file (.npy) with a hardcoded name "hasil.npy" you can turn into a numpy array using numpy.load(filename) 

To reset all the saved patterns, you can use Clear.