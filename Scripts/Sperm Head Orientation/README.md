SPERM HEAD ORIENTATION INSTRUCTIONS -- Replace directories in files with your own

 Transfer bounding box information to excel table using bbox_json_to_excel.py.

 Open ellipse.m , ellipse_detection.m and sperm_orientation.m in Matlab.

ellipse.m isn't essential, contains function that plots ellipse on sperm bounding box to visualize how well the ellipse fits the sperm head.


ellipse_detection.m contains the function that fits the ellipse on the sperm head. It is fully described in the .m file by the author. 

The parameters used for now are that the sperm head's large axis is between 30 and 90, the randomization is not used and only the best fit is selected. 

Some of these parameters might be interesting to play with, especially best fits, which returns the specified number of best fits (i.e 3 will return the three best fitting ellipses) or the randomization which instead of computing the N* N possible ellipses computes N* the specified value choosing them at random among the possible ellipses (much faster).

Code comes from "A New Efficient Ellipse Detection Method" (Yonghong Xie Qiang , Qiang Ji / 2002) and the randomization aspect comes from "Randomized Hough Transform for Ellipse Detection with Result Clustering" (CA Basca, M Talos, R Brad / 2005) and can be used to reduce the time complexity of the operation.

By running sperm_orientation.m in Matlab, a 2 column excel file will be created and saved where the first column will be the frame number and the second the orientation of the sperm head. 

LIMITS : 
The sperm head orientation can give wrong intuition regarding the path of a dead sperm as these will "go where the flow pushes them" rather than in the direction of the head

TO-DO 
Convert the excel file to pd dataframe in python to plot results and see their accuracy -- > Problem with multi valued index conversion

Determine direction of vector (as opposed to non-directed vector) -- > exploit consistent center position near bounding box borders (head near borders) 


