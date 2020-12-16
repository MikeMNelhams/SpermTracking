# Extra information:

## Data

Here is the data from two patients tp49 (low concentration) and tp57 (medium/high concentration).

For each patient two different aliquots were taken from one sample and analyzed separately (cover0 and cover1 videos). 
This means all videos were drawn from the same sperm distribution but if there was a preparation error, sample inhomogeneity or sampling error both aliquots may be slightly different.

For each aliquot you have 10 videos available each from a different random location on the slide.

The detections from the Deep Learning algorithm are given in `<video_name>_YOLO_NO_TRACKING_output/centroids_with_meta.json` in json format.

The data contained has the following:
- "centroids"        : a list of centroids in each frame (centroids[frame_nb] will give you access to the centroids detected in frame frame_nb)
- "extra_information": Only contains a resize_factor key (always equal to 2) which tells you the size of the frame on which detections where made (ie: if you want to refind the coordinates in the full frames multiply the values in centroids by the resize_factor)

Each centroid, or detection is represented as a dictionary with the following keys:
- "bbox"         : (x_min, y_min, w, h) of the detection bounding box
- "center"       : (x, y) of the center position,
- "class"        : type of object (sperm=1, your data should only contain sperm objects, disregard all other classes),
- "interpolated" : if the data point has been interpolated by a tracker or not, your original data should have no interpolated points,
- "id"           : a unique 4 character identifier for the detection.


## Output

Your tracking algorithm should output a new json in the same format as the one before. An association between two centroids is represented by having a shared id. This means that each id should represent a unique track.
If you choose to interpolate positions (that were occluded or not detected) you can do so by adding a new centroid object to the correct frame and giving it the same id as the rest of the track.
In this case the "interpolated" key should be set to true.


## Contact

If you have any questions feel free to email Robin at robin@nanovare.com
