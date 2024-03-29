Latest version
============
The latest version for the tracking algorithm is ```tracking_algorithm56.py```.
Running ```python run_main(algorithm, cover='cover', tp='tp')``` will write a .json file of the exact format of video detections for a given TP and cover.

Test files
-----
There are test files for verififying the functions work for all the data. There is a plot version to verify plotting works.

## Motility Calculations
The file:
```MotilityCalc.py``` produces the RMS distances travelled using 'Mike' based clustering. The algorithm doesn't have a mathematical name, but I would suggest calling it closest frame.

## Tracking Evaluations
The file:
```EvaluatingTracking.py``` has 4 functions for producing performance measures for any clustering algorithms for all of the data. 
The functions all take ```['algorithm1', 'algorithm2', ...]``` lists as their only input.  

* ```eval_U()``` Produces the U means and U standard deviations.
* ```eval_convex_hull()``` Produces the convex hull densities.
* ```eval_runtime()``` Uses date_time to accurately calculate the runtime to the nearest 10ms.
* ```plot_all_04()``` Is used to plot all the graphs for cover 04. Can easily be changed to suit all covers and TPs.

# Using the package to cluster data
Clustering the Data: 

To run an clustering algorithm `ALGO` on test patient (tp) `tp` and cover `cover`, the function ```run_main(algorithm=ALGO, tp=tp, cover=cover, plot=True, plot_type='2d')``` will calculate the clusters using the specified algorithm, return the cluster ID predictions and plot a 2D graph of the results. 

## Parameters 

### Algorithm 
Must be a string. 
Currently accepted are: ```["kmeans", "dbscan", "mike", "none", "hdbscan", "gmm", "htdbscan", "richard-dbscan"]``` 

### TP 
2 digit string. 

For the provided data ```['49', '57']```

Default `'49'`
### Cover 
2 digit string. 

For the provided data ```['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']```

Default `'57'`
### Plot 
Bool. 

Plot = True will plot the graph, Plot = False will not plot the graph 

Default `False`
### write_output 
Bool. 

write_output = True will write the data to the corresponding tp/cover file given that the Mojo tracking data bristol is saved locally within the same directory as ```tracking_algorithm55.py```

Default `True`
### Verbose
Bool. 

Verbose = True will print all information to console. Verbose = False will only print some information if any. 

Default `False`
### Plot type 
String. 

Currently accepted are: ```['2d', '3d', 'bar_graph']```

### Heatmap 
Bool. 

heatmap = True will change the colour scheme to using a heatmap where colour represents the U value accuracy. 

heatmap = False will use default colour scheme where every cluster is a different randomised cluster from palette rgb(0.05 to 0.7, 0.05 to 0.7, 0.05 to 0.7) .

### Legend
Bool. 

legend = True will include a nicely coloured and formatted legend on the right side. 

legend = False will rescale the graph and not include any legend.

## Examples:
### 1
```python 
run_main(algorithm='kmeans', cover='00', plot=True, plot_type='2d')
```
 
![](kmeans3x2d00.png)

### 2
```python 
run_main(algorithm='mike', cover='04', plot=true, plot_type='3d', heatmap=True
```

![](heatmap_example.png)
