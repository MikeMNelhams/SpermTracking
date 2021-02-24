clear;
Detections = cell(1,301); % main branch

%Table in form |sperm instance number(order of json)|frame number|x_centroid |y_centroid
vid0centers = readtable("C:\Users\Shane\Desktop\Year 3\Mathematical and Data Modelling\Phase B\mojo_sperm_tracking_data_bristol\mojo_sperm_tracking_data_bristol\tp49\cover0_0_YOLO_NO_TRACKING_output\vid_0_centers.xlsx");
vid0centers.Var4 = vid0centers.Var4 *2 ;
vid0centers.Var3 = vid0centers.Var3 *2 ; %resize factor
vid0centers = vid0centers{:,:};

% Store number of sperms in each frame of the video
nb_sperm = zeros(300,1);

for i = drange(1:300)
    nb_sperm(1) = sum(vid0centers(:,2) == 0);
    nb_sperm(i+1) = sum(vid0centers(:,2) == i) ;
    
end
counter = 1;
for i = drange(1:300)
    for j = drange(1:nb_sperm(i))
        measurement = [vid0centers(counter,3);vid0centers(counter,4);0];
        Detections{:,i}{j,1} = objectDetection(i-1,measurement);
        counter = counter + 1;
    end
   
end
Time = zeros(1,301);
for i = drange(1,301)
    Time(i) = i-1;
end

assignsperm = cell(3258,1);


% Create a multiObjectTracker (parameters to be added
tracker = trackerTOMHT();

x_values = zeros(3364,1);
y_values = zeros(3364,1);
%% Run the tracker
time = 0;
numSteps = numel(Time)-1;
i = 1;
assigner = 1;
while i < numSteps 
    i = i + 1;
    disp(i)
    % Current simulation time
    simTime = Time(i);
    
    scanBuffer = Detections{i};
    
    % Update tracker
    tic
    tracks = tracker(scanBuffer,simTime);
    time = time+toc;
    
    
    [pos,cov] = getTrackPositions(tracks,[1 0 0 0 0 0;0 0 1 0 0 0;0 0 0 0 1 0]); 
    labels = arrayfun(@(x)num2str(x.TrackID),tracks,'UniformOutput',false);
    
    for t = drange(1:numel(pos(:,1)))
        x_values(assigner)= pos(t,1);
        y_values(assigner)= pos(t,2);
        
        assignsperm{assigner,1}= string(labels(t));
        assigner = assigner + 1;
    end
end
lab = zeros(3364,1);
for d = drange(1:3228)
    lab(d) = assignsperm{d,1};
end
gscatter(x_values,y_values,lab)

%lab is the list of track labels

