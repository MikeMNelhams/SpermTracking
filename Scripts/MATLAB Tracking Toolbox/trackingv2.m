clear;
Detections = cell(1,301); % main branch
frames = 301;
%Table in form |sperm instance number(order of json)|frame number|x_centroid |y_centroid
vid0centers = readtable("D:\Uni work\Engineering Mathematics Work\MDM3\Mojo\mojo_sperm_tracking_data_bristol\tp49\cover0_3_YOLO_NO_TRACKING_output\vid0_3.xlsx");
% x and y values of the centres of detections
vid0centers.Var4 = vid0centers.Var4 *2 ;
vid0centers.Var3 = vid0centers.Var3 *2 ; %resize factor
vid0centers = vid0centers{:,:};

% Store number of sperms in each frame of the video
nb_sperm = zeros(frames-1,1);

for i = drange(1:frames-1)
    nb_sperm(1) = sum(vid0centers(:,2) == 0);
    nb_sperm(i+1) = sum(vid0centers(:,2) == i) ;
    
end

counter = 1;
for i = drange(1:frames)
    for j = drange(1:nb_sperm(i))
        measurement = [vid0centers(counter,3);vid0centers(counter,4);0];
        Detections{:,i}{j,1} = objectDetection(i-1,measurement);
        counter = counter + 1;
    end
   
end

    
Time = zeros(1,frames);

for i = drange(1,frames)
    Time(i) = i-1;
end
%create a final table x,y, tracktrack_labelel with of the sperm number sizew
assignsperm = cell(counter,1);

%count the number of detections
S = 0;
for k = 1:numel(Detections)
  S = S + size(Detections{k});
end


% Create a multiObjectTracker (parameters to be added
tracker = trackerJPDA('TrackLogic','Integrated','FilterInitializationFcn','initcvkf','DeathRate',0.08);

%initialise the x values and y values vector which is twice the size of the
%number of detections (counter) to be on the safe side since the tracker
%occasionally adds or removes points
x_values = zeros(2*counter,1);
y_values = zeros(2*counter,1);
detection_frame_num = zeros(2*counter,1);
%% Run the tracker
time = 0;
numSteps = numel(Time);
i = 0;
assigner = 1;
while i < numSteps 
    
    %disp(i)
    % Current simulation time
    simTime = Time(i+1);
    
    scanBuffer = Detections{i+1};
    
    % Update tracker
    tic
    tracks = tracker(scanBuffer,simTime);
    time = time+toc;
    
                                     
                                           %x position, y position, z position
    [pos,cov] = getTrackPositions(tracks,[1 0 0 0 0 0;0 0 1 0 0 0;0 0 0 0 0 0]); 
    track_labels = arrayfun(@(x)num2str(x.TrackID),tracks,'UniformOutput',false);
    
    %disp(size(pos))
    for t = drange(1:numel(pos(:,1)))
        x_values(assigner)= pos(t,1);
        y_values(assigner)= pos(t,2);
        detection_frame_num(assigner)= i;
        % assign sperm track_labels 
        assignsperm{assigner,1}= string(track_labels(t));
        assigner = assigner + 1;
    end
    i = i + 1;
end
%trim vectors to size since get track position creates more points than
%detections
x_values = x_values(1:assigner-1);
y_values = y_values(1:assigner-1);
detection_frame_num = detection_frame_num(1:assigner-1);
%track_label is the list of track labels
track_label = zeros(assigner-1,1);

for d = drange(1:assigner-1)
    track_label(d) = assignsperm{d,1};
end
%draw scatterplot
gscatter(x_values,y_values,track_label);

%flip y-axis to match orignial data
set(gca,'YDir','reverse')
file_matrix = [ x_values, y_values,detection_frame_num,track_label] 
file_write_path =  'D:\Uni work\Engineering Mathematics Work\MDM3\Mojo\labelledtracks.xlsx'
% will write a exccel file with the first column as the unique track
% identifier number, 2nd and 3rd columns are x values and y values accor
writematrix(file_matrix,file_write_path)