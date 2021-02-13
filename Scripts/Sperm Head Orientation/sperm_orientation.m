%===============================================================================
% Read excel file that has info from the json file.Use resize factor for correct dimensions
inf0 = readtable("C:\Users\Shane\Desktop\Year 3\Mathematical and Data Modelling\Phase B\mojo_sperm_tracking_data_bristol\mojo_sperm_tracking_data_bristol\tp49\cover0_0_YOLO_NO_TRACKING_output\vid_0.xlsx");
inf0.Var2 = inf0.Var2 *2 ;
inf0.Var3 = inf0.Var3 *2 ;
inf0.Var4 = inf0.Var4 *2 ;
inf0.Var5 = inf0.Var5 *2 ;
inf0 = inf0{:,:};

% Store number of sperms in each frame of the video
nb_sperm = zeros(300,1);
for i = drange(1:300)
    nb_sperm(1) = sum(inf0(:,1) == 0);
    nb_sperm(i+1) = sum(inf0(:,1) == i) + nb_sperm(i);
end
    
%From folder with each frame of a video as a file, load frames
folder = "C:\Users\Shane\Desktop\Year 3\Mathematical and Data Modelling\Phase B\MDM3_B\OrientationJ_Data\1st source\Video 0\Frames\"; 
index = 1;
orientations = zeros(3258,3);

%Iterate over the number of frames for the number of sperm in each frame
%and fit ellipse over head to determine angle
for frame_counter = drange(1:301)
    baseFileName = append('frame',int2str(frame_counter-1),'.jpg');
    fullFileName = fullfile(folder, baseFileName);
    rgbImage_raw = imread(fullFileName);
    nb = 0;
    for i = drange(index:nb_sperm(frame_counter))
        rgbImage = imcrop(rgbImage_raw,inf0(i,2:5));
        E = edge(rgb2gray(rgbImage),'canny',0.5);
        % note that the edge (or gradient) image is used
        bestFits = ellipseDetection(E, params);
        angle = bestFits(:,5);
        orientations(i,3) = angle;
        orientations(i,1) = frame_counter-1;
        orientations(i,2) = nb;
        nb = nb+1;
    index =  nb_sperm(frame_counter)+1;
    %fprintf('%d\n',index);
    
    %Displaying results, don't recommended using these if iterating over
    %lots of frames
    %fprintf('Output %d best fits.\n', size(bestFits,1));

    %fprintf('Angle is %d.',angle)
    %figure;
    %image(rgbImage);
    %ellipse drawing implementation: http://www.mathworks.com/matlabcentral/fileexchange/289 
    %ellipse(bestFits(:,3),bestFits(:,4),bestFits(:,5)*pi/180,bestFits(:,1),bestFits(:,2),'r');
    end
end
%store orientation results in excel file
writematrix(orientations,"C:\Users\Shane\Desktop\Year 3\Mathematical and Data Modelling\Phase B\mojo_sperm_tracking_data_bristol\mojo_sperm_tracking_data_bristol\tp49\cover0_0_YOLO_NO_TRACKING_output/orientations_0.xlsx");
