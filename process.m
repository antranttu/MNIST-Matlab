
function im = process(x)

% x = '9real.jpg';

img = imread(x);    % Read in image
threshold = 80;    % Define a threshold for converting to black and white
[M,N,clrs] = size(img);
subplot(1,3,1)
imshow(x)
title('original Image')

% Convert background to Black and the black digits to white
% to match with the MNIST dataset that the model was trained on
img = double(img);
filteredimage = zeros(M,N,clrs);
temp1 = (img(:,:,1) + img(:,:,2) + img(:,:,3))/3;
temp2 = temp1<threshold;

filteredimage(:,:,1) = 255*double(temp2);
filteredimage(:,:,2) = filteredimage(:,:,1);
filteredimage(:,:,3) = filteredimage(:,:,1);

newImage = double(filteredimage);

% Convert to RGB, then resize the image to 28x28
im = (rgb2gray(newImage));
subplot(1,3,2)
imshow(im)
title('boxed Image');

%detect the number region
L=bwlabel(im);  % Label the connection areas
stats=regionprops(L,'basic'); %get the basic property of the connected regions
areas=[stats.Area]; % get the area of the connected regions
rects=cat(1,stats.BoundingBox); % concat the boundingBox of the connected areas
[~,max_id]=max(areas); % find the max area of the connected areas
max_rect=rects(max_id,:); % the find the boundindBox of the max connected area
centroids=cat(1,stats.Centroid); %concat the contoid of the connected areas

% plot(centroids(:,1),centroids(:,2),'b*');

temp=[-150,-50,260,110]; %resize the boundingbox
max_rect=max_rect+temp;     
rectangle('position',max_rect,'EdgeColor','r');

cropImage=imcrop(im,max_rect);  %Crop the image
% figure
% imshow(cropImage)
% title('Crop Image');

im = imresize(cropImage,[28,28]);
im = im(:);
im = im./max(im);
im = reshape(im,28,28,1,size(im,2));

% Show actual image and the processed image

subplot(1,3,3)
imshow(im)
title('resized image');

% % 
end