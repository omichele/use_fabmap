% use this file to make precision recall curves for the results on fab-map

% results
% fabMapResult = load('/media/michele/Data/miche/file temporanei/DOCUMENTI MIEI/lavori università/robotics/master thesis/softwares/workspace/use_fabmap/results/fabMapResult_Nantes_cloudy_sunny_st_lucia_train.txt');

% fabMapResult = load('/home/michele/Documents/master thesis/softwares/openFABMAPsample/Nantes/normal_voc/results.txt');
% fabMapResult = load('/media/michele/Data/miche/file temporanei/DOCUMENTI MIEI/lavori università/robotics/master thesis/softwares/workspace/use_fabmap/results/fabMapResult_Nantes_cloudy_sunny_normal_voc_train.txt');
% fabMapResult = load('/media/michele/Data/miche/file temporanei/DOCUMENTI MIEI/lavori università/robotics/master thesis/softwares/workspace/use_fabmap/resultsfabMapResult.txt');
fabMapResult = load('D:\miche\file temporanei\DOCUMENTI MIEI\lavori università\robotics\master thesis\softwares\workspace\use_fabmap\resultsfabMapResult.txt');

truth_enlarged = dlmread('gt_enlarged.txt');
truth_unique = dlmread('gt_unique.txt');

% discard the first im
% truth_enlarged = truth_enlarged(2:596, 2:596);
% truth_unique = truth_unique(2:596, 2:596);
truth_enlarged = truth_enlarged(2:end, 2:end);
truth_unique = truth_unique(2:end, 2:end);

figure;
imagesc(truth_unique), colormap gray, ylabel('test images'), xlabel('memory images')
set(gca,'xaxisLocation','top')
figure;
imagesc(truth_enlarged), colormap gray, ylabel('test images'), xlabel('memory images')
set(gca,'xaxisLocation','top')

% subtract the diagonal elements (new places)
for i = 1:size(fabMapResult)
    for j = 1:size(fabMapResult)
        if i == j
            fabMapResult(i,j) = 0;
        end
    end
end

sum( sum ( truth ) )       % display the total number of recognitions


figure(1)
% imshow(fabMapResult);
% imagesc(fabMapResult), colormap gray
imagesc(fabMapResult), colormap default

tolerance = 40; % this is the number of recent frames we don't consider for the recognition
results = fabMapResult;
% subtract the diagonal elements (new places) or places to close in time
for ii = 1:size(fabMapResult)
    for jj = 1:size(fabMapResult)
        if ii == jj
            if jj ~= 1
                if jj < tolerance +1
                    ind = jj-1;
                    results(ii,jj-ind:jj) = zeros(1,ind+1);
                else
                    ind = tolerance;
                    results(ii,jj-ind:jj) = zeros(1,ind+1);
                end
            end
        end
    end
end

figure
imshow(results), colormap default

figure
imshow(truth)

figure
imshow(full(spones(fabMapResult)))

figure
imshow(results)

% place_recognized = fabMapResult > 0.99;

% figure(4)
% imshow(place_recognized)


%% PR curves

j = 1;

th1 = 0.999:-0.0001:0.99;
th2 = 0.99:-0.001:0.8;
th3 = 0.8:-0.1:0.1;
th4 = 0.1:-0.0001:0.0001;
th = [th1, th2, th3, th4];
% th = [th2, th3];
% th = th1;
[B, I] = max(results, [], 2);
ds.data.frame_to_frame_diff.position = 1.5;
ds.data.frame_to_frame_diff.orientation = 10;
ds.conf.params.tolerance_enlarged =  20;
ds.conf.params.frame_tolerance_enlarged = ceil(ds.conf.params.tolerance_enlarged / ds.data.frame_to_frame_diff.position);
for k = th
    
    tmp = zeros(size(results,1));
    for aa = 1:size(results, 1)
        tmp(aa, I(aa)) = results(aa, I(aa));
    end
    place_recognized{j} = tmp > k;     % remove the weakest matches
    
    place_recognized_enlarged = zeros(size(results,1));
    for aa = 1:size(place_recognized{j},1)
        for bb = 1:size(place_recognized{j},2)
            if(place_recognized{j}(aa,bb))
                if bb > ds.conf.params.frame_tolerance_enlarged
                    place_recognized_enlarged(aa, bb-ds.conf.params.frame_tolerance_enlarged:min(bb+ds.conf.params.frame_tolerance_enlarged-1, size(results,2))) = ones(1, numel(bb-ds.conf.params.frame_tolerance_enlarged:min(bb+ds.conf.params.frame_tolerance_enlarged-1, size(results,2))));
                    %ones(1, ds.conf.params.frame_tolerance_enlarged*2);
                else
                    place_recognized_enlarged(aa, [1:bb, bb:bb+ds.conf.params.frame_tolerance_enlarged-1]) = ones(1, size([1:bb, bb:bb+ds.conf.params.frame_tolerance_enlarged-1], 2));
                end
            end
        end
    end
    
    tp(j) = 0;
    % tn(j) = 0;
    fp(j) = 0;
    fn(j) = 0;
    
    fp(j) = sum( sum( (place_recognized{j} - truth_enlarged) == 1 ));
    
%     fn(j) = sum( sum( (truth_enlarged - place_recognized{j}) == 1 ));
    fn(j) = sum( sum( (truth_unique - place_recognized_enlarged) == 1 ));
%     fn(j) = sum( sum( (truth_unique - place_recognized{j}) == 1 ));

    
    tp(j) = sum( sum( (place_recognized{j} & truth_enlarged )));      % true positives
    
    precision(j) = tp(j) / (tp(j)+fp(j));
    
    recall(j) = tp(j) / (tp(j)+fn(j));
    
    j = j + 1;
end

stats.threshold = th;
stats.truePositivesNum = tp;
stats.falsePositivesNum = fp;
stats.falseNegativesNum = fn;
stats.precision = precision;
stats.recall = recall;

% stats(:,:,1) = [th; tp; fp; fn; precision; recall];

figure
plot(recall,precision,'o-'), axis([0 1 0 1]), xlabel('Recall'), ylabel('Precision')

figure
plot(recall,precision,'o-'), xlabel('Recall'), ylabel('Precision')

clear th tp fp fn place_recognized precision

%%

tmp2 = results > 0.00015;

figure
imshow(tmp2)
