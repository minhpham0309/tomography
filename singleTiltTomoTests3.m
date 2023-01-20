%%%% singleTiltTomoTests2.m
%%%% Generate projections from atomic model using a single tilt axis,
%%%% reconstruct with iterative algorithm. This code allows motion to be
%%%% incorporated into the atomic model when generating the projections.
%%%% 3D reconstruction is compared with 4D reconstruction.

clear all; close all;
addpath model;
addpath src;
addpath linear_projs_calc\input;
addpath linear_projs_calc\src;

%% Create projection images

modelType = 1;
if modelType == 1
    % Define atom positions and types
    model = [0 0 0; 3 3 3; -2 -6 -10]';
    atoms = [1 2 1];
    % Dimension of projection images
    volSize = 80;
    % Define tilt angles
    numTilts = 17;
    minAngle = -75;
    maxAngle = 75;
    % Resolution of projections
    Res = 0.467;
    % How much motion (expanded linearly from original model)
    maxMotionPct = 20;
    % Calculate projections
    [projections,angles] = generate_projs_single_tilt_motion(model,atoms,numTilts,minAngle,maxAngle,volSize,Res,maxMotionPct);
elseif modelType == 2
    % Load angles and projections (this model has many atoms)
    load projectionsMBWsingleTilt.mat
    angles = angles';
else
    error('Unknown model type.');
end

%% Reconstruct 3D volume from projections

% Set user-defined algorithm parameters
step_size = 1;
iterations = 300;
positivity = true;

% Set up variables, dimensions, and A matrix
dtype = 'single';
projections = cast(projections,dtype);
[dimx, dimy, Num_pj] = size(projections);
obj_dimx = dimx; obj_dimy = dimy; obj_dimz = dimx;
rec = zeros([obj_dimx,obj_dimy,obj_dimz]);
rec_vec = permute(rec,[1 3 2]);
rec_vec = reshape(rec_vec,[obj_dimx*obj_dimz,obj_dimy]);
dt = (step_size/Num_pj/obj_dimz);
A = constructSingleTiltAmatrix(dimx,obj_dimx,obj_dimz,angles);
projections_vec = permute(projections,[1,3,2]);
projections_vec = reshape(projections_vec,[dimx*Num_pj, dimy]);
sum_proj = sum(projections_vec(:));

% Initialize object
size_rec = size(rec_vec);
grad = zeros( size_rec ,dtype);
pj_cals = zeros( dimx* Num_pj, dimy);

% Tomography reconstruction: solve Ax=b
fprintf('Beginning 3D recon.\n');
for iter=1:iterations
    % forward projections via Radon transform, i.e. compute pj_cals = Au
    for l=1:dimy
        pj_1d = double( rec_vec(:,l) );
        pj_cals(:, l) =  A* pj_1d;
    end
    % back projections, i.e compute residual = Au-b, and grad = A^T(Au-b)
    residual = double(pj_cals - projections_vec);
    % compute R factor
    if  mod(iter,20) == 0
        err = sum(abs(residual(:)))/sum_proj;
        fprintf('Iter %d. Rfactor=%.4f\n',iter, err);
    end
    % compute gradient
    for l=1:dimy
        res_l = residual(:,l);
        grad(:, l) = (res_l'*A)';
    end
    % update object
    rec_vec = rec_vec - dt*grad;
    % constraint & regularizer
    if positivity
        rec_vec(rec_vec<10)=0;
    end
end
%% show result of 3D reconstruction
% Show the reconstructed 3D volume, slice by slice
figure(1); clf;
rec3D = reshape(rec_vec, [obj_dimx, obj_dimz, obj_dimy]);
img(rec3D,'3D recon','caxis',[0,max(rec3D(:))]);

% Show xz, yz, and xy projections of reconstructed 3D volume
figure(2); clf;
pjYZ = sum(permute(rec3D,[3,2,1]),3);
pjXY = sum(permute(rec3D,[1,3,2]),3);
pjXZ = sum(rec3D,3);
img(pjXZ,'3D recon proj XZ',pjYZ,'3D recon proj YZ',pjXY,'3D recon proj XY','size',[1 3]);

% Show residual errors in projections
projectionErrors = zeros(Num_pj,1);
for l=1:dimy
    pj_1d = double(rec_vec(:,l));
    pj_cals(:,l) = A*pj_1d;
end
residual = reshape(double(pj_cals - projections_vec),[dimx Num_pj dimy]);
residual = permute(residual,[1 3 2]);
for pIndex = 1:Num_pj
    projectionErrors(pIndex) = norm(squeeze(residual(:,:,pIndex)),'fro')/norm(squeeze(projections(:,:,pIndex)),'fro');
end
figure(3);clf;
img(residual);
title('3D recon residuals');
figure(4);clf;
plot(1:Num_pj,projectionErrors);
title('3D recon residual errors');
xlabel('Projection (tilt) number');

if 1
    % For the 17 tilts with max motion 20%, compute tilt-by-tilt error with
    % respect to the ground truths.
    load model1groundTruth0to20pct_17tilts
    rec3D = reshape(rec3D, [obj_dimx*obj_dimz,obj_dimy]);
    errors3D = zeros(numTilts,1);
    norms3D = zeros(numTilts,1);
    for tiltNum = 1:numTilts
        errors3D(tiltNum) = norm(rec3D - rec_vecs_ground_truth{tiltNum},'fro')^2;
        norms3D(tiltNum) = norm(rec_vecs_ground_truth{tiltNum},'fro')^2;
    end
    fprintf('Total NMSE relative to ground truth: %.3f\n',sum(errors3D)/sum(norms3D));
    figure(5);clf;
    plot(1:numTilts,errors3D./norms3D); %expansionFactorsPct
    xlabel('Motion percentage');
    title('3D recon error relative to ground truth');
    ylabel('Normalized MSE');
end

%%
%
rec_vecs_ground_truth_1 = permute(reshape( rec_vecs_ground_truth{1}, [obj_dimx,obj_dimz,obj_dimy]), [1,2,3]);
rec_vecs_ground_truth_9 = permute(reshape( rec_vecs_ground_truth{9}, [obj_dimx,obj_dimz,obj_dimy]), [1,2,3]);
rec_vecs_ground_truth_17 = permute(reshape( rec_vecs_ground_truth{17}, [obj_dimx,obj_dimz,obj_dimy]), [1,2,3]);
figure(8); img(sum(rec_vecs_ground_truth_1,3), '1', sum(rec_vecs_ground_truth_9,3),'9',sum(rec_vecs_ground_truth_17,3) ,'17')
%}
%% 4D Reconstruction

%% Set up interpolation operator and measurement matrix
%
numAnchors = 17;
interpType = 'linear'; % linear or nearest
interpWeights = zeros(numTilts,numAnchors);
anchorTimes = linspace(1,numTilts,numAnchors);
anchorDelta = anchorTimes(2)-anchorTimes(1);

switch interpType
    case 'linear'
        % Linear interpolation
        for aa = 1:numAnchors
            for tt = 1:numTilts
                interpWeights(tt,aa) = 1 - min(abs(tt-anchorTimes(aa))/anchorDelta,1);
            end
        end
    case 'nearest'
        % Nearest neighbor interpolation
        for tt = 1:numTilts
            [minVal,minIndex] = min(abs(anchorTimes-tt));
            interpWeights(tt,minIndex(1)) = 1;
        end
    otherwise
        error('interpType not recognized.');
end
%}
%% gausian weight
numAnchors = 17;
sigma=6;
interpWeights = zeros(numTilts,numAnchors);
for aa = 1:numAnchors
    for tt = 1:numTilts
        val = exp(-(tt-aa)^2/(2*sigma^2));
        if val< 1e-2, val=0; end
        interpWeights(tt,aa) = val;
    end    
end

for tt = 1:numTilts
    interpWeights(tt,:) = interpWeights(tt,:)/sum(interpWeights(tt,:));
end

%% Interpolation weights should sum to one for each tilt
if max(abs(sum(interpWeights,2)-1)) > 1e-3
    fprintf('Warning: invalid interpolation weights.\n');
end

A = constructSingleTiltAmatrix(dimx,obj_dimx,obj_dimz,angles); %size= [dimx*Num_pj, obj_dimx*obj_dimz]
AW = sparse(size(A,1),numAnchors*size(A,2));                   %size= [dimx*Num_pj, obj_dimx*obj_dimz*numAnchors]
% x(size) = [ obj_dimx,obj_dimz,numAnchors ]
measPerTilt = dimx;
for tiltNum = 1:numTilts
    rowEnd = measPerTilt*tiltNum;
    rowStart = rowEnd - measPerTilt + 1;
    for aa = 1:numAnchors
        colEnd = aa*size(A,2);
        colStart = (aa-1)*size(A,2) + 1;
        AW(rowStart:rowEnd,colStart:colEnd) = interpWeights(tiltNum,aa)*A(rowStart:rowEnd,:);
    end
end

%% Laplace operator
%size(vec_4D)= [obj_dimx*obj_dimz*numAnchors, dimy]

%grad_t = zeros([obj_dimx*obj_dimz*numAnchors,obj_dimy]); 
%grad_t(1: end-obj_dimx*obj_dimz,:) = rec_vec4D(obj_dimx*obj_dimz+1:end,:) - rec_vec4D(1: end-obj_dimx*obj_dimz,:);
Dt0 = full(gallery('tridiag',numAnchors,0,-1,1));
Dt0(end,end)=0;
Ixz = speye(obj_dimx*obj_dimz);
Dt = kron(Dt0,Ixz);
DtD = Dt'*Dt;

Dyt = gallery('tridiag',obj_dimy,0,-1,1);
Dyt(end,end)=0;
DyD = Dyt*Dyt';

Dx0 = gallery('tridiag',obj_dimx,0,-1,1);
Dx0(end,end)=0;
Izt = speye(obj_dimz*numAnchors);
Dx = kron(Izt,Dx0);
DxD = Dx'*Dx;

Dz0 = gallery('tridiag',obj_dimz,0,-1,1);
Dz0(end,end)=0;
Ix = speye(obj_dimx);
It = speye(numAnchors);
Dz1 = kron(Dz0, Ix);
Dz = kron(It, Dz1);
DzD = Dz'*Dz;

DxD_DzD = DxD+DzD;

%% Tomography reconstruction: solve Ax=b (USING AW INSTEAD OF A)
fprintf('\nBeginning 4D recon.\n');
%rec4D = zeros([obj_dimx,obj_dimy,obj_dimz,numAnchors]);

rec4D = repmat( permute(rec3D,[1,2,3]), [1,1,1,numAnchors]  );
rec_vec4D = reshape(permute(rec4D,[1 3 4 2]),[obj_dimx*obj_dimz*numAnchors,obj_dimy]);
size_rec4D = size(rec_vec4D);
grad = zeros( size_rec4D ,dtype);
pj_cals = zeros( dimx* Num_pj, dimy);

%% iterations
for iter=1:iterations
    % forward projections via Radon transform, i.e. compute pj_cals = Au
    for l=1:dimy
        pj_1d = double( rec_vec4D(:,l) );
        pj_cals(:, l) =  AW* pj_1d;
    end
    % back projections, i.e compute residual = Au-b, and grad = A^T(Au-b)
    residual = double(pj_cals - projections_vec);
    % compute R factor
    if  mod(iter,20) == 0
        err = sum(abs(residual(:)))/sum_proj;
        fprintf('Iter %d. Rfactor=%.4f\n',iter, err);
    end
    % compute gradient
    for l=1:dimy
        res_l = residual(:,l);
        grad(:, l) = (res_l'*AW)';
    end
    % update object
    %grad_t(1: end-obj_dimx*obj_dimz,:) = rec_vec4D(obj_dimx*obj_dimz+1:end,:) - rec_vec4D(1: end-obj_dimx*obj_dimz,:);
    % you can adjust the step-size for faster converence
    rec_vec4D = rec_vec4D - (6*numAnchors*dt)*(grad ); %+ 0.1*DtD*double(rec_vec4D)

    % constraint & regularizer: uncomment it when needed
    %{
    for l=1:dimy
        %rec_vec4D(:,l) = rec_vec4D(:,l) -  (0.2*numAnchors*dt)* DtD*double(rec_vec4D(:,l));
        rec_vec4D(:,l) = rec_vec4D(:,l) -  (0.4*numAnchors*dt)* (DxD+DzD+DtD)*double(rec_vec4D(:,l));
    end
    %}

    %{
    for y=1:obj_dimx*obj_dimz*numAnchors
        rec_vec4D(y,:) = rec_vec4D(y,:) - (0.4*numAnchors*dt)*double(rec_vec4D(y,:))*DyD;
    end
    %}

    %positivity constraint
    if positivity
        rec_vec4D(rec_vec4D<10)=0;
    end
end

%% Show 3D reconstructions of first and last anchors
figure(11); clf;
%rec_vec4D = reshape(rec_vec4D,[obj_dimx*obj_dimz*numAnchors,obj_dimy]);
rec = reshape(rec_vec4D, [obj_dimx, obj_dimz, numAnchors, obj_dimy]);
img(squeeze(rec(:,:,1,:)),'first 4D anchor', squeeze(rec(:,:,3,:)),'6th 4D anchor',...
    squeeze(rec(:,:,numAnchors,:)),'last 4D anchor', 'caxis', [0, max(rec(:))]);

% Show xz, yz, and xy projections of reconstructed 3D volume, tilt by tilt
figure(12); clf;
rec = reshape(rec_vec4D, [obj_dimx, obj_dimz, numAnchors, obj_dimy]);
rec = permute(rec,[1,2,4,3]);
pjYZ = squeeze(sum(permute(rec,[3,2,1,4]),3));
pjXY = squeeze(sum(permute(rec,[1,3,2,4]),3));
pjXZ = squeeze(sum(rec,3));
pjYZ = reshape(reshape(pjYZ,[obj_dimy*obj_dimz,numAnchors])*interpWeights',[obj_dimy, obj_dimz, numTilts]);
pjXY = reshape(reshape(pjXY,[obj_dimx*obj_dimy,numAnchors])*interpWeights',[obj_dimx, obj_dimy, numTilts]);
pjXZ = reshape(reshape(pjXZ,[obj_dimx*obj_dimz,numAnchors])*interpWeights',[obj_dimx, obj_dimz, numTilts]);
img(pjXZ,'4D recon proj XZ',pjYZ,'4D recon proj YZ',pjXY,'4D recon proj XY','size',[1 3]);

%% write video
writerObj2 = VideoWriter('results/rec_4D_3anchor.avi');
writerObj2.FrameRate = 10;
% set the seconds per image
secsPerImage = 5:5:numTilts*5;
% open the video writer
open(writerObj2);
% write the frames to the video
Fr(17) = struct('cdata',[],'colormap',[]);

for u=1:numTilts
    % convert the image to a frame
    frame_i =  mat2im(pjXY(:,:,u), jet(256), max(pjXY(:))) ;    

    [C, map] = rgb2ind(frame_i, 256);
    frame_ii = im2frame (C, map);
    Fr(u) = frame_ii;
    
    writeVideo(writerObj2, C);
end
% close the writer object
close(writerObj2);
 movie(Fr,2,5);
%% Show residual errors in projections
projectionErrors = zeros(Num_pj,1);
for l=1:dimy
    pj_1d = double(rec_vec4D(:,l));
    pj_cals(:,l) = AW*pj_1d;
end
residual = reshape(double(pj_cals - projections_vec),[dimx Num_pj dimy]);
residual = permute(residual,[1 3 2]);
pj_cals_3D = permute( reshape(pj_cals,[dimx Num_pj dimy]), [1,3,2]);

for pIndex = 1:Num_pj
    projectionErrors(pIndex) = norm(squeeze(residual(:,:,pIndex)),'fro')/norm(squeeze(projections(:,:,pIndex)),'fro');
end
figure(13);clf;
img(residual,'res',pj_cals_3D,'cal proj',projections,'meas proj', 'caxis', [0,max(projections(:))]);
title('4D recon residuals');
figure(14);clf;
plot(1:numTilts, projectionErrors);
title('4D recon residual errors');
xlabel('Projection (tilt) number');

if 1
    % For the 17 tilts with max motion 20%, compute tilt-by-tilt error with
    % respect to the ground truths.
    load model1groundTruth0to20pct_17tilts
    rec = reshape(rec, [obj_dimx*obj_dimz*obj_dimy,numAnchors]);
    errors4D = zeros(numTilts,1);
    norms4D = zeros(numTilts,1);
    for tiltNum = 1:numTilts
        recThisTilt = reshape(rec*interpWeights(tiltNum,:)',[obj_dimx*obj_dimz,obj_dimy]);
        errors4D(tiltNum) = norm(recThisTilt - rec_vecs_ground_truth{tiltNum},'fro')^2;
        norms4D(tiltNum) = norm(rec_vecs_ground_truth{tiltNum},'fro')^2;
    end
    fprintf('Total NMSE relative to ground truth: %.3f\n',sum(errors4D)/sum(norms4D));
    figure(16);clf;
    plot(1:numTilts,errors4D./norms4D);%expansionFactorsPct
    xlabel('Motion percentage');
    title('4D recon error relative to ground truth');
    ylabel('Normalized MSE');
end
    