%% Gradient Descent on tomography
addpath('functions');
%% inputs
% FePt atom model
%projections  = importdata('data/FePt_projections.mat');
%angles       = importdata('data/FePt_angles.mat');

% vesicle model
%data        = importdata('data\circle_2D.mat');
data        = importdata('data\vesicle_2D.mat');
angles      = data.thetas;
model       = data.model_2d;    %model = croppedOut(model,[50,25]);
projections = data.projs_radon; %projections = croppedOut(projections, [50,55]);
% you can try projections simulated by FST but it does not work well
%projections = data.projs_FST;

%projections = My_paddzero(projections, [72,64,61]);
%% parameter
step_size      = 2.;  %step_size <=1 but can be larger if sparse
iterations     = 4000;
showim         = true;
%% pre-process
projections = cast(projections,'double');

% extract size of projections & num of projections
[dimx, Num_pj] = size(projections);

% choose z-dimension size: dimz=dimx, or any arbitrary value
dimz     = dimx;  

% stepsize
dt  =(step_size/Num_pj/dimz);

% cartesian grid in 3d fourier space
ncx = round((dimx+1)/2); 
ncz = round((dimz+1)/2); 

%% compute matrix A
[ZZ,XX] = meshgrid( (1:1:dimz) - ncz, (1:1:dimx) - ncx );
XX = XX(:)'; ZZ = ZZ(:)';
Coord = [XX;ZZ];
clear XX ZZ

o_ratio = 4;
num_pts = size(Coord,2) ;
vec_ind = ( 1:num_pts )';
rot_mat_k = cell(Num_pj, 1);

tic
for k = 1:Num_pj
    theta = angles(k);
    
    % rotation matrix R with angle theta
    R = [cosd(theta), -sind(theta);
         sind(theta),  cosd(theta) ];    
    
    % compute rotated coordinate
    rotCoords = R(1,:) * Coord;    
    rot_mat_k{k} = sparse( dimx,num_pts) ;
    % each pt is split into 4 pts with coord shift +/- 0.25 in each direction
    for s=1:o_ratio
        [s1,s2] = ind2sub([2,2],s);
        % shift coordinate +/- 0.25 in each direction
        rotCoords_shift = R(1,:)* [ (-1)^s1; (-1)^s2 ] *.25 ;
    
        rot_x   = rotCoords + rotCoords_shift + ncx ; 
        rot_x   = rot_x';

        % coefficients of linear interpolation
        %{
        x1 = floor(rot_x); x2 = x1+1;
        goodInd = x1>=1 & x2<=dimx | x1==dimx;
        x2(x1==dimx) = dimx;
        x1(x1==dimx) = dimx-1;
        rot_x       = rot_x(goodInd); 
        vec_goodInd = vec_ind(goodInd);
        x1 = x1(goodInd); x2 = x2(goodInd);    
        b1 = x2-rot_x   ; b2 = rot_x-x1;
        
        masterSub = [ [x1;x2], [vec_goodInd;vec_goodInd] ] ;
        masterVal = [b1;b2];
        %}
        %
        % coefficients of linear interpolation
        x_foor = floor(rot_x);        
        goodInd = x_foor>=1 & x_foor<dimx;
        vec_goodInd1 = vec_ind(goodInd);
        x1 = x_foor(goodInd);
        b1 = x1 + 1 - rot_x(goodInd)   ;
        
        goodInd = x_foor==0;
        vec_goodInd2 = vec_ind(goodInd);
        x2 = x_foor(goodInd)+1;  %x2=1
        b2 = rot_x(goodInd)   ;        
        
        goodInd = x_foor==dimx;
        vec_goodInd3 = vec_ind(goodInd);
        x3 = x_foor(goodInd);
        b3 = 1+dimx - rot_x(goodInd) ;
        
        masterSub = [ [x1;x1+1; x2;x3], ...
            [vec_goodInd1;vec_goodInd1;vec_goodInd2;vec_goodInd3] ] ;
        masterVal = [b1;1-b1; b2;b3];
        %}
        rot_mat_k{k} = rot_mat_k{k} + accumarray(masterSub, masterVal, [dimx,num_pts],[],[], true );
    end        
end
toc
A = cell2mat( rot_mat_k )/4;
%At = transpose(A);
clear b1 b2 x1 x2 Coord masterSub masterVal rot_mat_k rotCoords ...
    rot_x vec_ind vec_goodInd goodInd
%% iteration: minimize ||Au-b||^2 by gradient descent
% initialize object
rec_vec     = zeros(dimx*dimz  ,1 );
pj_cals     = zeros(dimx*Num_pj,1);
%%
tic
for iter=1:iterations
    % forward projections: compute pj_cals = Au    
    pj_cals =  A* rec_vec;
    
    % back projections: compute residual = Au-b, and grad = A^T(Au-b)
    residual =  pj_cals - projections(:) ;
    grad = (residual'*A)' ;
    rec_vec = rec_vec - (dt)*grad;
    rec_vec = max(0,rec_vec);    
    
    % compute relative error   
    if mod(iter,100)==0        
        errF   = norm( residual  , 1)/ norm(projections(:),1);
        errF2  = norm( residual  , 2)/ norm(projections(:),2);
        l2_err = norm(rec_vec-model(:))/norm(model,'fro');
        fprintf('%d.R1=%.4f R2=%.4f PSNR=%0.2f l2_err=%.4f\n',iter, errF, errF2, psnr(rec,model), l2_err );
    end 
    
    % show result
    if showim && mod(iter,100)==1
        rec = reshape(rec_vec, [dimx,dimz]);
        figure(4012); img(model,'model',rec,'rec',model-rec,'diff', 'caxis',[0,max(model(:))]);
        drawnow();
    end

end
toc
%% show result
rec_rt = rec;


