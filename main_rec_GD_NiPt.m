
%addpath('src');
%addpath('cuda_code');
projections  = importdata([pwd '/data/NiPt_180823AA_t3_ys_gd_ms_pj_test3_allsurf_Pt_randmove.mat' ]);
angles       = importdata([pwd '/data/NiPt_180823AA_t3_ys_gd_spra2_refangle.mat' ]);

%% input
%projections = max(projections(:)) - projections;
rotation       = 'ZYX';  % Euler angles setting ZYZ
dtype          = 'single';
projections = cast(projections,dtype);
angles      = cast(angles,dtype);

% compute normal vector of rotation matrix
matR = zeros(3,3);
if length(rotation)~=3
    disp('rotation not recognized. Set rotation = ZYX\n'); rotation = 'ZYX';
end
for i=1:3
    switch rotation(i)
        case 'X',   matR(:,i) = [1;0;0];
        case 'Y',   matR(:,i) = [0;1;0];
        case 'Z',   matR(:,i) = [0;0;1];
        otherwise,  matR = [0,0,1;
                0,1,0;
                1,0,0];
            disp('Rotation not recognized. Set rotation = ZYX');
            break
    end
end
vec1 = matR(:,1); vec2 = matR(:,2); vec3 = matR(:,3);

% extract size of projections & num of projections
[dimx, dimy, Num_pj] = size(projections);

%% rotation matrix
Rs = zeros(3,3,Num_pj, dtype);
for k = 1:Num_pj
    phi   = angles(k,1);
    theta = angles(k,2);
    psi   = angles(k,3);
    
    % compute rotation matrix R w.r.t euler angles {phi,theta,psi}
    rotmat1 = MatrixQuaternionRot(vec1,phi);
    rotmat2 = MatrixQuaternionRot(vec2,theta);
    rotmat3 = MatrixQuaternionRot(vec3,psi);
    R =  single(rotmat1*rotmat2*rotmat3)';
    Rs(:,:,k) = R;
end

%% parameter
step_size      = 2.;  %step_size <=1 but can be larger is sparse
iterations     = 200;
dimz           = dimx;
positivity     = true;

%% iteration: minimize ||Au-b||^2 by gradient descent: run reconstruction the first time
% syntax 1: no initial rec
tic
[rec1] = RT3_1GPU( (projections), (Rs), (dimz), (iterations), (step_size) , (positivity));
toc
%% calculate projection
calprj = calculate3Dprojection_RT3_1GPU(rec1, Rs);
figure(1); img(rec1,'RESIRE rec' , 'caxis',[0,max(rec1(:))]);
figure(2); img(projections,'measured projections', calprj,'calculated projection', projections-calprj,'residual', ...
    'caxis',[min(projections(:)),max(projections(:))])


















