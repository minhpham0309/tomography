%%
addpath('functions')
%%
model = double( ReadMRC('data\vesicle.mrc') );
model = permute(model,[2,3,1]);
% model need to have odd dimension size in order for FST and RT are aligned
model = croppedOut(model, [63,63,63]);
%figure(1); img(model);
model_2d = squeeze( sum(model,2) );
model_2d = circshift(model_2d, [0,0]);
[dimx,dimz] = size(model_2d);
dimy = size(model,2);
%figure(2); img( model_2d );

o_ratio=3;

%model_pad3D = My_paddzero( model, [o_ratio*dimx,o_ratio*dimy,o_ratio*dimz]);
%model_K3D = my_fft( model_pad3D );

%% simulate projections by Radon transform (RT)
% rotation angle {theta}
thetas = -90:3:72; 

model_2d_rot = model_2d'; %model_2d_rot = model_2d_rot(end:-1:1, :);
model_RT = radon( model_2d_rot, thetas); %model_RT = circshift(model_RT, [1,0]);
%figure(3); img(model_RT);

projs_radon = croppedOut(model_RT, [dimx, length(thetas)]);
rec_FBP = iradon(model_RT, thetas);
figure(2); img(model_2d,'model',rec_FBP','FBP reconstruction')
%figure(4); img(permute(radon(model_2d, -90:3:87),[2,1]), 'sinogram', 'colormap','gray');
%% simulate projections by FST
dimx_big = round(o_ratio*dimx); if mod(dimx_big,2)==1, dimx_big=dimx_big+1;end
dimz_big = round(o_ratio*dimz); if mod(dimz_big,2)==1, dimz_big=dimz_big+1;end
ncx_big = round((dimx_big+1)/2); 
ncz_big = round((dimz_big+1)/2); 

model_pad = My_paddzero( model_2d, [dimx_big,dimz_big]);
model_K = my_fft( model_pad );

X_big =  single( (1:dimx_big) - ncx_big );
Num_pj = length(thetas);
projs_FST = zeros( dimx_big, Num_pj);

for k=1:Num_pj
    theta = thetas(k);
    R = [cosd(theta),  -sind(theta);
         sind(theta),  cosd(theta) ];  
    
    rotCoords = (R(1,:))' * X_big;
    xj = rotCoords(1,:)'  + ncx_big;
    zj = rotCoords(2,:)'  + ncz_big; 
    proj_k = interp2( model_K, zj, xj, 'linear') ;
    proj_k(isnan(proj_k)) = 0;
    projs_FST(:,k) = proj_k;
end
projs_FST = ifftshift( ifft( fftshift(projs_FST) ) );
projs_FST = croppedOut( projs_FST, [dimx,Num_pj] );
projs_FST = single( max( real(projs_FST), 0 ) );
%projs_FST = circshift(projs_FST, [0,0]);

figure(3); img(projs_radon, 'sinogram by RT', projs_FST,'sinogram by FST', ...
    projs_radon-projs_FST,'difference', 'caxis', [0,max(projs_FST(:))])
sum( abs( projs_FST(:) - projs_radon(:) ) ) / sum( abs(projs_radon(:)) )

%% simulate 3D-FST first, then compute projection
%{
Num_pj = length(thetas);
projs_FST_3D = zeros(o_ratio*dimx,o_ratio*dimy, length(thetas) );

for k = 1:Num_pj
    projs_FST_3D(:,:,k) = calculate3Dprojection_splinterp( model_K3D, 0,thetas(k),0 );
  
end
projs_FST_3D = croppedOut(projs_FST_3D, [dimx,dimy, length(thetas)]);
projs_FST_2d = squeeze(sum(projs_FST_3D,2));
%figure(2); img(projs_radon, '', projs_FST,'',...
%    projs_FST_2d,'',projs_FST_2d-projs_radon,'', 'caxis', [0,max(projs_FST(:))])
%}

%%

save('data\vesicle_2D.mat','projs_radon', 'projs_FST','thetas','model_2d')
