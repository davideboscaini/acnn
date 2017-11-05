clear;
close all;
clc;

paths.main = '/media/dboscaini/data/refactored/acnn_data/datasets/FAUST_registrations'; % path to the dataset
params.flag_recompute = 1;

% convert to '.mat' format
paths.input = fullfile(paths.main,'meshes','orig');
params.format_in = 'off';
params.format_out = 'mat';
paths.output = fullfile(paths.main,'meshes','orig');
run_compute_format_conversion(paths,params);

% rescale to unit diameter
paths.input = fullfile(paths.main,'meshes','orig');
paths.output = fullfile(paths.main,'meshes','diam=001');
run_rescale_shape(paths);

% rescale to diameter 200
paths.input = fullfile(paths.main,'meshes','diam=001');
params.flag_compute_scale_factor_to_unit_diam = 0;
params.avoid_geods = 0;
params.scale_factor = 200;
paths.output = fullfile(paths.main,'meshes','diam=200');
run_rescale_shape(paths,params);

% rotate the shape s.t. its maxium elongation will be along axis z
% and traslate it s.t. mean(shape.X)=0, mean(shape.Y)=0, min(shape.Z)=0
paths.input = fullfile(paths.main,'meshes','diam=200');
params.theta = 180;
params.idxs = [1,3,2];
params.signs = [-1,1,1];
paths.output = fullfile(paths.main,'meshes','diam=200');
run_compute_rototranslation(paths,params);

% compute the anisotropic Laplace-Beltrami operator
paths.input = fullfile(paths.main,'meshes','diam=200');
params.angles = linspace(0,pi,17);
params.angles = params.angles(1:end-1);
params.alpha = 100;
params.curv_smooth = 10;
paths.output = fullfile(paths.main,'data','diam=200','albos');
run_compute_albo(paths,params);

% compute the (anisotropic) eigen-decomposition
paths.input = fullfile(paths.main,'data','diam=200','albos');
params.k = 10 %00;
params.shift = 'SM';
paths.output = fullfile(paths.main,'data','diam=200','aeigendec',sprintf('k=%04d',params.k));
run_compute_aeigendec(paths,params);

% compute the anisotropic patches 
paths.input = fullfile(paths.main,'meshes','diam=200');
paths.eigendec = fullfile(paths.main,'data','diam=200','aeigendec',sprintf('k=%04d',params.k));
params.angles = linspace(0,pi,17);
params.angles = params.angles(1:end-1);
params.tvals = linspace(6,24,5);
params.flag_L2_norm  = 0;
params.flag_L1_norm  = 1;
params.flag_max_norm = 0;
params.thresh = 99.9;
params.flag_between_0_and_1 = 0;
paths.output = fullfile(paths.main,'data','diam=200','patch_aniso',...
    sprintf('alpha=%03.0f_nangles=%03.0f_ntvals=%03.0f_tmin=%03.3f_tmax=%03.3f_thresh=%03.3f_norm=L1',...
    params.alpha,length(params.angles),length(params.tvals),min(params.tvals),max(params.tvals),params.thresh));
run_compute_aniso_patches(paths,params);
