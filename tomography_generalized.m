%% TEST OF OUR CODE BUILT AS IN IRtool EXTENSION

close all;
close all hidden;
clear;
clc;

warning('off','all');
clear classes
cd('AIRToolsII/');
AIRToolsII_setup;
cd('..');
cd('IRtools/');
IRtools_setup;
cd('..');
warning('on','all');

FID = fopen('results_tomo_residual_stopping_generalized.dat','w');

%% ITERATION
for theproblem = 1:2
    switch theproblem
        case 1
            N = 50;
            theta = 0:1:179;
            p = round(sqrt(2)*N);
            d = p-1;
            [A,bl,x_true,theta,p,d] = paralleltomo(N,theta,p,d,0,0);
            problemname = 'paralleltomo';
        case 2
            N = 70;
            [A,bl,x_true,s,p] = seismictomo(N,N,2*N,0,0);
            problemname = 'seismictomo';
    end
    fprintf(FID,'Tomography Experiment %s launched on: %s\n',problemname,datestr(now));
    for NoiseLevel = [1e-4,1e-3,1e-2,1e-1,2e-1,5e-1,6e-1]
        close all;
        close all hidden;
        fprintf(FID,'%1.1e',NoiseLevel);
        rng(0);                         % Initialize random number generator.
        e = randn(size(bl));            % Gaussian white noise.
        e = NoiseLevel*norm(bl)*e/norm(e);   % Scale the noise vector.
        b = bl + e;                     % Add the noise to the pure data.
        
        
        %% MATRIX FUNCTION REGULARIZATION
        options  = IRfun('defaults');
        options.RegParam = NoiseLevel*1e-1;
        options.x_true = x_true;
        options.NoStop = 'off';
        options.RegBeta = 1e+9;
        options.eta = 1.01;
        options.NoiseLevel = NoiseLevel;
        options.MaxIter = 200;
        options.verbosity = false;
        options.IterBar = 'on';
        options.RegType = 'generalized';
        
        tic;
        [X,infofun] = IRfun(A,b,options);
        timefun = toc;
        
        options.NoStop = 'on';
        options.verbosity = 0;
        options.IterBar = 'off';
        [~,infofunbest] = IRfun(A,b,options);
        
        fprintf(FID,'& %d (%d) & %1.1e & %1.2f (%1.2f) \\\\\n',...
            infofun.StopReg.It,infofunbest.BestReg.It,timefun,...
            psnr(reshape(infofun.StopReg.X,size(x_true)),x_true),...
            psnr(reshape(infofunbest.BestReg.X,size(x_true)),x_true));
        
        
        
        clear options
    end
end
fclose(FID);