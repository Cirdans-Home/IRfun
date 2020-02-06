%% USING THE MATRIX FUNCTION ROUTINE TO DENOISE THE IMAGE:

clear; clc;

warning('off','all');
clear classes
cd('AIRToolsII/');
AIRToolsII_setup;
cd('..');
cd('IRtools/');
IRtools_setup;
cd('..');
warning('on','all');

FID = fopen('results_IRfun.dat','a');
fprintf(FID,'Experiment run on %s\n',datestr(now));

%% Launch iteration:

for imagename = {'phillips','gravity'} % ,
    fprintf(FID,'TEST CASE %s \n',imagename{1});
    %% Generate the blurred and noised signal:
    [A,bl,x_true] = feval(imagename{1},4*2^8);
    
    for NoiseLevel = [1e-3,1e-2,1e-1,3e-1,6e-1]
        fprintf(FID,'%1.1e',NoiseLevel);
        [b, NoiseInfo] = PRnoise(bl, 'gauss',NoiseLevel);
        
        %% Apply the CGLS to restore the signal:
        fprintf('CGLS Algorithm:\n');
        try
            options  = IRcgls('defaults');
            options.x_true = x_true;
            options.NoiseLevel = NoiseLevel;
            options.verbosity = 'off';
            options.IterBar = 'off';
            tic;
            [Xcgls,infocgls] = IRcgls(A,b,options);
            timecgls = toc;
            
            options.NoStop = 'on';
            
            [~,infocglsbest] = IRcgls(A,b,options);
            
            fprintf(FID, ' & %d (%d) & %1.1e & %1.2f (%1.2f)',...
                infocgls.StopReg.It,infocglsbest.BestReg.It,timecgls,...
                psnr(reshape(infocgls.StopReg.X,size(x_true)),x_true),...
                psnr(reshape(infocglsbest.BestReg.X,size(x_true)),x_true));
        catch
            fprintf(FID, ' & -- & -- & --');
        end
        %% Apply the Range-Restricted GMRES
        fprintf('\nRangeRestricted GMRES Algorithm:\n');
        try
            options  = IRrrgmres('defaults');
            options.x_true = x_true;
            options.NoiseLevel = NoiseLevel;
            options.verbosity = 'off';
            options.IterBar = 'off';
            tic;
            [Xrrgmres,inforrgmres] = IRrrgmres(A,b,options);
            timerrgmres = toc;
            
            options.NoStop = 'on';
            [~,inforrgmresbest] = IRrrgmres(A,b,options);
            
            fprintf(FID,' & %d (%d) & %1.1e & %1.2f (%1.2f)', ...
                inforrgmres.StopReg.It,...
                inforrgmresbest.BestReg.It,timerrgmres,...
                psnr(reshape(inforrgmres.StopReg.X,size(x_true)),x_true),...
                psnr(reshape(inforrgmresbest.BestReg.X,size(x_true)),x_true));
        catch
            fprintf(FID, ' & -- & -- & --');
        end
        %% CGLS for Tikhonov method
        fprintf('\nTikhonov CGLS method:\n');
        try
            options  = IRcgls('defaults');
            options.x_true = x_true;
            options.verbosity = 'off';
            options.IterBar = 'off';
            options.NoiseLevel = 'none';
            tic;
            options.RegParam = settikhonov(A,b);
            [Xtikhonov,infotikhonov] = IRcgls(A,b,options);
            timetikhonov = toc;
            
            fprintf(FID,'& %1.2e & %d & %1.1e & %1.2f',...
                options.RegParam,infotikhonov.StopReg.It,timetikhonov,...
                psnr(reshape(infotikhonov.StopReg.X,size(x_true)),x_true));
        catch
            fprintf(FID, ' & -- & -- & --');
        end
        %% Apply the matrix-function denoiser:
        fprintf('\nMatrix Function Regularizer:\n');
        try
            options  = IRfun('defaults');
            options.RegParam = NoiseLevel*1e-1;
            options.x_true = x_true;
            options.NoStop = 'off';
            options.RegBeta = 1e+6;
            options.verbosity = 0;
            options.NoiseLevel = NoiseLevel;
            options.MaxIter = 100;
            options.IterBar = 'off';
            options.Reorth = 'on';
            options.eta = 1.01;
            if norm(A-A') > 1e-6
                options.RegType = 'arnoldi';
            else
                options.RegType = 'lanczos';
            end
            fprintf(FID,' & %1.1e',options.RegParam);
            
            tic;
            [X,infofun] = IRfun(A,b,options);
            timefun = toc;
            
            options.NoStop = 'on';
            [~,infofunbest] = IRfun(A,b,options);
            
            fprintf(FID,'& %d (%d) & %1.1e & %1.2f (%1.2f)',...
                infofun.StopReg.It,infofunbest.BestReg.It,timefun,...
                psnr(reshape(infofun.StopReg.X,size(x_true)),x_true),...
                psnr(reshape(infofunbest.BestReg.X,size(x_true)),x_true));
        catch
            fprintf(FID, ' & -- & -- & --');
        end
        fprintf(FID,'\\\\\n');
    end
end
fprintf(FID,'\n\n\n\n');

fclose(FID);


