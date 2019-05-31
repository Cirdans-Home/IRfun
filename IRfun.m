function [X,info] = IRfun(A,b,varargin)
%IRFUN Regularization with the matrix function f(A)b.
%
% options  = IRfun('defaults')
% [X,info] = IRfun(A,b)
% [X,info] = IRfun(A,b,K)
% [X,info] = IRfun(A,b,options)
% [X,info] = IRfun(A,b,K,options)
%
% This function applies the polynomial Krylov algorithm for the computation
% of the matrix function f = 0.5(1+tanh(beta(x-alpha))/x on the matrix A
% times the vector b. We obtain a regularized solution by terminating the
% iterations.
%
% With 'defaults' as input returns the default options.  Otherwise outputs
% the iterates specified in K, using max(K) as MaxIter, and using all other
% default options.  With options as input: uses the user-specified options
% and all the other default options.
% Inputs:
%  A : either (a) a full or sparse matrix
%             (b) a matrix object that performs the matrix*vector operation
%             (c) user-defined function handle
%  b : right-hand side vector
%  K : (optional) integer vector that specifies which iterates are returned
%      in X; the maximum number of iterations is assumed to be max(K)
%      [ positive integer | vector of positive components ]
%  options : structure with the following fields (optional)
%      MaxIter    - maximum allowed number of iterations
%                   [ positive integer | {100} ]
%                   NOTE: K overrules MaxIter if both are assigned
%      x_true     - true solution; allows us to returns error norms with
%                   respect to x_true at each iteration
%                   [ array | {'none'} ]
%      NoiseLevel - norm of noise in rhs divided by norm of rhs
%                   [ {'none'} | nonnegative scalar]
%      eta        - safety factor for the discrepancy principle
%                   [ {1.01} | scalar greater than (and close to) 1 ]
%      RegParam   - Is the alpha value of the matrix function
%                   [ {0} | scalar greater than 0 but closer to 0 ]
%      RegBeta    - It the beta value of the matrix function, has to be
%                   smaller than alpha
%                   [ {1e-6} | scalar smaller than alpha greater than 0]
%      RegType    - the type of function that is computed inside the code,
%                   if 'classic' the function f_\alpha(A)b is computed
%                   assuming that A is a square matrix, or, the function
%                   handle of a square matrix, if 'normal' the function
%                   matrix f_\alpha(A^TA)A^Tb is computed, if A is a
%                   function handle then it has to admit a transpose flag
%                   such that A(x,'transp') computes A'*x, and
%                   A(x,'notransp') computes A*x, if 'generalized' then the
%                   generalized matrix function f_\alpha(A)b is computed
%                   for A a rectangular matrix.
%                   [ {'classic'} | 'normal' | 'generalized']
%      Reorth     - The Arnoldi iteration can be computed with or without
%                   reorthogonalization for enhancing the quality of the
%                   Krylov subspace.
%                   [{'on'} | 'off' ]
%      IterBar    - shows the progress of the iterations
%                   [ {'on'} | 'off' ]
%      NoStop     - specifies whether the iterations should proceed after
%                   a stopping criterion has been satisfied
%                   [ 'on' | {'off'} ]
% Note: the options structure can be inizialized with IRfun('defaults')
%
% Outputs:
%   X : computed solutions, stored column-wise (at the iterations listed in K)
%   info: structure with the following fields:
%      its      - number of the last computed iteration
%      saved_iterations - iteration numbers of iterates stored in X
%      StopFlag - a string that describes the stopping condition:
%                   * Reached maximum number of iterations
%                   * Residual tolerance satisfied (discrepancy principle)
%      StopReg  - struct containing information about the solution that
%                 satisfies the stopping criterion, with the fields:
%                   It   : iteration where the stopping criterion is satisfied
%                   X    : the solution satisfying the stopping criterion
%                   Enrm : the best relative error (requires x_true)
%      Rnrm     - relative residual norms at each iteration
%      Xnrm     - solution norms at each iteration
%      Enrm     - relative error norms (requires x_true) at each iteration
%      BestReg  - struct containing information about the solution that
%                 minimizes Enrm (requires x_true), with the fields:
%                   It   : iteration where the minimum is attained
%                   X    : best solution
%                   Enrm : best relative error
%
%
% This code is based upon the code by
% Silvia Gazzola, University of Bath
% Per Christian Hansen, Technical University of Denmark
% James G. Nagy, Emory University
% April, 2018.
% to be compatible and comparable with the IR Tools package, it is
% similarly distributed under the 3-Clause BSD License.
%
% A separate license file should be provided as part of the package.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
% "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
% TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
% PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
% HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
% SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
% TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
% OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
% OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
% NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
% SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

% Fabio Durastante
% Università di Pisa, Dipartimento di Informatica
% Stefano Cipolla
% Università di Padova, Dipartimento di Matematica
% May 2019

% Set default values for options.
defaultopt = struct('MaxIter',100 ,...
    'x_true','none',...
    'NoiseLevel','none',...
    'eta',1.01,...
    'IterBar','on', ...
    'NoStop','off', ...
    'RegParam',0,...
    'RegType','classic',...
    'RegBeta',1,...
    'Reorth','on',...
    'verbosity',true);

% If input is 'defaults,' return the default options in X.
if nargin==1 && nargout <= 1 && isequal(A,'defaults')
    X = defaultopt;
    return;
end

defaultopt.verbosity  = 'on';

% Check for acceptable number of optional input arguments.
switch length(varargin)
    case 0
        K = []; options = [];
    case 1
        if isa(varargin{1}, 'double')
            K = varargin{1}; options = [];
        else
            K = []; options = varargin{1};
        end
    case 2
        if isa(varargin{1}, 'double')
            K = varargin{1}; options = varargin{2};
        else
            K = varargin{2}; options = varargin{1};
        end
        if isfield(options, 'MaxIter') && ~isempty(options.MaxIter) && ...
                (~isempty(K) && options.MaxIter ~= max(K))
            warning('The value of MaxIter is discarded; the maximum value in K is taken as MaxIter')
        end
    otherwise
        error('Too many input parameters')
end

if isempty(options)
    options = defaultopt;
end

MaxIter    = IRget(options, 'MaxIter',    [], 'fast');
x_true     = IRget(options, 'x_true',     [], 'fast');
NoiseLevel = IRget(options, 'NoiseLevel', [], 'fast');
eta        = IRget(options, 'eta',        [], 'fast');
IterBar    = IRget(options, 'IterBar',    [], 'fast');
RegType    = IRget(options, 'RegType',    [], 'fast');
NoStop     = IRget(options, 'NoStop',     [], 'fast');
NoStop     = strcmp(NoStop, 'on');
verbose    = IRget(options, 'verbosity',  [], 'fast');
alphar     = IRget(options, 'RegParam',   [], 'fast');
beta       = IRget(options, 'RegBeta',    [], 'fast');
Reorth     = IRget(options, 'Reorth', [], 'fast');
Reorth     = strcmp(Reorth,'on');

if isempty(K)
    K = MaxIter;
end
% Sorting the iteration numbers (in case they are shuffled in input).
K = K(:); K = sort(K,'ascend'); K = unique(K);
if ~((isreal(K) && (all(K > 0)) && all(K == floor(K))))
    error('K must be a vector of positive real integers')
end
if K(end) ~= MaxIter
    MaxIter = K(end);
end

StopIt = MaxIter;

if isempty(NoiseLevel) || strcmp(NoiseLevel,'none')
    Rtol = 0;
else
    Rtol = eta*NoiseLevel;
end

%% Type of Computation
% The type of computation and the algorithms used depends on the RegType
% variable, and on the type of A that is given to the routine.

if strcmp(RegType,'classic')
    Amat = A;
    n = length(b);
elseif strcmp(RegType,'normal')
    if isa(A,'function_handle')
        Amat = @(x) A(A(x,'notransp'),'transp');
        b = A(b,'transp');
    else
        Amat = @(x) A'*(A*x);
        b = A'*b;
    end
    n = length(b);
elseif strcmp(RegType,'generalized')
    if isa(A,'function_handle')
        Amat = @(x) A(x,'notransp');
        Atransp = @(x) A(x,'transp');
    else
        Amat = @(x) A*x;
        Atransp = @(x) A'*x;
    end
    n = length(Atransp(b));
else
    error('The RegType has to be a classic, normal, or generalized');
end

Rtol = Rtol/norm(b);                         % Use the relative NoiseLevel

%% Prepare the output
% We have two type of output, one in the case in which the x_true has been
% given, and one in which it is not available
X = zeros(n,length(K));
Xnrm    = zeros(MaxIter,1);
Rnrm    = zeros(MaxIter,1);
saved_iterations = zeros(1, length(K));

if strcmp(x_true,'none')
    errornorms = false;
else
    errornorms = true;
    Enrm = zeros(MaxIter,1);
    nrmtrue = norm(x_true(:));
    BestReg.It = [];
    BestReg.X = [];
    BestReg.Xnrm = [];
    BestReg.Rnrm = [];
    BestReg.Enrm = [];
    BestEnrm = 1e10;
end

%% START THE MATRIX FUNCTION COMPUTATION
% This is the matrix function computation based on the Arnoldi Algorithm
noIterBar = strcmp(IterBar,{'off'});
if ~noIterBar
    h_wait = waitbar(0, 'Running iterations, please wait ...');
end
j = 0;
f = @(x) 0.5*(1+tanh((x-alphar).*beta))./x;

if strcmp(RegType,'classic') || strcmp(RegType,'normal')
    % In the case of either the classic or the normal equation approach the
    % Arnoldi based routine for the computation of f_\alpha(x) is used.
    if strcmp(RegType,'normal')
        Rtol = Rtol*1e-1;
    end
    nr = norm(b);
    v(:,1)=b/nr;
    for k=1:MaxIter
        if ~noIterBar
            waitbar(k/MaxIter, h_wait)
        end
        if isa(A,'function_handle') % Check if A is a matrix or a handle
            z = Amat(v(:,k));
        else
            z = Amat*v(:,k);
        end
        for l=1:k                   % Modified Grahm-Schmidt
            H(l,k)=v(:,l)'*z;
            z=z-H(l,k)*v(:,l);
        end
        if Reorth
            for i=1:k               % Reorth
                tmp=( v(:,i) )' * z;
                z = z - tmp*v(:,i);
                H(i,k)=H(i,k)+tmp;
            end
        end
        H(k+1,k) = norm(z);
        if abs(H(k+1,k)) > 1e-13
            v(:,k+1)=z/H(k+1,k);
        else
            % Lucky Breakdown: stop due to invariant subspace
            % Observe: we are never lucky, this is an overkill.
            tmp1=funm2(H(1:k,:),f);
            tmp2=nr*(tmp1(:,1));
            yout = v(:,1:k)*tmp2;
            % Compute norms.
            Xnrm(k)    = norm(yout);           % Norm of the Solution
            if isa(A,'function_handle')        % Is A a matrix or a handle?
                Rnrm(k) = norm(Amat(yout) - b)/nr;% Norm of the residual
            else
                Rnrm(k) = norm(Amat*yout - b)/nr; % Norm of the residual
            end
            
            if verbose
                disp('Lucky breakdown!')
            end
            StopFlag = 'Lucky breakdown: found invariant subspace';
            if ~AlreadySaved && ~NoStop
                j = j+1;
                X(:,j) = yout;
                saved_iterations(j) = k;
                AlreadySaved = 1;
            end
            StopIt = k;
            StopReg.It = k;
            StopReg.X = yout;
            StopReg.Xnrm = Xnrm(k);
            StopReg.Rnrm = Rnrm(k);
            if errornorms
                StopReg.Enrm = Enrm(k);
            end
            if ~ NoStop
                Xnrm    = Xnrm(1:k);
                Rnrm    = Rnrm(1:k);
                if errornorms
                    Enrm = Enrm(1:k);
                end
                X = X(:,1:j);
                break
            end
        end
        
        tmp1=funm2(H(1:k,:),f);
        tmp2=nr*(tmp1(:,1));
        yout = v(:,1:k)*tmp2;
        % Compute norms.
        Xnrm(k)    = norm(yout);        % Norm of the Solution
        if isa(A,'function_handle')     % Is A a matrix or a handle?
            Rnrm(k) = norm(Amat(yout) - b)/nr;      % Norm of the residual
        else
            Rnrm(k) = norm(Amat*yout - b)/nr;       % Norm of the residual
        end
        
        if k >= 2
            Stop1 = Rnrm(k);
            Stop2 = abs(Rnrm(k)-Rnrm(k-1));
        else
            Stop1 = 1;
            Stop2 = 1;
        end
        
        if verbose
            fprintf('Iteration %d Rnrm(%d) = %1.1e Residual Difference = %1.1e Rtol = %1.1e\n',...
                k,k,Rnrm(k),Stop2,Rtol);
        end
        if errornorms
            Enrm(k) = norm(x_true(:) - yout)/nrmtrue;
            if Enrm(k)<BestEnrm
                BestReg.It = k;
                BestReg.X = yout;
                BestReg.Xnrm = Xnrm(k);
                BestReg.Rnrm = Rnrm(k);
                BestEnrm = Enrm(k);
                BestReg.Enrm = BestEnrm;
            end
        end
        AlreadySaved = 0;
        % Save the iteration if one of the required by the K vector
        if any(k==K)
            j = j+1;
            X(:,j) = yout;
            saved_iterations(j) = k;
            if k == MaxIter
                AlreadySaved = 1;
            end
        end
        %% Stopping criterium
        if ( (Stop1 <= Rtol) || (Stop2 <= Rtol)  ) && (StopIt == MaxIter)
            if verbose
                if Stop1 <= Rtol
                    disp('Residual tolerance satisfied')
                elseif Stop2 <= Rtol
                    disp('Successive residuals near than the Residual tolerance');
                end
            end
            if Stop1 <= Rtol
                StopFlag = 'Residual tolerance satisfied';
            elseif Stop2 <= Rtol
                StopFlag = 'Successive Residual tolerance satisfied';
            end
            if ~AlreadySaved && ~NoStop
                j = j+1;
                X(:,j) = yout;
                saved_iterations(j) = k;
                AlreadySaved = 1;
            end
            StopIt = k;
            StopReg.It = k;
            StopReg.X = yout;
            StopReg.Xnrm = Xnrm(k);
            StopReg.Rnrm = Rnrm(k);
            if errornorms
                StopReg.Enrm = Enrm(k);
            end
            if ~ NoStop
                Xnrm    = Xnrm(1:k);
                Rnrm    = Rnrm(1:k);
                if errornorms
                    Enrm = Enrm(1:k);
                end
                X = X(:,1:j);
                break
            end
        end
    end
elseif strcmp(RegType,'generalized')
    % We are using the generalized matrix function, thus we move from the
    % Arnoldi based algorithm to the Golub-Kahan bidiagonalization
    alpha = zeros(MaxIter,1);
    beta = zeros(MaxIter+1,1);
    nr = norm(b,2);
    p=b./nr;
    beta(1)=1;
    U = zeros(size(b,1),MaxIter);
    V = zeros(size(Atransp(b),1),MaxIter);
    for k=1:MaxIter
        if ~noIterBar
            waitbar(k/MaxIter, h_wait)
        end
        
        if abs(beta(k))>1e-12
            U(:,k)=p./beta(k);
        else
            % Lucky Breakdown: stop due to invariant subspace
            % Observe: we are never lucky, this is an overkill.
            % Compute norms.
            
            E=spdiags([beta(2:k+1) alpha],-1:0, k,k); %E=U'*A*V;
            [Ur,S,Vr] = svds(E,size(E,1));
            tmp = Vr*diag(f(diag(S)))*Ur';
            yout = V(:,1:k)*(tmp*(U(:,1:k)'*b));
            Xnrm(k)    = norm(yout);              % Norm of the Solution
            Rnrm(k) = norm(Amat(yout) - b);       % Norm of the residual
            
            if verbose
                disp('Lucky breakdown!')
            end
            StopFlag = 'Lucky breakdown: found invariant subspace';
            if ~AlreadySaved && ~NoStop
                j = j+1;
                X(:,j) = yout;
                saved_iterations(j) = k;
                AlreadySaved = 1;
            end
            StopIt = k;
            StopReg.It = k;
            StopReg.X = yout;
            StopReg.Xnrm = Xnrm(k);
            StopReg.Rnrm = Rnrm(k);
            if errornorms
                StopReg.Enrm = Enrm(k);
            end
            if ~ NoStop
                Xnrm    = Xnrm(1:k);
                Rnrm    = Rnrm(1:k);
                if errornorms
                    Enrm = Enrm(1:k);
                end
                X = X(:,1:j);
                break
            end
        end
        
        if k==1
            r=Atransp(U(:,k));
        else
            r=Atransp(U(:,k))-beta(k).*V(:,k-1);
            for i=2:k
                tmp=V(:,i)'*r;
                r=r-tmp.*V(:,i);
            end
        end
        
        alpha(k)=norm(r);
        V(:,k)=r./alpha(k);
        p=Amat(V(:,k))-alpha(k).*U(:,k);
        for i=1:k
            tmp=U(:,i)'*p;
            p=p-tmp.*U(:,i);
        end
        beta(k+1)=norm(p,2);
        
        E=spdiags([beta(2:k+1) alpha(1:k)],-1:0, k,k); %E=U'*A*V;
        [Ur,S,Vr] = svds(E,size(E,1));
        tmp = Vr*diag(f(diag(S)))*Ur';
        yout = V(:,1:k)*(tmp*(U(:,1:k)'*b));
        
        % Compute norms.
        Xnrm(k)    = norm(yout);                     % Norm of the Solution
        Rnrm(k) = norm(Amat(yout) - b)/nr;           % Norm of the residual
        
        if k >= 2
            Stop1 = Rnrm(k);
            Stop2 = abs(Rnrm(k)-Rnrm(k-1));
        else
            Stop1 = 1;
            Stop2 = 1;
        end
        
        if verbose
            fprintf('Iteration %d Rnrm(%d) = %1.1e Stop = %1.1e Rtol = %1.1e\n',...
                k,k,Rnrm(k),Stop2,Rtol);
        end
        if errornorms
            Enrm(k) = norm(x_true(:) - yout)/nrmtrue;
            if Enrm(k)<BestEnrm
                BestReg.It = k;
                BestReg.X = yout;
                BestReg.Xnrm = Xnrm(k);
                BestReg.Rnrm = Rnrm(k);
                BestEnrm = Enrm(k);
                BestReg.Enrm = BestEnrm;
            end
        end
        AlreadySaved = 0;
        % Save the iteration if one of the required by the K vector
        if any(k==K)
            j = j+1;
            X(:,j) = yout;
            saved_iterations(j) = k;
            if k == MaxIter
                AlreadySaved = 1;
            end
        end
        %% Stopping criterium
        if ( (Stop1 <= Rtol) || (Stop2 <= Rtol)  ) && (StopIt == MaxIter)
            if verbose
                if Stop1 <= Rtol
                    disp('Residual tolerance satisfied')
                elseif Stop2 <= Rtol
                    disp('Successive residuals near than the Residual tolerance');
                end
            end
            if Stop1 <= Rtol
                StopFlag = 'Residual tolerance satisfied';
            elseif Stop2 <= Rtol
                StopFlag = 'Successive Residual tolerance satisfied';
            end
            if ~AlreadySaved && ~NoStop
                j = j+1;
                X(:,j) = yout;
                saved_iterations(j) = k;
                AlreadySaved = 1;
            end
            StopIt = k;
            StopReg.It = k;
            StopReg.X = yout;
            StopReg.Xnrm = Xnrm(k);
            StopReg.Rnrm = Rnrm(k);
            if errornorms
                StopReg.Enrm = Enrm(k);
            end
            if ~ NoStop
                Xnrm    = Xnrm(1:k);
                Rnrm    = Rnrm(1:k);
                if errornorms
                    Enrm = Enrm(1:k);
                end
                X = X(:,1:j);
                break
            end
        end
    end
else
    error('The RegType has to be a classic, normal, or generalized');
end


if k == MaxIter
    if StopIt == MaxIter
        % Stop because max number of iterations reached.
        if verbose
            disp('Reached maximum number of iterations')
        end
        StopFlag = 'Reached maximum number of iterations';
        if ~AlreadySaved
            j = j+1;
            X(:,j) = yout;
            saved_iterations(j) = k;
        end
        StopReg.It = k;
        StopReg.X = yout;
        StopReg.Xnrm = Xnrm(k);
        StopReg.Rnrm = Rnrm(k);
        if errornorms
            StopReg.Enrm = Enrm(k);
        end
        Xnrm    = Xnrm(1:k);
        Rnrm    = Rnrm(1:k);
        if errornorms
            Enrm = Enrm(1:k);
        end
        X = X(:,1:j);
        saved_iterations = saved_iterations(1:j);
    end
end

if ~noIterBar, close(h_wait), end
if nargout==2
    info.its = k;
    info.saved_iterations = saved_iterations(1:j);
    info.StopFlag = StopFlag;
    info.StopReg = StopReg;
    info.Rnrm = Rnrm(1:k);
    info.Xnrm = Xnrm(1:k);
    if errornorms
        info.Enrm = Enrm(1:k);
        info.BestReg = BestReg;
    end
end

end

%% Schur-Parlet Algorithm:
function [F,esterr] = funm2(A,fun)
%FUNM Evaluate general matrix function.
%   F = FUNM(A,FUN) for a square matrix argument A, evaluates the
%   matrix version of the function FUN. For matrix exponentials,
%   logarithms and square roots, use EXPM(A), LOGM(A) and SQRTM(A)
%   instead.
%
%   FUNM uses a potentially unstable algorithm.  If A is close to a
%   matrix with multiple eigenvalues and poorly conditioned eigenvectors,
%   FUNM may produce inaccurate results.  An attempt is made to detect
%   this situation and print a warning message.  The error detector is
%   sometimes too sensitive and a message is printed even though the
%   the computed result is accurate.
%
%   [F,ESTERR] = FUNM(A,FUN) does not print any message, but returns
%   a very rough estimate of the relative error in the computed result.
%
%   If A is symmetric or Hermitian, then its Schur form is diagonal and
%   FUNM is able to produce an accurate result.
%
%   L = LOGM(A) uses FUNM to do its computations, but it can get more
%   reliable error estimates by comparing EXPM(L) with A.
%   S = SQRTM(A) and E = EXPM(A) use completely different algorithms.
%
%   Example
%      FUN can be specified using @:
%         F = funm(magic(3),@sin)
%      is the matrix sine of the 3-by-3 magic matrix.
%
%   See also EXPM, SQRTM, LOGM, @.

%   C.B. Moler 12-2-85, 7-21-86, 7-11-92, 5-2-95.
%   Copyright 1984-2001 The MathWorks, Inc.
%   $Revision: 5.15 $  $Date: 2001/04/15 12:01:37 $

% Parlett's method.  See Golub and VanLoan (1983), p. 384.

if isstr(A), error('The first argument must be a matrix.'); end

[Q,T] = schur(A);
[Q,T] = rsf2csf(Q,T);
F = diag(feval(fun,diag(T)));
[n,~] = size(A);
dmin = abs(T(1,1));
for p = 1:n-1
    for i = 1:n-p
        j = i+p;
        s = T(i,j)*(F(j,j)-F(i,i));
        if p > 1
            k = i+1:j-1;
            s = s + T(i,k)*F(k,j) - F(i,k)*T(k,j);
        end
        d = T(j,j) - T(i,i);
        if d ~= 0
            s = s/d;
        end
        F(i,j) = s;
        dmin = min(dmin,abs(d));
    end
end
F = Q*F*Q';
if isreal(A) && norm(imag(F),1) <= 10*n*eps*norm(F,1)
    F = real(F);
end

esterr=0;
end

