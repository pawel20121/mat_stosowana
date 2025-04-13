function [h, error_history] = irls_filter_design(N, wc, p, transition_width)
    % Implements IRLS algorithm for P-norm optimal linear-phase FIR filter
    % 
    % Parameters:
    %   N: filter order (filter length = N+1)
    %   wc: cutoff frequency in radians
    %   p: p-norm parameter
    %   transition_width: width of transition band in radians
    %
    % Returns:
    %   h: filter coefficients
    %   error_history: error at each iteration
    
    % Number of frequency points for discretization
    num_points = 1024;
    
    % Frequency grid from 0 to π
    w = linspace(0, pi, num_points);
    
    % Define desired frequency response (ideal low-pass)
    Hd = zeros(1, num_points);
    Hd(w <= wc) = 1;
    
    % Define transition band for weighting
    transition_band = (w >= (wc - transition_width/2)) & (w <= (wc + transition_width/2));
    
    % Create weighting function (we give less weight to transition band)
    weight = ones(1, num_points);
    weight(transition_band) = 0.01;  % Lower weight in transition band
    
    % For linear phase, we only need to calculate half of the coefficients
    % For even order N, we need (N/2)+1 coefficients
    % For odd order N, we need (N+1)/2 coefficients
    if mod(N, 2) == 0
        M = N/2 + 1;
    else
        M = (N+1)/2;
    end
    
    % Create cosine matrix for linear phase
    A = zeros(num_points, M);
    for i = 1:num_points
        for k = 1:M
            if k == 1 && mod(N, 2) == 0
                A(i, k) = cos((k-1) * w(i) * (N/2));
            else
                if mod(N, 2) == 0
                    A(i, k) = 2*cos((k-1) * w(i) * (N/2));
                else
                    A(i, k) = 2*cos((k-1-0.5) * w(i) * (N/2));
                end
            end
        end
    end
    
    % Initialize weight matrix S
    S = eye(num_points);
    
    % Maximum number of iterations
    max_iter = 50;
    
    % Convergence tolerance
    tol = 1e-6;
    
    % Initialize error history
    error_history = zeros(1, max_iter);
    
    % Start IRLS iterations
    for iter = 1:max_iter
        % Apply current weights to the problem
        W = S' * S;
        W_diag = diag(W);
        
        % Create weighted matrices
        A_weighted = A .* sqrt(W_diag)';
        Hd_weighted = Hd .* sqrt(W_diag)';
        
        % Compute coefficient vector using weighted least squares
        c = (A_weighted' * A_weighted) \ (A_weighted' * Hd_weighted');
        
        % Compute error at current iteration
        e = A * c - Hd';
        error_p_norm = sum(weight .* abs(e).^p)^(1/p);
        error_history(iter) = error_p_norm;
        
        % Check convergence
        if iter > 1 && abs(error_history(iter) - error_history(iter-1)) < tol
            fprintf('Converged after %d iterations\n', iter);
            break;
        end
        
        % Update weight matrix S for next iteration
        % weights are |e_i|^(p-2)
        w_new = abs(e).^(p-2);
        
        % Handle very small errors to avoid numerical issues
        w_new(w_new > 1e10) = 1e10;
        
        % Update S matrix
        S = diag(w_new);
    end
    
    % Extract full filter coefficients from c (restore linear phase)
    h = zeros(1, N+1);
    if mod(N, 2) == 0
        % Even order filter (symmetric)
        h(1:M) = c';
        h(N+1:-1:N+2-M) = c';
    else
        % Odd order filter (symmetric)
        h(1:M) = c';
        h(N+1:-1:M+1) = c(2:end)';
    end
    
    fprintf('Final error (p-norm): %f\n', error_history(iter));
end