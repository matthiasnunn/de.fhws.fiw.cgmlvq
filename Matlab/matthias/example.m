addpath( '..' );
addpath( '../benchmarks(PATH)' );
addpath( '../display(PATH)' );
addpath( '../fourier(PATH)' );


A = load( '../iris.mat' );  % csvread for complex valued data


X_train = A.fvec( 1:120, : );
y_train = A.lbl( 1:120, : );

X_test = A.fvec( 121:150, : );
y_test = A.lbl( 121:150, : );



%%% CGMLVQ with fft %%%

% [output_system, training_curves, param_set, backProts] = CGMLVQ( X_train, y_train, 2, 50, unique(y_train)' );

% [X_test] = Fourier( X_test, 2 );

% [crisp, ~] = classify_gmlvq( output_system, X_test, 1, y_test );

% display( crisp );



%%% CGMLVQ without fft %%%

[gmlvq_system] = run_single( X_train, y_train, 50, unique(y_train)' );

[crisp, ~, ~, ~] = classify_gmlvq( gmlvq_system, X_test, 1, y_test );

display( crisp );
