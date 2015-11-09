package multivariatelr;

import Jama.Matrix;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

class LogReg extends MultivariateLR{
	


	/**
     * GRADIENTDESCENTMULTI Performs gradient descent to learn theta
        theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
        taking num_iters gradient steps with learning rate alpha
     * @param X
     * @param y
     * @param theta
     * @param alpha
     * @param numIterations
     * @return 
     */
    GradientDescentValues gradientDescentMulti(Matrix X, Matrix y, Matrix theta, double alpha, int numIterations){
        //Write equivalent Java code for the Octave code below.
        
        //Initialize some useful values.
        //Octave: m = length(y); % number of training examples

        //create a matrix that stores cost history
        //Octave: J_history = zeros(num_iters, 1);

        //Loop thru numIterations
        //Octave:for iter = 1:num_iters
        
        // ====================== YOUR CODE HERE ======================
        // Instructions: Perform a single gradient step on the parameter vector
        //               theta. 
        //
        // Hint: While debugging, it can be useful to print out the values
        //       of the cost function (computeCostMulti) and gradient here.
        //
        GradientDescentValues GDV = new GradientDescentValues();
        int m = y.getRowDimension();
        Matrix jHistory = new Matrix(numIterations, 1);
        
        for(int iter=0; iter<numIterations ;iter++){
            
            jHistory.set(iter,0, costFunctionGD(X, y, theta));
            
            Matrix hypothesis = X.times(theta);
            Matrix error = hypothesis.minus(y);
            Matrix difference = error.times(alpha/m);
            Matrix changeTheta = X.transpose().times(difference);
            theta = theta.minus(changeTheta);
           
        } 
        
        GDV.setTheta(theta);
        GDV.setCostHistory(jHistory);
        return GDV;

        // Save the cost J in every iteration    
        //Octave: J_history(iter) = costFunction(theta, X, y);
        
    }

    /**
		%COSTFUNCTION Compute cost and gradient for logistic regression
		%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
		%   parameter for logistic regression and the gradient of the cost
		%   w.r.t. to the parameters.
	*/
	double costFunctionGD(Matrix theta, Matrix X, Matrix y){
		/**
			% Initialize some useful values
			m = length(y); % number of training examples

			% You need to return the following variables correctly 
			J = 0;
			grad = zeros(size(theta));

			% ====================== YOUR CODE HERE ======================
			% Instructions: Compute the cost of a particular choice of theta.
			%               You should set J to the cost.
			%               Compute the partial derivatives and set grad to the partial
			%               derivatives of the cost w.r.t. each parameter in theta
			%
			% Note: grad should have the same dimensions as theta
			%
			% h = compute hypothesis
			% J = cost function
			%
		*/
            int m = y.getRowDimension();
            Matrix grad = theta;
            double J =0;
            return 0;
	}

	/**
		%COSTFUNCTION Compute cost and gradient for logistic regression
		%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
		%   parameter for logistic regression and the gradient of the cost
		%   w.r.t. to the parameters.
	*/
        CostFunctionValues costFunction(Matrix theta, Matrix X, Matrix y){
		/**
		
	% Initialize some useful values
			m = length(y); % number of training examples

			% You need to return the following variables correctly 
			J = 0;
			grad = zeros(size(theta));

			% ====================== YOUR CODE HERE ======================
			% Instructions: Compute the cost of a particular choice of theta.
			%               You should set J to the cost.
			%               Compute the partial derivatives and set grad to the partial
			%               derivatives of the cost w.r.t. each parameter in theta
			%
			% Note: grad should have the same dimensions as theta
			%
			% h = compute hypothesis
			% J = cost function
			% grad = gradient
			%
		*/
            CostFunctionValues CFV = new CostFunctionValues();
            int m = y.getRowDimension();
            Matrix grad = new Matrix(theta.getRowDimension(),theta.getColumnDimension());
            double cost = 0;
            
            CFV.setGrad(grad);
            CFV.setJ(cost);
            
            return CFV;
	}

	/*
	function g = sigmoid(z)
	%SIGMOID Compute sigmoid functoon
	%   J = SIGMOID(z) computes the sigmoid of z.
	*/
	double sigmoid(Matrix z){
		/*
		% You need to return the following variables correctly 
		g = zeros(size(z));

		% ====================== YOUR CODE HERE ======================
		% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
		%               vector or scalar).

		g = 1./(1+exp(-z));
		% ============================================================
		*/


            Matrix g = new Matrix(z.getRowDimension(),1);
            for (int i = 0; i<z.getColumnDimension(); i++){
                for (int j = 0; j<z.getRowDimension(); j++){
                    g.set(j, i, 1./(1+ Math.exp(-1*z.get(j,i))));
                }
            }
            return g;
	}
	/*
	function p = predict(theta, X)
	%PREDICT Predict whether the label is 0 or 1 using learned logistic 
	%regression parameters theta
	%   p = PREDICT(theta, X) computes the predictions for X using a 
	%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
	*/
	Matrix predict(Matrix theta, Matrix X){
		/*
		m = size(X, 1); % Number of training examples

		% You need to return the following variables correctly
		p = zeros(m, 1);

		% ====================== YOUR CODE HERE ======================
		% Instructions: Complete the following code to make predictions using
		%               your learned logistic regression parameters. 
		%               You should set p to a vector of 0's and 1's
		%
		for i=1:m
		  h = sigmoid(X(i,:)*theta)
		  if h >= 0.5
		     p(i) = 1;
		  else
		     p(i) = 0;
		  end
		end
		% =========================================================================
		*/
            
            int m = X.getRowDimension();
            Matrix p = new Matrix (m,1);
            double h = 0;
            for (int i = 0; i < m ; i++){
                h = sigmoid(X.times(theta));
                if (h >= 0.5)
                    p.set(i, 0, 1);
                else
                    p.set(i, 0, 0);
            }
            
            return p;
	}


	/**
		fprintf('Train Accuracy: %f\n', );		
	*/
	double accuracy(Matrix p, Matrix y){
		//mean(double(p == y)) * 100
            

            return 0;
	}



	public static void main(String[] args) throws FileNotFoundException, UnsupportedEncodingException{
		/**
			%% Load Data
			%  The first two columns contains the exam scores and the third column
			%  contains the label.

			data = load('ex2data1.txt');
			X = data(:, [1, 2]); y = data(:, 3);
		*/
		//===== JAVA CODE HERE ====
            List data = new ArrayList();
            InputStream dataStream = TestMLR.class.getResourceAsStream("ex2data1.txt");
            BufferedReader br = new BufferedReader(new InputStreamReader(dataStream, "utf-8"));

            String strLine;

            try {
                while((strLine = br.readLine()) != null){
                    data.add(strLine);
                }
            } catch (IOException ex) {
                Logger.getLogger(TestMLR.class.getName()).log(Level.SEVERE, null, ex);
            }

            Matrix X = new Matrix(data.size(),2);
            Matrix y = new Matrix(data.size(),1);
            int m = y.getRowDimension();

            //Set values to Matrices X and y
            for(int i = 0; i<data.size();i++){
                String line = (String) data.get(i);
                String[] temp = line.split(",");
                double [][] tempX= {{Double.valueOf(temp[0]),Double.valueOf(temp[1])}};
                double [][] tempY = {{Double.valueOf(temp[2])}};

                X.setMatrix(i, i, 0, (X.getColumnDimension())-1, new Matrix(tempX));
                y.setMatrix(i, i, 0, (y.getColumnDimension())-1, new Matrix(tempY));
            }
		/*
		[X mu sigma] = featureNormalize(X);
		*/
		//===== JAVA CODE HERE ====
            MultivariateLR MLR = new MultivariateLR();
            FeatureNormalizationValues FNV = MLR.featureNormalize(X);

		/**
			plotData(X, y);

			% Put some labels 
			hold on;
			% Labels and Legend
			xlabel('Exam 1 score')
			ylabel('Exam 2 score')

			% Specified in plot order
			legend('Admitted', 'Not admitted')
		*/
		//===== JAVA CODE HERE ====




		/**

			Compute Cost and Gradient
			%  Setup the data matrix appropriately, and add ones for the intercept term
			[m, n] = size(X);

			% Add intercept term to x and X_test
			X = [ones(m, 1) X];

			% Initialize fitting parameters
			initial_theta = zeros(n + 1, 1);

			% Compute and display initial cost and gradient
			[cost, grad] = costFunction(initial_theta, X, y);

			fprintf('Cost at initial theta (zeros): %f\n', cost);
			fprintf('Gradient at initial theta (zeros): \n');
			fprintf(' %f \n', grad);

			Cost at initial theta (zeros): 0.693147
			Gradient at initial theta (zeros):
			 -0.100000
			 -12.009217
			 -11.262842

		*/
		//===== JAVA CODE HERE ====
            m = X.getRowDimension();
            int n = X.getColumnDimension();
            
            Matrix newMatrix = new Matrix(X.getRowDimension(),3);
            for (int i = 0; i<X.getRowDimension();i++){
                newMatrix.set(i, 0, 1);
                newMatrix.set(i, 1, X.get(i, 0));
                newMatrix.set(i, 2, X.get(i, 1));
            }
            X = newMatrix;
            
            Matrix initial_theta = new Matrix(n+1,1);
            LogReg LR = new LogReg();
            CostFunctionValues CFV = LR.costFunction(initial_theta, X, y);
            double cost = CFV.getJ();
            Matrix grad = CFV.getGrad();
		/**
			Optimizing using GD
	        % Choose some alpha value
	        alpha = 0.1;
	        num_iters = 1000;

	        % Init Theta and Run Gradient Descent 
	        theta = zeros(3, 1);
	        [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);
	        
	        % Plot the convergence graph
	        figure;
	        plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
	        xlabel('Number of iterations');
	        ylabel('Cost J');

	        % Display gradient descent's result
	        fprintf('Theta computed from gradient descent: \n');
	        fprintf(' %f \n', theta);
	        */ 
		

	    //===== JAVA CODE HERE ====
            double alpha = 0.1;
            int num_iters = 1000;
            
            Matrix theta = new Matrix(3,1);
            GradientDescentValues GDV = LR.gradientDescentMulti(X, y, theta, alpha, num_iters);
            theta = GDV.getTheta();
            Matrix J_history = GDV.getCostHistory();
            


	    /**
	    Predict and Accuracies
	    prob = sigmoid([1 45 85] * theta);
		fprintf(['For a student with scores 45 and 85, we predict an admission ' ...
		         'probability of %f\n\n'], prob);

		% Compute accuracy on our training set
		p = predict(theta, X);

	    */
		//===== JAVA CODE HERE ====
            Matrix s = new Matrix(3,1);
            s.set(0, 0, 1);
            s.set(1, 0, 45);
            s.set(2, 0, 85);
            double prob = LR.sigmoid(s.times(theta));
            System.out.println("For a student with scores 45 and 85, we predict an admission"
                    + "probablity of" + prob);
            Matrix p = LR.predict(theta, X);
		/**
		fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
		*/
		//===== JAVA CODE HERE ====
            
            
	}	
}



class CostFunctionValues{
	double J;
	Matrix grad;

        public Matrix getGrad(){
            return grad;
        }
        public double getJ(){
            return J;
        }
        public void setJ(double j){
            this.J = j;
        }
        public void setGrad(Matrix g){
            this.grad = g;
        }
}