package multivariatelr;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author MiriamMarie
 */
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
import com.xeiam.xchart.Chart;
import com.xeiam.xchart.QuickChart;
import com.xeiam.xchart.SwingWrapper;

/**
 * Implements multivariate linear regression. 
 * @author MiriamMarie
 */
public class MultivariateLR {
//    
//    public static void main(String[] args) throws FileNotFoundException, UnsupportedEncodingException {
//        new TestMLR().main(args);
//    }
    double alpha = 0.01;
    int numIterations = 400;
    
    /**
     *FEATURENORMALIZE Normalizes the features in X 
    *   FEATURENORMALIZE(X) returns a normalized version of X where
    *   the mean value of each feature is 0 and the standard deviation
    *   is 1. This is often a good preprocessing step to do when
    *   working with learning algorithms.
     * working with learning algorithms.
     * @param X the matrix to be normalized
     * @return the object the contains the matrix values of X, mu and sigma
     */
    FeatureNormalizationValues featureNormalize(Matrix X){
        
        FeatureNormalizationValues FNV = new FeatureNormalizationValues();
        
        Matrix X_norm = X;
        Matrix mu = new Matrix(1, X.getColumnDimension());
        Matrix sigma = new Matrix(1, X.getColumnDimension());
       
        int column = X.getColumnDimension();
        for (int i = 0;i<column;i++){
            Matrix matrix = X.getMatrix(0, (X.getRowDimension())-1, i, i);
            
            //Compute mean
            double sum=0;
            for (int j = 0;j<matrix.getRowDimension();j++){
                sum += matrix.get(j,0);   
            }
            
            double mean = sum/matrix.getRowDimension();
            mu.set(0, i , mean); //Add the value mean in the matrix mu
            
            //Subtract the mean value from every element of the the feature/column 
            for (int j=0;j<matrix.getRowDimension();j++){
                double newElement = matrix.get(j,0) - mean;
                X_norm.set(j, i, newElement);
                
            }
            
            //Compute std
            sum = 0;
            for (int k = 0;k<matrix.getRowDimension();k++){
                sum += Math.pow((matrix.get(k,0) - mean),2);
            }
            
            double var = sum/((matrix.getRowDimension())-1);
            double std = Math.sqrt(var);
            sigma.set(0,i,std);//Add the value std in the matrix sigma
            
            //Divide each element in X_norm by the standard deviation of the feature/column
            for (int k = 0; k<matrix.getRowDimension();k++){
                double updatedElement = X_norm.get(k, i)/std;
                X_norm.set(k, i, updatedElement);
            }
        }  

        FNV.setMu(mu);
        FNV.setX(X_norm);
        FNV.setSigma(sigma);
        
        return FNV;
    }
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
    GradientDescentValues gradientDescent(Matrix X, Matrix y, Matrix theta, double alpha, int numIterations){

        GradientDescentValues GDV = new GradientDescentValues();
        
        int m = y.getRowDimension();
        Matrix jHistory = new Matrix(numIterations,1);
        
        for(int iter=0; iter<numIterations ;iter++){
            
            jHistory.set(iter,0, computeCostMulti(X, y, theta));
            
            Matrix hypothesis = X.times(theta);
            Matrix error = hypothesis.minus(y);
            Matrix difference = error.times(alpha/m);
            Matrix changeTheta = X.transpose().times(difference);
            theta = theta.minus(changeTheta);
           
        }   
        
        GDV.setTheta(theta);
        GDV.setCostHistory(jHistory);
        return GDV;
    }
    /**
     *COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
    *   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
    *   parameter for linear regression to fit the data points in X and y
     * @param X
     * @param y
     * @param theta
     * @return 
     */
    double computeCostMulti(Matrix X, Matrix y, Matrix theta){
        int m = y.getRowDimension();
        double j = 0;
        
        Matrix hypothesis = X.times(theta);
        Matrix error = hypothesis.minus(y);
        Matrix squareDiff = error.transpose().times(error).times(1.d/(2.d*m));
        
        j = squareDiff.get(0, 0);
        return j;
    }
    /**
     NORMALEQN Computes the closed-form solution to linear regression 
       NORMALEQN(X,y) computes the closed-form solution to linear 
       regression using the normal equations.
     * @param X
     * @param y
     * @return 
     */
    
    Matrix normalEqn(Matrix X, Matrix y){
        return (((X.transpose()).times(X)).inverse())
                .times((X.transpose()).times(y));
    }
    
}
class GradientDescentValues{
    Matrix theta;
    Matrix costHistory;

    public Matrix getTheta() {
        return theta;
    }

    public void setTheta(Matrix theta) {
        this.theta = theta;
    }

    public Matrix getCostHistory() {
        return costHistory;
    }

    public void setCostHistory(Matrix costHistory) {
        this.costHistory = costHistory;
    }
    
}
class FeatureNormalizationValues{
    Matrix X;
    Matrix mu;
    Matrix sigma;

    public Matrix getX() {
        return X;
    }

    public void setX(Matrix X) {
        this.X = X;
    }

    public Matrix getMu() {
        return mu;
    }

    public void setMu(Matrix mu) {
        this.mu = mu;
    }

    public Matrix getSigma() {
        return sigma;
    }

    public void setSigma(Matrix sigma) {
        this.sigma = sigma;
    }
    
}

class TestMLR extends Thread{
    
    //Threading for plotting 
    double[] J;
    public TestMLR(double[] _J){
        J = _J;
    }
    
    @Override
    public void run(){
        Chart c = QuickChart.getChart("Cost x iterations plot","iterations",
                "J(theta)",null,null,J);
        new SwingWrapper(c).displayChart();
    }
    
    public static void main(String[] args) throws FileNotFoundException, UnsupportedEncodingException {
        //Write the corresponding Java code for the Octave code below between /**...*/
        
        long start = System.currentTimeMillis();
        Matrix theta = new Matrix(3,1);
        double price = 0;    
        Matrix sample = new Matrix(1,3); 
        
        System.out.println("Loading data...");
        
        List data = new ArrayList();
        InputStream dataStream = TestMLR.class.getResourceAsStream("ex1data2.txt");
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
        
        System.out.println("First ten examples: ");
        System.out.println("Features (X Values)");
        X.getMatrix(0, 9, 0, 1).print(10, 3);
        System.out.println("Actual cost (Y Values)");
        y.getMatrix(0, 9 ,0 ,0).print(10,3);
        
        System.out.println("Normalizing features..");
        
        //Normalize features of matrix X
        MultivariateLR MLR = new MultivariateLR();
        FeatureNormalizationValues FNV = MLR.featureNormalize(X);
        
        Matrix newMatrix = new Matrix(X.getRowDimension(),3);
        for (int i = 0; i<X.getRowDimension();i++){
            newMatrix.set(i, 0, 1);
            newMatrix.set(i, 1, X.get(i, 0));
            newMatrix.set(i, 2, X.get(i, 1));
        }
        X = newMatrix;
        
        System.out.println("Running gradient descent...");
     
        double alpha = 0.01;
        int num_iters = 400;
        GradientDescentValues GDV = MLR.gradientDescent(X, y, theta, alpha, num_iters);

        //Plotting
        double [] jHistory = GDV.getCostHistory().getRowPackedCopy();
        double [] x = new double[num_iters];
        double [] J = new double[num_iters];
        new Thread(new TestMLR(jHistory)).start();
        
        //Print theta computed from GD
        theta = GDV.getTheta();      
        System.out.println("Theta computed from the gradient descent");
        theta.print(10, 3);

        //==============
        System.out.println("Estimating price...");
        
        //Predict price using gradient descent
        sample.set(0, 0, 1);
        sample.set(0, 1, ((1650-FNV.mu.get(0, 0))/FNV.sigma.get(0, 0)));
        sample.set(0, 2, ((3-FNV.mu.get(0, 1))/FNV.sigma.get(0, 1)));
        price = sample.times(theta).get(0, 0);
        System.out.print("Predicted price of a 1650 sq-ft, "
                + "3 br house (using gradient descent): ");
        System.out.println(price);

        //==============
        
        System.out.println("Solving with normal equations");

        //Reset
        data = new ArrayList();
        dataStream = TestMLR.class.getResourceAsStream("ex1data2.txt");
        br = new BufferedReader(new InputStreamReader(dataStream, "utf-8"));
        
        try {
            while((strLine = br.readLine()) != null){
                data.add(strLine);
            }
        } catch (IOException ex) {
            Logger.getLogger(TestMLR.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        X = new Matrix(data.size(),2);
        y = new Matrix(data.size(),1);
        m = y.getRowDimension();
        
        //Set values to Matrices X and y
        for(int i = 0; i<data.size();i++){
            String line = (String) data.get(i);
            String[] temp = line.split(",");
            double [][] tempX= {{Double.valueOf(temp[0]),Double.valueOf(temp[1])}};
            double [][] tempY = {{Double.valueOf(temp[2])}};

            X.setMatrix(i, i, 0, (X.getColumnDimension())-1, new Matrix(tempX));
            y.setMatrix(i, i, 0, (y.getColumnDimension())-1, new Matrix(tempY));
        }       
        
        newMatrix = new Matrix(X.getRowDimension(),3);
        for (int i = 0; i<X.getRowDimension();i++){
            newMatrix.set(i, 0, 1);
            newMatrix.set(i, 1, X.get(i, 0));
            newMatrix.set(i, 2, X.get(i, 1));
        }
        X = newMatrix;
        
        //Print theta computed from NE
        theta = MLR.normalEqn(X, y);
        System.out.println("Theta computed from the normal equations: ");
        theta.print(10, 3);
        
        //Predict price using normal equation
        sample.set(0, 0, 1);
        sample.set(0, 1, 1650);
        sample.set(0, 2, 3);
        price = sample.times(theta).get(0, 0);
        System.out.print("Predicted price of a 1650 sq-ft, 3 br house "
                + "using normal equation: ");
        System.out.println(price);
        
        //=========================================================
        for (int i = 0; i < 1000000000; i++) {
            double a = Math.sqrt((i+5.9)*(i*i));
        }
        
        long end = System.currentTimeMillis();
        
        long dif = end-start;
        if(dif>1000){
            dif = (end-start)/1000;
            System.out.println("Speed:"+dif+" seconds");
        }else{
            System.out.println("Speed:"+dif+" milliseconds");
        }
    }
}