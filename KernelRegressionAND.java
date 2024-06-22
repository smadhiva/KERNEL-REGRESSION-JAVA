import java.util.Arrays;

public class KernelRegressionAND extends Formulas {


        public static void main(String[] args) {
            double[][] A = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
            double[] Y = {0, 0, 0, 1};

            // Polynomial Kernel for data
            double[][] K = new double[A.length][A.length];
            for (int i = 0; i < A.length; i++) {
                for (int j = 0; j < A.length; j++) {
                    K[i][j] = Math.pow(A[i][0] * A[j][0] + A[i][1] * A[j][1] + 1, 3);
                }
            }

            // alpha values
            double[] alpha = calculateAlpha(K, Y);

            System.out.println("Predicted label for all data");
            double[] predictedLabels = calculatePredictedLabels(K, alpha);
            System.out.println(Arrays.toString(predictedLabels));

            System.out.println("Actual label for all data");
            System.out.println(Arrays.toString(Y));

            // Predictions for specific data points
            double[] x1 = {0, 0}; // Belongs to Class 0
            double x1PredLabel = calculatePrediction(x1, A, alpha);
            if(x1PredLabel<0) {
                x1PredLabel = 0;
            }
            System.out.println("Prediction for x1: " + x1PredLabel);

            double[] x2 = {0, 1}; // Belongs to Class 1
            double x2PredLabel = calculatePrediction(x2, A, alpha);
            if(x2PredLabel<0) {
                x2PredLabel = 0;
            }
            System.out.println("Prediction for x2: " + x2PredLabel);
        }


    }






