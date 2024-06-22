public class Formulas {

    // Calculate the alpha values using the kernel matrix K and the label vector Y
    // where Alpla = Pinv(K)*Y;
    public static double[] calculateAlpha(double[][] K, double[] Y) {
        // Multiply the pseudo-inverse of K with Y to get alpha
        return multiplyMatrices(pseudoInverse(K), Y);
    }

    // Calculating the predicted labels using the kernel matrix K and alpha values
    // Y = K*alpha;
    public static double[] calculatePredictedLabels(double[][] K, double[] alpha) {
        // Multiply the kernel matrix K with alpha to get the predicted labels
        return multiplyMatrices(K, alpha);
    }

    // Calculating the prediction for a single data point x
    //
    public static double calculatePrediction(double[] x, double[][] A, double[] alpha) {
        double[] row = new double[A.length];
        // Projecting the x on the A to get the transformed vector and making it in terms of the polynomial equation.
        for (int i = 0; i < A.length; i++) {
            row[i] = Math.pow(x[0] * A[i][0] + x[1] * A[i][1] + 1, 3);
        }
        // Compute the dot product of the transformed feature vector with alpha to get the prediction
        return dotProduct(row, alpha);
    }

    // Computing the pseudo-inverse of a matrix
    public static double[][] pseudoInverse(double[][] matrix) {
        // Transpose the input matrix
        double[][] transpose = transpose(matrix);
        // Multiply the transpose with the original matrix
        double[][] product = multiply(transpose, matrix);
        // Compute the inverse of the product
        double[][] inverse = inverse(product);
        // Here All the steps are done just to get the inverse(A*A') where A IS THE MATRIX I created.
        // Multiply the inverse with the transpose to get the pseudo-inverse
        return multiply(inverse, transpose);
    }

    // Transpose a matrix
    public static double[][] transpose(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] transpose = new double[cols][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transpose[j][i] = matrix[i][j];
            }
        }
        return transpose;
    }


    public static double[][] multiply(double[][] matrix1, double[][] matrix2) {
        int rows1 = matrix1.length;
        int cols1 = matrix1[0].length;
        int cols2 = matrix2[0].length;
        double[][] result = new double[rows1][cols2];
        for (int i = 0; i < rows1; i++) {
            for (int j = 0; j < cols2; j++) {
                double sum = 0;
                for (int k = 0; k < cols1; k++) {
                    sum += matrix1[i][k] * matrix2[k][j];
                }
                result[i][j] = sum;
            }
        }
        return result;
    }

    // Computing the inverse of a matrix
    public static double[][] inverse(double[][] matrix) {
        int n = matrix.length;
        double[][] inverseMatrix = new double[n][n];
         // We know that inverse of the Matrix A is equal to 1/Determinant of the (A) * Adjoint(A)
        // Calculate the determinant of the matrix
        double determinant = determinant(matrix);

        // If the determinant is zero, the matrix is singular and does not have an inverse
        if (determinant == 0) {
            throw new IllegalArgumentException("Matrix is singular, cannot compute inverse");
        }

        // Calculate the adjoint matrix
        double[][] adjointMatrix = adjoint(matrix);

        // Calculate the inverse matrix by dividing adjugate by determinant
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                inverseMatrix[i][j] = adjointMatrix[i][j] / determinant;
            }
        }

        return inverseMatrix;
    }

    // Computing the determinant of a matrix
    public static double determinant(double[][] matrix) {
        int n = matrix.length;
        // Base case: if the matrix is 1x1, return the single element as the determinant
        if (n == 1) {
            return matrix[0][0];
        }
        double det = 0;
        // If the matrix is 2x2, compute the determinant directly
        if (n == 2) {
            det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
        } else {
            // For larger matrices we are computing the  determinant using cofactor expansion.
            for (int j = 0; j < n; j++) {
                det += matrix[0][j] * cofactor(matrix, 0, j);
            }
        }
        return det;
    }

    // Computing the cofactor of an element in a matrix
    // The cofactor of the element in the A matrix is the (-1)^(i+j)*determinant(Minor(A[i][j]))
    public static double cofactor(double[][] matrix, int row, int col) {
        // Cofactor is (-1)^(row+col) * determinant of the minor matrix
        return Math.pow(-1, row + col) * determinant(minor(matrix, row, col));
    }

    // The Minor can be obtained by removing the i row and the j th colum elements and making matrix with the dimention-1;
    public static double[][] minor(double[][] matrix, int row, int col) {
        int n = matrix.length;
        double[][] minor = new double[n - 1][n - 1];
        // Populate the minor matrix
        for (int i = 0, p = 0; i < n; i++) {
            if (i == row) {
                continue;
            }
            for (int j = 0, q = 0; j < n; j++) {
                if (j == col) {
                    continue;
                }
                minor[p][q] = matrix[i][j];
                q++;
            }
            p++;
        }
        return minor;
    }

    // Compute the adjugate of a matrix
    public static double[][] adjoint(double[][] matrix) {
        int n = matrix.length;
        double[][] adjoint = new double[n][n];
        // Populate the adjugate matrix with cofactors
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                adjoint[j][i] = Math.pow(-1, i + j) * determinant(minor(matrix, i, j));
            }
        }
        return adjoint;
    }

    // Multiply a matrix with a vector
    public static double[] multiplyMatrices(double[][] matrix1, double[] matrix2) {
        int rows1 = matrix1.length;
        int cols1 = matrix1[0].length;
        int length2 = matrix2.length;

        // Check if multiplication is possible
        if (cols1 != length2) {
            throw new IllegalArgumentException("Matrix multiplication not possible: Number of columns in matrix1 must be equal to the length of matrix2.");
        }

        double[] result = new double[rows1];

        // Perform matrix-vector multiplication
        for (int i = 0; i < rows1; i++) {
            double sum = 0;
            for (int j = 0; j < cols1; j++) {
                sum += matrix1[i][j] * matrix2[j];
            }
            result[i] = sum;
        }

        return result;
    }

    // Compute the dot product of two vectors
    public static double dotProduct(double[] vector1, double[] vector2) {
        int length1 = vector1.length;
        int length2 = vector2.length;

        // Check if dot product is possible
        if (length1 != length2) {
            throw new IllegalArgumentException("Dot product calculation not possible: Vectors must be of equal length.");
        }

        double dotProductValue = 0;

        // Calculate the dot product
        for (int i = 0; i < length1; i++) {
            dotProductValue += vector1[i] * vector2[i];
        }

        return dotProductValue;
    }
}
