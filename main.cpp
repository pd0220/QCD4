// chi squared method implementation for function fit to correlated data sets

// used headers/libraries
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <math.h>
#include <numeric>

// ------------------------------------------------------------------------------------------------------------

// read given file
// expected structure: x11 | y11 | y_err11 | y_jck11...
//                     x21 | y21 | y_err21 | y_jck21...
//                     x12 | y12 | y_err12 | y_jck12...
//                     x22 | y22 | y_err22 | y_jcj22...
//                     ... | ... |   ...   |   ...
Eigen::MatrixXd ReadFile(std::string fileName)
{
    // start reading
    std::ifstream fileToRead;
    fileToRead.open(fileName);

    // determine number of columns (3 + N_jck)
    std::string firstLine;
    std::getline(fileToRead, firstLine);
    std::stringstream firstLineStream(firstLine);

    // number of columns in given file
    int numOfCols = 0;
    std::string temp;
    // count number of writes to temporary string container
    while (firstLineStream >> temp)
    {
        numOfCols++;
    }
    fileToRead.close();

    // string for lines
    std::string line;

    // data structure to store data
    // first column is for name
    Eigen::MatrixXd DataMat(0, numOfCols - 1);

    // reopen file
    fileToRead.open(fileName);
    // check if open
    if (fileToRead.is_open())
    {
        // read line by line
        int i = 0;
        while (std::getline(fileToRead, line))
        {
            // using stringstream to write matrix
            std::stringstream dataStream(line);
            DataMat.conservativeResize(i + 1, numOfCols - 1);
            // buffer the name
            std::string tmp;
            dataStream >> tmp;
            for (int j = 0; j < numOfCols - 1; j++)
            {
                dataStream >> DataMat(i, j);
            }
            i++;
        }
        // close file
        fileToRead.close();
    }
    // error check
    else
    {
        std::cout << "Unable to open given file." << std::endl;
        std::exit(-1);
    }

    // return raw data
    return DataMat;
}

// ------------------------------------------------------------------------------------------------------------

double CorrCoeff(Eigen::VectorXd const &one, Eigen::VectorXd const &two, double meanOne, double meanTwo)
{
    // number of jackknife samples
    double NJck = (double)one.size();

    // calculate correlation
    double corr = 0;
    for (int i = 0; i < NJck; i++)
    {
        corr += (one(i) - meanOne) * (two(i) - meanTwo);
    }

    return corr * (NJck - 1) / NJck;
}

// ------------------------------------------------------------------------------------------------------------

// block from the blockdiagonal covariance matrix
Eigen::MatrixXd BlockC(Eigen::MatrixXd const &rawDataMat, int const &numOfQs, int const &Q)
{
    // need jackknife samples to calculate correlations
    Eigen::MatrixXd JCKs(numOfQs, rawDataMat.cols() - 3);
    for (int i = 0; i < JCKs.cols(); i++)
    {
        for (int j = 0; j < JCKs.rows(); j++)
        {
            JCKs(j, i) = rawDataMat(Q * numOfQs + j, i + 3);
        }
    }
    // get y_err data to compare with correlation results
    Eigen::VectorXd Errs(numOfQs);
    for (int i = 0; i < numOfQs; i++)
    {
        Errs(i) = rawDataMat(Q * numOfQs + i, 2);
    }

    // means to calculate correlations
    std::vector<double> means(numOfQs, 0.);
    for (int i = 0; i < numOfQs; i++)
    {
        means[i] = JCKs.row(i).mean();
    }

    // covariance matrix block
    Eigen::MatrixXd C(numOfQs, numOfQs);
    for (int i = 0; i < numOfQs; i++)
    {
        for (int j = i; j < numOfQs; j++)
        {
            // triangular part
            C(j, i) = CorrCoeff(JCKs.row(i), JCKs.row(j), means[i], means[j]);
            // using symmetris
            if(i != j)
                C(i, j) = C(j, i);
            // compare correlation results with y_err data
            if(i == j && std::abs(Errs(i) * Errs(i) - C(i, j)) / Errs(i) / Errs(i) > 1e-2)
            {
                std::cout << "WARNING\nProblem might occur with covariance matrix." << std::endl;
            }
        }
    }

    // return inverse covariance matrix block
    return C.inverse();
}

// ------------------------------------------------------------------------------------------------------------

// main function
// argv[1] is datafile to fit
//       1st col --> some physical quantity (x)
//       2nd col --> data (y)
//       3rd col --> err (sigma)
//  rest of cols --> Jackknife samples (y_jck)
// argv[2] is the number of measured quantities
// rest of argv --> if given degree should be fitted (true/false <--> 1/0)
int main(int argc, char **argv)
{
    // file name
    std::string fileName = "None";
    // check for arguments
    fileName = argv[1];
    // container for polynomial degress
    std::vector<bool> degContainer(argc - 3, 0);
    int numOfBasis = 0;
    for (int i = 0; i < argc - 3; i++)
    {
        degContainer[i] = std::stoi(argv[i + 3]);
        if (degContainer[i])
        {
            numOfBasis++;
        }
    }
    //number of measured quantities
    int const numOfQs = std::stoi(argv[2]);

    // error check
    if (fileName == "None")
    {
        std::cout << "No file was given, or the file dose not exist or unavailable." << std::endl;
        std::exit(-1);
    }
    if (argc < 4)
    {
        std::cout << "No polynomial degrees were given." << std::endl;
        std::exit(-1);
    }

    // read file to matrix
    Eigen::MatrixXd const rawDataMat = ReadFile(fileName);

    std::cout << BlockC(rawDataMat, numOfQs, 1) << std::endl;
}
