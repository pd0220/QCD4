// chi squared method implementation for function fit to correlated data sets

// used headers/libraries
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <math.h>

// ------------------------------------------------------------------------------------------------------------

// read given file
// expected structure: x11 | y11 | y_err11 | y_jck11...
//                     x21 | y21 | y_err21 | y_jck21...
//                     ... | ... |   ...   |   ...
Eigen::MatrixXd readFile(std::string fileName)
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
    // first col is for name
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

// separate the raw data to specific quantities and collect them in a vector
std::vector<Eigen::MatrixXd> SeparateQs(Eigen::MatrixXd const &rawDataMat, int const &numOfQs)
{
    // sizes of raw data matrix
    int rows = rawDataMat.rows(), cols = rawDataMat.cols();
    // container for separated matrices
    std::vector<Eigen::MatrixXd> DataMatVec(numOfQs, Eigen::MatrixXd(rawDataMat.rows() / numOfQs, rawDataMat.cols()));
    // fill container according to number of quantities
    for (int i = 0; i < numOfQs; i++)
    {
        int qIndex = i;
        for (int j = 0; j < rows / numOfQs; j++)
        {
            for (int k = 0; k < cols; k++)
            {
                DataMatVec[i](j, k) = rawDataMat(qIndex, k);
            }
            qIndex += numOfQs;
        }
    }

    // return matrix container
    return DataMatVec;
}

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
    Eigen::MatrixXd const rawDataMat = readFile(fileName);

    // separation for specific quantities
    std::vector<Eigen::MatrixXd> const DataMatVec = SeparateQs(rawDataMat, numOfQs);

    
}
