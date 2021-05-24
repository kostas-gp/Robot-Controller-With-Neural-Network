// neural_network.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <math.h> 
#include <iostream>
#include <stdio.h>
#include <ctime>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <list> 
#include <iterator>
#include <string>
#include <sstream>
#include <iterator>
#include "stdafx.h"
#include <Aria.h>
using namespace std;
int main(int argc, char** argv)
{

		//create Lists for save the data
		list <double> data_x1;
		list <double> data_x2;
		list <double> data_y1;
		list <double> data_y2;
		list <double> validation_errors;

		//create 2 table for the weights
		double weight_h[3][3];
		double weight_y[2][3];

		srand((unsigned)time(0));

		//initialize the variable
		double n = 0.6, l = 0.2, a = 0.1, input_0 = 1, weight_random;

		//initialize the tables with random weight in a range 0-1
		//table for hiden_layer
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				weight_h[i][j] = static_cast<float> (rand()) / static_cast<float> (RAND_MAX);
			}
		}

		//table for output
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 4; j++) {
				weight_y[i][j] = static_cast<float> (rand()) / static_cast<float> (RAND_MAX);
			}
		}
		//read an csv file with the data and save the data in  lists
		ifstream ip("neural.csv");

		if (!ip.is_open()) std : cout << "Error:File not open" << '\n';
		double x1, x2, y1, y2 = 0;
		string x_csv1, x_csv2, y_csv1, y_csv2 = "";

		while (ip.good()) {

			getline(ip, x_csv1, ',');
			stringstream geek1(x_csv1);
			geek1 >> x1;

			getline(ip, x_csv2, ',');
			stringstream geek2(x_csv2);
			geek2 >> x2;

			getline(ip, y_csv1, ',');
			stringstream geek3(y_csv1);
			geek3 >> y1;

			getline(ip, y_csv2);
			stringstream geek4(y_csv2);
			geek4 >> y2;

			data_x1.push_back(x1);
			data_x2.push_back(x2);
			data_y1.push_back(y1);
			data_y2.push_back(y2);
		}
		ip.close();

		//the total data is 3268 so I use for training the 2300 data and the rest for validation
		double rmse_prev, rmse,sum_error, v1h, v2h, v3h, h0, h1, h2, v1, v2, y1_new, y2_new, e1, e2, d1, d2, d1_h, d2_h, d3_h, val_e = 0;
		
		rmse = 0;
		rmse_prev = 100;
	

		while (rmse_prev > rmse) {
			cout << "Train" << endl;
			rmse_prev = rmse;

			//start the training
			//compute the neural network

			for (int i = 0; i < 2288; i++) {
			
				//retrieve the element from the lists
				auto input1 = next(data_x1.begin(), i);
				auto input2 = next(data_x2.begin(), i);
				auto output1 = next(data_y1.begin(), i);
				auto output2 = next(data_y2.begin(), i);

				v1h = 1 * weight_h[0][0] + *input1 * weight_h[0][1] + *input2 * weight_h[0][2];
				v2h = 1 * weight_h[1][0] + *input1 * weight_h[1][1] + *input2 * weight_h[1][2];
				v3h = 1 * weight_h[2][0] + *input1 * weight_h[2][1] + *input2 * weight_h[2][2];

				h0 = 1 / 1 + exp(-l * v1h);
				h1 = 1 / 1 + exp(-l * v2h);
				h2 = 1 / 1 + exp(-l * v3h);

				v1 = h0 * weight_y[0][0] + h1 * weight_y[0][1] + h2 * weight_y[0][2] +1* weight_y[0][3];
				v2 = h0 * weight_y[1][0] + h1 * weight_y[1][1] + h2 * weight_y[1][2] +1 * weight_y[1][3];

				y1_new = 1 / 1 + exp(-l * v1);
				y2_new = 1 / 1 + exp(-l * v2);

				//compute error
				e1 = *output1 - y1_new;
				e2 = *output2 - y2_new;

				//local gradient
				d1 = l * y1_new * (1 - y1_new) * e1;
				d2 = l * y2_new * (1 - y2_new) * e2;

				d1_h = l * h0 * (1 - h0)*(d1 * weight_y[0][0] + d2 * weight_y[1][0]);
				d2_h = l * h1 * (1 - h1)*(d1 * weight_y[0][1] + d2 * weight_y[1][1]);
				d3_h = l * h2 * (1 - h2)*(d1 * weight_y[0][2] + d2 * weight_y[1][2]);

				//updates the weights
				weight_y[0][0] = n * d1 * weight_y[0][0] + weight_y[0][0];
				weight_y[0][1] = n * d1 * weight_y[0][1] + weight_y[0][1];
				weight_y[0][2] = n * d1 * weight_y[0][2] + weight_y[0][2];
				weight_y[0][3] = n * d1 * weight_y[0][3] + weight_y[0][3];
				weight_y[1][0] = n * d2 * weight_y[1][0] + weight_y[1][0];
				weight_y[1][1] = n * d2 * weight_y[1][1] + weight_y[1][1];
				weight_y[1][2] = n * d2 * weight_y[1][2] + weight_y[1][2];
				weight_y[1][3] = n * d2 * weight_y[1][3] + weight_y[1][3];

				weight_h[0][0] = n * d1_h * weight_h[0][0] + weight_h[0][0];
				weight_h[0][1] = n * d1_h * weight_h[0][1] + weight_h[0][1];
				weight_h[0][2] = n * d1_h * weight_h[0][2] + weight_h[0][2];
				weight_h[1][0] = n * d2_h * weight_h[1][0] + weight_h[1][0];
				weight_h[1][1] = n * d2_h * weight_h[1][1] + weight_h[1][1];
				weight_h[1][2] = n * d2_h * weight_h[1][2] + weight_h[1][2];
				weight_h[2][0] = n * d3_h * weight_h[2][0] + weight_h[2][0];
				weight_h[2][1] = n * d3_h * weight_h[2][1] + weight_h[2][1];
				weight_h[2][2] = n * d3_h * weight_h[2][2] + weight_h[2][2];



			}// final training

			 //update the file with the weights

			std::ofstream out1("weight_h.csv");
			std::ofstream out2("weight_y.csv");

			for (auto& row : weight_y) {
				for (auto col : row)
					out1 << col << ',';
				out1 << '\n';
			}

			for (auto& row : weight_h) {
				for (auto col : row)
					out2 << col << ',';
				out2 << '\n';
			}

			out1.close();
			out2.close();

			//start validation
			for (int i = 2301; i < 3268; i++) {
				//retrieve the element from the lists
				auto input1 = next(data_x1.begin(), i);
				auto input2 = next(data_x2.begin(), i);
				auto output1 = next(data_y1.begin(), i);
				auto output2 = next(data_y2.begin(), i);

				v1h = 1 * weight_h[0][0] + *input1 * weight_h[0][1] + *input2 * weight_h[0][2];
				v2h = 1 * weight_h[1][0] + *input1 * weight_h[1][1] + *input2 * weight_h[1][2];
				v3h = 1 * weight_h[2][0] + *input1 * weight_h[2][1] + *input2 * weight_h[2][2];

				h0 = 1 / 1 + exp(-l * v1h);
				h1 = 1 / 1 + exp(-l * v2h);
				h2 = 1 / 1 + exp(-l * v3h);

				v1 = h0 * weight_y[0][0] + h1 * weight_y[0][1] + h2 * weight_y[0][2] + 1 * weight_y[0][3];
				v2 = h0 * weight_y[1][0] + h1 * weight_y[1][1] + h2 * weight_y[1][2] + 1 * weight_y[1][3];
				
				y1_new = 1 / 1 + exp(-l * v1);
				y2_new = 1 / 1 + exp(-l * v2);

				e1 = *output1 - y1_new;
				e2 = *output2 - y2_new;

				//save the error in order to compute the RMSE
				val_e = pow(e1, 2) + pow(e2, 2);
				validation_errors.push_back(val_e);
			}

			//compute the rmse
			sum_error = 0;
			for (int i = 0; i < validation_errors.size(); i++) {
				auto error = next(data_x1.begin(), i);
				sum_error = pow(*error, 2) + sum_error;
			}
			rmse = sqrt(sum_error / validation_errors.size());
			cout << rmse << endl;
		}

	//start the code of aria
		Aria::init();
		ArRobot robot;
		ArArgumentParser argParser(&argc, argv);
		argParser.loadDefaultArguments();
		ArRobotConnector robotConnector(&argParser, &robot);

		if (robotConnector.connectRobot())
			std::cout << "Robot Connected!" << std::endl;
		robot.runAsync(false);
		robot.lock();
		robot.enableMotors();
		robot.unlock();
		int x_1, x_2,y_1,y_2;
		ArSensorReading *sonarSensor[8];

		int sonarRange[8];
		for (int i = 0; i < 8; i++) {
			sonarSensor[i] = robot.getSonarReading(i);
			sonarRange[i] = sonarSensor[i]->getRange();
		}
		
		while (true)
		{
			ArSensorReading *sonarSensor[8];
			//robot.setVel2( 50, 50);
			ArUtil::sleep(100);

			int smallest = 99999;
			double sonarRange0, sonarRange1, sonarRange2, sonarRange3, sonarRange4;

			for (int i = 0; i < 8; i++) { //read the sonar and save it

				sonarSensor[i] = robot.getSonarReading(i);
			}

			sonarRange0 = sonarSensor[0]->getRange();
			sonarRange1 = sonarSensor[1]->getRange();
			sonarRange2 = sonarSensor[2]->getRange();
			sonarRange3 = sonarSensor[3]->getRange();
			sonarRange4 = sonarSensor[4]->getRange();

			//calculate the input
			
			
			if (sonarRange0<smallest)
			{
				smallest = sonarRange0;
			}
			else if (sonarRange1 > smallest)
			{
				smallest = sonarRange1;
			}
			else if (sonarRange2 > smallest)
			{
				smallest = sonarRange2;
			}

			x_1 = smallest;

			smallest = 99999;

			if (sonarRange3<smallest)
			{
				smallest = sonarRange3;
			}
			else if (sonarRange4 > smallest)
			{
				smallest = sonarRange4;
			}
			
			x_2 = smallest;

			//compute the output
			v1h = 1 * weight_h[0][0] + x_1 * weight_h[0][1] + x_2 * weight_h[0][2];
			v2h = 0 * weight_h[1][0] + x_1 * weight_h[1][1] + x_2 * weight_h[1][2];
			v3h = 0 * weight_h[2][0] + x_1 * weight_h[2][1] + x_2 * weight_h[2][2];

			h0 = 1 / 1 + exp(-l * v1h);
			h1 = 1 / 1 + exp(-l * v2h);
			h2 = 1 / 1 + exp(-l * v3h);

			v1 = h0 * weight_y[0][0] + h1 * weight_y[0][1] + h2 * weight_y[0][2];
			v2 = h0 * weight_y[1][0] + h1 * weight_y[1][1] + h2 * weight_y[2][2];

			y_1 = 1 / 1 + exp(-l * v1);
			y_2 = 1 / 1 + exp(-l * v2);

			robot.setVel2(y_1, y_2);
		
		}
		Aria::exit();
		
	}
	


