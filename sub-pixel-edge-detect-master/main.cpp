#include <fstream>
#include "edgeTest.h"
#include "WriteFile.h"
#include <ctime>
using namespace std;
using namespace cv;

struct IMAGERECT
{
	int x0;
	int y0;
	int x1;
	int y1;
	IMAGERECT(void)
	{
		x0 = 0;
		y0 = 0;
		x1 = 0;
		y1 = 0;
	}
};

struct PT2D{
	double dX;
	double dY;
	PT2D(){
		dX = 0;
		dY = 0;
	}
};

int findConnectedSet(vector<IMAGERECT>& vRectList, cv::Mat bwImage);
int FitEllipsePara(double &dCx, double &dCy, double &dFitDev, vector<PT2D> vEdgePtList);

int main()
{
	Mat srcImage, grayImage, bwImage, dstImage;
	srcImage = imread("L1.bmp", 0);
	if (srcImage.empty())
	{
		cout << "load error!" << endl;
		return -1;
	}
	int iHeight = srcImage.rows;
	int iWidth = srcImage.cols;
	//parameters setting
	double * x;          /* x[n] y[n] coordinates of result contour point n */
	double * y;
	int * curve_limits;  /* limits of the curves in the x[] and y[] */
	int N, M;         /* result: N contour points, forming M curves */
	double S = 0/*1.5*/; /* default sigma=0 */
	double H = 4.2; /* default th_h=0  */
	double L = 0.81; /* default th_l=0  */
	double W = 1.0; /* default W=1.3   */
	char * pdf_out = "output.pdf";  /*pdf filename*/
	char * txt_out = "output.txt";
	//otsu
	srcImage.copyTo(grayImage);
	cv::threshold(grayImage, bwImage, 0, 255, CV_THRESH_OTSU);

	vector<IMAGERECT> vRectList;
	int iRectNum = findConnectedSet(vRectList, bwImage);
	//Each Rect
	int iRectSize = vRectList.size();
	ofstream fout("Center.txt");
	for (int i = 0; i < iRectSize; i++)
	{
		int iX0 = vRectList[i].x0;
		int iY0 = vRectList[i].y0;
		int iRectWid = vRectList[i].x1 - vRectList[i].x0 + 1;
		int iRectHei = vRectList[i].y1 - vRectList[i].y0 + 1;
		Mat roi = grayImage(Rect(iX0, iY0, iRectWid, iRectHei));
		Mat roiCopy;
		roi.copyTo(roiCopy);
		Mat roiDst;
		uchar* pSrc = roiCopy.data;
		uchar* pDst = roiDst.data;
		devernay(&x, &y, &N, &curve_limits, &M, pSrc, pDst, iRectWid, iRectHei, S, H, L);

		for (int k = 0; k<M; k++)
		{
			if (x[curve_limits[k]] == x[curve_limits[k + 1] - 1] && y[curve_limits[k]] == y[curve_limits[k + 1] - 1])
			{
				vector<PT2D> vEdgePt;
				for (int j = curve_limits[k]; j < curve_limits[k + 1]; j++)
				{
					PT2D pt2d;
					pt2d.dX = x[j];
					pt2d.dY = y[j];
					vEdgePt.push_back(pt2d);
				}
				if (vEdgePt.size() > 15 && vEdgePt.size() < 200)
				{
					double dCx = 0, dCy = 0;
					double dFitDev = 0;
					FitEllipsePara(dCx, dCy, dFitDev, vEdgePt);
					if (dFitDev < 2.0)
					{
						dCx += iX0;
						dCy += iY0;
						fout << dCx << " " << dCy << endl;
					}
				}
			}
		}

		/*if (pdf_out != NULL)
		{
		leaf_write_curves_pdf(x, y, curve_limits, M, pdf_out, iRectWid, iRectHei, W);
		}
		if (txt_out != NULL)
		{
		write_curves_txt(x, y, curve_limits, M, txt_out);
		}*/
	}
	fout.close();
	return 0;
}


//int main()
//{
//	Mat srcImage, grayImage, dstImage;
//	srcImage = imread("R1.bmp");
//	if (srcImage.empty())
//	{
//		cout << "load error!" << endl;
//		return -1;
//	}
//
//	//parameters setting
//	double * x;          /* x[n] y[n] coordinates of result contour point n */
//	double * y;
//	int * curve_limits;  /* limits of the curves in the x[] and y[] */
//	int N, M;         /* result: N contour points, forming M curves */
//	double S = 1.5; /* default sigma=0 */
//	double H = 4.2; /* default th_h=0  */
//	double L = 0.81; /* default th_l=0  */
//	double W = 1.0; /* default W=1.3   */
//	char * pdf_out = "output.pdf";  /*pdf filename*/
//	char * txt_out = "output.txt";
//
//	cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);
//	dstImage = grayImage;
//	const int iHeight = dstImage.rows;
//	const int iWidth = dstImage.cols;
//	uchar* pSrc = grayImage.data;//new uchar[iHeight*iWidth];
//	uchar* pDst = dstImage.data;
//
//	//imshow("input image", grayImage);
//	clock_t start, finish;
//	start = clock();
//	devernay(&x, &y, &N, &curve_limits, &M, pSrc, pDst, iWidth, iHeight, S, H, L);
//	finish = clock();
//	cout << finish - start << endl;
//	if (pdf_out != NULL) write_curves_pdf(x, y, curve_limits, M, pdf_out, iWidth, iHeight, W);
//	if (txt_out != NULL) write_curves_txt(x, y, curve_limits, M, txt_out);
//
//	//imshow("gaussion filtered image", dstImage);
//	waitKey();
//	system("pause");
//	return 0;
//}

int findConnectedSet(vector<IMAGERECT>& vRectList, cv::Mat bwImage)
{
	int iHeight = bwImage.rows;
	int iWidth = bwImage.cols;
	cv::Mat I = cv::Mat(iHeight, iWidth, CV_8UC3);
	for (int i = 0; i < iHeight; i++)
	{
		for (int j = 0; j < iWidth; j++)
		{
			bwImage.at<uchar>(i, j) = 255 - bwImage.at<uchar>(i, j);
			I.at<cv::Vec3b>(i, j)[0] = bwImage.at<uchar>(i, j);
			I.at<cv::Vec3b>(i, j)[1] = bwImage.at<uchar>(i, j);
			I.at<cv::Vec3b>(i, j)[2] = bwImage.at<uchar>(i, j);
		}
	}

	int NumberOfRuns = 0;
	vector<int> stRun;
	vector<int> endRun;
	vector<int> rowRun;
	vector<int> numRun;
	/********查找团*******/
	for (int i = 0; i < iHeight; i++)
	{
		unsigned char *rowData = bwImage.ptr<uchar>(i);
		if (rowData[0] == 0)
		{
			NumberOfRuns++;
			stRun.push_back(0);
			rowRun.push_back(i);
		}
		for (int j = 1; j < iWidth; j++)
		{
			if (rowData[j - 1] == 255 && rowData[j] == 0)
			{
				NumberOfRuns++;
				stRun.push_back(j);
				rowRun.push_back(i);
			}
			else if (rowData[j - 1] == 0 && rowData[j] == 255)
			{
				endRun.push_back(j - 1);
			}
		}
		if (rowData[iWidth - 1] == 0)
		{
			endRun.push_back(iWidth - 1);
		}
	}

	int numTmp = 0;
	/********标记团和等价对*******/
	vector<int> runLabels;
	vector<pair<int, int>> equivalences;
	runLabels.assign(NumberOfRuns, 0);
	int offset = 0;
	int idxLabel = 1;
	int curRowIdx = 0;
	int firstRunOnCur = 0;
	int firstRunOnPre = 0;
	int lastRunOnPre = -1;
	for (int i = 0; i < NumberOfRuns; i++)
	{
		numTmp = endRun[i] - stRun[i] + 1;
		numRun.push_back(numTmp);

		if (rowRun[i] != curRowIdx)
		{
			curRowIdx = rowRun[i];
			firstRunOnPre = firstRunOnCur;
			lastRunOnPre = i - 1;
			firstRunOnCur = i;

		}
		for (int j = firstRunOnPre; j <= lastRunOnPre; j++)
		{
			if (stRun[i] <= endRun[j] + offset && endRun[i] >= stRun[j] - offset)
			{
				if (runLabels[i] == 0) // 没有被标号过
					runLabels[i] = runLabels[j];
				else if (runLabels[i] != runLabels[j])// 已经被标号             
					equivalences.push_back(make_pair(runLabels[i], runLabels[j])); // 保存等价对
			}
		}
		if (runLabels[i] == 0) // 没有与前一行的任何run重合
		{
			runLabels[i] = idxLabel++;
		}
	}
	/********等价对处理*******/
	int maxLabel = *max_element(runLabels.begin(), runLabels.end());
	vector< vector< bool>> eqTab(maxLabel, vector< bool>(maxLabel, false));
	vector< pair< int, int>>::iterator vecPairIt = equivalences.begin();
	while (vecPairIt != equivalences.end())
	{
		eqTab[vecPairIt->first - 1][vecPairIt->second - 1] = true;
		eqTab[vecPairIt->second - 1][vecPairIt->first - 1] = true;
		vecPairIt++;
	}
	vector< int> labelFlag(maxLabel, 0);
	vector< vector< int>> equaList;
	vector< int> tempList;
	for (int i = 1; i <= maxLabel; i++)
	{
		if (labelFlag[i - 1])
		{
			continue;
		}
		labelFlag[i - 1] = equaList.size() + 1;
		tempList.push_back(i);
		for (vector< int>::size_type j = 0; j < tempList.size(); j++)
		{
			for (vector< bool>::size_type k = 0; k != eqTab[tempList[j] - 1].size(); k++)
			{
				if (eqTab[tempList[j] - 1][k] && !labelFlag[k])
				{
					tempList.push_back(k + 1);
					labelFlag[k] = equaList.size() + 1;
				}
			}
		}
		equaList.push_back(tempList);
		tempList.clear();
	}
	for (vector< int>::size_type i = 0; i != runLabels.size(); i++)
	{
		runLabels[i] = labelFlag[runLabels[i] - 1];    //对团重新遍历标号
	}

	int num = equaList.size();
	vector <int> sumRun(num, 0);
	vector <vector<double>> boundBox(num, vector<double>(4, 0));
	int label, RectX0, RectY0, RectX1, RectY1;
	for (int i = 0; i<runLabels.size(); i++)
	{
		label = runLabels[i];
		sumRun[label - 1] += numRun[i];   //每个所占的像素个数
		RectX0 = stRun[i];
		RectX1 = endRun[i];
		RectY0 = rowRun[i];
		RectY1 = rowRun[i];
		if (!boundBox[label - 1][0] && !boundBox[label - 1][1] && !boundBox[label - 1][2] && !boundBox[label - 1][3])
		{
			boundBox[label - 1][0] = RectX0;
			boundBox[label - 1][1] = RectX1;
			boundBox[label - 1][2] = RectY0;
			boundBox[label - 1][3] = RectY1;
		}
		else
		{
			if (RectX0 < boundBox[label - 1][0])
				boundBox[label - 1][0] = RectX0;
			if (RectY0 < boundBox[label - 1][2])
				boundBox[label - 1][2] = RectY0;
			if (RectX1 > boundBox[label - 1][1])
				boundBox[label - 1][1] = RectX1;
			if (RectY1 > boundBox[label - 1][3])
				boundBox[label - 1][3] = RectY1;
		}
	}

	for (int i = 0; i < num; i++)
	{
		double dEqualDiam = sqrt(sumRun[i] / (4 * 3.14));
		//	double dRatio = (boundBox[i][1] - boundBox[i][0]) / (boundBox[i][3] - boundBox[i][2]);
		if (dEqualDiam > 2 && dEqualDiam < 20 /*&& dRatio >0.33 && dRatio < 3*/)
		{
			IMAGERECT imgRect;
			imgRect.x0 = boundBox[i][0] - 5;
			imgRect.x1 = boundBox[i][1] + 5;
			imgRect.y0 = boundBox[i][2] - 5;
			imgRect.y1 = boundBox[i][3] + 5;
			if (imgRect.x0 >= 0 && imgRect.x1 <= iWidth && imgRect.y0 >= 0 && imgRect.y1 <= iHeight)
			{
				vRectList.push_back(imgRect);
				cv::Rect rect(imgRect.x0, imgRect.y0, abs(imgRect.x1 - imgRect.x0 + 1), abs(imgRect.y1 - imgRect.y0 + 1));
				cv::rectangle(I, rect, cv::Scalar(0, 0, 255), 1, 8, 0);
			}
		}
	}

	return (int)vRectList.size();
}

int FitEllipsePara(double &dCx, double &dCy, double &dFitDev, vector<PT2D> vEdgePtList)
{
	int nPt = (int)vEdgePtList.size();
	if (nPt < 6)
	{
		return 0;
	}
	double *dEllipsePara = new double[5];
	double dLa;
	double dLb;
	//Matrix mA(5, 5), mC(5, 1), mB(5, 1);
	double  mA[5][5] = { 0 }, mB[5] = { 0 };
	int i;
	double Py, Py2, Py3, Px, Px2, Px3;
	for (i = 0; i < nPt; i++)
	{
		Px = vEdgePtList[i].dX;
		Py = vEdgePtList[i].dY;

		Py2 = Py*Py;
		Py3 = Py2*Py;
		Px2 = Px*Px;
		Px3 = Px2*Px;

		mA[0][0] += Px2*Py2;
		mA[0][1] += Px*Py3;
		mA[0][2] += Px2*Py;
		mA[0][3] += Px*Py2;
		mA[0][4] += Px*Py;

		mA[1][0] += Px*Py3;
		mA[1][1] += Py*Py3;
		mA[1][2] += Px*Py2;
		mA[1][3] += Py3;
		mA[1][4] += Py2;

		mA[2][0] += Px2*Py;
		mA[2][1] += Px*Py2;
		mA[2][2] += Px2;
		mA[2][3] += Px*Py;
		mA[2][4] += Px;

		mA[3][0] += Px*Py2;
		mA[3][1] += Py3;
		mA[3][2] += Px*Py;
		mA[3][3] += Py2;
		mA[3][4] += Py;

		mA[4][0] += Px*Py;
		mA[4][1] += Py2;
		mA[4][2] += Px;
		mA[4][3] += Py;
		mA[4][4] += 1;

		mB[0] += (-1)*Px3*Py;
		mB[1] += (-1)*Px2*Py2;
		mB[2] += (-1)*Px3;
		mB[3] += (-1)*Px2*Py;
		mB[4] += (-1)*Px2;
	}

	int j, k;
	for (k = 0; k < 4; k++)
	{
		int uk = k;
		for (i = k; i < 5; i++)
		{
			if (fabs(mA[i][k]) > fabs(mA[uk][k]))
			{
				uk = i;
			}
		}
		if (mA[uk][k] == 0)
		{
			return 0;
		}
		double tmpa;
		for (j = 0; j < 5; j++)
		{
			tmpa = mA[k][j];
			mA[k][j] = mA[uk][j];
			mA[uk][j] = tmpa;
		}
		tmpa = mB[k];
		mB[k] = mB[uk];
		mB[uk] = tmpa;

		for (i = k + 1; i < 5; i++)
		{
			if (mA[k][k] == 0)
			{
				return 0;
			}
			mA[i][k] /= mA[k][k];
			for (j = k + 1; j < 5; j++)
			{
				mA[i][j] -= mA[i][k] * mA[k][j];
			}
			mB[i] -= mA[i][k] * mB[k];
		}
	}

	if (mA[4][4] == 0)
	{
		return 0;
	}
	dEllipsePara[4] = mB[4] / mA[4][4];
	for (k = 3; k >= 0; k--)
	{
		double sum = 0;
		for (j = k + 1; j < 5; j++)
		{
			sum += mA[k][j] * dEllipsePara[j];
		}
		if (mA[k][k] == 0)
		{
			return 0;
		}
		dEllipsePara[k] = (mB[k] - sum) / mA[k][k];
	}

	//拟合偏差
	for (int i = 0; i < nPt; i++)
	{
		Px = vEdgePtList[i].dX;
		Py = vEdgePtList[i].dY;
		dFitDev += abs(Px*Px + dEllipsePara[0] * Px*Py + dEllipsePara[1] * Py*Py + dEllipsePara[2] * Px + dEllipsePara[3] * Py + dEllipsePara[4]);
	}
	dFitDev /= nPt;

	// 椭圆特征参数
	double mC[5] = { 0 };
	memcpy(mC, dEllipsePara, sizeof(double)* 5);
	double temp1 = mC[0] * mC[0] - 4 * mC[1];
	double temp2 = mC[0] * mC[2] * mC[3] - mC[1] * mC[2] * mC[2] - mC[3] * mC[3] + 4 * mC[1] * mC[4] - mC[0] * mC[0] * mC[4];
	double temp3 = mC[0] * mC[0] + (1 - mC[1])*(1 - mC[1]);

	if (fabs(temp1)<1e-7)
	{
		return 0;
	}

	dCx = (2 * mC[1] * mC[2] - mC[0] * mC[3]) / temp1;
	dCy = (2 * mC[3] - mC[0] * mC[2]) / temp1;

	double temp4 = temp1*(mC[1] - sqrt(temp3) + 1);
	if (fabs(temp4) < 1e-7)
	{
		return 0;
	}
	dLa = sqrt(fabs(2 * temp2 / temp4));

	double temp5 = temp1*(mC[1] + sqrt(temp3) + 1);
	if (fabs(temp4) < 1e-7)
	{
		return 0;
	}
	dLb = sqrt(fabs(2 * temp2 / temp5));

	delete[] dEllipsePara;
	dEllipsePara = NULL;
	return 1;
}
