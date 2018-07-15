//
// Created by phyorch on 30/06/18.
//

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include </home/phyorch/PROJECT/hand-eye_calibration/pose_estimation.cpp>
// #include "extra.h" // use this if in OpenCV2
using namespace std;
using namespace cv;


Mat skew(Mat A)
{
    CV_Assert(A.cols == 1 && A.rows == 3);
    Mat B(3, 3, CV_64FC1);

    B.at<double>(0, 0) = 0.0;
    B.at<double>(0, 1) = -A.at<double>(2, 0);
    B.at<double>(0, 2) = A.at<double>(1, 0);

    B.at<double>(1, 0) = A.at<double>(2, 0);
    B.at<double>(1, 1) = 0.0;
    B.at<double>(1, 2) = -A.at<double>(0, 0);

    B.at<double>(2, 0) = -A.at<double>(1, 0);
    B.at<double>(2, 1) = A.at<double>(0, 0);
    B.at<double>(2, 2) = 0.0;

    return B;
}

void Tsai_HandEye(Mat Hcg, vector<Mat> Hgij, vector<Mat> Hcij)
{
    CV_Assert(Hgij.size() == Hcij.size());
    int nStatus = Hgij.size();

    Mat Rgij(3, 3, CV_64FC1);
    Mat Rcij(3, 3, CV_64FC1);

    Mat rgij(3, 1, CV_64FC1);
    Mat rcij(3, 1, CV_64FC1);

    double theta_gij;
    double theta_cij;

    Mat rngij(3, 1, CV_64FC1);
    Mat rncij(3, 1, CV_64FC1);

    Mat Pgij(3, 1, CV_64FC1);
    Mat Pcij(3, 1, CV_64FC1);

    Mat tempA(3, 3, CV_64FC1);
    Mat tempb(3, 1, CV_64FC1);

    Mat A;
    Mat b;
    Mat pinA;

    Mat Pcg_prime(3, 1, CV_64FC1);
    Mat Pcg(3, 1, CV_64FC1);
    Mat PcgTrs(1, 3, CV_64FC1);

    Mat Rcg(3, 3, CV_64FC1);
    Mat eyeM = Mat::eye(3, 3, CV_64FC1);

    Mat Tgij(3, 1, CV_64FC1);
    Mat Tcij(3, 1, CV_64FC1);

    Mat tempAA(3, 3, CV_64FC1);
    Mat tempbb(3, 1, CV_64FC1);

    Mat AA;
    Mat bb;
    Mat pinAA;

    Mat Tcg(3, 1, CV_64FC1);

    for (int i = 0; i < nStatus; i++)
    {
        Hgij[i](Rect(0, 0, 3, 3)).copyTo(Rgij);
        Hcij[i](Rect(0, 0, 3, 3)).copyTo(Rcij);

        Rodrigues(Rgij, rgij);
        Rodrigues(Rcij, rcij);

        theta_gij = norm(rgij);
        theta_cij = norm(rcij);

        rngij = rgij / theta_gij;
        rncij = rcij / theta_cij;

        Pgij = 2 * sin(theta_gij / 2)*rngij;
        Pcij = 2 * sin(theta_cij / 2)*rncij;

        tempA = skew(Pgij + Pcij);
        tempb = Pcij - Pgij;

        A.push_back(tempA);
        b.push_back(tempb);
    }

    //Compute rotation
    invert(A, pinA, DECOMP_SVD);

    Pcg_prime = pinA * b;
    Pcg = 2 * Pcg_prime / sqrt(1 + norm(Pcg_prime) * norm(Pcg_prime));
    PcgTrs = Pcg.t();
    Rcg = (1 - norm(Pcg) * norm(Pcg) / 2) * eyeM + 0.5 * (Pcg * PcgTrs + sqrt(4 - norm(Pcg)*norm(Pcg))*skew(Pcg));

    //Computer Translation
    for (int i = 0; i < nStatus; i++)
    {
        Hgij[i](Rect(0, 0, 3, 3)).copyTo(Rgij);
        Hcij[i](Rect(0, 0, 3, 3)).copyTo(Rcij);
        Hgij[i](Rect(3, 0, 1, 3)).copyTo(Tgij);
        Hcij[i](Rect(3, 0, 1, 3)).copyTo(Tcij);

        tempAA = Rgij - eyeM;
        tempbb = Rcg * Tcij - Tgij;

        AA.push_back(tempAA);
        bb.push_back(tempbb);
    }

    invert(AA, pinAA, DECOMP_SVD);
    Tcg = pinAA * bb;

    Rcg.copyTo(Hcg(Rect(0, 0, 3, 3)));
    Tcg.copyTo(Hcg(Rect(3, 0, 1, 3)));
    Hcg.at<double>(3, 0) = 0.0;
    Hcg.at<double>(3, 1) = 0.0;
    Hcg.at<double>(3, 2) = 0.0;
    Hcg.at<double>(3, 3) = 1.0;

}


int main(){
    Mat img_l1 = imread("/home/phyorch/PROJECT/hand-eye_calibration/data/left_images/image1.jpg", CV_LOAD_IMAGE_COLOR);
    Mat img_l2 = imread("/home/phyorch/PROJECT/hand-eye_calibration/data/left_images/image2.jpg", CV_LOAD_IMAGE_COLOR);
    Mat img_r1 = imread("/home/phyorch/PROJECT/hand-eye_calibration/data/right_images/image1.jpg", CV_LOAD_IMAGE_COLOR);
    Mat img_r2 = imread("/home/phyorch/PROJECT/hand-eye_calibration/data/right_images/image2.jpg", CV_LOAD_IMAGE_COLOR);

    Mat R_l , t_l, R_r, t_r;
    estimate(img_l1, img_l2, R_l, t_l);
    estimate(img_r1, img_r2, R_r, t_r);
    cout << R_l.type();
    vector<Mat> g;
    vector<Mat> c;
    Mat gg = Mat::zeros(4, 4, CV_64FC1);
    Mat cc = Mat::zeros(4, 4, CV_64FC1);
    Mat result = Mat::ones(4, 4, CV_64FC1);
    cout << "here" << gg(Rect(0, 0, 3, 3)) <<endl;
    R_l.copyTo(gg(Rect(0, 0, 3, 3)));
    t_l.copyTo(gg(Rect(3, 0, 1, 3)));
    gg.at<double>(3, 3) = 1;
    R_r.copyTo(cc(Rect(0, 0, 3, 3)));
    t_r.copyTo(cc(Rect(3, 0, 1, 3)));
    cc.at<double>(3, 3) = 1;
    cout << "here" << gg <<endl;
    cout << "here" << cc <<endl;
    g.push_back(gg);
    c.push_back(cc);
    Tsai_HandEye(result, g, c);
    cout << "here" << result <<endl;
    return 0;

}