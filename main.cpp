#include <fmt/format.h>
#include <filesystem>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

constexpr int chess_width = 9;
constexpr int chess_height = 6;

bool get_subpix_chess_corner(const cv::Mat &im, const cv::Size2i &pattern, std::vector<cv::Point2f> &corners) {
    bool found = cv::findChessboardCorners(im, pattern, corners);
    if (!found) return false;
    const cv::TermCriteria term = {cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 100, 1e-3};
    cv::Mat gray;
    cv::cvtColor(im, gray, cv::COLOR_BGR2GRAY);
    cv::cornerSubPix(gray, corners, {11, 11}, {-1, -1}, term);
    return true;
}

void calibration(const std::string &path, cv::Mat &K, cv::Mat &C) {
    std::vector<cv::Point3f> object_point;
    for (int h = 0; h < chess_height; h++) {
        for (int w = 0; w < chess_width; w++) {
            object_point.emplace_back(w, h, 0);
        }
    }
    std::vector<std::vector<cv::Point3f>> object_points;

    std::vector<std::vector<cv::Point2f>> image_points;

    cv::Size2i image_size;

    for (const auto &fd : fs::directory_iterator(path)) {
        const auto &name = fd.path().string();
        cv::Mat src = cv::imread(name);
        image_size.width = src.cols;
        image_size.height = src.rows;
        std::vector<cv::Point2f> corners;
        bool found = get_subpix_chess_corner(src, {chess_width, chess_height}, corners);
        if (!found) {
            fmt::print("'{}' not found!\n", name);
            continue;
        }
        cv::drawChessboardCorners(src, {chess_width, chess_height}, corners, found);
//        cv::imshow("chess", src);
//        if (cv::waitKey(0) == 'q') {
//            fmt::print("'{}' skip!\n", name);
//            continue;
//        }
        fmt::print("'{}' found!\n", name);
        object_points.emplace_back(object_point);
        image_points.emplace_back(corners);
    }
    cv::Mat rvec, tvec;
    cv::calibrateCamera(object_points, image_points, image_size, K, C, rvec, tvec);
}

void get_camera_pos(const cv::Mat &im, const cv::Mat &K, const cv::Mat &C, cv::Mat &T) {
    std::vector<cv::Point2f> corners;
    get_subpix_chess_corner(im, {chess_width, chess_height}, corners);

    std::vector<cv::Point3f> object_point;
    for (int h = 0; h < chess_height; h++) {
        for (int w = 0; w < chess_width; w++) {
            object_point.emplace_back(w, h, 0);
        }
    }
    cv::Mat rvec, tvec;

    cv::solvePnPRansac(object_point, corners, K, C, rvec, tvec);

    cv::Mat R;
    cv::Rodrigues(rvec, R);
    T = cv::Mat(3, 4, CV_32F);
    R.copyTo(T({0, 0, 3, 3}));
    tvec.copyTo(T({3, 0, 1, 3}));
}

void get_object_points(const cv::Mat &im0, const cv::Mat &im1, const cv::Mat &K, const cv::Mat &C,
                       const cv::Mat &T0, const cv::Mat &T1) {
    std::vector<cv::Point2f> image_points0, image_points1;
    get_subpix_chess_corner(im0, {chess_width, chess_height}, image_points0);
    get_subpix_chess_corner(im1, {chess_width, chess_height}, image_points1);

    cv::undistortPoints(image_points0, image_points0, K, C);

    cv::undistortPoints(image_points1, image_points1, K, C);
    cv::Mat p4d;

    cv::triangulatePoints(T0, T1, image_points0, image_points1, p4d);


    cv::Mat p3d(3, p4d.cols, p4d.type());
    for (int c = 0; c < p4d.cols; c++) {
        p3d.col(c) = p4d.col(c).rowRange(0, 3) / p4d.at<float>(3, c);
    }
    std::cout << p4d << std::endl;
    std::cout << p3d << std::endl;
}


void matches(const cv::Mat &src0, const cv::Mat &src1, const cv::Ptr<cv::FeatureDetector> &detector) {
    std::vector<cv::KeyPoint> kps0;
    cv::Mat dest0;
    detector->detectAndCompute(src0, {}, kps0, dest0);

    std::vector<cv::KeyPoint> kps1;
    cv::Mat dest1;
    detector->detectAndCompute(src1, {}, kps1, dest1);

    auto matcher = cv::BFMatcher::create();
    std::vector<std::vector<cv::DMatch>> match;
    matcher->knnMatch(dest0, dest1, match, 2);

    std::vector<cv::DMatch> good_match;
    for (int i = 0; i < match.size(); i++) {
        if (match[i][0].distance < match[i][1].distance / 2) {
            good_match.emplace_back(match[i][0]);
        }
    }

    cv::Mat K = (cv::Mat_<float>(3, 3) << 1114.1804893712708, 0.0, 1074.2415297217708,
            0.0, 1113.4568392254073, 608.6477877664104,
            0.0, 0.0, 1.0);

    std::vector<cv::Point2f> left_points, right_points;
    for (int i = 0; i < good_match.size(); i++) {
        left_points.emplace_back(kps0[good_match[i].trainIdx].pt);
        right_points.emplace_back(kps0[good_match[i].queryIdx].pt);
    }

    cv::Mat mask;
    cv::Mat E = cv::findEssentialMat(left_points, right_points, K, cv::RANSAC, 0.999, 1.0, mask);

    std::vector<cv::Point2f> good_left, good_right;
    for (int i = 0; i < mask.cols; i++) {
        if (mask.at<uint8_t>(i) == 1) {
            good_left.emplace_back(left_points[i]);
            good_right.emplace_back(right_points[i]);
        }
    }

    cv::Mat im2show;
    cv::drawMatches(src0, kps0, src1, kps1, good_match, im2show);
    cv::resize(im2show, im2show, {-1, -1}, 0.5, 0.5);
    cv::imshow("match", im2show);
    cv::waitKey(0);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fmt::print("usage: {} <path to data>\n", argv[0]);
        return -1;
    }
    std::string path = argv[1];

    cv::Mat K, C;
    calibration(path, K, C);

    cv::Mat im0 = cv::imread(path + "/0.jpg");
    cv::Mat im1 = cv::imread(path + "/1.jpg");
    cv::Mat T0, T1;

    get_camera_pos(im0, K, C, T0);

    get_camera_pos(im1, K, C, T1);

    get_object_points(im0, im1, K, C, T0, T1);

    return 0;
}
