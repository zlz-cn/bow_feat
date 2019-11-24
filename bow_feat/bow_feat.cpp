#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ml/ml.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <iostream>
#include <fstream>
#include <cstring>
#include <iterator>
#include <vector>
#include <map>
#include<fstream>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;
using namespace cv::ml;

#define DATA_FOLDER "data/"
#define TRAIN_FOLDER "data/train_images/"
#define TEMPLATE_FOLDER "data/templates/"
#define TEST_FOLDER "data/test_image"
#define RESULT_FOLDER "data/result_image/"


class categorizer
{
private:
	// //从类目名称到数据的map映射
	// map<string,Mat> result_objects;    
	//存放所有训练图片的BOW
	map<string, Mat> allsamples_bow;
	//从类目名称到训练图集的映射，关键字可以重复出现
	multimap<string, Mat> train_set;
	// 训练得到的SVM
	Ptr<SVM>* stor_svms;
	//类目名称，也就是TRAIN_FOLDER设置的目录名
	vector<string> category_name;
	//类目数目
	int categories_size;
	//用SURF特征构造视觉词库的聚类数目
	int clusters;
	//存放训练图片词典
	Mat vocab;

	Ptr<SURF> featureDecter;
	Ptr<BOWKMeansTrainer> bowtrainer;
	Ptr<BFMatcher> descriptorMacher;
	Ptr<BOWImgDescriptorExtractor> bowDescriptorExtractor;

	//构造训练集合
	void make_train_set();
	// 移除扩展名，用来讲模板组织成类目
	string remove_extention(string);

public:
	//构造函数
	categorizer(int);
	// 聚类得出词典
	void bulid_vacab();
	//构造BOW
	void compute_bow_image();
	//训练分类器
	void trainSvm();
	//将测试图片分类
	void category_By_svm();
};

// 移除扩展名，用来讲模板组织成类目
string categorizer::remove_extention(string full_name)
{
	//find_last_of找出字符最后一次出现的地方
	int last_index = full_name.find_last_of(".");
	string name = full_name.substr(0, last_index);
	return name;
}

// 构造函数
categorizer::categorizer(int _clusters)
{
	cout << "开始初始化..." << endl;
	clusters = _clusters;
	//初始化指针
	int minHessian = 400;
	featureDecter = SURF::create(minHessian);
	bowtrainer = new BOWKMeansTrainer(clusters);
	descriptorMacher = BFMatcher::create();
	bowDescriptorExtractor = new BOWImgDescriptorExtractor(featureDecter, descriptorMacher);
	cout << "初始化完毕..." << endl;
	//读取训练集
	make_train_set();
}

//构造训练集合
void categorizer::make_train_set()
{
	cout << "读取训练集..." << endl;
	string categor;
	//递归迭代rescursive 直接定义两个迭代器：i为迭代起点（有参数），end_iter迭代终点
	for (boost::filesystem::recursive_directory_iterator i(TRAIN_FOLDER), end_iter; i != end_iter; i++)
	{
		// level == 0即为目录，因为TRAIN__FOLDER中设置如此
		if (i.level() == 0)
		{
			// 将类目名称设置为目录的名称
			if ((i->path()).filename().string() != ".DS_Store") {
				categor = (i->path()).filename().string();
				category_name.push_back(categor);
			}
		}
		else
		{
			// 读取文件夹下的文件。level 1表示这是一副训练图，通过multimap容器来建立由类目名称到训练图的一对多的映射
			string filename = string(TRAIN_FOLDER) + categor + string("/") + (i->path()).filename().string();
			if ((i->path()).filename().string() != ".DS_Store") {
				Mat temp = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
				pair<string, Mat> p(categor, temp);
				//得到训练集
				train_set.insert(p);
			}
		}
	}
	categories_size = category_name.size();
	cout << "发现 " << categories_size << "种类别物体..." << endl;
}

// 训练图片feature聚类，得出词典
void categorizer::bulid_vacab()
{
	FileStorage vacab_fs(DATA_FOLDER "vocab.xml", FileStorage::READ);

	//如果之前已经生成好，就不需要重新聚类生成词典
	if (vacab_fs.isOpened())
	{
		cout << "图片已经聚类，词典已经存在.." << endl;
		vacab_fs.release();
	}
	else
	{
		Mat vocab_descriptors;
		// 对于每一幅模板，提取SURF算子，存入到vocab_descriptors中
		multimap<string, Mat> ::iterator i = train_set.begin();
		for (; i != train_set.end(); i++)
		{
			vector<KeyPoint>kp;
			Mat templ = (*i).second;
			Mat descrip;
			featureDecter->detect(templ, kp);

			featureDecter->compute(templ, kp, descrip);
			//push_back(Mat);在原来的Mat的最后一行后再加几行,元素为Mat时， 其类型和列的数目 必须和矩阵容器是相同的
			vocab_descriptors.push_back(descrip);
		}
		// vocab_descriptors.convertTo(vocab_descriptors, CV_32F);
		cout << "训练图片开始聚类..." << endl;
		//将每一副图的ORB特征加入到bowTraining中去,就可以进行聚类训练了
		// 对ORB描述子进行聚类
		vocab = bowtrainer->cluster(vocab_descriptors);
		cout << "聚类完毕，得出词典..." << endl;

		//以文件格式保存词典
		FileStorage file_stor(DATA_FOLDER "vocab.xml", FileStorage::WRITE);
		file_stor << "vocabulary" << vocab;
		file_stor.release();
	}
}

//构造bag of words
void categorizer::compute_bow_image()
{
	cout << "构造bag of words..." << endl;
	FileStorage va_fs(DATA_FOLDER "vocab.xml", FileStorage::READ);
	//如果词典存在则直接读取
	if (va_fs.isOpened())
	{
		Mat temp_vacab;
		va_fs["vocabulary"] >> temp_vacab;
		bowDescriptorExtractor->setVocabulary(temp_vacab);
		va_fs.release();
	}
	else
	{
		//对每张图片的特征点，统计这张图片各个类别出现的频率，作为这张图片的bag of words
		bowDescriptorExtractor->setVocabulary(vocab);
	}

	//如果bow.txt已经存在说明之前已经训练过了，下面就不用重新构造BOW
	string bow_path = string(DATA_FOLDER) + string("bow.txt");
	boost::filesystem::ifstream read_file(bow_path);
	// //如BOW已经存在，则不需要构造
	if (read_file.is_open())
	{
		cout << "BOW 已经准备好..." << endl;
	}
	else {
		// 对于每一幅模板，提取SURF算子，存入到vocab_descriptors中
		multimap<string, Mat> ::iterator i = train_set.begin();
		for (; i != train_set.end(); i++)
		{
			vector<KeyPoint>kp;
			string cate_nam = (*i).first;
			Mat tem_image = (*i).second;
			Mat imageDescriptor;
			featureDecter->detect(tem_image, kp);
			bowDescriptorExtractor->compute(tem_image, kp, imageDescriptor);
			//push_back(Mat);在原来的Mat的最后一行后再加几行,元素为Mat时， 其类型和列的数目 必须和矩阵容器是相同的
			allsamples_bow[cate_nam].push_back(imageDescriptor);
		}
		//简单输出一个文本，为后面判断做准备
		boost::filesystem::ofstream ous(bow_path);
		ous << "flag";
		cout << "bag of words构造完毕..." << endl;
	}
}

//训练分类器

void categorizer::trainSvm()
{
	int flag = 0;
	for (int k = 0; k < categories_size; k++)
	{
		string svm_file_path = string(DATA_FOLDER) + category_name[k] + string("SVM.xml");
		FileStorage svm_fil(svm_file_path, FileStorage::READ);
		//判断训练结果是否存在
		if (svm_fil.isOpened())
		{
			svm_fil.release();
			continue;
		}
		else
		{
			flag = -1;
			break;
		}
	}
	//如果训练结果已经存在则不需要重新训练
	if (flag != -1)
	{
		cout << "分类器已经训练完毕..." << endl;
	}
	else

	{
		stor_svms = new Ptr<SVM>[categories_size];

		cout << "训练分类器..." << endl;
		for (int i = 0; i < categories_size; i++)
		{
			Mat tem_Samples(0, allsamples_bow.at(category_name[i]).cols, allsamples_bow.at(category_name[i]).type());
			Mat responses(0, 1, CV_32SC1);
			tem_Samples.push_back(allsamples_bow.at(category_name[i]));
			Mat posResponses(allsamples_bow.at(category_name[i]).rows, 1, CV_32SC1, Scalar::all(1));
			responses.push_back(posResponses);

			for (map<string, Mat>::iterator itr = allsamples_bow.begin(); itr != allsamples_bow.end(); ++itr)
			{
				if (itr->first == category_name[i]) {
					continue;
				}
				tem_Samples.push_back(itr->second);
				Mat response(itr->second.rows, 1, CV_32SC1, Scalar::all(-1));
				responses.push_back(response);
			}
			//设置训练参数
			stor_svms[i] = SVM::create();
			stor_svms[i]->setType(SVM::C_SVC);
			stor_svms[i]->setKernel(SVM::LINEAR);
			stor_svms[i]->setGamma(3);
			stor_svms[i]->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER, 100, 1e-6));
			stor_svms[i]->train(tem_Samples, ROW_SAMPLE, responses);
			//存储svm
			string svm_filename = string(DATA_FOLDER) + category_name[i] + string("SVM.xml");
			cout << svm_filename.c_str() << endl;
			stor_svms[i]->save(svm_filename.c_str());
			tem_Samples.release();
			responses.release();
			posResponses.release();
		}
		cout << "分类器训练完毕..." << endl;
	}
}


//对测试图片进行分类

void categorizer::category_By_svm()
{
	cout << "物体分类开始..." << endl;
	Mat gray_pic;
	Mat threshold_image;
	string prediction_category;
	float curConfidence;

	boost::filesystem::directory_iterator begin_train(TEST_FOLDER);
	boost::filesystem::directory_iterator end_train;

	for (; begin_train != end_train; ++begin_train)
	{

		//获取该目录下的图片名
		string train_pic_name = (begin_train->path()).filename().string();
		string train_pic_path = string(TEST_FOLDER) + string("/") + (begin_train->path()).filename().string();

		//读取图片
		if ((begin_train->path()).filename().string() == ".DS_Store") {
			continue;
		}
		Mat input_pic = imread(train_pic_path);
		cvtColor(input_pic, gray_pic, CV_BGR2GRAY);

		// 提取BOW描述子
		vector<KeyPoint>kp;
		Mat test;
		featureDecter->detect(gray_pic, kp);
		bowDescriptorExtractor->compute(gray_pic, kp, test);
		int sign = 0;
		float best_score = -2.0f;
		for (int i = 0; i < categories_size; i++)
		{
			string cate_na = category_name[i];
			string f_path = string(DATA_FOLDER) + cate_na + string("SVM.xml");
			FileStorage svm_fs(f_path, FileStorage::READ);
			//读取SVM.xml
			if (svm_fs.isOpened())
			{
				svm_fs.release();
				Ptr<SVM> st_svm = Algorithm::load<SVM>(f_path.c_str());
				if (sign == 0)
				{
					float score_Value = st_svm->predict(test, noArray(), true);
					float class_Value = st_svm->predict(test, noArray(), false);
					sign = (score_Value < 0.0f) == (class_Value < 0.0f) ? 1 : -1;
				}
				curConfidence = sign * st_svm->predict(test, noArray(), true);
			}
			else
			{
				if (sign == 0)
				{
					float scoreValue = stor_svms[i]->predict(test, noArray(), true);
					float classValue = stor_svms[i]->predict(test, noArray(), false);
					sign = (scoreValue < 0.0f) == (classValue < 0.0f) ? 1 : -1;
				}
				curConfidence = sign * stor_svms[i]->predict(test, noArray(), true);
			}
			if (curConfidence > best_score)
			{
				best_score = curConfidence;
				prediction_category = cate_na;
			}
		}
		//将图片写入相应的文件夹下
		boost::filesystem::directory_iterator begin_iterater(RESULT_FOLDER);
		boost::filesystem::directory_iterator end_iterator;
		//获取该目录下的文件名
		for (; begin_iterater != end_iterator; ++begin_iterater)
		{

			if (begin_iterater->path().filename().string() == prediction_category)
			{
				string filename = string(RESULT_FOLDER) + prediction_category + string("/") + train_pic_name;
				imwrite(filename, input_pic);
			}
		}
		cout << train_pic_name << "这张图属于:" << prediction_category << endl;
	}
}


int main(void)
{
	int clusters = 1000;
	categorizer c(clusters);
	//特征聚类
	c.bulid_vacab();
	//构造BOW
	c.compute_bow_image();
	//训练分类器
	c.trainSvm();
	//将测试图片分类
	c.category_By_svm();
	system("pause");
	return 0;
}