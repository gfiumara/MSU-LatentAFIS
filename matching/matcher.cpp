/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include <assert.h>

#include <sstream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <memory>
#include <tuple>
#include <vector>
#include <math.h>
#include <chrono>
#include <stdlib.h>
#include "matcher.h"
#include <Eigen/Dense>
//delete this probably
#include <ctime>
#include "boost/filesystem.hpp"
namespace fs = boost::filesystem;

using namespace std;
using namespace Eigen;

namespace PQ
{


Matcher::Matcher(string code_file)
{
    N = 200;
    description.push_back("minutiae similarity");

    description.push_back("obtaining corr");

    description.push_back("second order fast");

    description.push_back("second order original");

    dist_N = 50;
    table_dist.resize(dist_N*dist_N);
    max_nrof_templates = 0;
    int n = 0;
    for(int i=0;i<dist_N; ++i)
    {
        for(int j=i;j<dist_N;++j)
        {
            table_dist[i*dist_N+j] = sqrt((i*16.0)*(i*16.0) +  (j*16.0)*(j*16.0));
            table_dist[j*dist_N+i] = table_dist[i*dist_N+j];
        }
    }
    //    load code book
    ifstream is;
    is.open(code_file, ifstream::binary);
    // get length of file:
    is.seekg(0, ios::end);
    int length = is.tellg();

    if( length<=0 )
    {
        cout<<"codebook is empty!"<<endl;
    }
    is.seekg(0, ios::beg);

    nrof_subs = 0;
    nrof_clusters=0;
    sub_dim = 0;

    is.read(reinterpret_cast<char*>(&nrof_subs),sizeof(short));
    is.read(reinterpret_cast<char*>(&nrof_clusters),sizeof(short));
    is.read(reinterpret_cast<char*>(&sub_dim),sizeof(short));

    int len = nrof_subs*nrof_clusters*sub_dim;
    if(len<=0)
    {
        cout<<"codebook is empty!"<<endl;
    }

    codewords = std::vector<float>(len);
    float *pword = codewords.data();
    for(int i=0;i<nrof_subs; ++i)
    {
        for(int j=0; j<nrof_clusters; ++j)
        {
            is.read(reinterpret_cast<char*>(pword),sizeof(float)*sub_dim);
            pword += sub_dim;
        }
    }
}


int Matcher::List2List_matching(string latent_path, string rolled_path, string score_path)
{
    string template_file, score_file;

    int nrof_latents = 0;
    vector<fs::path> latent_template_files;
    fs::directory_iterator end_itr;
    for(fs::directory_iterator dir_itr(latent_path); dir_itr != end_itr; ++dir_itr)
    {
        if(dir_itr->path().extension() == ".dat")
        {
            latent_template_files.push_back(dir_itr->path());
            cout<<"latent template file"<<dir_itr->path()<<endl;
            ++nrof_latents;
        }
    }
    if(nrof_latents <= 0)
    {
        cout<<"No latent templates found in directory: "<<latent_path<<endl;
        return -1;
    }

    int i,j,k;

    int nrof_rolled = 0;
    vector<fs::path> rolled_template_files;
    for(fs::directory_iterator dir_itr(rolled_path); dir_itr != end_itr; ++dir_itr)
    {
    if(dir_itr->path().extension() == ".dat")
        {
            rolled_template_files.push_back(dir_itr->path());
            cout<<"rolled template file"<<dir_itr->path()<<endl;
            ++nrof_rolled;
        }
    }
    if(nrof_rolled <= 0)
    {
        cout<<"No rolled templates found in directory: "<<rolled_path<<endl;
        return -1;
    }
    cout<<"Gallery size: "<<nrof_rolled<<endl;
    {
        using namespace std::chrono;
        vector<high_resolution_clock::time_point> t(10);
        duration<double, std::milli> time_span;
        t[0] = high_resolution_clock::now();

        for(i=0;i<nrof_latents; ++i)
        {
            vector<float> scores(nrof_rolled, -1);
            cout<<latent_template_files[i]<<endl;

            //load latent original template and create a latent FP object
            LatentFPTemplate latent_FP{};
            try {
	            latent_FP = load_latent_template(latent_template_files[i].string());
	    } catch (const std::exception&) {
	    	cout<<"No minutiae or texture templates found"<<endl;
                ofstream output;
                output.open(score_path + latent_template_files[i].stem().string() + ".csv");

                output<<0<<endl;
                output.close();

                continue;
	    }
            cout<<"Latent minutiae templates: "<<latent_FP.m_minu_templates.size()<<endl;
            cout<<"Latent texture templates: "<<latent_FP.m_texture_templates.size()<<endl;
            if(latent_FP.m_minu_templates.size()<=0 && latent_FP.m_texture_templates.size()<=0)
            {
				cout<<"No minutiae or texture templates found"<<endl;
                ofstream output;
                output.open(score_path + latent_template_files[i].stem().string() + ".csv");

                output<<0<<endl;
                output.close();

                continue;
            }

            using namespace std::chrono;
            high_resolution_clock::time_point t_start = high_resolution_clock::now();
            int result = 0;
	        #pragma omp parallel for num_threads(8) schedule(static,16)
            for(j=0;j<nrof_rolled; ++j)
            {

                RolledFPTemplate rolled_FP;
                try {
                	rolled_FP = load_rolled_template(rolled_template_files[j].string());
                } catch (const std::exception&) {
                    rolled_FP.m_minu_templates.clear();
                    rolled_FP.m_texture_templates.clear();
                }

				vector<float> score;
				result = One2One_matching_selected_templates(latent_FP,rolled_FP,score);
                if(result == 1){
                    continue;
                }
                else if(result == 2){
                    cout<<"Comparison failed: rolled template is empty. Skipping."<<endl;
                    continue;
                }
				float final_score = score[0] + score[1] + score[2] + score[28]*0.3;
                scores[j] = final_score;
            }
            if(result == 1){
                cout<<"Matching failed: latent template is empty. Skipping."<<endl;
                continue;
            }
            auto t_end = high_resolution_clock::now();
            duration<double, std::milli> duration = (t_end - t_start);

			ofstream output;
			output.open(score_path + latent_template_files[i].stem().string() + ".csv");

			for(j=0;j<nrof_rolled; ++j)
			{
                output<<rolled_template_files[j]<<","<<std::setprecision(3)<<std::fixed<<scores[j]<<endl;
			}
			output.close();
        }
        t[1] = high_resolution_clock::now();
        time_span = t[1] - t[0];
        cout<<"Total matching duration (ms): "<<time_span.count()<<endl;

		auto timenow = chrono::system_clock::to_time_t(chrono::system_clock::now());
    }
    return 0;
}

int Matcher::One2List_matching(string latent_template_file_string, string rolled_list_file, string score_path)
{

    int i,j,k;

    fs::path latent_template_file(latent_template_file_string);
    string score_file = score_path + latent_template_file.stem().string() + ".csv";

    int nrof_rolled = 0;
    vector<fs::path> rolled_template_files;
    fs::directory_iterator end_itr;
    for(fs::directory_iterator dir_itr(rolled_list_file); dir_itr != end_itr; ++dir_itr)
    {
    if(dir_itr->path().extension() == ".dat")
        {
            rolled_template_files.push_back(dir_itr->path());
            ++nrof_rolled;
        }
    }
    if(nrof_rolled <= 0)
    {
        cout<<"No rolled templates found in directory: "<<rolled_list_file<<endl;
        return -1;
    }

    // Create a vector of indices
    // Allows us to output a sorted score list for use with the GUI
    vector<int> ind(rolled_template_files.size(), 0);
    for(int n = 0; n != rolled_template_files.size(); n++){
        ind[n] = n;
    }

    {
        using namespace std::chrono;
        vector<high_resolution_clock::time_point> t(10);
        duration<double, std::milli> time_span;
        t[0] = high_resolution_clock::now();

        vector<float> scores(nrof_rolled, -1);
        cout<<"Latent Query: "<<latent_template_file<<endl;
        cout<<"Gallery size: "<<nrof_rolled<<endl;


        LatentFPTemplate latent_FP;
        try {
        	latent_FP = load_latent_template(latent_template_file.string());
        } catch (const std::exception&) {
            ofstream output;
            output.open(score_file);

            output<<0<<endl;
            output.close();
        }
        if(latent_FP.m_minu_templates.size()<=0 && latent_FP.m_texture_templates.size()<=0)
        {
            ofstream output;
            output.open(score_file);

            output<<0<<endl;
            output.close();

        }

        using namespace std::chrono;
        high_resolution_clock::time_point t_start = high_resolution_clock::now();
        int result = 0;
        #pragma omp parallel for num_threads(8) schedule(static,16)
        for(j=0;j<nrof_rolled; ++j)
        {

            RolledFPTemplate rolled_FP;
            try {
            	rolled_FP = load_rolled_template(rolled_template_files[j].string());

            } catch (const std::exception&) {
                rolled_FP.m_minu_templates.clear();
                rolled_FP.m_texture_templates.clear();
            }

            vector<float> score;
            result = One2One_matching_selected_templates(latent_FP,rolled_FP,score);
            if(result == 1){
                continue;
            }
            else if(result == 2){
                cout<<"Comparison failed: rolled template is empty. Skipping."<<endl;
                continue;
            }
            float final_score = score[0] + score[1] + score[2] + score[28]*0.3;
            scores[j] = final_score;
        }
        if(result == 1){
            cout<<"Matching failed: latent template is empty. Exiting."<<endl;
            return 1;
        }
        auto t_end = high_resolution_clock::now();
        duration<double, std::milli> duration = (t_end - t_start);
        ofstream output;
        output.open(score_file);

        // Sort scores to create rank list
        sort(ind.begin(), ind.end(), [&](const int& a, const int& b){
                return (scores[a] > scores[b]);
            }
        );
        output<<"filename,score"<<endl;
        //generate correspondence files for top 24 only
        cout<<"Match Results"<<endl;
        cout<<"----------------"<<endl;
        cout<<"Rank     Filename      Score"<<endl;
        for(j=0; j<24; ++j)
        {
            if(j >= nrof_rolled){
                break;
            }
            output<<to_string(j+1)<<rolled_template_files[ind[j]]<<","<<scores[ind[j]]<<endl;
            RolledFPTemplate rolled_FP;
            try {
	            rolled_FP = load_rolled_template(rolled_template_files[ind[j]].string());
	    } catch (const std::exception&) {
		    rolled_FP.m_minu_templates.clear();
                    rolled_FP.m_texture_templates.clear();
	    }
            string latent_fname = latent_template_file.stem().string();
            string rolled_fname = rolled_template_files[ind[j]].stem().string();
            string corr_file = "/LatentAFIS/scores/corr" + latent_fname + "_" + rolled_fname;
            vector<float> score;
            One2One_matching_selected_templates(latent_FP,rolled_FP,score, true, corr_file);
            cout<<to_string(j+1)<<"        "<<rolled_template_files[ind[j]].filename()<<"       "<<scores[ind[j]]<<endl;
        }
        output.close();
        t[1] = high_resolution_clock::now();
        time_span = t[1] - t[0];
        cout<<"Total matching duration (ms): "<<time_span.count()<<endl;

    }
    return 0;
}

int Matcher::One2One_matching_all_templates(const LatentFPTemplate &latent_template, const RolledFPTemplate &rolled_template, vector<float> & score)
{

    score.resize(latent_template.m_minu_templates.size() + latent_template.m_texture_templates.size());
    std::fill(score.begin(), score.end(), 0);

   if(latent_template.m_minu_templates.size()<=0 && latent_template.m_texture_templates.size()<=0)
   {
        return 1;
    }

    if(rolled_template.m_minu_templates.size()<=0 && rolled_template.m_texture_templates.size()<=0)
    {
        return 2;
    }
    int i,j;

    using namespace std::chrono;
    vector<high_resolution_clock::time_point> t(10);

    t[0] = high_resolution_clock::now();

    for(i=0;i<latent_template.m_minu_templates.size() && rolled_template.m_minu_templates.size(); ++i)
    {
        float s = One2One_minutiae_matching(latent_template.m_minu_templates[i], rolled_template.m_minu_templates[0]);
        score[i] = s;
    }
    t[1] = high_resolution_clock::now();

    for(i=0;i<latent_template.m_texture_templates.size() && rolled_template.m_texture_templates.size()>0 ; ++i)
    {
        float s = One2One_texture_matching(latent_template.m_texture_templates[i], rolled_template.m_texture_templates[0]);
        score[i+latent_template.m_minu_templates.size()] = s;
    }

   return 0;
}

int Matcher::One2One_matching_selected_templates(const LatentFPTemplate &latent_template, const RolledFPTemplate &rolled_template, vector<float> & score, bool save_corr, string corr_file)
const
{
    score.resize(latent_template.m_minu_templates.size() + latent_template.m_texture_templates.size());
    std::fill(score.begin(), score.end(), 0);
    vector<int> selected_ind{27-1, 3-1, 12-1};


    if(latent_template.m_minu_templates.size()<=selected_ind[0] && latent_template.m_texture_templates.size()<=0)
    {
        return 1;
    }

    if(rolled_template.m_minu_templates.size()<=0 && rolled_template.m_texture_templates.size()<=0)
    {
        return 2;
    }
    int i,j;

    using namespace std::chrono;
    vector<high_resolution_clock::time_point> t(10);

    t[0] = high_resolution_clock::now();


    for(i=0;i<selected_ind.size() && rolled_template.m_minu_templates.size()>0; ++i)
    {
        int ind = selected_ind[i];
        if(latent_template.m_minu_templates.size()<=ind)
            continue;
        string one_corr_file = corr_file + "_" + to_string(i) + ".csv";
        float s = One2One_minutiae_matching(latent_template.m_minu_templates[ind], rolled_template.m_minu_templates[0], save_corr, one_corr_file);
        score[i] = s;
    }
    t[1] = high_resolution_clock::now();

    for(i=0;i<min(1ul,latent_template.m_texture_templates.size()) && rolled_template.m_texture_templates.size()>0 ; ++i)
    {
        float s = One2One_texture_matching(latent_template.m_texture_templates[i], rolled_template.m_texture_templates[0]);
        score[i+latent_template.m_minu_templates.size()] = s;
    }

    return 0;
}


float Matcher::One2One_minutiae_matching(const MinutiaeTemplate &latent_minu_template_in, const MinutiaeTemplate &rolled_minu_template_in, bool save_corr, string corr_file)
const
{
	// XXX: Can we avoid the manipulation from eigen below and not dup?
	MinutiaeTemplate latent_minu_template(latent_minu_template_in);
	MinutiaeTemplate rolled_minu_template(rolled_minu_template_in);

    // step 1: compute pairwise similarity between descriptors

    int n_time = 0;
    int i,j,k;

    int des_len = rolled_minu_template.des_length();
    if(des_len!=latent_minu_template.des_length()){
        cout<<latent_minu_template.des_length()<<endl;
	cout<<rolled_minu_template.des_length()<<endl;
	}
    assert(des_len == latent_minu_template.des_length());

    float simi = 0.0;

    using namespace std::chrono;
    vector<high_resolution_clock::time_point> t(10);

    Matrix<float, Eigen::Dynamic, Eigen::Dynamic> aa =  Map<Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(latent_minu_template.m_des.data(),latent_minu_template.m_minutiae.size(),des_len);
    Matrix<float, Eigen::Dynamic, Eigen::Dynamic> bb =  Map<Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(rolled_minu_template.m_des.data(),rolled_minu_template.m_minutiae.size(),des_len);

    MatrixXf  simi_matrix=aa*bb.transpose();

    for(i=0; i<latent_minu_template.m_minutiae.size(); ++i)
    {
        for(j = 0; j<rolled_minu_template.m_minutiae.size(); ++j)
        {
            if(simi_matrix(i,j)<0)
                simi_matrix(i,j) = 0;
        }
    }

    //  step 2:  similarity normalization
    VectorXf  rolled_simi_sum = simi_matrix.colwise().sum();
    VectorXf  latent_simi_sum = simi_matrix.rowwise().sum();

    int ind_1, ind_2 ;
    vector<float> norm_simi_matrix(latent_minu_template.m_minutiae.size()*rolled_minu_template.m_minutiae.size());
    float norm_simi=0.0;
    for(i=0; i<latent_minu_template.m_minutiae.size(); ++i)
    {
        ind_1 = i*rolled_minu_template.m_minutiae.size();
        for(j = 0; j<rolled_minu_template.m_minutiae.size(); ++j)
        {
            ind_2  = ind_1 + j;
            norm_simi = simi_matrix(i,j)/(latent_simi_sum(i) + rolled_simi_sum(j) - simi_matrix(i,j)+0.000001); // //simi_matrix[ind_2]*
            norm_simi_matrix[ind_2] = norm_simi;
        }
    }
    // step 3: find top N correspondences using norm_simi_matrix;
    // the sorting part can be replaced by a min-heap
    std::vector<int> y(norm_simi_matrix.size());
    std::iota(y.begin(), y.end(), 0);
    auto comparator = [&norm_simi_matrix](int a, int b){ return norm_simi_matrix[a] > norm_simi_matrix[b]; };
    std::sort(y.begin(), y.end(), comparator);

    std::vector<tuple<float, int, int>>corr;
    int topN = 120;
    if(rolled_minu_template.m_minutiae.size()*latent_minu_template.m_minutiae.size()<topN)
        topN = rolled_minu_template.m_minutiae.size()*latent_minu_template.m_minutiae.size();
    for(i=0; i<topN ; ++i)
    {
        ind_1 = y[i]/rolled_minu_template.m_minutiae.size(); // latent minutiae  index
        ind_2 = y[i] - ind_1*rolled_minu_template.m_minutiae.size(); // rolled minutiae index
        simi = simi_matrix(ind_1,ind_2);
        corr.push_back(make_tuple(simi,ind_1,ind_2));
    }

     // step 4: remove false correspondences using two graph matching
    int d_thr = 30;
    vector<tuple<float, int, int>> corr2 = LSS_R_Fast2_Dist_eigen(corr, latent_minu_template, rolled_minu_template, d_thr);


    vector<tuple<float, int, int>> corr3  = LSS_R_Fast2(corr2, latent_minu_template, rolled_minu_template, d_thr);

    if (save_corr){
        ofstream output;
        output.open(corr_file);
        for(int i = 0; i < corr3.size(); i++){
            output<<latent_minu_template.m_minutiae[get<1>(corr3[i])].x<<","<<latent_minu_template.m_minutiae[get<1>(corr3[i])].y
                <<","<<rolled_minu_template.m_minutiae[get<2>(corr3[i])].x<<","<<rolled_minu_template.m_minutiae[get<2>(corr3[i])].y<<endl;
        }
        output.close();
    }


    float score = 0.0;

    for(i=0; i<corr3.size(); ++i)
    {
        score += get<0>(corr3[i]);
    }
    return score;

}

int Matcher::One2One_matching(string latent_file, string rolled_file)
{
    LatentFPTemplate latent_template{};
    try {
    	load_latent_template(latent_file);
    } catch (const std::exception&) {
    	return (1);
    }

    RolledFPTemplate rolled_template{};
    try {
	    rolled_template = load_rolled_template(rolled_file);
    } catch (const std::exception&) {
    	return (2);
    }

    vector<float> score;
    int ret = One2One_matching_selected_templates(latent_template, rolled_template, score);

    return ret;
}

float Matcher::One2One_texture_matching(const LatentTextureTemplate &latent_texture_template, const RolledTextureTemplatePQ &rolled_texture_template)
const
{
    // step 1: compute pairwise similarity between descriptors
    int n_time = 0;
    int i,j,k;

    int des_len = rolled_texture_template.des_length();

   std::unique_ptr<float[]> simi_matrix{new float[MaxNRolledMinu*MaxNLatentMinu]};
   memset(simi_matrix.get(),0,MaxNRolledMinu*MaxNLatentMinu*sizeof(float));

    const int numLatentMinutiae = std::min(latent_texture_template.m_minutiae.size(), MaxNLatentMinu);
    const int numExemplarMinutiae = std::min(rolled_texture_template.m_minutiae.size(), MaxNRolledMinu);

    float simi = 0.0;
    float *p_latent_des, *p_latent_des0, *p_rolled_des;

    using namespace std::chrono;
    vector<high_resolution_clock::time_point> t(10);

    t[n_time++] = high_resolution_clock::now();
    float dist0=0.0, dist1= 0.0, dist2= 0, dist3=0.0, dist4 = 0.0; //, dist5, dist6,dist7, dist8;
    int code1 = 0, code2 = 0, code3 = 0, code4=0;
    float *p_dist_codewords0 = NULL, *p_dist_codewords1 = NULL, *p_dist_codewords2 =NULL;

    /* Make a copy of the codewords from the latent texture template */
    std::unique_ptr<float[]> latent_texture_template_m_dist_codewords_copy(new float[latent_texture_template.m_dist_codewords.size()]);
    std::copy(latent_texture_template.m_dist_codewords.begin(), latent_texture_template.m_dist_codewords.end(), latent_texture_template_m_dist_codewords_copy.get());

    int n=0;
    int nrof_clusters3 = nrof_clusters*3, nrof_clusters2 = nrof_clusters*2;
    const int method = 1;
    if(method == 1)
    {
        for(i=0; i<numLatentMinutiae; ++i)
        {
            p_dist_codewords0 = latent_texture_template_m_dist_codewords_copy.get() + i*nrof_subs*nrof_clusters;
            for(j=0; j<numExemplarMinutiae; ++j)
            {
                dist1 = 6.;
                dist2 = 0.;
                dist3 = 0.;
                dist4 = 0.;
                p_dist_codewords1 = p_dist_codewords0;
                const unsigned char *p_des0 = rolled_texture_template.m_desPQ.data() + j* rolled_texture_template.des_length();
                for(k=0; k<nrof_subs; k+=4, p_dist_codewords1+=4*nrof_clusters)
                {
                    code1 = *(p_des0+k);
                    dist1 -= *(p_dist_codewords1 + code1);

                    code2 = *(p_des0+k+1);
                    dist2 -= *(p_dist_codewords1 + code2 + nrof_clusters);

                    code3 = *(p_des0+k+2);
                    dist3 -= *(p_dist_codewords1 + code3 + nrof_clusters2);

                    code4 = *(p_des0+k+3);
                    dist4 -= *(p_dist_codewords1 + code4 + nrof_clusters3);

                }
                simi_matrix[n++] = (dist1+dist2)+ (dist3+dist4);
            }
        }
    }
    else if (method==2)
    {
        int B1=64, B2 = 64;
        for(i=0; i<numLatentMinutiae-B1; i+=B1)
        {

            for(j=0; j<numExemplarMinutiae-B2; j += B2)
            {

                for(int ii=i; ii<i+B1; ++ii)
                {
                    p_dist_codewords0 = latent_texture_template_m_dist_codewords_copy.get() + ii*nrof_subs*nrof_clusters;
                    for(int jj=j; jj<j+B2; ++jj)
                    {
                       dist1 = 6.;
                       dist2 = 0.;
                       dist3 = 0.;
                       dist4 = 0.;
                       p_dist_codewords1 = p_dist_codewords0;
                       const unsigned char *p_des0 = rolled_texture_template.m_desPQ.data() + jj* rolled_texture_template.des_length();


                        for(k=0; k<nrof_subs; k+=4, p_dist_codewords1+=4*nrof_clusters)
                        {
                            code1 = *(p_des0+k);
                            dist1 -= *(p_dist_codewords1 + code1);

                            code2 = *(p_des0+k+1);
                            dist2 -= *(p_dist_codewords1 + code2 + nrof_clusters);

                            code3 = *(p_des0+k+2);
                            dist3 -= *(p_dist_codewords1 + code3 + nrof_clusters2);

                            code4 = *(p_des0+k+3);
                            dist4 -= *(p_dist_codewords1 + code4 + nrof_clusters3);

                        }
                        simi_matrix[ii*numExemplarMinutiae+jj] = (dist1+dist2)+ (dist3+dist4);
                    }
                }
            }
        }
    }
    else if(method == 3)
    {
         int B1=64, B2 = 64;
        for(i=0; i<numLatentMinutiae-B1; i+=B1)
        {
            p_dist_codewords0 = latent_texture_template_m_dist_codewords_copy.get() + i*nrof_subs*nrof_clusters;
            for(j=0; j<numExemplarMinutiae-B2; j += B2)
            {
                p_dist_codewords1 = p_dist_codewords0;
                const unsigned char *p_des0 = rolled_texture_template.m_desPQ.data() + j* rolled_texture_template.des_length();
                for(int ii=i; ii<i+B1; ++ii)
                {
                    const unsigned char *p_des1 = p_des0;
                    for(int jj=j; jj<j+B2; ++jj)
                    {
                       p_dist_codewords2 = p_dist_codewords1;
                       dist1 = 6.;
                       dist2 = 0.;
                       dist3 = 0.;
                       dist4 = 0.;

                        for(k=0; k<nrof_subs; k+=4, p_dist_codewords2+=4*nrof_clusters)
                        {
                            code1 = *(p_des1+k);
                            dist1 -= *(p_dist_codewords2 + code1);

                            code2 = *(p_des1+k+1);
                            dist2 -= *(p_dist_codewords2 + code2 + nrof_clusters);

                            code3 = *(p_des1+k+2);
                            dist3 -= *(p_dist_codewords2 + code3 + nrof_clusters2);

                            code4 = *(p_des1+k+3);
                            dist4 -= *(p_dist_codewords2 + code4 + nrof_clusters3);

                        }
                        simi_matrix[ii*numExemplarMinutiae+jj] = (dist1+dist2)+ (dist3+dist4);
                        p_des1 +=  rolled_texture_template.des_length();
                    }
                    p_dist_codewords1 += nrof_subs*nrof_clusters;
                }
            }
        }
    }
    else if(method==4)
    {
        for(i=0; i<numLatentMinutiae; ++i)
        {
            p_dist_codewords0 = latent_texture_template_m_dist_codewords_copy.get() + i*nrof_subs*nrof_clusters;
            for(k=0; k<nrof_subs; ++k)
            {
                for(j=0; j<numExemplarMinutiae-4; j+=4)
                {
                    n = i*numExemplarMinutiae + j;
                    const unsigned char *p_des0 = rolled_texture_template.m_desPQ.data() + j* rolled_texture_template.des_length() + k;
                    const unsigned char *p_des1 = p_des0 + rolled_texture_template.des_length();
                    const unsigned char *p_des2 = p_des1 + rolled_texture_template.des_length();
                    const unsigned char *p_des3 = p_des2 + rolled_texture_template.des_length();

                    dist0 -= *(p_dist_codewords0 + *p_des0);

                    dist1 -= *(p_dist_codewords0 + *p_des1);

                    dist2 -= *(p_dist_codewords0 + *p_des2);

                    dist3 -= *(p_dist_codewords0 + *p_des3);

                    simi_matrix[n] += dist0;
                    simi_matrix[n+1] += dist1;
                    simi_matrix[n+2] += dist2;
                    simi_matrix[n+3] += dist3;
                }
            }
        }
    }
    t[n_time] = high_resolution_clock::now();
    duration<double, std::milli> time_span = t[n_time] - t[n_time-1];

    vector<float> time(10);
    time[n_time-1]+=time_span.count() ;  // minutiae similarity
    n_time++;

//
    std::vector<tuple<float, int, int>>tmp_corr(numLatentMinutiae), corr(N);
    float max_val;
    float *psimi = simi_matrix.get();
    int max_index;
    for(i=0;i<numLatentMinutiae; ++i)
    {

        max_index = std::distance(psimi, std::max_element(psimi, psimi+numExemplarMinutiae));
        max_val = *(psimi + max_index);
        tmp_corr[i] = make_tuple(max_val,i,max_index);

        psimi += numExemplarMinutiae;
    }
    if(tmp_corr.size()>N)
    {
        std::vector<int> y(tmp_corr.size());
        std::iota(y.begin(), y.end(), 0);
        auto comparator = [&tmp_corr](int a, int b){ return get<0>(tmp_corr[a]) > get<0>(tmp_corr[b]); };
        std::sort(y.begin(), y.end(), comparator);

        for(i=0;i<N; ++i)
        {
            corr[i] = tmp_corr[y[i]];
        }
    }
    else
        corr = tmp_corr;
    t[n_time] = high_resolution_clock::now();

    time_span = t[n_time] - t[n_time-1];

    time[n_time-1]+=time_span.count() ;  // obtaining initial correspondences
    n_time++;

     // step 4: remove false correspondences using two graph matching
    int d_thr = 30;
    vector<tuple<float, int, int>> corr2 = LSS_R_Fast2_Dist_lookup(corr, latent_texture_template, rolled_texture_template, d_thr);


    t[n_time] = high_resolution_clock::now();
    time_span = t[n_time] - t[n_time-1];
    time[n_time-1]+=time_span.count() ;   // second order graph matching: distance
    n_time++;

    vector<tuple<float, int, int>> corr3  = LSS_R_Fast2(corr2, latent_texture_template, rolled_texture_template, d_thr);


    t[n_time] = high_resolution_clock::now();
    time_span = t[n_time] - t[n_time-1];
    time[n_time-1]+=time_span.count() ;   // second order graph matching: original
    n_time++;

    float score = 0.0;

    for(i=0; i<corr3.size(); ++i)
    {
        score += get<0>(corr3[i]);
    }
    return score;

}

LatentFPTemplate Matcher::load_latent_template(const std::string &tname) const
{
	std::ifstream is{tname, ifstream::binary};
	if (!is)
		throw std::runtime_error{"Could not open " + tname};

	is.seekg(0, std::ios_base::end);
	std::vector<uint8_t> buf(is.tellg());
	is.seekg(0, std::ios_base::beg);

	is.read(reinterpret_cast<char *>(buf.data()), buf.size());
	return (load_latent_template(buf));
}

LatentFPTemplate Matcher::load_latent_template(const std::vector<uint8_t> &buf) const
{
    LatentFPTemplate fp_template{};
    /*
     * FIXME: When using dynamic-sized containers, these maximums aren't
     *        necessary. Preserving them for historical reasons (we probably
     *        don't want to search when there's >2000 minutiae anyway).
     */
    static const short Max_Nrof_Minutiae = 2*1000; // including virtual minutiae. We only consider top 1000 minutiae including both real and virtual minutiae for each template.
    static const short Max_Des_Length = 192;
    static const short Max_BlkSize = 100;

    std::istringstream is{
        {reinterpret_cast<const char *>(buf.data()), buf.size()},
        std::istringstream::binary};

    // get length of file:
    is.seekg(0, ios::end);
    int length = is.tellg();

    if( length<=0 )
    {
            throw std::runtime_error{"Length of latent template is 0"};
    }
    is.seekg(0, ios::beg);
    short header[12];
    short h,w,blkH,blkW;
    unsigned char nrof_minu_template,nrof_texture_template;
    short nrof_minutiae;


    short nrof_minutiae_feature;
    short des_len;
    int i,j;

    std::vector<short> x{};
    std::vector<short> y{};
    std::vector<float> ori{};
    std::vector<float> des{};

    for(int i=0; i<12; i++){
        is.read(reinterpret_cast<char*>(&header[i]),sizeof(short));
    }
    /*
     * FIXME: Seen at least once where a template appears corrupt (possibly
     *        because an image was too large? it was a palm), so fail fast
     *        before we trigger a segfault allocating one of the internal
     *        structures.
     */
    static const short expectedHeader[12] = {
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    if (memcmp(header, expectedHeader, 12 * sizeof(short)) != 0) {
    	throw std::runtime_error{"Template appears corrupt (invalid header)"};
    }

    is.read(reinterpret_cast<char*>(&h),sizeof(short));
    is.read(reinterpret_cast<char*>(&w),sizeof(short));
    is.read(reinterpret_cast<char*>(&blkH),sizeof(short));
    is.read(reinterpret_cast<char*>(&blkW),sizeof(short));
    is.read(reinterpret_cast<char*>(&nrof_minu_template),sizeof(unsigned char));
    if(blkH>50)
        blkH = 50;
    if(blkW>50)
        blkW = 50;
    for(i=0;i<nrof_minu_template; ++i)
    {
        is.read(reinterpret_cast<char*>(&nrof_minutiae),sizeof(short));
        if(nrof_minutiae<=0)
            continue;
        if(nrof_minutiae>Max_Nrof_Minutiae)
        {
            throw std::runtime_error{"Number of minutiae is larger than "
                "Max Number of Minutiae (latent): " +
                std::to_string(nrof_minutiae) + ">" +
                std::to_string(Max_Nrof_Minutiae)};
        }
        if(blkH>Max_BlkSize || blkW>Max_BlkSize)
        {
            throw std::runtime_error{"The size of the ridge flow is larger "
                "than maximum size: " + std::to_string(Max_BlkSize)};
        }
        x.resize(nrof_minutiae);
        is.read(reinterpret_cast<char*>(x.data()),sizeof(short)*nrof_minutiae);
        y.resize(nrof_minutiae);
        is.read(reinterpret_cast<char*>(y.data()),sizeof(short)*nrof_minutiae);
        ori.resize(nrof_minutiae);
        is.read(reinterpret_cast<char*>(ori.data()),sizeof(float)*nrof_minutiae);
        is.read(reinterpret_cast<char*>(&des_len),sizeof(short));

        des.resize(nrof_minutiae * des_len);
        is.read(reinterpret_cast<char*>(des.data()),sizeof(float)*nrof_minutiae*des_len);

        MinutiaeTemplate minu_template(nrof_minutiae,x,y,ori,des,blkH, blkW);
        fp_template.add_template(minu_template);
    }

    is.read(reinterpret_cast<char*>(&nrof_texture_template),sizeof(unsigned char));

   for(i=0;i<nrof_texture_template; ++i)
    {
        is.read(reinterpret_cast<char*>(&nrof_minutiae),sizeof(short));
        if(nrof_minutiae<=0)
            continue;
        if(nrof_minutiae>Max_Nrof_Minutiae)
        {
            throw std::runtime_error{"Number of minutiae is larger than Max "
                "Number of Minutiae: " + std::to_string(nrof_minutiae) + ">" +
                std::to_string(Max_Nrof_Minutiae)};
        }
        x.resize(nrof_minutiae);
        is.read(reinterpret_cast<char*>(x.data()),sizeof(short)*nrof_minutiae);
        y.resize(nrof_minutiae);
        is.read(reinterpret_cast<char*>(y.data()),sizeof(short)*nrof_minutiae);
        ori.resize(nrof_minutiae);
        is.read(reinterpret_cast<char*>(ori.data()),sizeof(float)*nrof_minutiae);
        is.read(reinterpret_cast<char*>(&des_len),sizeof(short));
        des.resize(nrof_minutiae * des_len);
        is.read(reinterpret_cast<char*>(des.data()),sizeof(float)*nrof_minutiae*des_len);


        LatentTextureTemplate texture_template(nrof_minutiae,x,y,ori,des);
        texture_template.compute_dist_to_codewords(this->codewords, nrof_subs,  sub_dim,  nrof_clusters);
        fp_template.add_texture_template(texture_template);
    }

    return (fp_template);
}

RolledFPTemplate Matcher::load_rolled_template(const string &tname) const
{
	std::ifstream is{tname, ifstream::binary};
	if (!is)
		throw std::runtime_error{"Could not open " + tname};

	is.seekg(0, std::ios_base::end);
	std::vector<uint8_t> buf(is.tellg());
	is.seekg(0, std::ios_base::beg);

	is.read(reinterpret_cast<char *>(buf.data()), buf.size());
	return (load_rolled_template(buf));
}

RolledFPTemplate Matcher::load_rolled_template(const std::vector<uint8_t> &buf) const
{
    RolledFPTemplate fp_template{};
    /*
     * FIXME: When using dynamic-sized containers, these maximums aren't
     *        necessary. Preserving them for historical reasons (we probably
     *        don't want to search when there's >2000 minutiae anyway).
     */
    static const short Max_Nrof_Minutiae = 2*1000; // including virtual minutiae. We only consider top 1000 minutiae including both real and virtual minutiae for each template.
    static const short Max_Des_Length = 192;
    static const short Max_BlkSize = 100;

    std::istringstream is{
        {reinterpret_cast<const char *>(buf.data()), buf.size()},
        std::istringstream::binary};

    // get length of file:
    is.seekg(0, ios::end);
    int length = is.tellg();

    if( length<=10 )
    {
        throw std::runtime_error{"Size of rolled template is " +
            std::to_string(length) + " (<= 10)"};
    }
    is.seekg(0, ios::beg);
    short header[12];
    short h,w,blkH,blkW;
    unsigned char nrof_minu_template,nrof_texture_template;
    short nrof_minutiae;


    short nrof_minutiae_feature;
    short des_len=96;
    int i,j;

    std::vector<short> x{};
    std::vector<short> y{};
    std::vector<float> ori{};

    std::vector<float> des{};

    for(int i=0; i<12; i++){
        is.read(reinterpret_cast<char*>(&header[i]),sizeof(short));
    }
    /*
     * FIXME: Seen at least once where a template appears corrupt (possibly
     *        because an image was too large? it was a palm), so fail fast
     *        before we trigger a segfault allocating one of the internal
     *        structures.
     */
    static const short expectedHeader[12] = {
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    if (memcmp(header, expectedHeader, 12 * sizeof(short)) != 0) {
    	throw std::runtime_error{"Template appears corrupt (invalid header)"};
    }

    is.read(reinterpret_cast<char*>(&h),sizeof(short));
    is.read(reinterpret_cast<char*>(&w),sizeof(short));
    is.read(reinterpret_cast<char*>(&blkH),sizeof(short));
    is.read(reinterpret_cast<char*>(&blkW),sizeof(short));
    is.read(reinterpret_cast<char*>(&nrof_minu_template),sizeof(unsigned char));
    if(blkH>50)
        blkH = 50;
    if(blkW>50)
        blkW = 50;
    for(i=0;i<nrof_minu_template; ++i)
    {
        is.read(reinterpret_cast<char*>(&nrof_minutiae),sizeof(short));
        if(nrof_minutiae<=0)
            continue;
        if(nrof_minutiae>Max_Nrof_Minutiae)
        {
            throw std::runtime_error{"Number of minutiae is larger than Max "
                "Number of Minutiae: " + std::to_string(nrof_minutiae) + ">" +
                std::to_string(Max_Nrof_Minutiae)};
        }
        if(blkH>Max_BlkSize || blkW>Max_BlkSize)
        {
            throw std::runtime_error{"The size of the ridge flow is larger "
                "than maximum size: " + std::to_string(Max_BlkSize)};
        }
        x.resize(nrof_minutiae);
        is.read(reinterpret_cast<char*>(x.data()),sizeof(short)*nrof_minutiae);
        y.resize(nrof_minutiae);
        is.read(reinterpret_cast<char*>(y.data()),sizeof(short)*nrof_minutiae);
        ori.resize(nrof_minutiae);
        is.read(reinterpret_cast<char*>(ori.data()),sizeof(float)*nrof_minutiae);
        is.read(reinterpret_cast<char*>(&des_len),sizeof(short));

        des.resize(nrof_minutiae * des_len);
        is.read(reinterpret_cast<char*>(des.data()),sizeof(float)*nrof_minutiae*des_len);

        MinutiaeTemplate minu_template(nrof_minutiae,x,y,ori,des,blkH, blkW);
        fp_template.add_template(minu_template);
    }

    is.read(reinterpret_cast<char*>(&nrof_texture_template),sizeof(unsigned char));

   for(i=0;i<nrof_texture_template; ++i)
    {
        is.read(reinterpret_cast<char*>(&nrof_minutiae),sizeof(short));
        if(nrof_minutiae<=0)
            continue;
        if(nrof_minutiae>Max_Nrof_Minutiae)
        {
            throw std::runtime_error{"Number of minutiae is larger than Max "
                "Number of Minutiae: " + std::to_string(nrof_minutiae) + ">" +
                std::to_string(Max_Nrof_Minutiae)};
        }
        x.resize(nrof_minutiae);
        is.read(reinterpret_cast<char*>(x.data()),sizeof(short)*nrof_minutiae);
        y.resize(nrof_minutiae);
        is.read(reinterpret_cast<char*>(y.data()),sizeof(short)*nrof_minutiae);
        ori.resize(nrof_minutiae);
        is.read(reinterpret_cast<char*>(ori.data()),sizeof(float)*nrof_minutiae);
        is.read(reinterpret_cast<char*>(&des_len),sizeof(short));
        des.resize(nrof_minutiae * des_len);
        is.read(reinterpret_cast<char*>(des.data()),sizeof(float)*nrof_minutiae*des_len);

        RolledTextureTemplatePQ texture_template(nrof_minutiae,x,y,ori,des);
        fp_template.add_texture_template(texture_template);
    }

    return (fp_template);
}

vector<tuple<float, int, int>>  Matcher::LSS_R_Fast2_Dist(vector<tuple<float, int, int>> &corr, const SingleTemplate & latent_template, const SingleTemplate & rolled_template, float d_thr)
const
{
    int num = corr.size();
    vector<float> H(num*num);

    vector<short> flag_latent(latent_template.m_minutiae.size()),flag_rolled(rolled_template.m_minutiae.size());

    int i,j,k;

    float dist_1, dist_2, dist;
    float dx_1, dy_1, dx_2, dy_2;

    for(i=0; i<num-1; ++i)
    {
        const MinuPoint p_latent_minutia_1 = latent_template.m_minutiae[get<1>(corr[i])];
        const MinuPoint p_rolled_minutia_1 = rolled_template.m_minutiae[get<2>(corr[i])];
        for(j=i+1; j<num;++j)
        {
            const MinuPoint p_latent_minutia_2 = latent_template.m_minutiae[get<1>(corr[j])];
            const MinuPoint p_rolled_minutia_2 = rolled_template.m_minutiae[get<2>(corr[j])];

            dx_1 = p_latent_minutia_1.x-p_latent_minutia_2.x;
            dx_2 = p_rolled_minutia_1.x-p_rolled_minutia_2.x;

            dy_1 = p_latent_minutia_1.y-p_latent_minutia_2.y;
            dy_2 = p_rolled_minutia_1.y-p_rolled_minutia_2.y;

            dist_1 = (dx_1*dx_1)+(dy_1*dy_1);
            dist_1 = sqrt(dist_1);


            dist_2 = (dx_2*dx_2)+(dy_2*dy_2);
            dist_2 = sqrt(dist_2);

            dist = fabs(dist_1-dist_2);

            H[i*num+j] = (30-dist)/(25.0);
            if(H[i*num+j]>1)
                H[i*num+j] = 1.0;
            else if(H[i*num+j]<0)
                H[i*num+j] = 0.0;

            H[j*num+i] = H[i*num+j];
        }
    }

    vector<float> S(num),S1(num);

    float s0 = 1.0/num;
    for(i=0; i<num; ++i)
        S[i] = get<0>(corr[i]);


    float sum = 0.0;
    for(i=0;i<5 ; ++i)
    {
        sum = 0.0;
        for(j=0;j<num; ++j)
        {
            S1[j] = 0;
            for(k=0; k<num;++k)
            {
                //if(H[j*num+k])
                S1[j] += H[j*num+k]*S[k];
            }
            sum += S1[j];
        }
        sum = 1.0/(sum+0.0001);
        for(j=0;j<num; ++j)
        {
            S[j] = S1[j]*sum;
        }
    }

    // sort the S
    // the sorting part can be replaced by a min-heap
    std::vector<int> y(S.size());
    std::iota(y.begin(), y.end(), 0);
    auto comparator = [&S](int a, int b){ return S[a] > S[b]; };
    std::sort(y.begin(), y.end(), comparator);

    vector<tuple<float, int, int>>  new_corr;
    vector<int>  selected_ind;
    short ind;
    for(i=0; i<num; ++i)
    {
        ind = y[i];
        if(S[ind]<0.0001)
            break;
        if((flag_latent[get<1>(corr[ind])] == 1) || (flag_rolled[get<2>(corr[ind])] == 1))
            continue;

        if(i==0)
        {
            selected_ind.push_back(ind);
            new_corr.push_back(make_tuple(get<0>(corr[ind]),get<1>(corr[ind]),get<2>(corr[ind])));

            flag_latent[get<1>(corr[ind])] = 1;
            flag_rolled[get<2>(corr[ind])] = 1;
        }
        else
        {
            int found =0;
            for(j=0;j<selected_ind.size(); ++j)
            {
                if(H[ind*num+selected_ind[j]]<0.00001)
                {
                    found = 1;
                    break;
                }
            }
            if(found==0)
            {
                selected_ind.push_back(ind);
                new_corr.push_back(make_tuple(get<0>(corr[ind]),get<1>(corr[ind]),get<2>(corr[ind])));

                flag_latent[get<1>(corr[ind])] = 1;
                flag_rolled[get<2>(corr[ind])] = 1;
            }
        }
    }

    return new_corr;
}

vector<tuple<float, int, int>>  Matcher::LSS_R_Fast2_Dist_lookup(vector<tuple<float, int, int>> &corr, const SingleTemplate & latent_template, const SingleTemplate & rolled_template, float d_thr)
const
{
    int num = corr.size();
    std::unique_ptr<float[]> H{new float [num*num]()};
    vector<short> flag_latent(latent_template.m_minutiae.size()),flag_rolled(rolled_template.m_minutiae.size());

    int i,j,k;

    float dist_1, dist_2, dist;
    int  dx_1, dy_1, dx_2, dy_2;

    for(i=0; i<num-1; ++i)
    {
        const MinuPoint p_latent_minutia_1 = latent_template.m_minutiae[get<1>(corr[i])];
        const MinuPoint p_rolled_minutia_1 = rolled_template.m_minutiae[get<2>(corr[i])];
        for(j=i+1; j<num;++j)
        {
            const MinuPoint p_latent_minutia_2 = latent_template.m_minutiae[get<1>(corr[j])];
            const MinuPoint p_rolled_minutia_2 = rolled_template.m_minutiae[get<2>(corr[j])];

            dx_1 = p_latent_minutia_1.x-p_latent_minutia_2.x;
            dx_2 = p_rolled_minutia_1.x-p_rolled_minutia_2.x;

            dx_1 = abs(dx_1);
            dx_2 = abs(dx_2);
            dy_1 = p_latent_minutia_1.y-p_latent_minutia_2.y;
            dy_2 = p_rolled_minutia_1.y-p_rolled_minutia_2.y;

            dy_1 = abs(dy_1);
            dy_2 = abs(dy_2);

            if((dx_1>=dist_N) || (dx_2>=dist_N) || (dy_1>=dist_N) || (dy_2>=dist_N))
                continue;

            dist_1 = table_dist[dx_1*dist_N+dy_1];

            dist_2 = table_dist[dx_2*dist_N+dy_2];

            dist = fabs(dist_1-dist_2);
            if(dist>d_thr)
                continue;

            H[i*num+j] = (30-dist)/(25.0);
            if(H[i*num+j]>1)
                H[i*num+j] = 1.0;
            else if(H[i*num+j]<0)
                H[i*num+j] = 0.0;
            H[j*num+i] = H[i*num+j];
        }
    }

    Matrix<float, Eigen::Dynamic, Eigen::Dynamic> aa =  Map<Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(H.get(),num,num);

    float sum = 0.0;
    VectorXf b(num);
    VectorXf c;
    for(i=0; i<num; ++i)
        b(i) = get<0>(corr[i]);
    for(i=0;i<3 ; ++i)
    {
        c = aa*b;
        sum = c.sum();
        b = c*(1./(sum+0.00001));
    }

    vector<float> S(num);
    for(i=0;i<num; ++i)
    {
        S[i] = b(i);
    }

    // sort S
    std::vector<int> y(S.size());
    std::iota(y.begin(), y.end(), 0);
    auto comparator = [&S](int a, int b){ return S[a] > S[b]; };
    std::sort(y.begin(), y.end(), comparator);


    vector<tuple<float, int, int>>  new_corr;
    vector<int>  selected_ind;
    short ind;
    for(i=0; i<num; ++i)
    {

        ind = y[i];
        if(S[ind]<0.0001)
            break;
        if((flag_latent[get<1>(corr[ind])] == 1) || (flag_rolled[get<2>(corr[ind])] == 1))
            continue;

        if(i==0)
        {
            selected_ind.push_back(ind);
            new_corr.push_back(make_tuple(get<0>(corr[ind]),get<1>(corr[ind]),get<2>(corr[ind])));

            flag_latent[get<1>(corr[ind])] = 1;
            flag_rolled[get<2>(corr[ind])] = 1;
        }
        else
        {
            int found = 0;
            for(j=0;j<selected_ind.size(); ++j)
            {
                if(H[ind*num+selected_ind[j]]<0.00001)
                {
                    found = 1;
                    break;
                }
            }
            if(found==0)
            {
                selected_ind.push_back(ind);
                new_corr.push_back(make_tuple(get<0>(corr[ind]),get<1>(corr[ind]),get<2>(corr[ind])));

                flag_latent[get<1>(corr[ind])] = 1;
                flag_rolled[get<2>(corr[ind])] = 1;
            }
        }
    }

    return new_corr;
}

vector<tuple<float, int, int>>  Matcher::LSS_R_Fast2_Dist_eigen(vector<tuple<float, int, int>> &corr, const SingleTemplate & latent_template, const SingleTemplate & rolled_template, float d_thr)
const
{
    int num = corr.size();
    std::unique_ptr<float[]> H{new float [num*num]()};

    vector<short> flag_latent(latent_template.m_minutiae.size()),flag_rolled(rolled_template.m_minutiae.size());

    int i,j,k;

    MinuPoint *p_latent_minutia_1, *p_latent_minutia_2, *p_rolled_minutia_1, *p_rolled_minutia_2;
    float dist_1, dist_2, dist;
    float dx_1, dy_1, dx_2, dy_2;

    for(i=0; i<num-1; ++i)
    {
        const MinuPoint p_latent_minutia_1 = latent_template.m_minutiae[get<1>(corr[i])];
        const MinuPoint p_rolled_minutia_1 = rolled_template.m_minutiae[get<2>(corr[i])];
        for(j=i+1; j<num;++j)
        {
            const MinuPoint p_latent_minutia_2 = latent_template.m_minutiae[get<1>(corr[j])];
            const MinuPoint p_rolled_minutia_2 = rolled_template.m_minutiae[get<2>(corr[j])];

            dx_1 = p_latent_minutia_1.x-p_latent_minutia_2.x;
            dx_2 = p_rolled_minutia_1.x-p_rolled_minutia_2.x;


            dy_1 = p_latent_minutia_1.y-p_latent_minutia_2.y;
            dy_2 = p_rolled_minutia_1.y-p_rolled_minutia_2.y;


            dist_1 = (dx_1*dx_1)+(dy_1*dy_1);
            dist_1 = sqrt(dist_1);

            dist_2 = (dx_2*dx_2)+(dy_2*dy_2);
            dist_2 = sqrt(dist_2);
            dist = fabs(dist_1-dist_2);
           if(dist>d_thr)
                continue;

            H[i*num+j] = (30-dist)/(25.0);
            if(H[i*num+j]>1)
                H[i*num+j] = 1.0;
            else if(H[i*num+j]<0)
                H[i*num+j] = 0.0;

            H[j*num+i] = H[i*num+j];
        }
    }

    Matrix<float, Eigen::Dynamic, Eigen::Dynamic> aa =  Map<Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(H.get(),num,num);

    float sum = 0.0;
    VectorXf b(num);
    VectorXf c;
    for(i=0; i<num; ++i)
        b(i) = get<0>(corr[i]);
    for(i=0;i<5 ; ++i)
    {
        c = aa*b;
        sum = c.sum();
        b = c*(1./(sum+0.00001));
    }

    vector<float> S(num);
    for(i=0;i<num; ++i)
    {
        S[i] = b(i);
    }
    // sort the S
    // the sorting part can be replaced by a min-heap
    std::vector<int> y(S.size());
    std::iota(y.begin(), y.end(), 0);
    auto comparator = [&S](int a, int b){ return S[a] > S[b]; };
    std::sort(y.begin(), y.end(), comparator);

    vector<tuple<float, int, int>>  new_corr;
    vector<int>  selected_ind;
    short ind;
    for(i=0; i<num; ++i)
    {

        ind = y[i];
        if(S[ind]<0.0001)
            break;
        if((flag_latent[get<1>(corr[ind])] == 1) || (flag_rolled[get<2>(corr[ind])] == 1))
            continue;

        if(i==0)
        {
            selected_ind.push_back(ind);
            new_corr.push_back(make_tuple(get<0>(corr[ind]),get<1>(corr[ind]),get<2>(corr[ind])));

            flag_latent[get<1>(corr[ind])] = 1;
            flag_rolled[get<2>(corr[ind])] = 1;
        }
        else
        {
            int found =0;
            for(j=0;j<selected_ind.size(); ++j)
            {
                if(H[ind*num+selected_ind[j]]<0.00001)
                {
                    found = 1;
                    break;
                }
            }
            if(found==0)
            {
                selected_ind.push_back(ind);
                new_corr.push_back(make_tuple(get<0>(corr[ind]),get<1>(corr[ind]),get<2>(corr[ind])));

                flag_latent[get<1>(corr[ind])] = 1;
                flag_rolled[get<2>(corr[ind])] = 1;
            }
        }
    }

    return new_corr;
}

vector<tuple<float, int, int>>  Matcher::LSS_R_Fast2(vector<tuple<float, int, int>> &corr, const SingleTemplate & latent_template, const SingleTemplate & rolled_template, int d_thr)
const
{
    int num = corr.size();
    vector<bool> H(num*num);
    vector<short> flag_latent(latent_template.m_minutiae.size()),flag_rolled(rolled_template.m_minutiae.size());

    int i,j,k;

    float dist_1, dist_2;
    float angle_1, angle_2, angle_diff;
    float line_angle_1, line_angle_2;
    float dx_1,dx_2,dy_1,dy_2;


    for(i=0; i<num-1; ++i)
    {
        const MinuPoint p_latent_minutia_1 = latent_template.m_minutiae[get<1>(corr[i])];
        const MinuPoint p_rolled_minutia_1 = rolled_template.m_minutiae[get<2>(corr[i])];
        for(j=i+1; j<num;++j)
        {
            const MinuPoint p_latent_minutia_2 = latent_template.m_minutiae[get<1>(corr[j])];
            const MinuPoint p_rolled_minutia_2 = rolled_template.m_minutiae[get<2>(corr[j])];

            angle_1 = p_latent_minutia_1.ori-p_latent_minutia_2.ori;
            angle_1 = adjust_angle(angle_1);

            angle_2 = p_rolled_minutia_1.ori-p_rolled_minutia_2.ori;
            angle_2 = adjust_angle(angle_2);

            angle_diff = fabs(angle_1 - angle_2);

            if(angle_diff>PI)
               angle_diff = 2*PI - angle_diff;



            if(angle_diff>PI/4.)
                continue;

            dx_1 = p_latent_minutia_1.x-p_latent_minutia_2.x;
            dy_1 = p_latent_minutia_1.y-p_latent_minutia_2.y;



            line_angle_1 = -atan2(dy_1,dx_1);
            angle_1 = p_latent_minutia_1.ori - line_angle_1;
            angle_1 = adjust_angle(angle_1);


            dx_2 = p_rolled_minutia_1.x-p_rolled_minutia_2.x;
            dy_2 = p_rolled_minutia_1.y-p_rolled_minutia_2.y;

            line_angle_2 = -atan2(dy_2,dx_2);
            angle_2 = p_rolled_minutia_1.ori - line_angle_2;
            angle_2 = adjust_angle(angle_2);

            angle_diff = fabs(angle_1 - angle_2);

            if(angle_diff>PI)
               angle_diff = 2*PI - angle_diff;
            if(angle_diff>PI/6.)
                continue;



            angle_1 = p_latent_minutia_2.ori - line_angle_1;
            angle_1 = adjust_angle(angle_1);


            angle_2 = p_rolled_minutia_2.ori - line_angle_2;
            angle_2 = adjust_angle(angle_2);

            angle_diff = fabs(angle_1 - angle_2);

            if(angle_diff>PI)
               angle_diff = 2*PI - angle_diff;
            if(angle_diff>PI/6.)
                continue;

            H[i*num+j] = true;
            H[j*num+i] = true;
        }
    }

    vector<float> S(num),S1(num);

    float s0 = 1.0/num;
    for(i=0; i<num; ++i)
        S[i] = s0;

    float sum = 0.0;
    for(i=0;i<5 ; ++i)
    {
        sum = 0.0;
        for(j=0;j<num; ++j)
        {
            S1[j] = 0;
            for(k=0; k<num;++k)
            {
                if(H[j*num+k])
                   S1[j] += S[k];
            }
            sum += S1[j];
        }
        sum = 1.0/(sum+0.00001);
        for(j=0;j<num; ++j)
        {
            S[j] = S1[j]*sum;
        }
    }

    s0 = 0.0;

    // sort the S
    // the sorting part can be replaced by a min-heap
    std::vector<int> y(S.size());
    std::iota(y.begin(), y.end(), 0);
    auto comparator = [&S](int a, int b){ return S[a] > S[b]; };
    std::sort(y.begin(), y.end(), comparator);


    vector<tuple<float, int, int>>  new_corr;
    vector<int> selected_ind;
    short ind;
    for(i=0; i<num; ++i)
    {
        ind = y[i];
        if(S[ind]<0.001)
            break;
        if((flag_latent[get<1>(corr[ind])] == 1) || (flag_rolled[get<2>(corr[ind])] == 1))
            continue;

          if(i==0)
        {
            selected_ind.push_back(ind);
            new_corr.push_back(make_tuple(get<0>(corr[ind]),get<1>(corr[ind]),get<2>(corr[ind])));

            flag_latent[get<1>(corr[ind])] = 1;
            flag_rolled[get<2>(corr[ind])] = 1;
        }
        else
        {
            int found =0;
            for(j=0;j<selected_ind.size(); ++j)
            {
                if(!H[ind*num+selected_ind[j]])
                {
                    found = 1;
                    break;
                }
            }
            if(found==0)
            {
                selected_ind.push_back(ind);
                new_corr.push_back(make_tuple(get<0>(corr[ind]),get<1>(corr[ind]),get<2>(corr[ind])));

                flag_latent[get<1>(corr[ind])] = 1;
                flag_rolled[get<2>(corr[ind])] = 1;
            }

        }
    }

    return new_corr;
}

float Matcher::adjust_angle(float angle)
const
{
    if(angle>PI)
        angle -= 2*PI;
    else if (angle<-PI)
    {
        angle += 2*PI;
    }
    return angle;
}
}
