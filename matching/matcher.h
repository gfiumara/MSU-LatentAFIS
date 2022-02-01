/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   matcher_int.h
 * Author: cori
 *
 * Created on November 4, 2018, 4:49 PM
 */

#ifndef MATCHER_INT_H
#define MATCHER_INT_H
#include <cstdlib>
#include <type_traits>
#include <string>
#include <iostream>
#include <vector>
#include <chrono>
#include"include.h"

using namespace std;
using namespace std::chrono;
#define PI 3.1415926
#define EIGEN_DONT_PARALLELIZE

namespace PQ
{
    const int MaxNRolledMinu = 1000;
    const int MaxNLatentMinu = 1000;

class Matcher {


public:
    Matcher(string code_file);
    int List2List_matching(string latent_list_file, string rolled_list_file, string output_file);
    int One2List_matching(string latent_template_file, string rolled_list_file, string score_file);
    float One2One_minutiae_matching(const MinutiaeTemplate  &latent_minu_template, const MinutiaeTemplate  &rolled_minu_template, bool save_corr = false, string corr_file = "") const;
    float One2One_texture_matching(const LatentTextureTemplate &latent_texture_template, const RolledTextureTemplatePQ &rolled_minu_template) const;
    int One2One_matching_selected_templates(const LatentFPTemplate &latent_template, const RolledFPTemplate &rolled_template, vector<float> & score, bool save_corr = false, string corr_file = "") const;
    int One2One_matching_all_templates(const LatentFPTemplate &latent_template, const RolledFPTemplate &rolled_template, vector<float> & score);
    int One2One_matching(string latent_file, string rolled_file);
    int load_single_template(string tname, TextureTemplate& texture_template);
    int load_FP_template(string tname, LatentFPTemplate & fp_template);
    int load_FP_template(const std::vector<uint8_t> &buf, LatentFPTemplate & fp_template) const;
    int load_FP_template(string tname, RolledFPTemplate & fp_template);
    int load_FP_template(const std::vector<uint8_t> &buf, RolledFPTemplate & fp_template) const;
    int load_single_PQ_template(string tname, RolledTextureTemplatePQ& minu_template);
    Matcher(const Matcher& orig);
    virtual ~Matcher();

private:
    vector<tuple<float, int, int>>  LSS_R_Fast2_Dist(vector<tuple<float, int, int>> &corr, const SingleTemplate & latent_template,const SingleTemplate & rolled_template, float d_thr=30.0) const;
    vector<tuple<float, int, int>>  LSS_R_Fast2_Dist_eigen(vector<tuple<float, int, int>> &corr, const SingleTemplate & latent_template, const SingleTemplate & rolled_template, float d_thr = 30.0) const;
    vector<tuple<float, int, int>>  LSS_R_Fast2_Dist_lookup(vector<tuple<float, int, int>> &corr, const SingleTemplate & latent_template, const SingleTemplate & rolled_template, float d_thr = 30.0) const;
    vector<tuple<float, int, int>>  LSS_R_Fast2(vector<tuple<float, int, int>> &corr, const SingleTemplate & latent_template, const SingleTemplate & rolled_template, int d_thr=3) const;
    float adjust_angle(float angle) const;

private:
    int N; // top N minutiae correspondences for matching

    vector<string> description;
    vector<float> table_dist;
    int dist_N;
    int max_nrof_templates;
    int nrof_subs;
    int nrof_clusters;
    int sub_dim;
    float *codewords;
    float a;
};
}
#endif /* MATCHER_H */
